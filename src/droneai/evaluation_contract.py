from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Literal, Protocol

import numpy as np

from droneai.integrity import is_sha256

if TYPE_CHECKING:
    from droneai.model_brief import ModelBrief

OutputType = Literal["count", "density", "points", "hybrid"]
Point = tuple[float, float]


@dataclass(frozen=True)
class ZoneBox:
    zone_id: str
    x0: float
    y0: float
    x1: float
    y1: float

    def __post_init__(self) -> None:
        if not self.zone_id or self.x1 <= self.x0 or self.y1 <= self.y0:
            raise ValueError("zone requires an id and positive rectangle")

    def contains(self, x: float, y: float) -> bool:
        return self.x0 <= x < self.x1 and self.y0 <= y < self.y1


@dataclass(frozen=True)
class EvaluationSample:
    sample_id: str
    dataset_id: str
    split_id: str
    image_path: Path
    source_sha256: str
    width: int
    height: int
    ground_truth_count: float
    annotation_sha256: str | None = None
    ground_truth_points: tuple[Point, ...] = ()
    ground_truth_density: np.ndarray | None = field(default=None, compare=False)
    condition_tags: dict[str, str] = field(default_factory=dict, compare=False)
    zones: tuple[ZoneBox, ...] = ()
    has_point_annotations: bool = False

    def __post_init__(self) -> None:
        if not self.sample_id or not self.dataset_id or not self.split_id:
            raise ValueError("sample identity is required")
        if not is_sha256(self.source_sha256):
            raise ValueError("sample source SHA-256 is required")
        if self.annotation_sha256 is not None and not is_sha256(self.annotation_sha256):
            raise ValueError("sample annotation SHA-256 must be valid when provided")
        if self.width <= 0 or self.height <= 0:
            raise ValueError("sample dimensions must be positive")
        if any(zone.x0 < 0 or zone.y0 < 0 or zone.x1 > self.width or zone.y1 > self.height for zone in self.zones):
            raise ValueError("sample zones must stay inside image bounds")
        if not math.isfinite(self.ground_truth_count) or self.ground_truth_count < 0:
            raise ValueError("ground-truth count must be finite and non-negative")
        if self.ground_truth_points and not self.has_point_annotations:
            raise ValueError("point coordinates require point-annotation status")
        if any(not (0 <= x < self.width and 0 <= y < self.height) for x, y in self.ground_truth_points):
            raise ValueError("ground-truth points must stay inside image bounds")
        if self.has_point_annotations and (
            not math.isclose(self.ground_truth_count, round(self.ground_truth_count), abs_tol=1e-6)
            or len(self.ground_truth_points) != round(self.ground_truth_count)
        ):
            raise ValueError("point count must equal ground-truth count")
        if self.ground_truth_density is not None:
            density = np.asarray(self.ground_truth_density)
            if density.ndim != 2 or not np.isfinite(density).all():
                raise ValueError("ground-truth density must be a finite 2D array")
            tolerance = max(1e-4, 1e-5 * max(self.ground_truth_count, 1.0))
            if abs(float(density.sum()) - self.ground_truth_count) > tolerance:
                raise ValueError("ground-truth density sum must preserve ground-truth count")


@dataclass(frozen=True)
class NativePrediction:
    sample_id: str
    output_type: OutputType
    predicted_count: float | None
    latency_ms: float
    peak_vram_mb: float
    density: np.ndarray | None = field(default=None, compare=False)
    points: tuple[Point, ...] = ()
    point_confidences: tuple[float, ...] = ()
    confidence: float | None = None
    failure_state: str | None = None
    coordinate_space: str = "original_pixels"
    metadata: dict[str, str | int | float | bool | None] = field(
        default_factory=dict, compare=False
    )

    def __post_init__(self) -> None:
        if not self.sample_id:
            raise ValueError("prediction sample identity is required")
        if (
            not math.isfinite(self.latency_ms)
            or not math.isfinite(self.peak_vram_mb)
            or self.latency_ms <= 0
            or self.peak_vram_mb < 0
        ):
            raise ValueError("runtime values must be non-negative and latency positive")
        if any(
            not isinstance(key, str)
            or not key.strip()
            or not isinstance(value, (str, int, float, bool, type(None)))
            or (isinstance(value, float) and not math.isfinite(value))
            for key, value in self.metadata.items()
        ):
            raise ValueError("prediction metadata requires named finite JSON scalars")
        if self.failure_state:
            if self.predicted_count is not None or self.density is not None or self.points:
                raise ValueError("failed predictions cannot contain a count or native output")
            return
        if self.predicted_count is None or not math.isfinite(self.predicted_count) or self.predicted_count < 0:
            raise ValueError("successful predictions require a finite non-negative count")
        if self.density is not None:
            density = np.asarray(self.density)
            if density.ndim != 2 or not np.isfinite(density).all():
                raise ValueError("density output must be a finite 2D density map")
            tolerance = max(1e-4, 1e-5 * max(self.predicted_count, 1.0))
            if abs(float(density.sum()) - self.predicted_count) > tolerance:
                raise ValueError("predicted density sum must preserve predicted count")
        if self.point_confidences and len(self.point_confidences) != len(self.points):
            raise ValueError("point confidences must align with points")
        if self.output_type == "density" and self.density is None:
            raise ValueError("density output type requires a density map")
        if self.output_type == "points" and abs(self.predicted_count - len(self.points)) > 1e-6:
            raise ValueError("point output count must equal predicted point count")
        if self.output_type == "hybrid" and self.density is None and not self.points:
            raise ValueError("hybrid output requires density or points")


@dataclass(frozen=True)
class ScalarEvaluation:
    sample_id: str
    ground_truth_count: float
    predicted_count: float | None
    signed_error: float | None
    absolute_error: float | None
    normalized_error: float | None
    density_band: str
    latency_ms: float
    peak_vram_mb: float
    output_type: OutputType
    failure_state: str | None
    spatial_metric_name: str | None = None
    spatial_metric_value: float | None = None
    extra_metrics: dict[str, float] = field(default_factory=dict, compare=False)
    condition_values: dict[str, str] = field(default_factory=dict, compare=False)


class ModelAdapter(Protocol):
    def brief(self) -> "ModelBrief":
        raise NotImplementedError

    def predict(self, sample: EvaluationSample, *, retain_native: bool) -> NativePrediction:
        raise NotImplementedError
