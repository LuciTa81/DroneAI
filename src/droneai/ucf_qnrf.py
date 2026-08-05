"""Test-isolated UCF-QNRF indexing and deterministic smoke selection."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
from PIL import Image
from scipy.io import loadmat

from droneai.evaluation_contract import EvaluationSample, Point, ZoneBox
from droneai.integrity import sha256_file

DATASET_ID = "ucf-qnrf-kaggle-apache"
DEFAULT_DENSITY_THRESHOLDS = (500, 1500)


@dataclass(frozen=True)
class UCFQNRFRecord:
    sample_id: str
    image_path: Path
    annotation_path: Path
    image_sha256: str
    annotation_sha256: str
    width: int
    height: int
    points: tuple[Point, ...]
    density_band: str
    normalization_corrections: int = 0

    @property
    def count(self) -> int:
        return len(self.points)


def read_ucf_qnrf_points(annotation_path: str | Path) -> tuple[Point, ...]:
    """Read the public UCF-QNRF ``annPoints`` matrix without opening images."""

    path = Path(annotation_path)
    payload = loadmat(path)
    if "annPoints" not in payload:
        raise ValueError(f"UCF-QNRF annotation has no annPoints matrix: {path}")
    points = np.asarray(payload["annPoints"], dtype=np.float64)
    if points.size == 0:
        return ()
    if points.ndim != 2 or points.shape[1] != 2 or not np.isfinite(points).all():
        raise ValueError(f"UCF-QNRF annPoints must be a finite Nx2 matrix: {path}")
    return tuple((float(x), float(y)) for x, y in points)


def density_band(
    count: int, thresholds: tuple[int, int] = DEFAULT_DENSITY_THRESHOLDS
) -> str:
    low_max, medium_max = thresholds
    if low_max <= 0 or medium_max <= low_max:
        raise ValueError("density thresholds must be increasing positive integers")
    if count < low_max:
        return "low"
    if count < medium_max:
        return "medium"
    return "high"


def _annotation_path(image_path: Path, annotation_root: Path) -> Path:
    candidates = (
        annotation_root / f"{image_path.stem}_ann.mat",
        annotation_root / f"{image_path.stem}.mat",
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"annotation missing for {image_path.name}")


def _normalized_points(
    points: Sequence[Point], *, width: int, height: int
) -> tuple[tuple[Point, ...], int]:
    if width <= 0 or height <= 0:
        raise ValueError("image dimensions must be positive")
    normalized: list[Point] = []
    corrections = 0
    max_x = math.nextafter(float(width), 0.0)
    max_y = math.nextafter(float(height), 0.0)
    for x, y in points:
        if x < 0.0 or x > width or y < 0.0 or y > height:
            corrections += 1
            continue
        clipped = (min(float(x), max_x), min(float(y), max_y))
        corrections += clipped != (x, y)
        normalized.append(clipped)
    return tuple(normalized), corrections


def index_ucf_qnrf_partition(
    image_root: str | Path,
    *,
    annotation_root: str | Path | None = None,
    thresholds: tuple[int, int] = DEFAULT_DENSITY_THRESHOLDS,
    partition: str,
    expected_samples: int | None = None,
) -> tuple[UCFQNRFRecord, ...]:
    """Index one explicitly supplied UCF-QNRF partition directory."""

    if partition not in {"train", "validation", "test"}:
        raise ValueError("partition must be train, validation, or test")
    root = Path(image_root)
    annotations = Path(annotation_root) if annotation_root is not None else root
    if not root.is_dir() or not annotations.is_dir():
        raise FileNotFoundError(
            f"explicit UCF-QNRF {partition} image and annotation roots are required"
        )
    images = sorted(
        path
        for path in root.iterdir()
        if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    if not images:
        raise ValueError(f"no UCF-QNRF {partition} images found under {root}")
    records: list[UCFQNRFRecord] = []
    for image_path in images:
        annotation_path = _annotation_path(image_path, annotations)
        with Image.open(image_path) as image:
            width, height = image.size
        raw_points = read_ucf_qnrf_points(annotation_path)
        points, corrections = _normalized_points(raw_points, width=width, height=height)
        records.append(
            UCFQNRFRecord(
                sample_id=image_path.stem,
                image_path=image_path.resolve(),
                annotation_path=annotation_path.resolve(),
                image_sha256=sha256_file(image_path),
                annotation_sha256=sha256_file(annotation_path),
                width=width,
                height=height,
                points=points,
                density_band=density_band(len(points), thresholds),
                normalization_corrections=corrections,
            )
        )
    sample_ids = [record.sample_id for record in records]
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError("duplicate UCF-QNRF sample identity")
    if expected_samples is not None and len(records) != expected_samples:
        raise ValueError(
            f"UCF-QNRF {partition} expected {expected_samples} samples, "
            f"observed {len(records)}"
        )
    return tuple(records)


def index_ucf_qnrf_train(
    train_root: str | Path,
    *,
    annotation_root: str | Path | None = None,
    thresholds: tuple[int, int] = DEFAULT_DENSITY_THRESHOLDS,
) -> tuple[UCFQNRFRecord, ...]:
    """Index only the explicitly supplied training directory.

    This compatibility entry point still has no dataset-root or test-root
    argument, so Round 1 validation preparation cannot discover the official
    test tree.
    """

    return index_ucf_qnrf_partition(
        train_root,
        annotation_root=annotation_root,
        thresholds=thresholds,
        partition="train",
    )


def _stable_key(sample_id: str, seed: int, namespace: str) -> tuple[str, str]:
    payload = f"{namespace}:{seed}:{sample_id}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest(), sample_id


def _split_ids(path: str | Path) -> tuple[str, ...]:
    values = Path(path).read_text(encoding="utf-8").split()
    sample_ids = tuple(Path(value).stem for value in values)
    if not sample_ids or len(sample_ids) != len(set(sample_ids)):
        raise ValueError(f"official split list must be non-empty and unique: {path}")
    return sample_ids


def apply_official_train_validation_split(
    records: Iterable[UCFQNRFRecord],
    *,
    train_list_path: str | Path,
    validation_list_path: str | Path,
) -> tuple[tuple[UCFQNRFRecord, ...], tuple[UCFQNRFRecord, ...]]:
    rows = tuple(records)
    indexed = {row.sample_id: row for row in rows}
    if len(indexed) != len(rows):
        raise ValueError("indexed UCF-QNRF records must have unique sample IDs")
    train_ids = _split_ids(train_list_path)
    validation_ids = _split_ids(validation_list_path)
    overlap = set(train_ids) & set(validation_ids)
    if overlap:
        raise ValueError(f"official train/validation split overlap: {sorted(overlap)}")
    listed = set(train_ids) | set(validation_ids)
    available = set(indexed)
    if listed != available:
        raise ValueError(
            "official split must be complete for the indexed training partition: "
            f"missing={sorted(available - listed)} unknown={sorted(listed - available)}"
        )
    return (
        tuple(indexed[sample_id] for sample_id in train_ids),
        tuple(indexed[sample_id] for sample_id in validation_ids),
    )


def select_stratified_smoke(
    records: Iterable[UCFQNRFRecord],
    *,
    sample_count: int,
    seed: int,
) -> tuple[UCFQNRFRecord, ...]:
    rows = tuple(records)
    if sample_count <= 0 or len(rows) < sample_count:
        raise ValueError("smoke selection requires enough rows for the requested count")
    bands = ("low", "medium", "high")
    base, remainder = divmod(sample_count, len(bands))
    selected: list[UCFQNRFRecord] = []
    selected_ids: set[str] = set()
    for index, band in enumerate(bands):
        quota = base + (1 if index < remainder else 0)
        candidates = sorted(
            (row for row in rows if row.density_band == band),
            key=lambda row: _stable_key(row.sample_id, seed, f"smoke-{band}"),
        )
        for row in candidates[:quota]:
            selected.append(row)
            selected_ids.add(row.sample_id)
    if len(selected) < sample_count:
        remaining = sorted(
            (row for row in rows if row.sample_id not in selected_ids),
            key=lambda row: _stable_key(row.sample_id, seed, "smoke-fill"),
        )
        selected.extend(remaining[: sample_count - len(selected)])
    if len(selected) != sample_count:
        raise ValueError("unable to build the requested stratified smoke subset")
    return tuple(sorted(selected, key=lambda row: _stable_key(row.sample_id, seed, "smoke-order")))


def _point_density(
    points: Sequence[Point], *, width: int, height: int, downsample: int
) -> np.ndarray:
    if downsample <= 0:
        raise ValueError("density downsample must be positive")
    density_height = math.ceil(height / downsample)
    density_width = math.ceil(width / downsample)
    density = np.zeros((density_height, density_width), dtype=np.float32)
    for x, y in points:
        column = min(int(x / downsample), density_width - 1)
        row = min(int(y / downsample), density_height - 1)
        density[row, column] += 1.0
    return density


def _quadrant_zones(width: int, height: int) -> tuple[ZoneBox, ...]:
    x_mid, y_mid = width / 2.0, height / 2.0
    return (
        ZoneBox("q00", 0.0, 0.0, x_mid, y_mid),
        ZoneBox("q01", x_mid, 0.0, float(width), y_mid),
        ZoneBox("q10", 0.0, y_mid, x_mid, float(height)),
        ZoneBox("q11", x_mid, y_mid, float(width), float(height)),
    )


def prepare_evaluation_sample(
    record: UCFQNRFRecord,
    *,
    split_id: str,
    dataset_id: str = DATASET_ID,
    density_downsample: int = 8,
    source_partition: str = "official_validation",
) -> EvaluationSample:
    return EvaluationSample(
        sample_id=record.sample_id,
        dataset_id=dataset_id,
        split_id=split_id,
        image_path=record.image_path,
        source_sha256=record.image_sha256,
        width=record.width,
        height=record.height,
        ground_truth_count=float(record.count),
        annotation_sha256=record.annotation_sha256,
        ground_truth_points=record.points,
        ground_truth_density=_point_density(
            record.points,
            width=record.width,
            height=record.height,
            downsample=density_downsample,
        ),
        condition_tags={
            "density_band": record.density_band,
            "source_partition": source_partition,
        },
        zones=_quadrant_zones(record.width, record.height),
        has_point_annotations=True,
    )
