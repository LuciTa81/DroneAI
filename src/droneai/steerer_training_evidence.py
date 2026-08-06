"""Deterministic metric and evidence primitives for STEERER training stages."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np


T0_METRICS = frozenset({"train_loss"})
T1_METRICS = frozenset(
    {
        "mae",
        "rmse",
        "signed_bias",
        "reference_mape",
        "game_l1",
        "quadrant_zone_mae",
        "localization_precision_16px",
        "localization_recall_16px",
        "localization_f1_16px",
        "median_latency_ms",
        "p95_latency_ms",
        "fps",
        "peak_vram_mb",
        "density_sum_count_difference",
        "train_loss",
        "validation_loss",
        "training_window_improvement",
    }
)


@dataclass(frozen=True)
class ValidationSampleObservation:
    """Measured values for one approved validation sample."""

    sample_id: str
    target_count: float
    predicted_count: float
    game_l1: float
    quadrant_zone_mae: float
    localization_tp: int
    localization_fp: int
    localization_fn: int
    latency_ms: float
    peak_vram_mb: float
    density_sum_count_difference: float


def _finite_number(value: Any, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, np.number)):
        raise ValueError(f"{name} must be a finite number")
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ValueError(f"{name} must be finite")
    return numeric


def _mean(values: Sequence[float], *, name: str) -> float:
    if not values:
        raise ValueError(f"{name} requires at least one observation")
    return float(sum(values) / len(values))


def build_t0_metrics(update: object) -> dict[str, float]:
    """Return the exact T0 metric set from the real optimizer update."""

    return {"train_loss": _finite_number(getattr(update, "loss", None), name="train_loss")}


def _localization_metrics(samples: Sequence[ValidationSampleObservation]) -> tuple[float, float, float]:
    tp = sum(sample.localization_tp for sample in samples)
    fp = sum(sample.localization_fp for sample in samples)
    fn = sum(sample.localization_fn for sample in samples)
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value < 0
        for sample in samples
        for value in (sample.localization_tp, sample.localization_fp, sample.localization_fn)
    ):
        raise ValueError("localization counts must be non-negative integers")
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    return float(precision), float(recall), float(f1)


def build_t1_metrics(
    epoch: object,
    validation: object,
    *,
    expected_samples: int,
) -> dict[str, float]:
    """Aggregate the exact scoreable T1 metric schema from measured observations."""

    if isinstance(expected_samples, bool) or not isinstance(expected_samples, int) or expected_samples <= 0:
        raise ValueError("expected_samples must be a positive integer")
    samples = tuple(getattr(validation, "samples", ()))
    observed_count = getattr(validation, "sample_count", None)
    if len(samples) != expected_samples or observed_count != expected_samples:
        raise ValueError("validation observations must cover the complete approved split")
    sample_ids = [sample.sample_id for sample in samples]
    if any(not isinstance(sample_id, str) or not sample_id for sample_id in sample_ids):
        raise ValueError("validation sample IDs must be non-empty strings")
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError("validation sample IDs must be unique")

    errors: list[float] = []
    normalized_errors: list[float] = []
    game_values: list[float] = []
    quadrant_values: list[float] = []
    latencies: list[float] = []
    peak_vram: list[float] = []
    density_differences: list[float] = []
    for sample in samples:
        target = _finite_number(sample.target_count, name="target_count")
        predicted = _finite_number(sample.predicted_count, name="predicted_count")
        if target < 0 or predicted < 0:
            raise ValueError("validation counts must be non-negative")
        error = predicted - target
        errors.append(error)
        normalized_errors.append(abs(error) / max(target, 1.0))
        for value, name, output in (
            (sample.game_l1, "game_l1", game_values),
            (sample.quadrant_zone_mae, "quadrant_zone_mae", quadrant_values),
            (sample.latency_ms, "latency_ms", latencies),
            (sample.peak_vram_mb, "peak_vram_mb", peak_vram),
            (
                sample.density_sum_count_difference,
                "density_sum_count_difference",
                density_differences,
            ),
        ):
            numeric = _finite_number(value, name=name)
            if numeric < 0:
                raise ValueError(f"{name} must be non-negative")
            output.append(numeric)

    precision, recall, f1 = _localization_metrics(samples)
    median_latency = float(np.median(np.asarray(latencies, dtype=np.float64)))
    if median_latency <= 0:
        raise ValueError("median_latency_ms must be positive")
    losses = tuple(
        _finite_number(value, name="training loss")
        for value in tuple(getattr(epoch, "losses", ()))
    )
    if not losses:
        raise ValueError("training loss observations are required")
    window = max(1, int(math.ceil(len(losses) * 0.1)))

    metrics = {
        "mae": _mean([abs(error) for error in errors], name="mae"),
        "rmse": float(math.sqrt(_mean([error * error for error in errors], name="rmse"))),
        "signed_bias": _mean(errors, name="signed_bias"),
        "reference_mape": _mean(normalized_errors, name="reference_mape") * 100.0,
        "game_l1": _mean(game_values, name="game_l1"),
        "quadrant_zone_mae": _mean(quadrant_values, name="quadrant_zone_mae"),
        "localization_precision_16px": precision,
        "localization_recall_16px": recall,
        "localization_f1_16px": f1,
        "median_latency_ms": median_latency,
        "p95_latency_ms": float(np.percentile(np.asarray(latencies, dtype=np.float64), 95)),
        "fps": 1000.0 / median_latency,
        "peak_vram_mb": max(peak_vram),
        "density_sum_count_difference": _mean(
            density_differences, name="density_sum_count_difference"
        ),
        "train_loss": _mean(list(losses), name="train_loss"),
        "validation_loss": _finite_number(
            getattr(validation, "loss", None), name="validation_loss"
        ),
        "training_window_improvement": _mean(
            list(losses[:window]), name="first training window"
        )
        - _mean(list(losses[-window:]), name="last training window"),
    }
    if set(metrics) != T1_METRICS:
        raise RuntimeError("T1 metric aggregation produced the wrong schema")
    return {name: _finite_number(value, name=name) for name, value in metrics.items()}


def strict_metrics_payload(
    *, run_id: str, stage: str, metrics: Mapping[str, float]
) -> dict[str, object]:
    """Validate and return the immutable runner metrics payload."""

    if not isinstance(run_id, str) or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", run_id) is None:
        raise ValueError("run_id must be one safe path component")
    expected = T0_METRICS if stage == "T0" else T1_METRICS if stage in {"T1", "T5", "T50"} else None
    if expected is None:
        raise ValueError("stage must be T0, T1, T5, or T50")
    if set(metrics) != expected:
        raise ValueError("metrics must contain the exact stage schema")
    validated = {name: _finite_number(value, name=name) for name, value in metrics.items()}
    return {
        "schema_version": 1,
        "run_id": run_id,
        "stage": stage,
        "metrics": validated,
    }


__all__ = [
    "T0_METRICS",
    "T1_METRICS",
    "ValidationSampleObservation",
    "build_t0_metrics",
    "build_t1_metrics",
    "strict_metrics_payload",
]
