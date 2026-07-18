from __future__ import annotations

import math
from collections import defaultdict
from typing import Iterable

import numpy as np
from scipy.optimize import linear_sum_assignment
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

from droneai.evaluation_contract import EvaluationSample, NativePrediction, ScalarEvaluation


def _point_metrics(
    gt: tuple[tuple[float, float], ...],
    pred: tuple[tuple[float, float], ...],
    radius: float,
) -> dict[str, float]:
    if not gt and not pred:
        return {
            "localization_precision": 1.0,
            "localization_recall": 1.0,
            "localization_f1": 1.0,
            "localization_mean_distance": 0.0,
        }
    if not gt or not pred:
        return {
            "localization_precision": 0.0,
            "localization_recall": 0.0,
            "localization_f1": 0.0,
            "localization_mean_distance": float("inf"),
        }
    gt_array = np.asarray(gt, dtype=np.float64)
    pred_array = np.asarray(pred, dtype=np.float64)
    distances = np.linalg.norm(gt_array[:, None, :] - pred_array[None, :, :], axis=2)
    gt_count, pred_count = distances.shape
    penalty = max(radius, 1.0) + 1.0
    invalid = penalty * 4.0 + float(distances.max(initial=0.0))
    costs = np.full((gt_count + pred_count, pred_count + gt_count), invalid)
    costs[:gt_count, :pred_count] = np.where(
        distances <= radius,
        distances,
        invalid,
    )
    for index in range(gt_count):
        costs[index, pred_count + index] = penalty
    for index in range(pred_count):
        costs[gt_count + index, index] = penalty
    costs[gt_count:, pred_count:] = 0.0
    gt_indices, pred_indices = linear_sum_assignment(costs)
    matched = [
        float(distances[g, p])
        for g, p in zip(gt_indices, pred_indices)
        if g < gt_count and p < pred_count and distances[g, p] <= radius
    ]
    true_positive = len(matched)
    precision = true_positive / len(pred)
    recall = true_positive / len(gt)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    mean_distance = float(np.mean(matched)) if matched else float("inf")
    return {
        "localization_precision": precision,
        "localization_recall": recall,
        "localization_f1": f1,
        "localization_mean_distance": mean_distance,
    }


def _point_zone_mae(sample: EvaluationSample, prediction: NativePrediction) -> float:
    if not sample.zones or not sample.has_point_annotations:
        raise ValueError("point zone metrics require zones and point annotations")
    errors = []
    for zone in sample.zones:
        ground_truth = sum(zone.contains(x, y) for x, y in sample.ground_truth_points)
        predicted = sum(zone.contains(x, y) for x, y in prediction.points)
        errors.append(abs(ground_truth - predicted))
    return float(np.mean(errors))


def _density_zone_mae(
    sample: EvaluationSample,
    ground_truth: np.ndarray,
    predicted: np.ndarray,
) -> float:
    if not sample.zones:
        raise ValueError("density zone metrics require zones")
    height, width = ground_truth.shape
    errors = []
    for zone in sample.zones:
        x_edges = np.linspace(0.0, sample.width, width + 1)
        y_edges = np.linspace(0.0, sample.height, height + 1)
        x_overlap = np.maximum(
            0.0,
            np.minimum(x_edges[1:], zone.x1) - np.maximum(x_edges[:-1], zone.x0),
        ) / (sample.width / width)
        y_overlap = np.maximum(
            0.0,
            np.minimum(y_edges[1:], zone.y1) - np.maximum(y_edges[:-1], zone.y0),
        ) / (sample.height / height)
        weights = y_overlap[:, None] * x_overlap[None, :]
        ground_truth_mass = float(np.sum(ground_truth * weights))
        predicted_mass = float(np.sum(predicted * weights))
        errors.append(abs(ground_truth_mass - predicted_mass))
    return float(np.mean(errors))


def _game_l1(ground_truth: np.ndarray, predicted: np.ndarray) -> float:
    if ground_truth.shape != predicted.shape:
        raise ValueError("density maps must share shape")
    height, width = ground_truth.shape
    y_mid, x_mid = height // 2, width // 2
    slices = (
        (slice(0, y_mid), slice(0, x_mid)),
        (slice(0, y_mid), slice(x_mid, width)),
        (slice(y_mid, height), slice(0, x_mid)),
        (slice(y_mid, height), slice(x_mid, width)),
    )
    return float(
        sum(
            abs(float(ground_truth[s].sum()) - float(predicted[s].sum()))
            for s in slices
        )
    )


def _density_ssim(
    ground_truth: np.ndarray,
    predicted: np.ndarray,
    *,
    data_range: float,
) -> float:
    minimum_side = min(ground_truth.shape)
    window = min(7, minimum_side)
    if window % 2 == 0:
        window -= 1
    if window < 3:
        raise ValueError("density maps must be at least 3 by 3 for SSIM")
    return float(
        structural_similarity(
            ground_truth,
            predicted,
            data_range=data_range,
            win_size=window,
        )
    )


def evaluate_sample(
    sample: EvaluationSample,
    prediction: NativePrediction,
    *,
    localization_radius: float,
) -> ScalarEvaluation:
    if prediction.sample_id != sample.sample_id:
        raise ValueError("sample and prediction IDs differ")
    if not math.isfinite(localization_radius) or localization_radius < 0:
        raise ValueError("localization radius must be finite and non-negative")
    if prediction.points and prediction.coordinate_space != "original_pixels":
        raise ValueError("predicted points must use original image coordinates")
    if any(
        not (0 <= x < sample.width and 0 <= y < sample.height)
        for x, y in prediction.points
    ):
        raise ValueError("predicted points must use original image coordinates")
    density_band = sample.condition_tags.get("density_band", "unknown")
    if prediction.failure_state:
        return ScalarEvaluation(
            sample.sample_id,
            sample.ground_truth_count,
            None,
            None,
            None,
            None,
            density_band,
            prediction.latency_ms,
            prediction.peak_vram_mb,
            prediction.output_type,
            prediction.failure_state,
            condition_values=dict(sample.condition_tags),
        )
    predicted = float(prediction.predicted_count)
    signed = predicted - sample.ground_truth_count
    extra: dict[str, float] = {}
    spatial_name = None
    spatial_value = None
    if sample.ground_truth_density is not None and prediction.density is not None:
        gt_density = np.asarray(sample.ground_truth_density, dtype=np.float64)
        pred_density = np.asarray(prediction.density, dtype=np.float64)
        if gt_density.shape != pred_density.shape:
            raise ValueError("density maps must share shape")
        data_range = max(float(gt_density.max() - gt_density.min()), 1.0)
        extra["density_psnr"] = (
            float("inf")
            if np.array_equal(gt_density, pred_density)
            else float(
                peak_signal_noise_ratio(
                    gt_density,
                    pred_density,
                    data_range=data_range,
                )
            )
        )
        if min(gt_density.shape) >= 3:
            extra["density_ssim"] = _density_ssim(
                gt_density,
                pred_density,
                data_range=data_range,
            )
        extra["predicted_mass"] = float(pred_density.sum())
        extra["game_l1"] = _game_l1(gt_density, pred_density)
        if sample.zones:
            density_zone_mae = _density_zone_mae(
                sample,
                gt_density,
                pred_density,
            )
            extra["density_zone_mae"] = density_zone_mae
            extra["zone_mae"] = density_zone_mae
        spatial_name, spatial_value = "game_l1", extra["game_l1"]
    point_output_available = prediction.output_type == "points" or (
        prediction.output_type == "hybrid" and bool(prediction.points)
    )
    if sample.has_point_annotations and point_output_available:
        extra.update(
            _point_metrics(
                sample.ground_truth_points,
                prediction.points,
                localization_radius,
            )
        )
        extra["localization_radius"] = localization_radius
        if sample.zones:
            point_zone_mae = _point_zone_mae(sample, prediction)
            extra["point_zone_mae"] = point_zone_mae
            if "density_zone_mae" not in extra:
                extra["zone_mae"] = point_zone_mae
        if spatial_name is None:
            spatial_name, spatial_value = (
                "localization_f1",
                extra["localization_f1"],
            )
    return ScalarEvaluation(
        sample.sample_id,
        sample.ground_truth_count,
        predicted,
        signed,
        abs(signed),
        abs(signed) / max(sample.ground_truth_count, 1.0),
        density_band,
        prediction.latency_ms,
        prediction.peak_vram_mb,
        prediction.output_type,
        None,
        spatial_name,
        spatial_value,
        extra,
        dict(sample.condition_tags),
    )


def summarize_records(
    records: Iterable[ScalarEvaluation],
    *,
    expected_samples: int,
) -> dict[str, object]:
    if (
        isinstance(expected_samples, bool)
        or not isinstance(expected_samples, int)
        or expected_samples <= 0
    ):
        raise ValueError("expected_samples must be a positive integer")
    rows = list(records)
    sample_ids = [row.sample_id for row in rows]
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError("duplicate sample_id in evaluation records")
    if len(rows) > expected_samples:
        raise ValueError("recorded samples cannot exceed expected_samples")
    for row in rows:
        is_success = (
            row.predicted_count is not None
            and row.failure_state is None
            and row.signed_error is not None
            and row.absolute_error is not None
            and row.normalized_error is not None
        )
        is_failure = (
            row.predicted_count is None
            and bool(row.failure_state)
            and row.signed_error is None
            and row.absolute_error is None
            and row.normalized_error is None
        )
        if not (is_success or is_failure):
            raise ValueError("each record must be exactly one success or explicit failure")
    valid = [row for row in rows if row.predicted_count is not None]
    explicit_failures = [row for row in rows if row.failure_state is not None]
    errors = np.asarray([row.signed_error for row in valid], dtype=np.float64)
    normalized_errors = np.asarray(
        [row.normalized_error for row in valid], dtype=np.float64
    )
    extra_values: dict[str, list[float]] = defaultdict(list)
    for row in valid:
        for key, value in row.extra_metrics.items():
            numeric = float(value)
            if math.isfinite(numeric):
                extra_values[key].append(numeric)
    by_band: dict[str, list[float]] = defaultdict(list)
    by_condition: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    all_by_band: dict[str, list[ScalarEvaluation]] = defaultdict(list)
    all_by_condition: dict[str, dict[str, list[ScalarEvaluation]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for row in rows:
        all_by_band[row.density_band].append(row)
        for key, value in row.condition_values.items():
            all_by_condition[key][value].append(row)
    for row in valid:
        by_band[row.density_band].append(float(row.signed_error))
        for key, value in row.condition_values.items():
            by_condition[key][value].append(float(row.signed_error))
    median_latency = (
        float(np.median([row.latency_ms for row in valid]))
        if valid
        else float("inf")
    )

    def coverage(group: list[ScalarEvaluation]) -> dict[str, float | int]:
        successful = sum(row.predicted_count is not None for row in group)
        failures = sum(row.failure_state is not None for row in group)
        return {
            "expected_samples": len(group),
            "recorded_samples": len(group),
            "successful_samples": successful,
            "explicit_failures": failures,
            "coverage": successful / len(group) if group else 0.0,
        }

    return {
        "expected_samples": expected_samples,
        "recorded_samples": len(rows),
        "successful_samples": len(valid),
        "failures": len(explicit_failures),
        "explicit_failures": len(explicit_failures),
        "accounting_complete": len(rows) == expected_samples,
        "coverage": len(valid) / expected_samples,
        "mae": float(np.mean(np.abs(errors))) if len(errors) else float("inf"),
        "rmse": (
            float(np.sqrt(np.mean(np.square(errors))))
            if len(errors)
            else float("inf")
        ),
        "signed_bias": float(np.mean(errors)) if len(errors) else float("inf"),
        "mape_reference": (
            100.0 * float(np.mean(normalized_errors))
            if len(normalized_errors)
            else float("inf")
        ),
        "mape_reference_unit": "percent",
        "extra_metric_means": {
            key: float(np.mean(values))
            for key, values in sorted(extra_values.items())
        },
        "extra_metric_finite_samples": {
            key: len(values) for key, values in sorted(extra_values.items())
        },
        "median_latency_ms": median_latency,
        "throughput_fps_batch1": (
            1000.0 / median_latency
            if math.isfinite(median_latency) and median_latency > 0
            else 0.0
        ),
        "peak_vram_mb": max(
            (row.peak_vram_mb for row in valid),
            default=float("inf"),
        ),
        "band_signed_bias": {
            band: float(np.mean(values)) for band, values in sorted(by_band.items())
        },
        "band_coverage": {
            band: coverage(group) for band, group in sorted(all_by_band.items())
        },
        "condition_signed_bias": {
            key: {
                value: float(np.mean(errors))
                for value, errors in sorted(groups.items())
            }
            for key, groups in sorted(by_condition.items())
        },
        "condition_coverage": {
            key: {
                value: coverage(group)
                for value, group in sorted(groups.items())
            }
            for key, groups in sorted(all_by_condition.items())
        },
    }
