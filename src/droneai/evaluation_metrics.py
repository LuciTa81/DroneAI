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
    gt_indices, pred_indices = linear_sum_assignment(distances)
    matched = [
        float(distances[g, p])
        for g, p in zip(gt_indices, pred_indices)
        if distances[g, p] <= radius
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
        x0 = round(zone.x0 * width / sample.width)
        x1 = round(zone.x1 * width / sample.width)
        y0 = round(zone.y0 * height / sample.height)
        y1 = round(zone.y1 * height / sample.height)
        region = (slice(y0, y1), slice(x0, x1))
        errors.append(
            abs(float(ground_truth[region].sum()) - float(predicted[region].sum()))
        )
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
        extra["density_ssim"] = _density_ssim(
            gt_density,
            pred_density,
            data_range=data_range,
        )
        extra["predicted_mass"] = float(pred_density.sum())
        extra["game_l1"] = _game_l1(gt_density, pred_density)
        if sample.zones:
            extra["zone_mae"] = _density_zone_mae(sample, gt_density, pred_density)
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
            extra["zone_mae"] = _point_zone_mae(sample, prediction)
        spatial_name, spatial_value = "localization_f1", extra["localization_f1"]
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
    rows = list(records)
    valid = [row for row in rows if row.predicted_count is not None]
    errors = np.asarray([row.signed_error for row in valid], dtype=np.float64)
    by_band: dict[str, list[float]] = defaultdict(list)
    by_condition: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for row in valid:
        by_band[row.density_band].append(float(row.signed_error))
        for key, value in row.condition_values.items():
            by_condition[key][value].append(float(row.signed_error))
    median_latency = (
        float(np.median([row.latency_ms for row in valid]))
        if valid
        else float("inf")
    )
    return {
        "expected_samples": expected_samples,
        "recorded_samples": len(rows),
        "successful_samples": len(valid),
        "failures": len(rows) - len(valid),
        "coverage": len(valid) / expected_samples if expected_samples else 0.0,
        "mae": float(np.mean(np.abs(errors))) if len(errors) else float("inf"),
        "rmse": (
            float(np.sqrt(np.mean(np.square(errors))))
            if len(errors)
            else float("inf")
        ),
        "signed_bias": float(np.mean(errors)) if len(errors) else float("inf"),
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
        "condition_signed_bias": {
            key: {
                value: float(np.mean(errors))
                for value, errors in sorted(groups.items())
            }
            for key, groups in sorted(by_condition.items())
        },
    }
