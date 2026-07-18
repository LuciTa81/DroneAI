import warnings
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from droneai.evaluation_contract import (
    EvaluationSample,
    NativePrediction,
    ScalarEvaluation,
    ZoneBox,
)
from droneai.evaluation_metrics import evaluate_sample, summarize_records


def _sample(tmp_path: Path) -> EvaluationSample:
    return EvaluationSample(
        sample_id="dense-1",
        dataset_id="fixture-v1",
        split_id="validation",
        image_path=tmp_path / "dense-1.png",
        source_sha256="a" * 64,
        width=8,
        height=8,
        ground_truth_count=2.0,
        ground_truth_points=((1.0, 1.0), (6.0, 6.0)),
        ground_truth_density=np.asarray(
            [[1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
            dtype=np.float32,
        ),
        condition_tags={"density_band": "high"},
        zones=(ZoneBox("left", 0, 0, 4, 8), ZoneBox("right", 4, 0, 8, 8)),
        has_point_annotations=True,
    )


def test_density_metrics_and_count_error_are_recorded(tmp_path: Path) -> None:
    prediction = NativePrediction(
        sample_id="dense-1",
        output_type="density",
        predicted_count=2.0,
        latency_ms=3.0,
        peak_vram_mb=100.0,
        density=np.asarray(
            [[1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
            dtype=np.float32,
        ),
    )
    record = evaluate_sample(_sample(tmp_path), prediction, localization_radius=2.0)
    assert record.absolute_error == 0.0
    assert record.extra_metrics["density_psnr"] == pytest.approx(float("inf"))
    assert record.extra_metrics["density_ssim"] == pytest.approx(1.0)
    assert record.extra_metrics["predicted_mass"] == pytest.approx(2.0)
    assert record.extra_metrics["game_l1"] == pytest.approx(0.0)
    assert record.extra_metrics["zone_mae"] == pytest.approx(0.0)
    assert "localization_f1" not in record.extra_metrics


def test_point_matching_and_zone_error_are_recorded(tmp_path: Path) -> None:
    prediction = NativePrediction(
        sample_id="dense-1",
        output_type="points",
        predicted_count=2.0,
        latency_ms=3.0,
        peak_vram_mb=100.0,
        points=((1.5, 1.5), (6.5, 6.5)),
        point_confidences=(0.9, 0.8),
    )
    record = evaluate_sample(_sample(tmp_path), prediction, localization_radius=2.0)
    assert record.extra_metrics["localization_precision"] == 1.0
    assert record.extra_metrics["localization_recall"] == 1.0
    assert record.extra_metrics["localization_f1"] == 1.0
    assert record.extra_metrics["localization_radius"] == 2.0
    assert record.extra_metrics["zone_mae"] == 0.0
    assert "density_ssim" not in record.extra_metrics


def test_failed_prediction_reduces_coverage_without_becoming_zero(tmp_path: Path) -> None:
    failed = NativePrediction(
        sample_id="dense-1",
        output_type="count",
        predicted_count=None,
        latency_ms=1.0,
        peak_vram_mb=0.0,
        failure_state="fixture_failure",
    )
    record = evaluate_sample(_sample(tmp_path), failed, localization_radius=2.0)
    summary = summarize_records([record], expected_samples=1)
    assert record.predicted_count is None
    assert summary["coverage"] == 0.0
    assert summary["failures"] == 1


def test_density_zone_metric_is_unavailable_without_zones(tmp_path: Path) -> None:
    sample = replace(_sample(tmp_path), zones=())
    prediction = NativePrediction(
        sample_id="dense-1",
        output_type="density",
        predicted_count=2.0,
        latency_ms=3.0,
        peak_vram_mb=100.0,
        density=np.asarray(sample.ground_truth_density, dtype=np.float32),
    )

    record = evaluate_sample(sample, prediction, localization_radius=2.0)

    assert "zone_mae" not in record.extra_metrics


def test_point_zone_metric_is_unavailable_without_zones(tmp_path: Path) -> None:
    sample = replace(_sample(tmp_path), zones=())
    prediction = NativePrediction(
        sample_id="dense-1",
        output_type="points",
        predicted_count=2.0,
        latency_ms=3.0,
        peak_vram_mb=100.0,
        points=((1.5, 1.5), (6.5, 6.5)),
    )

    record = evaluate_sample(sample, prediction, localization_radius=2.0)

    assert "zone_mae" not in record.extra_metrics


def test_density_only_hybrid_does_not_fabricate_localization_metrics(
    tmp_path: Path,
) -> None:
    sample = _sample(tmp_path)
    prediction = NativePrediction(
        sample_id="dense-1",
        output_type="hybrid",
        predicted_count=2.0,
        latency_ms=3.0,
        peak_vram_mb=100.0,
        density=np.asarray(sample.ground_truth_density, dtype=np.float32),
    )

    record = evaluate_sample(sample, prediction, localization_radius=2.0)

    assert "localization_f1" not in record.extra_metrics
    assert record.spatial_metric_name == "game_l1"


def test_point_metrics_require_original_pixel_coordinates(tmp_path: Path) -> None:
    prediction = NativePrediction(
        sample_id="dense-1",
        output_type="points",
        predicted_count=2.0,
        latency_ms=3.0,
        peak_vram_mb=100.0,
        points=((0.1, 0.1), (0.8, 0.8)),
        coordinate_space="normalized",
    )

    with pytest.raises(ValueError, match="original image coordinates"):
        evaluate_sample(_sample(tmp_path), prediction, localization_radius=2.0)


@pytest.mark.parametrize("radius", [float("nan"), float("inf"), -1.0])
def test_localization_radius_must_be_finite_and_non_negative(
    tmp_path: Path,
    radius: float,
) -> None:
    prediction = NativePrediction(
        sample_id="dense-1",
        output_type="points",
        predicted_count=2.0,
        latency_ms=3.0,
        peak_vram_mb=100.0,
        points=((1.5, 1.5), (6.5, 6.5)),
    )

    with pytest.raises(ValueError, match="localization radius"):
        evaluate_sample(_sample(tmp_path), prediction, localization_radius=radius)


def test_identical_density_maps_do_not_emit_runtime_warning(tmp_path: Path) -> None:
    sample = _sample(tmp_path)
    prediction = NativePrediction(
        sample_id="dense-1",
        output_type="density",
        predicted_count=2.0,
        latency_ms=3.0,
        peak_vram_mb=100.0,
        density=np.asarray(sample.ground_truth_density, dtype=np.float32),
    )

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        record = evaluate_sample(sample, prediction, localization_radius=2.0)

    assert record.extra_metrics["density_psnr"] == float("inf")


def test_density_metrics_require_matching_map_shapes(tmp_path: Path) -> None:
    predicted_density = np.zeros((4, 4), dtype=np.float32)
    predicted_density[0, 0] = 1.0
    predicted_density[-1, -1] = 1.0
    prediction = NativePrediction(
        sample_id="dense-1",
        output_type="density",
        predicted_count=2.0,
        latency_ms=3.0,
        peak_vram_mb=100.0,
        density=predicted_density,
    )

    with pytest.raises(ValueError, match="density maps must share shape"):
        evaluate_sample(_sample(tmp_path), prediction, localization_radius=2.0)


def test_summary_preserves_signed_bias_groups_and_success_runtime(
    tmp_path: Path,
) -> None:
    sample_high = _sample(tmp_path)
    sample_low = replace(
        sample_high,
        sample_id="dense-2",
        ground_truth_count=1.0,
        ground_truth_points=((1.0, 1.0),),
        ground_truth_density=None,
        condition_tags={"density_band": "low", "lighting": "night"},
    )
    high_prediction = NativePrediction(
        sample_id="dense-1",
        output_type="count",
        predicted_count=1.0,
        latency_ms=2.0,
        peak_vram_mb=50.0,
    )
    low_prediction = NativePrediction(
        sample_id="dense-2",
        output_type="count",
        predicted_count=3.0,
        latency_ms=4.0,
        peak_vram_mb=75.0,
    )
    failed_prediction = NativePrediction(
        sample_id="dense-3",
        output_type="count",
        predicted_count=None,
        latency_ms=100.0,
        peak_vram_mb=500.0,
        failure_state="fixture_failure",
    )
    records = [
        evaluate_sample(sample_high, high_prediction, localization_radius=2.0),
        evaluate_sample(sample_low, low_prediction, localization_radius=2.0),
        evaluate_sample(
            replace(sample_low, sample_id="dense-3"),
            failed_prediction,
            localization_radius=2.0,
        ),
    ]

    summary = summarize_records(records, expected_samples=3)

    assert summary["coverage"] == pytest.approx(2 / 3)
    assert summary["mae"] == pytest.approx(1.5)
    assert summary["rmse"] == pytest.approx(np.sqrt(2.5))
    assert summary["signed_bias"] == pytest.approx(0.5)
    assert summary["median_latency_ms"] == 3.0
    assert summary["throughput_fps_batch1"] == pytest.approx(1000.0 / 3.0)
    assert summary["peak_vram_mb"] == 75.0
    assert summary["band_signed_bias"] == {"high": -1.0, "low": 2.0}
    assert summary["condition_signed_bias"] == {
        "density_band": {"high": -1.0, "low": 2.0},
        "lighting": {"night": 2.0},
    }


def test_summary_aggregates_mape_reference_and_native_metric_means() -> None:
    records = [
        ScalarEvaluation(
            "one",
            100.0,
            110.0,
            10.0,
            10.0,
            0.10,
            "low",
            2.0,
            10.0,
            "density",
            None,
            "game_l1",
            12.0,
            {
                "density_psnr": 20.0,
                "density_ssim": 0.8,
                "density_zone_mae": 5.0,
                "game_l1": 12.0,
            },
        ),
        ScalarEvaluation(
            "two",
            200.0,
            240.0,
            40.0,
            40.0,
            0.20,
            "medium",
            4.0,
            20.0,
            "density",
            None,
            "game_l1",
            18.0,
            {
                "density_psnr": 30.0,
                "density_ssim": 0.6,
                "density_zone_mae": 9.0,
                "game_l1": 18.0,
            },
        ),
    ]

    summary = summarize_records(records, expected_samples=2)

    assert summary["mape_reference"] == pytest.approx(15.0)
    assert summary["mape_reference_unit"] == "percent"
    assert summary["extra_metric_means"] == {
        "density_psnr": 25.0,
        "density_ssim": 0.7,
        "density_zone_mae": 7.0,
        "game_l1": 15.0,
    }
    assert summary["extra_metric_finite_samples"] == {
        "density_psnr": 2,
        "density_ssim": 2,
        "density_zone_mae": 2,
        "game_l1": 2,
    }


def test_point_matching_maximizes_in_radius_matches_before_distance(
    tmp_path: Path,
) -> None:
    sample = replace(
        _sample(tmp_path),
        width=4,
        height=4,
        ground_truth_count=2.0,
        ground_truth_points=((0.0, 0.0), (1.0, 0.0)),
        ground_truth_density=None,
        zones=(),
    )
    prediction = NativePrediction(
        sample_id="dense-1",
        output_type="points",
        predicted_count=2.0,
        latency_ms=1.0,
        peak_vram_mb=1.0,
        points=((0.0, 0.0), (0.0, 0.6)),
    )

    record = evaluate_sample(sample, prediction, localization_radius=1.1)

    assert record.extra_metrics["localization_recall"] == 1.0
    assert record.extra_metrics["localization_precision"] == 1.0


def test_condition_coverage_exposes_concentrated_failures(tmp_path: Path) -> None:
    sample = replace(
        _sample(tmp_path),
        condition_tags={"density_band": "high", "lighting": "night"},
    )
    success = NativePrediction("dense-1", "count", 2.0, 1.0, 1.0)
    failed = NativePrediction(
        "dense-2",
        "count",
        None,
        1.0,
        1.0,
        failure_state="oom",
    )
    records = [
        evaluate_sample(sample, success, localization_radius=1.0),
        evaluate_sample(
            replace(sample, sample_id="dense-2"),
            failed,
            localization_radius=1.0,
        ),
    ]

    summary = summarize_records(records, expected_samples=2)

    assert summary["band_coverage"]["high"] == {
        "expected_samples": 2,
        "recorded_samples": 2,
        "successful_samples": 1,
        "explicit_failures": 1,
        "coverage": 0.5,
    }
    assert summary["condition_coverage"]["lighting"]["night"]["coverage"] == 0.5


def test_narrow_density_zone_uses_fractional_cell_mass(tmp_path: Path) -> None:
    sample = replace(
        _sample(tmp_path),
        zones=(ZoneBox("narrow", 0.0, 0.0, 0.5, 8.0),),
    )
    predicted_density = np.asarray(sample.ground_truth_density, dtype=np.float32).copy()
    predicted_density[0, 0] = 0.0
    predicted_density[1, 1] += 1.0
    prediction = NativePrediction(
        "dense-1",
        "density",
        2.0,
        1.0,
        1.0,
        density=predicted_density,
    )

    record = evaluate_sample(sample, prediction, localization_radius=1.0)

    assert record.extra_metrics["density_zone_mae"] > 0.0


def test_hybrid_preserves_density_and_point_spatial_metrics(tmp_path: Path) -> None:
    sample = _sample(tmp_path)
    prediction = NativePrediction(
        "dense-1",
        "hybrid",
        2.0,
        1.0,
        1.0,
        density=np.asarray(sample.ground_truth_density, dtype=np.float32),
        points=((1.0, 1.0), (6.0, 6.0)),
    )

    record = evaluate_sample(sample, prediction, localization_radius=1.0)

    assert record.spatial_metric_name == "game_l1"
    assert record.extra_metrics["density_zone_mae"] == 0.0
    assert record.extra_metrics["point_zone_mae"] == 0.0
    assert record.extra_metrics["game_l1"] == 0.0
    assert record.extra_metrics["localization_f1"] == 1.0


def test_summary_rejects_duplicate_sample_ids(tmp_path: Path) -> None:
    sample = _sample(tmp_path)
    prediction = NativePrediction("dense-1", "count", 2.0, 1.0, 1.0)
    record = evaluate_sample(sample, prediction, localization_radius=1.0)

    with pytest.raises(ValueError, match="duplicate sample_id"):
        summarize_records([record, record], expected_samples=2)


def test_summary_marks_missing_records_as_incomplete(tmp_path: Path) -> None:
    sample = _sample(tmp_path)
    prediction = NativePrediction("dense-1", "count", 2.0, 1.0, 1.0)
    record = evaluate_sample(sample, prediction, localization_radius=1.0)

    summary = summarize_records([record], expected_samples=2)

    assert summary["accounting_complete"] is False
    assert summary["coverage"] == 0.5
    assert summary["explicit_failures"] == 0


def test_summary_rejects_ambiguous_success_or_failure_rows(tmp_path: Path) -> None:
    sample = _sample(tmp_path)
    prediction = NativePrediction("dense-1", "count", 2.0, 1.0, 1.0)
    record = evaluate_sample(sample, prediction, localization_radius=1.0)
    ambiguous = replace(record, predicted_count=None, failure_state=None)

    with pytest.raises(ValueError, match="success or explicit failure"):
        summarize_records([ambiguous], expected_samples=1)


def test_small_density_map_keeps_non_ssim_metrics(tmp_path: Path) -> None:
    density = np.asarray([[1.0, 1.0]], dtype=np.float32)
    sample = replace(
        _sample(tmp_path),
        ground_truth_density=density,
        ground_truth_count=2.0,
        ground_truth_points=((1.0, 1.0), (6.0, 6.0)),
    )
    prediction = NativePrediction(
        "dense-1",
        "density",
        2.0,
        1.0,
        1.0,
        density=density.copy(),
    )

    record = evaluate_sample(sample, prediction, localization_radius=1.0)

    assert "density_ssim" not in record.extra_metrics
    assert record.extra_metrics["predicted_mass"] == 2.0
