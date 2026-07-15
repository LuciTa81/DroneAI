import warnings
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from droneai.evaluation_contract import EvaluationSample, NativePrediction, ZoneBox
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
        sample_id="dense-2",
        output_type="count",
        predicted_count=None,
        latency_ms=100.0,
        peak_vram_mb=500.0,
        failure_state="fixture_failure",
    )
    records = [
        evaluate_sample(sample_high, high_prediction, localization_radius=2.0),
        evaluate_sample(sample_low, low_prediction, localization_radius=2.0),
        evaluate_sample(sample_low, failed_prediction, localization_radius=2.0),
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
