from pathlib import Path

import numpy as np
import pytest

from droneai.evaluation_contract import EvaluationSample, NativePrediction, ZoneBox


def test_explicit_failure_cannot_be_serialized_as_zero_prediction(tmp_path: Path) -> None:
    prediction = NativePrediction(
        sample_id="sample-1",
        output_type="count",
        predicted_count=None,
        latency_ms=2.0,
        peak_vram_mb=4.0,
        failure_state="cuda_out_of_memory",
    )
    assert prediction.predicted_count is None
    with pytest.raises(ValueError, match="failed predictions cannot contain a count"):
        NativePrediction(
            sample_id="sample-1",
            output_type="count",
            predicted_count=0.0,
            latency_ms=2.0,
            peak_vram_mb=4.0,
            failure_state="cuda_out_of_memory",
        )


def test_sample_rejects_ground_truth_count_point_mismatch(tmp_path: Path) -> None:
    image_path = tmp_path / "image.png"
    image_path.write_bytes(b"fixture")
    with pytest.raises(ValueError, match="point count"):
        EvaluationSample(
            sample_id="sample-1",
            dataset_id="fixture-v1",
            split_id="validation",
            image_path=image_path,
            source_sha256="a" * 64,
            width=32,
            height=24,
            ground_truth_count=2.0,
            ground_truth_points=((4.0, 5.0),),
            has_point_annotations=True,
        )


def test_density_prediction_requires_two_dimensional_finite_map() -> None:
    with pytest.raises(ValueError, match="finite 2D density"):
        NativePrediction(
            sample_id="sample-1",
            output_type="density",
            predicted_count=1.0,
            latency_ms=2.0,
            peak_vram_mb=4.0,
            density=np.asarray([[[float("nan")]]], dtype=np.float32),
        )


def test_sample_rejects_invalid_source_hash(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="source SHA-256"):
        EvaluationSample(
            sample_id="sample-1",
            dataset_id="fixture-v1",
            split_id="validation",
            image_path=tmp_path / "image.png",
            source_sha256="short",
            width=32,
            height=24,
            ground_truth_count=0.0,
        )


def test_sample_rejects_invalid_optional_annotation_hash(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="annotation SHA-256"):
        EvaluationSample(
            sample_id="sample-1",
            dataset_id="fixture-v1",
            split_id="validation",
            image_path=tmp_path / "image.png",
            source_sha256="a" * 64,
            annotation_sha256="short",
            width=32,
            height=24,
            ground_truth_count=0.0,
        )


def test_sample_rejects_density_whose_sum_does_not_preserve_count(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="density sum"):
        EvaluationSample(
            sample_id="sample-1",
            dataset_id="fixture-v1",
            split_id="validation",
            image_path=tmp_path / "image.png",
            source_sha256="a" * 64,
            width=32,
            height=24,
            ground_truth_count=2.0,
            ground_truth_density=np.ones((3, 3), dtype=np.float32),
        )


def test_failed_prediction_still_rejects_invalid_runtime() -> None:
    with pytest.raises(ValueError, match="runtime values"):
        NativePrediction(
            sample_id="sample-1",
            output_type="count",
            predicted_count=None,
            latency_ms=-1.0,
            peak_vram_mb=0.0,
            failure_state="fixture_failure",
        )


def test_prediction_metadata_accepts_only_named_json_scalars() -> None:
    with pytest.raises(ValueError, match="metadata"):
        NativePrediction(
            sample_id="sample-1",
            output_type="count",
            predicted_count=1.0,
            latency_ms=1.0,
            peak_vram_mb=0.0,
            metadata={"processed_shape": [32, 32]},
        )


@pytest.mark.parametrize(
    ("failure_state", "predicted_count"),
    [(None, 1.0), ("fixture_failure", None)],
)
@pytest.mark.parametrize(
    ("latency_ms", "peak_vram_mb"),
    [
        (float("nan"), 0.0),
        (float("inf"), 0.0),
        (1.0, float("nan")),
        (1.0, float("inf")),
    ],
)
def test_prediction_rejects_non_finite_runtime_values(
    failure_state: str | None,
    predicted_count: float | None,
    latency_ms: float,
    peak_vram_mb: float,
) -> None:
    with pytest.raises(ValueError, match="runtime values"):
        NativePrediction(
            sample_id="sample-1",
            output_type="count",
            predicted_count=predicted_count,
            latency_ms=latency_ms,
            peak_vram_mb=peak_vram_mb,
            failure_state=failure_state,
        )


def test_zone_box_contains_right_and_bottom_exclusively() -> None:
    zone = ZoneBox("zone-a", 0, 0, 10, 10)
    assert zone.contains(0, 0)
    assert zone.contains(9.9, 9.9)
    assert not zone.contains(10, 10)
