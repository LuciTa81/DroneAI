from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from droneai.evaluation_contract import (
    EvaluationSample,
    NativePrediction,
    ScalarEvaluation,
    ZoneBox,
)
from droneai.evaluation_panels import render_review_panel, zone_counts
from droneai.integrity import sha256_file


def _sample(tmp_path: Path) -> EvaluationSample:
    image_path = tmp_path / "source.png"
    Image.new("RGB", (64, 48), "gray").save(image_path)
    return EvaluationSample(
        sample_id="s1",
        dataset_id="fixture",
        split_id="validation",
        image_path=image_path,
        source_sha256=sha256_file(image_path),
        width=64,
        height=48,
        ground_truth_count=2.0,
        ground_truth_points=((10.0, 10.0), (40.0, 30.0)),
        ground_truth_density=np.ones((12, 16), dtype=np.float32) / 96.0,
        condition_tags={"density_band": "medium"},
        zones=(
            ZoneBox("a", 0, 0, 32, 48),
            ZoneBox("b", 32, 0, 64, 48),
        ),
        has_point_annotations=True,
    )


def _density_prediction() -> NativePrediction:
    return NativePrediction(
        "s1",
        "density",
        2.0,
        5.0,
        100.0,
        density=np.ones((12, 16), dtype=np.float32) / 96.0,
    )


def _density_record() -> ScalarEvaluation:
    return ScalarEvaluation(
        "s1",
        2.0,
        2.0,
        0.0,
        0.0,
        0.0,
        "medium",
        5.0,
        100.0,
        "density",
        None,
        "game_l1",
        0.0,
        {"density_ssim": 1.0},
    )


def _point_record(output_type: str = "points") -> ScalarEvaluation:
    return ScalarEvaluation(
        "s1",
        2.0,
        2.0,
        0.0,
        0.0,
        0.0,
        "medium",
        5.0,
        100.0,
        output_type,
        None,
        "localization_f1",
        1.0,
        {"localization_f1": 1.0},
    )


def test_density_panel_has_four_equal_columns(tmp_path: Path) -> None:
    path = render_review_panel(
        _sample(tmp_path),
        _density_prediction(),
        _density_record(),
        tmp_path / "panel.png",
        model_id="fixture",
        checkpoint_sha256="b" * 64,
        zone_warning_count=1.0,
        zone_critical_count=2.0,
        category="typical",
    )

    with Image.open(path) as rendered:
        assert rendered.size == (4 * 320, 365)
        assert rendered.mode == "RGB"


def test_density_panel_uses_declared_operational_output_label(tmp_path: Path) -> None:
    prediction = replace(
        _density_prediction(),
        metadata={"spatial_output_label": "operational clipped density"},
    )

    path = render_review_panel(
        _sample(tmp_path),
        prediction,
        _density_record(),
        tmp_path / "operational-density.png",
        model_id="fixture",
        checkpoint_sha256="b" * 64,
        zone_warning_count=1.0,
        zone_critical_count=2.0,
        category="compatibility",
    )

    with Image.open(path) as rendered:
        columns = rendered.info["panel_columns"]
        assert "operational clipped density" in columns
        assert "native predicted density" not in columns


def test_panel_records_visible_count_error_and_category_annotations(
    tmp_path: Path,
) -> None:
    path = render_review_panel(
        _sample(tmp_path),
        _density_prediction(),
        _density_record(),
        tmp_path / "annotations.png",
        model_id="fixture",
        checkpoint_sha256="b" * 64,
        zone_warning_count=1.0,
        zone_critical_count=2.0,
        category="typical",
    )

    with Image.open(path) as rendered:
        annotations = rendered.info["panel_annotations"]
        assert "category=typical" in annotations
        assert "density_band=medium" in annotations
        assert "gt=2.00 pred=2.00" in annotations
        assert "signed_error=0.00 abs_error=0.00" in annotations
        assert "game_l1=0.000" in annotations
        assert "confidence=unavailable" in annotations


def test_repeated_render_is_byte_deterministic(tmp_path: Path) -> None:
    sample = _sample(tmp_path)
    paths = []
    for name in ("first.png", "second.png"):
        paths.append(
            render_review_panel(
                sample,
                _density_prediction(),
                _density_record(),
                tmp_path / name,
                model_id="fixture",
                checkpoint_sha256="b" * 64,
                zone_warning_count=1.0,
                zone_critical_count=2.0,
                category="best",
            )
        )

    assert sha256_file(paths[0]) == sha256_file(paths[1])


def test_point_panel_uses_native_points_without_derived_density_claim(
    tmp_path: Path,
) -> None:
    prediction = NativePrediction(
        "s1",
        "points",
        2.0,
        5.0,
        100.0,
        points=((11.0, 11.0), (41.0, 31.0)),
    )
    record = ScalarEvaluation(
        "s1",
        2.0,
        2.0,
        0.0,
        0.0,
        0.0,
        "medium",
        5.0,
        100.0,
        "points",
        None,
        "localization_f1",
        1.0,
        {"localization_f1": 1.0},
    )

    path = render_review_panel(
        _sample(tmp_path),
        prediction,
        record,
        tmp_path / "points.png",
        model_id="fixture",
        checkpoint_sha256="b" * 64,
        zone_warning_count=1.0,
        zone_critical_count=2.0,
        category="strong",
    )

    with Image.open(path) as rendered:
        columns = rendered.info["panel_columns"]
        assert "native predicted points" in columns
        assert "derived" not in columns
        predicted_column = np.asarray(rendered.crop((640, 30, 960, 270)))
        red_point_pixels = (
            (predicted_column[:, :, 0] > 200)
            & (predicted_column[:, :, 1] < 100)
            & (predicted_column[:, :, 2] < 100)
        )
        assert red_point_pixels.any()


def test_count_only_spatial_outputs_are_explicitly_unavailable(
    tmp_path: Path,
) -> None:
    sample = replace(
        _sample(tmp_path),
        ground_truth_points=(),
        ground_truth_density=None,
        has_point_annotations=False,
    )
    prediction = NativePrediction("s1", "count", 2.0, 5.0, 100.0)
    record = ScalarEvaluation(
        "s1",
        2.0,
        2.0,
        0.0,
        0.0,
        0.0,
        "medium",
        5.0,
        100.0,
        "count",
        None,
    )

    path = render_review_panel(
        sample,
        prediction,
        record,
        tmp_path / "count-only.png",
        model_id="fixture",
        checkpoint_sha256="b" * 64,
        zone_warning_count=1.0,
        zone_critical_count=2.0,
    )

    with Image.open(path) as rendered:
        columns = rendered.info["panel_columns"]
        assert "ground-truth spatial output unavailable" in columns
        assert "native spatial output unavailable" in columns
        assert "operator zones unavailable" in columns
        assert "category=unavailable" in rendered.info["panel_annotations"]


def test_operator_column_marks_missing_declared_zones_unavailable(
    tmp_path: Path,
) -> None:
    sample = replace(_sample(tmp_path), zones=())

    path = render_review_panel(
        sample,
        _density_prediction(),
        _density_record(),
        tmp_path / "no-zones.png",
        model_id="fixture",
        checkpoint_sha256="b" * 64,
        zone_warning_count=1.0,
        zone_critical_count=2.0,
    )

    with Image.open(path) as rendered:
        assert "operator zones unavailable: none declared" in rendered.info[
            "panel_columns"
        ]


def test_failed_prediction_is_unavailable_and_never_rendered_as_zero(
    tmp_path: Path,
) -> None:
    prediction = NativePrediction(
        "s1",
        "count",
        None,
        5.0,
        100.0,
        failure_state="cuda_out_of_memory",
    )
    record = ScalarEvaluation(
        "s1",
        2.0,
        None,
        None,
        None,
        None,
        "medium",
        5.0,
        100.0,
        "count",
        "cuda_out_of_memory",
    )

    path = render_review_panel(
        _sample(tmp_path),
        prediction,
        record,
        tmp_path / "failure.png",
        model_id="fixture",
        checkpoint_sha256="b" * 64,
        zone_warning_count=1.0,
        zone_critical_count=2.0,
        category="failure",
    )

    with Image.open(path) as rendered:
        columns = rendered.info["panel_columns"]
        annotations = rendered.info["panel_annotations"]
        assert "prediction unavailable: cuda_out_of_memory" in columns
        assert "operator zones unavailable" in columns
        assert "pred=unavailable" in annotations
        assert "abs_error=unavailable" in annotations
        assert "failure=cuda_out_of_memory" in annotations
        assert "pred=0" not in annotations

    with pytest.raises(ValueError, match="zone counts unavailable"):
        zone_counts(_sample(tmp_path), prediction)


def test_zone_counts_use_the_retained_native_family_output(tmp_path: Path) -> None:
    sample = _sample(tmp_path)
    density_counts = zone_counts(sample, _density_prediction())
    point_counts = zone_counts(
        sample,
        NativePrediction(
            "s1",
            "points",
            2.0,
            5.0,
            100.0,
            points=((11.0, 11.0), (41.0, 31.0)),
        ),
    )

    assert density_counts == pytest.approx({"a": 1.0, "b": 1.0})
    assert point_counts == {"a": 1.0, "b": 1.0}


def test_count_output_ignores_stray_spatial_payload(tmp_path: Path) -> None:
    sample = _sample(tmp_path)
    prediction = NativePrediction(
        "s1",
        "count",
        2.0,
        5.0,
        100.0,
        density=np.ones((12, 16), dtype=np.float32) / 96.0,
        points=((11.0, 11.0), (20.0, 20.0)),
    )
    record = replace(_density_record(), output_type="count")

    path = render_review_panel(
        sample,
        prediction,
        record,
        tmp_path / "count-with-stray-spatial.png",
        model_id="fixture",
        checkpoint_sha256="b" * 64,
        zone_warning_count=1.0,
        zone_critical_count=2.0,
    )

    with Image.open(path) as rendered:
        assert "native spatial output unavailable" in rendered.info[
            "panel_columns"
        ]
        assert "operator zones unavailable" in rendered.info["panel_columns"]
    with pytest.raises(ValueError, match="zone counts unavailable"):
        zone_counts(sample, prediction)


def test_point_output_uses_point_gt_and_ignores_stray_density(
    tmp_path: Path,
) -> None:
    sample = _sample(tmp_path)
    prediction = NativePrediction(
        "s1",
        "points",
        2.0,
        5.0,
        100.0,
        density=np.ones((12, 16), dtype=np.float32) / 96.0,
        points=((11.0, 11.0), (20.0, 20.0)),
    )

    path = render_review_panel(
        sample,
        prediction,
        _point_record(),
        tmp_path / "points-with-stray-density.png",
        model_id="fixture",
        checkpoint_sha256="b" * 64,
        zone_warning_count=1.0,
        zone_critical_count=2.0,
    )

    with Image.open(path) as rendered:
        columns = rendered.info["panel_columns"]
        assert "ground-truth points" in columns
        assert "native predicted points" in columns
        assert "predicted density" not in columns
    assert zone_counts(sample, prediction) == {"a": 2.0, "b": 0.0}


def test_point_derived_ground_truth_density_is_labelled_derived(
    tmp_path: Path,
) -> None:
    path = render_review_panel(
        _sample(tmp_path),
        _density_prediction(),
        _density_record(),
        tmp_path / "derived-ground-truth.png",
        model_id="fixture",
        checkpoint_sha256="b" * 64,
        zone_warning_count=1.0,
        zone_critical_count=2.0,
    )

    with Image.open(path) as rendered:
        assert "derived ground-truth density" in rendered.info["panel_columns"]


def test_hybrid_output_uses_density_precedence_with_explicit_label(
    tmp_path: Path,
) -> None:
    sample = _sample(tmp_path)
    prediction = NativePrediction(
        "s1",
        "hybrid",
        2.0,
        5.0,
        100.0,
        density=np.ones((12, 16), dtype=np.float32) / 96.0,
        points=((11.0, 11.0), (20.0, 20.0)),
    )

    path = render_review_panel(
        sample,
        prediction,
        replace(_density_record(), output_type="hybrid"),
        tmp_path / "hybrid-both.png",
        model_id="fixture",
        checkpoint_sha256="b" * 64,
        zone_warning_count=1.0,
        zone_critical_count=2.0,
    )

    with Image.open(path) as rendered:
        columns = rendered.info["panel_columns"]
        assert "native predicted density (hybrid)" in columns
        assert "operator zones (hybrid density)" in columns
    assert zone_counts(sample, prediction) == pytest.approx({"a": 1.0, "b": 1.0})


def test_point_only_hybrid_uses_points_with_explicit_label(tmp_path: Path) -> None:
    sample = _sample(tmp_path)
    prediction = NativePrediction(
        "s1",
        "hybrid",
        2.0,
        5.0,
        100.0,
        points=((11.0, 11.0), (20.0, 20.0)),
    )

    path = render_review_panel(
        sample,
        prediction,
        replace(_point_record(), output_type="hybrid"),
        tmp_path / "hybrid-points.png",
        model_id="fixture",
        checkpoint_sha256="b" * 64,
        zone_warning_count=1.0,
        zone_critical_count=2.0,
    )

    with Image.open(path) as rendered:
        columns = rendered.info["panel_columns"]
        assert "ground-truth points" in columns
        assert "native predicted points (hybrid)" in columns
        assert "operator zones (hybrid points)" in columns
    assert zone_counts(sample, prediction) == {"a": 2.0, "b": 0.0}


@pytest.mark.parametrize(
    ("record_field", "mismatched_value"),
    (
        ("ground_truth_count", 3.0),
        ("predicted_count", 3.0),
    ),
)
def test_record_numeric_fields_must_match_sample_and_prediction(
    tmp_path: Path,
    record_field: str,
    mismatched_value: float,
) -> None:
    target = tmp_path / "numeric-mismatch.png"

    with pytest.raises(ValueError, match=rf"{record_field} mismatch"):
        render_review_panel(
            _sample(tmp_path),
            _density_prediction(),
            replace(_density_record(), **{record_field: mismatched_value}),
            target,
            model_id="fixture",
            checkpoint_sha256="b" * 64,
            zone_warning_count=1.0,
            zone_critical_count=2.0,
        )

    assert not target.exists()


def test_selected_rerun_runtime_can_differ_from_first_pass_record(tmp_path: Path) -> None:
    path = render_review_panel(
        _sample(tmp_path),
        _density_prediction(),
        replace(_density_record(), latency_ms=6.0, peak_vram_mb=101.0),
        tmp_path / "runtime-rerun.png",
        model_id="fixture",
        checkpoint_sha256="b" * 64,
        zone_warning_count=1.0,
        zone_critical_count=2.0,
    )

    assert path.is_file()


@pytest.mark.parametrize(
    ("record_field", "nonfinite_value"),
    (
        ("ground_truth_count", float("nan")),
        ("predicted_count", float("inf")),
        ("latency_ms", float("nan")),
        ("peak_vram_mb", float("inf")),
    ),
)
def test_record_numeric_integrity_rejects_nonfinite_values(
    tmp_path: Path,
    record_field: str,
    nonfinite_value: float,
) -> None:
    with pytest.raises(ValueError, match=rf"{record_field} must be finite"):
        render_review_panel(
            _sample(tmp_path),
            _density_prediction(),
            replace(_density_record(), **{record_field: nonfinite_value}),
            tmp_path / "nonfinite.png",
            model_id="fixture",
            checkpoint_sha256="b" * 64,
            zone_warning_count=1.0,
            zone_critical_count=2.0,
        )


def test_predicted_count_none_must_match_across_prediction_and_record(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="predicted_count mismatch"):
        render_review_panel(
            _sample(tmp_path),
            _density_prediction(),
            replace(_density_record(), predicted_count=None),
            tmp_path / "none-mismatch.png",
            model_id="fixture",
            checkpoint_sha256="b" * 64,
            zone_warning_count=1.0,
            zone_critical_count=2.0,
        )


@pytest.mark.parametrize(
    "mismatch",
    ("prediction_sample_id", "record_sample_id", "output_type", "failure_state"),
)
def test_cross_object_identity_fields_must_match(
    tmp_path: Path,
    mismatch: str,
) -> None:
    prediction = _density_prediction()
    record = _density_record()
    expected = mismatch
    if mismatch == "prediction_sample_id":
        prediction = replace(prediction, sample_id="other")
        expected = "sample_id mismatch"
    elif mismatch == "record_sample_id":
        record = replace(record, sample_id="other")
        expected = "sample_id mismatch"
    elif mismatch == "output_type":
        record = replace(record, output_type="points")
        expected = "output_type mismatch"
    else:
        record = replace(record, failure_state="fixture_failure")
        expected = "failure_state mismatch"

    with pytest.raises(ValueError, match=expected):
        render_review_panel(
            _sample(tmp_path),
            prediction,
            record,
            tmp_path / "identity-mismatch.png",
            model_id="fixture",
            checkpoint_sha256="b" * 64,
            zone_warning_count=1.0,
            zone_critical_count=2.0,
        )


def test_panel_output_requires_png_extension(tmp_path: Path) -> None:
    target = tmp_path / "panel.jpg"

    with pytest.raises(ValueError, match="output path must end with .png"):
        render_review_panel(
            _sample(tmp_path),
            _density_prediction(),
            _density_record(),
            target,
            model_id="fixture",
            checkpoint_sha256="b" * 64,
            zone_warning_count=1.0,
            zone_critical_count=2.0,
        )

    assert not target.exists()


def test_panel_output_cannot_resolve_to_source_image(tmp_path: Path) -> None:
    sample = _sample(tmp_path)
    source_hash = sha256_file(sample.image_path)

    with pytest.raises(ValueError, match="output path resolves to source image"):
        render_review_panel(
            sample,
            _density_prediction(),
            _density_record(),
            sample.image_path,
            model_id="fixture",
            checkpoint_sha256="b" * 64,
            zone_warning_count=1.0,
            zone_critical_count=2.0,
        )

    assert sha256_file(sample.image_path) == source_hash


def test_panel_output_refuses_to_overwrite_existing_target(tmp_path: Path) -> None:
    target = tmp_path / "existing.png"
    Image.new("RGB", (3, 2), "purple").save(target)
    existing_hash = sha256_file(target)

    with pytest.raises(FileExistsError, match="panel target already exists"):
        render_review_panel(
            _sample(tmp_path),
            _density_prediction(),
            _density_record(),
            target,
            model_id="fixture",
            checkpoint_sha256="b" * 64,
            zone_warning_count=1.0,
            zone_critical_count=2.0,
        )

    assert sha256_file(target) == existing_hash
