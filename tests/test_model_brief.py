from dataclasses import replace

import pytest

from droneai.model_brief import ModelBrief, render_model_brief


def _brief(review_status: str = "approved") -> ModelBrief:
    return ModelBrief(
        model_id="fixture-density",
        paper="Fixture Density Paper",
        role="density baseline",
        family="density",
        backbone="three-layer fixture CNN",
        parameter_count=128,
        blocks=("frontend", "density head"),
        feature_scales=("stride 4 density features",),
        input_contract="RGB NCHW float32 normalized to [0,1]",
        preprocessing_policy="resize to 32x32 without tiling",
        coordinate_transform="density stride 4 to original pixels",
        native_output="one-channel density map at stride 4",
        count_derivation="sum of native density map",
        zone_derivation="integrate density inside scaled zone boxes",
        original_losses=("pixel MSE weight=1.0",),
        official_protocol="fixture validation split",
        official_reported_metrics=("fixture MAE 0.0",),
        strengths=("deterministic",),
        failure_modes=("texture false positives",),
        runtime_risks=("none for fixture",),
        rights_status="PASS_RESEARCH_ONLY",
        code_rights_status="project fixture code",
        dataset_rights_status="project synthetic data",
        checkpoint_rights_status="project synthetic checkpoint",
        deployment_rights_status="not production evidence",
        upstream_commit="a" * 40,
        checkpoint_path="fixture-checkpoint.bin",
        checkpoint_sha256="b" * 64,
        reviewed_paths=("src/droneai/model_brief.py",),
        review_status=review_status,
    )


def test_approved_brief_renders_required_architecture_and_io() -> None:
    rendered = render_model_brief(_brief())
    assert "three-layer fixture CNN" in rendered
    assert "one-channel density map" in rendered
    assert "sum of native density map" in rendered
    assert "pixel MSE weight=1.0" in rendered
    assert "stride 4 density features" in rendered
    assert "integrate density inside scaled zone boxes" in rendered


def test_pending_brief_cannot_authorize_full_run() -> None:
    brief = _brief("pending")
    with pytest.raises(ValueError, match="approved model brief"):
        brief.require_full_run_approval()


def test_brief_rejects_missing_architecture_lists() -> None:
    with pytest.raises(ValueError, match="architecture and review lists"):
        replace(_brief(), blocks=())
