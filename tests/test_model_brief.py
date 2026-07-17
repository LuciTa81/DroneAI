import json
from dataclasses import replace
from pathlib import Path

import pytest

from droneai.model_brief import ModelBrief, render_model_brief, write_model_brief


REPO_ROOT = Path(__file__).parents[1]


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


def test_approved_brief_renders_complete_required_structure() -> None:
    rendered = render_model_brief(_brief())
    required_fragments = (
        "# Model brief: fixture-density",
        "- Paper: Fixture Density Paper",
        "- Role: density baseline",
        "- Family: density",
        "- Backbone: three-layer fixture CNN",
        "- Parameter count: 128",
        "- Input: RGB NCHW float32 normalized to [0,1]",
        "- Preprocessing: resize to 32x32 without tiling",
        "- Coordinate transform: density stride 4 to original pixels",
        "- Native output: one-channel density map at stride 4",
        "- Count derivation: sum of native density map",
        "- Zone derivation: integrate density inside scaled zone boxes",
        "- Official protocol: fixture validation split",
        "- Rights status: PASS_RESEARCH_ONLY",
        "- Code rights: project fixture code",
        "- Dataset rights: project synthetic data",
        "- Checkpoint rights: project synthetic checkpoint",
        "- Deployment rights: not production evidence",
        f"- Upstream commit: `{'a' * 40}`",
        "- Checkpoint path: `fixture-checkpoint.bin`",
        f"- Checkpoint SHA-256: `{'b' * 64}`",
        "## Major blocks\n\n- frontend\n- density head",
        "## Feature scales\n\n- stride 4 density features",
        "## Original losses\n\n- pixel MSE weight=1.0",
        "## Official reported metrics\n\n- fixture MAE 0.0",
        "## Strengths\n\n- deterministic",
        "## Failure modes\n\n- texture false positives",
        "## Runtime risks\n\n- none for fixture",
        "## Reviewed code\n\n- src/droneai/model_brief.py",
        "Review status: **approved**",
    )
    assert all(fragment in rendered for fragment in required_fragments)


def test_approved_brief_with_reviewed_code_authorizes_full_run() -> None:
    assert _brief().require_full_run_approval() is None


def test_pending_brief_cannot_authorize_full_run() -> None:
    brief = _brief("pending")
    with pytest.raises(ValueError, match="approved model brief"):
        brief.require_full_run_approval()


@pytest.mark.parametrize(
    "reviewed_paths",
    [(), ("",), (" \t ",)],
    ids=("empty-tuple", "blank-string", "whitespace-only"),
)
def test_approved_brief_requires_nonblank_reviewed_code_paths(
    reviewed_paths: tuple[str, ...],
) -> None:
    brief = replace(_brief(), reviewed_paths=reviewed_paths)
    with pytest.raises(ValueError, match="reviewed code paths"):
        brief.require_full_run_approval()


def test_brief_rejects_missing_architecture_lists() -> None:
    with pytest.raises(ValueError, match="architecture and review lists"):
        replace(_brief(), blocks=())


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("model_id", ""),
        ("backbone", " \t "),
        ("checkpoint_path", "\n"),
    ),
)
def test_brief_rejects_blank_required_text(field: str, value: str) -> None:
    with pytest.raises(ValueError, match="fields cannot be empty"):
        replace(_brief(), **{field: value})


def test_brief_rejects_negative_parameter_count() -> None:
    with pytest.raises(ValueError, match="parameter count must be non-negative"):
        replace(_brief(), parameter_count=-1)


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("upstream_commit", "not-a-commit"),
        ("checkpoint_sha256", "not-a-checkpoint-hash"),
    ),
)
def test_brief_rejects_invalid_source_identity(field: str, value: str) -> None:
    with pytest.raises(ValueError, match="commit and checkpoint SHA-256"):
        replace(_brief(), **{field: value})


def test_write_model_brief_persists_rendered_markdown(tmp_path: Path) -> None:
    brief = _brief()
    target = tmp_path / "briefs" / "fixture-density.md"

    written = write_model_brief(target, brief)

    assert written == target
    assert target.read_text(encoding="utf-8") == render_model_brief(brief)


def test_steerer_contract_separates_paper_and_harness_localization() -> None:
    config = json.loads((REPO_ROOT / "configs/models/steerer.official.json").read_text())
    assert config["architecture"]["backbone"] == "HRNet-W48"
    assert config["architecture"]["output_type"] == "hybrid"
    assert config["architecture"]["blocks"] == [
        "HRNet-W48 multi-resolution backbone",
        "multi-resolution counting heads",
        "selective inheritance/upsample module",
        "Gaussian density/local-maximum point decoder",
    ]
    assert config["architecture"]["feature_scales"] == ["x1", "x4", "x8"]
    assert config["inference"]["long_side_cap"] == 3072
    assert config["inference"]["den_factor"] == 100
    assert config["preprocessing"] == {
        "input_color_space": "RGB",
        "normalization": {
            "policy": "ImageNet mean/std",
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
        },
        "resize_policy": {
            "preserve_native_size_when_long_side_lte": 3072,
            "resize_only_when_long_side_gt": 3072,
            "resized_long_side": 3072,
        },
        "padding_policy": {
            "mode": "zero",
            "dimensions": ["height", "width"],
            "multiple": 32,
        },
    }
    assert config["original_training_losses"] == [
        "multi-resolution MSE",
        "selective inheritance routing loss",
        "uncertainty weighting where configured",
    ]
    assert config["reported_results"]["ucf_qnrf"] == {
        "mae": 77.8, "rmse": 138.0, "f1": 75.6, "precision": 79.7, "recall": 72.0
    }
    assert config["measured_localization_metric"] == "harness F1@16px in original-image coordinates"
    assert config["failure_modes"] == [
        "tiny heads",
        "extreme perspective",
        "occlusion",
        "domain shift",
        "padding artifacts",
        "threshold-sensitive localization",
    ]

    brief = (REPO_ROOT / "docs/models/STEERER.md").read_text()
    required_preprocessing = (
        "- Input color space: RGB",
        "- Normalization: ImageNet mean `[0.485, 0.456, 0.406]` and std `[0.229, 0.224, 0.225]`",
        "- Resize: preserve native size when long side <= 3072; resize only when above 3072, setting the long side to 3072",
        "- Padding: zero-pad both dimensions to multiples of 32",
    )
    assert all(line in brief for line in required_preprocessing)
    output_semantics = "\n".join(
        (
            "estimated_count = sum(highest_resolution_density / 100)",
            "localization_points = official local maxima merged from x1/x4/x8",
            "zone_count = integral of count-preserving density in the calibrated image zone",
            "paper localization F1 != harness F1@16px",
        )
    )
    assert f"```text\n{output_semantics}\n```" in brief


def test_pet_contract_is_point_only_and_research_only() -> None:
    config = json.loads((REPO_ROOT / "configs/models/pet.official.json").read_text())

    assert config["repository"] == {
        "url": "https://github.com/cxliu0/PET",
        "commit": "5b4dd7da8b11568a3305a88bb7c99a7fc831a998",
        "rights_scope": "PASS_RESEARCH_ONLY",
    }
    assert config["architecture"]["backbone"] == "VGG16-BN"
    assert config["architecture"]["output_type"] == "points"
    assert config["architecture"]["query_stride"] == 8
    assert config["architecture"]["maximum_quadtree_depth"] == 2
    assert config["architecture"]["hidden_dimension"] == 256
    assert config["architecture"]["attention_heads"] == 8
    assert config["architecture"]["encoder_layers"] == 4
    assert config["architecture"]["decoder_layers"] == 2
    assert config["inference"]["point_probability_threshold"] == 0.5
    assert config["inference"]["ucf_qnrf_long_side_cap"] == 1536
    assert config["outputs"] == {
        "native": [
            "point/non-point class probabilities",
            "normalized point coordinates",
            "quadtree split map",
        ],
        "count_derivation": "number of point queries above the point-class probability threshold",
        "density_map_available": False,
        "localization_available": True,
        "zone_derivation": "count predicted points whose original-image coordinates fall inside each calibrated zone",
    }
    assert config["reported_results"]["ucf_qnrf"] == {"mae": 79.53, "rmse": 144.32}

    brief = (REPO_ROOT / "docs/models/PET.md").read_text()
    required_fragments = (
        "# Model brief: pet-official-ucf-qnrf",
        "- Rights scope: `PASS_RESEARCH_ONLY`",
        "- Native output: point/non-point probabilities, normalized point coordinates, and quadtree split map",
        "- Density map: unavailable natively",
        "estimated_count = number of point queries with point_probability > 0.5",
        "zone_count = number of predicted points inside the calibrated image zone",
        "Commercial fine-tuning and deployment are blocked",
        "python scripts/setup_pet_home5090.py --verify-only",
        "MAE: 79.53",
        "RMSE: 144.32",
    )
    assert all(fragment in brief for fragment in required_fragments)
