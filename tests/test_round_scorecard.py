from __future__ import annotations

import json
from pathlib import Path

import pytest

from droneai.round_scorecard import load_scorecard, piecewise, score_technical


CONFIG = Path("configs/evaluation/round1_cctv_scorecard.json")


def _aggregates() -> dict[str, float]:
    return {
        "mean_zone_mae": 75.0,
        "spatial_metric_value": 250.0,
        "max_absolute_band_bias": 200.0,
        "mae": 125.0,
        "rmse": 200.0,
        "mape_reference": 20.0,
        "high_band_bias": -200.0,
        "median_latency_ms": 66.67,
        "peak_vram_mb": 8192.0,
    }


def test_repository_scorecard_is_frozen_and_totals_100() -> None:
    config = load_scorecard(CONFIG)

    assert config.required_model_ids == (
        "dm-count",
        "steerer",
        "pet",
        "mpcount",
        "apgcc",
        "csrnet",
    )
    assert sum(config.category_weights.values()) == 100
    assert config.dataset_id == "ucf-qnrf-kaggle-apache"
    assert config.split_id == "dm-count-upstream-qnrf-val-smoke-36-v1"
    assert config.split_role == "validation"
    assert config.expected_samples == 36
    assert config.ranking_eligible is False


def test_piecewise_has_exact_full_linear_and_zero_boundaries() -> None:
    assert piecewise(50.0, 66.67, 500.0) == 1.0
    assert piecewise(500.0, 66.67, 500.0) == 0.0
    assert piecewise((66.67 + 500.0) / 2, 66.67, 500.0) == pytest.approx(0.5)


def test_density_score_uses_frozen_anchors() -> None:
    config = load_scorecard(CONFIG)
    result = score_technical(
        family="density",
        spatial_metric_name="game_l1",
        aggregates=_aggregates(),
        capabilities={
            "zone_aggregation": True,
            "density_map": True,
            "point_localization": False,
            "confidence": False,
        },
        checkpoint_training_split_status="VERIFIED_DISJOINT",
        coverage_complete=True,
        artifacts_verified=True,
        config=config,
    )

    categories = result["categories"]
    assert categories["zone_spatial"] == pytest.approx(17.5)
    assert categories["count_stability"] == pytest.approx(12.5)
    assert categories["runtime_resources"] == pytest.approx(15.0)
    assert categories["operator_integration"] == pytest.approx(10.0)
    assert categories["evidence_quality"] == pytest.approx(10.0)
    assert result["technical_score"] == pytest.approx(65.0)


def test_point_score_uses_localization_f1_without_density_credit() -> None:
    config = load_scorecard(CONFIG)
    aggregates = _aggregates()
    aggregates["spatial_metric_value"] = 0.4

    result = score_technical(
        family="points",
        spatial_metric_name="localization_f1",
        aggregates=aggregates,
        capabilities={
            "zone_aggregation": True,
            "density_map": False,
            "point_localization": True,
            "confidence": False,
        },
        checkpoint_training_split_status="UNKNOWN",
        coverage_complete=True,
        artifacts_verified=True,
        config=config,
    )

    assert result["categories"]["zone_spatial"] == pytest.approx(17.5)
    assert result["categories"]["operator_integration"] == pytest.approx(9.0)
    assert result["categories"]["evidence_quality"] == pytest.approx(8.0)


def test_unknown_family_is_rejected() -> None:
    config = load_scorecard(CONFIG)

    with pytest.raises(ValueError, match="unsupported model family"):
        score_technical(
            family="detector",
            spatial_metric_name="game_l1",
            aggregates=_aggregates(),
            capabilities={},
            checkpoint_training_split_status="UNKNOWN",
            coverage_complete=True,
            artifacts_verified=True,
            config=config,
        )


def test_scorecard_rejects_non_100_weight_total(tmp_path: Path) -> None:
    payload = json.loads(CONFIG.read_text(encoding="utf-8"))
    payload["category_weights"]["evidence_quality"] = 9
    changed = tmp_path / "scorecard.json"
    changed.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="category weights must total 100"):
        load_scorecard(changed)
