from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from droneai.round_comparison import build_round_comparison, deployment_label
from droneai.round_evidence import ModelRunEvidence, load_verified_round
from droneai.round_scorecard import ScorecardConfig
from tests.round_comparison_helpers import make_round_fixture


@pytest.fixture
def verified_round(
    tmp_path: Path,
) -> tuple[tuple[ModelRunEvidence, ...], ScorecardConfig]:
    queue, results, config = make_round_fixture(tmp_path)
    return load_verified_round(queue, results, config), config


def test_deployment_labels_never_infer_production_permission() -> None:
    assert deployment_label("PRODUCTION_APPROVED") == "PRODUCT_ELIGIBLE"
    assert (
        deployment_label("PASS_COMMERCIAL_CANDIDATE")
        == "PENDING_DUE_DILIGENCE"
    )
    assert deployment_label("PASS_RESEARCH_ONLY") == "RESEARCH_ONLY"
    assert deployment_label("BLOCKED_EXPLICIT_RESTRICTION") == "BLOCKED"
    with pytest.raises(ValueError, match="unsupported rights scope"):
        deployment_label("UNKNOWN")


def test_rights_change_does_not_change_technical_score(
    verified_round: tuple[tuple[ModelRunEvidence, ...], ScorecardConfig],
) -> None:
    runs, config = verified_round
    first = build_round_comparison(runs, config, git_commit="a" * 40)
    changed = tuple(replace(run, rights_scope="PASS_RESEARCH_ONLY") for run in runs)
    second = build_round_comparison(changed, config, git_commit="a" * 40)

    assert [row["technical_score"] for row in first["models"]] == [
        row["technical_score"] for row in second["models"]
    ]
    assert {row["deployment_label"] for row in second["models"]} == {
        "RESEARCH_ONLY"
    }


def test_top_three_uses_unrounded_score_and_exposes_independent_rights(
    verified_round: tuple[tuple[ModelRunEvidence, ...], ScorecardConfig],
) -> None:
    runs, config = verified_round
    comparison = build_round_comparison(runs, config, git_commit="b" * 40)

    assert comparison["shortlist"]["label"] == (
        "PROVISIONAL_TECHNICAL_SHORTLIST"
    )
    assert len(comparison["shortlist"]["model_ids"]) == 3
    assert comparison["ranking_eligible"] is False
    rows = {row["model_id"]: row for row in comparison["models"]}
    assert all(
        "categories" in rows[model_id]
        for model_id in comparison["shortlist"]["model_ids"]
    )
    assert set(rows["csrnet"]["rights_components"]) == {
        "code",
        "dataset",
        "pretrained_weights",
        "derived_weights",
        "deployment",
    }
    assert rows["csrnet"]["rights_components"]["pretrained_weights"]["status"] == (
        "PENDING"
    )
    assert rows["csrnet"]["representative_panels"][0].startswith("../csrnet/")
    assert comparison["dataset"]["canonical_split_sha256"] == (
        runs[0].canonical_split_sha256
    )


def test_unrounded_score_precedes_frozen_tie_breakers(
    verified_round: tuple[tuple[ModelRunEvidence, ...], ScorecardConfig],
) -> None:
    runs, config = verified_round
    base = runs[0]
    almost = replace(
        runs[1],
        aggregates={**base.aggregates, "mean_zone_mae": 9.999999},
        metrics={**base.metrics},
        capabilities={**base.capabilities},
        family=base.family,
    )
    tied = replace(
        runs[2],
        aggregates={**base.aggregates, "mean_zone_mae": 10.0},
        metrics={**base.metrics},
        capabilities={**base.capabilities},
        family=base.family,
    )
    comparison = build_round_comparison(
        (base, almost, tied, *runs[3:]), config, git_commit="c" * 40
    )

    assert comparison["shortlist"]["model_ids"][0] == almost.model_id


def test_comparison_requires_full_git_commit_and_frozen_run_order(
    verified_round: tuple[tuple[ModelRunEvidence, ...], ScorecardConfig],
) -> None:
    runs, config = verified_round
    with pytest.raises(ValueError, match="full clean Git commit"):
        build_round_comparison(runs, config, git_commit="abc123")
    with pytest.raises(ValueError, match="run order"):
        build_round_comparison(tuple(reversed(runs)), config, git_commit="d" * 40)
