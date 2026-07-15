from dataclasses import replace

import pytest

from droneai.evaluation_gate import (
    EvidenceArtifact,
    EvaluationEvidence,
    build_evaluation_checks,
    run_evaluation_gate,
)


def _evidence() -> EvaluationEvidence:
    return EvaluationEvidence(
        run_id="fixture-run",
        expected_samples=12,
        recorded_samples=12,
        successful_samples=12,
        split_verified=True,
        leakage_free=True,
        mae=0.1,
        mae_max=1.0,
        rmse=0.2,
        rmse_max=2.0,
        absolute_bias=0.0,
        bias_max=1.0,
        spatial_pass=True,
        robustness_pass=True,
        median_latency_ms=2.0,
        latency_max_ms=10.0,
        peak_vram_mb=100.0,
        vram_max_mb=1000.0,
        provenance_verified=True,
        model_brief_approved=True,
        review_artifacts_verified=True,
        review_budget_ok=True,
        explicit_failures=0,
        artifact_references=(
            EvidenceArtifact("predictions.csv", "a" * 64),
            EvidenceArtifact("metrics.json", "b" * 64),
        ),
        spatial_target="game_l1 <= 1.0",
        spatial_observed="game_l1=0.1",
        robustness_target="all condition coverage >= 0.95",
        robustness_observed="minimum condition coverage=1.0",
    )


def test_evaluation_checks_total_one_hundred_and_pass() -> None:
    evidence = _evidence()
    checks = build_evaluation_checks(evidence)

    assert sum(check.weight for check in checks) == 100
    report = run_evaluation_gate(evidence)
    assert report.score == 100
    assert report.status == "PASS_RESEARCH_ONLY"
    assert report.decision_scope == "research_only"


def test_incomplete_sample_accounting_is_blocked_even_above_threshold() -> None:
    evidence = replace(_evidence(), recorded_samples=11)
    report = run_evaluation_gate(evidence)

    assert report.score == 90
    assert report.status == "BLOCKED"
    assert "coverage.accounting" in report.to_dict()["failed_blockers"]


def test_failure_count_must_reconcile_with_success_count() -> None:
    evidence = replace(_evidence(), successful_samples=11, explicit_failures=0)

    assert run_evaluation_gate(evidence).status == "BLOCKED"


def test_rights_status_is_not_in_technical_points() -> None:
    evidence = _evidence()

    assert not any("rights" in check.check_id for check in build_evaluation_checks(evidence))


def test_exactly_eighty_points_passes_without_blockers() -> None:
    evidence = replace(
        _evidence(),
        mae=2.0,
        spatial_pass=False,
    )

    report = run_evaluation_gate(evidence)
    assert report.score == 65
    assert report.status == "REVIEW"

    evidence = replace(evidence, spatial_pass=True)
    report = run_evaluation_gate(evidence)
    assert report.score == 80
    assert report.status == "PASS_RESEARCH_ONLY"


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("expected_samples", 0),
        ("recorded_samples", -1),
        ("successful_samples", -1),
        ("explicit_failures", -1),
    ],
)
def test_sample_counts_must_be_valid(field: str, value: int) -> None:
    with pytest.raises(ValueError, match="sample counts"):
        replace(_evidence(), **{field: value})


def test_frozen_thresholds_must_be_finite_and_non_negative() -> None:
    with pytest.raises(ValueError, match="threshold"):
        replace(_evidence(), mae_max=float("inf"))


def test_gate_flags_must_be_actual_booleans() -> None:
    with pytest.raises(ValueError, match="boolean"):
        replace(_evidence(), split_verified="yes")


@pytest.mark.parametrize("value", [float("inf"), "10.0", True])
def test_observed_metrics_must_be_finite_real_numbers(value: object) -> None:
    with pytest.raises(ValueError, match="observed metrics"):
        replace(_evidence(), mae=value)


def test_median_latency_must_be_positive() -> None:
    with pytest.raises(ValueError, match="latency"):
        replace(_evidence(), median_latency_ms=0.0)


def test_run_id_must_be_a_nonblank_string() -> None:
    with pytest.raises(ValueError, match="run_id"):
        replace(_evidence(), run_id=123)


def test_gate_checks_carry_artifact_and_frozen_target_evidence() -> None:
    checks = {check.check_id: check for check in build_evaluation_checks(_evidence())}

    assert "predictions.csv" in checks["provenance.bundle"].evidence
    assert checks["spatial.quality"].expected == "game_l1 <= 1.0"
    assert checks["spatial.quality"].observed == "game_l1=0.1"
    assert checks["condition.robustness"].expected == (
        "all condition coverage >= 0.95"
    )


def test_artifact_references_are_required_and_hash_validated() -> None:
    with pytest.raises(ValueError, match="artifact reference"):
        replace(_evidence(), artifact_references=())
    with pytest.raises(ValueError, match="artifact reference"):
        replace(
            _evidence(),
            artifact_references=(EvidenceArtifact("../escape.json", "c" * 64),),
        )
