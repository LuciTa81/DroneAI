from __future__ import annotations

import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
RESULT_ROOT = REPO_ROOT / "results/steerer/steerer-qnrf-official-a1-20260810-v2"


def test_compact_a1_result_is_complete_checkpoint_free_and_scoped() -> None:
    summary = json.loads((RESULT_ROOT / "result-summary.json").read_text(encoding="utf-8"))

    assert summary["stage"] == "A1"
    assert summary["status"] == "PASS_COMMERCIAL_CANDIDATE"
    assert summary["harness_score"] == 100
    assert summary["model_performance_scope"] == "epoch-one diagnostic only"
    assert summary["protocol"]["optimizer_steps"] == 150
    assert summary["protocol"]["train_samples_consumed"] == 1200
    assert summary["protocol"]["validation_population"] == 0
    assert summary["protocol"]["test_access_count"] == 334
    assert summary["protocol"]["test_selection"] == "official Test, test-selected"
    assert summary["protocol"]["official_steerer_checkpoint_loaded"] is False
    assert summary["observation"]["checkpoint_round_trip"] is True
    assert summary["observation"]["rng_round_trip"] is True
    assert summary["repeatability"]["bitwise_deterministic"] is False
    assert summary["ssd_artifacts"]["checkpoint"]["global_step"] == 150
    assert summary["rights"]["production_approved"] is False
    assert not list(RESULT_ROOT.glob("*.pth"))
    assert not list(RESULT_ROOT.glob("*.pt"))
