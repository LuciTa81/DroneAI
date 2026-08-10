from __future__ import annotations

import json
from pathlib import Path

from droneai.integrity import sha256_file


REPO_ROOT = Path(__file__).resolve().parents[1]
RESULT_ROOT = (
    REPO_ROOT
    / "results/steerer/steerer-qnrf-official-a0-20260810-v1"
)


def test_compact_a0_result_is_complete_hash_verified_and_checkpoint_free() -> None:
    summary = json.loads((RESULT_ROOT / "result-summary.json").read_text(encoding="utf-8"))

    assert summary["stage"] == "A0"
    assert summary["status"] == "PASS_COMMERCIAL_CANDIDATE"
    assert summary["score"] == 100
    assert summary["protocol"]["optimizer_steps"] == 1
    assert summary["protocol"]["validation_population"] == 0
    assert summary["protocol"]["test_access_count"] == 0
    assert summary["protocol"]["official_steerer_checkpoint_loaded"] is False
    assert summary["observation"]["checkpoint_round_trip"] is True
    assert summary["observation"]["rng_round_trip"] is True
    for artifact in summary["local_artifacts"].values():
        path = RESULT_ROOT / artifact["path"]
        assert path.is_file()
        assert sha256_file(path) == artifact["sha256"]
    score = json.loads((RESULT_ROOT / "stage-score.json").read_text(encoding="utf-8"))
    assert score["score"] == 100
    assert score["status"] == "PASS_COMMERCIAL_CANDIDATE"
    assert not list(RESULT_ROOT.glob("*.pth"))
    assert not list(RESULT_ROOT.glob("*.pt"))
