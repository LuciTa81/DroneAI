from __future__ import annotations

import json
from pathlib import Path

import pytest

from droneai.integrity import sha256_file


ROOT = (
    Path(__file__).resolve().parents[1]
    / "results/steerer/steerer-a-g1-official-checkpoint-test334-preprocessed-v2"
)


def _json(name: str) -> dict[str, object]:
    return json.loads((ROOT / name).read_text(encoding="utf-8"))


def test_g1_compact_result_preserves_complete_passing_evidence() -> None:
    metrics = _json("metrics.json")
    gate = _json("compatibility-gate.json")
    score = _json("score.json")
    preparation = _json("test-preparation.json")

    assert metrics["expected_samples"] == 334
    assert metrics["recorded_samples"] == 334
    assert metrics["successful_samples"] == 334
    assert metrics["explicit_failures"] == 0
    assert metrics["mae"] == pytest.approx(77.5164929266124)
    assert metrics["rmse"] == pytest.approx(139.08674421137636)
    assert gate["status"] == "PASS_RESEARCH_ONLY"
    assert gate["training_authorized"] is True
    assert score["score"] == 100
    assert score["failed_blockers"] == []
    assert preparation["sample_count"] == 334


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        (
            "metrics.json",
            "6520a833bda0ff3c90d292e6ff62220dd938c917df97593f5e4fa5789c106a6c",
        ),
        (
            "compatibility-gate.json",
            "10c0ec192021a919dc96098af5fa779aa261ce43f332744505288bfca06e5649",
        ),
        (
            "score.json",
            "18c7ad3d29f5342b6cf1d11ca712074152bd510003b2bddd4f6d3610ba4886f3",
        ),
        (
            "evidence-manifest.json",
            "af069987d24bb1c59e2c292e3a8eb26e3f920903a253717f713773e6950de82f",
        ),
        (
            "test-preparation.json",
            "1a56761009f13c04eac4e0731d2a841acd2e3d9afdb62b3627edcc6384f57c66",
        ),
    ],
)
def test_g1_compact_artifact_hashes_are_frozen(name: str, expected: str) -> None:
    assert sha256_file(ROOT / name) == expected
