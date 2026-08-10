from __future__ import annotations

import json
from pathlib import Path

import pytest

from droneai.integrity import sha256_file


ROOT = (
    Path(__file__).resolve().parents[1]
    / "results/steerer/steerer-a-g2-official-train1201-test334-data-v1"
)


def _json(name: str) -> dict[str, object]:
    return json.loads((ROOT / name).read_text(encoding="utf-8"))


def test_g2_compact_result_preserves_full_train_test_integrity() -> None:
    manifest = _json("dataset-preparation.json")
    verification = _json("g2-verification.json")
    score = _json("g2-score.json")
    provenance = _json("provenance.json")

    assert manifest["train"]["sample_count"] == 1201
    assert manifest["validation"] == {
        "list_path": None,
        "sample_count": 0,
        "split_role": "not_used",
    }
    assert manifest["test"]["sample_count"] == 334
    assert manifest["test"]["selection"] == "test_selected"
    assert verification["artifact_hashes_verified"] is True
    assert verification["split_overlap"] == []
    assert score["score"] == 100
    assert score["status"] == "PASS_COMMERCIAL_CANDIDATE"
    assert score["failed_blockers"] == []
    assert provenance["dataset"]["train_ground_truth_total"] == 1011515
    assert provenance["dataset"]["test_ground_truth_total"] == 240127
    assert provenance["training_profile"][
        "official_steerer_checkpoint_training_allowed"
    ] is False


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        (
            "dataset-preparation.json",
            "57c5b26205f8146a5593dfa98a17a007aaa7582cb91c85c4938f88f444706a35",
        ),
        (
            "g2-verification.json",
            "9aedc9568cb788f2665831f0ec1720b4eb7ee92a60518f11590fad3e5376357a",
        ),
        (
            "g2-score.json",
            "0af20a47e5051c16eeb1bb7bb89dccf6c941d21778595fb70aa0c8a0600bd61e",
        ),
        (
            "g2-score.md",
            "be819d724c792adff126a817c238735dbadd0bdb21fa053fe503be811dbeb04b",
        ),
        (
            "provenance.json",
            "dca136fdc1c9c13fc55de51e606e162fe442ee82d0536a835871a18e53431b9c",
        ),
        (
            "summary.md",
            "ebfec627efe201a12cbe861bf16dd480bec52417564c220d9ac0cc9ff08844d9",
        ),
    ],
)
def test_g2_compact_artifact_hashes_are_frozen(name: str, expected: str) -> None:
    assert sha256_file(ROOT / name) == expected
