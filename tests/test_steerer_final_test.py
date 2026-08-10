from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from droneai.steerer_final_test import (
    build_b_final_test_protocol,
    load_b_final_test_profile,
    validate_b_final_test_profile,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
PROFILE = REPO_ROOT / "configs/evaluation/steerer_b_ucf_qnrf_final_test.json"


def _payload() -> dict[str, object]:
    return json.loads(PROFILE.read_text(encoding="utf-8"))


def test_profile_freezes_three_validation_selected_checkpoints() -> None:
    profile = load_b_final_test_profile(PROFILE)

    assert [(row.role, row.epoch, row.filename, row.sha256) for row in profile.checkpoints] == [
        (
            "primary-best-val-mae",
            625,
            "best-mae.pth",
            "e41df787f546a5195e078486ec6d1bd62ffcb2e3e77010436f74b65045667ad2",
        ),
        (
            "secondary-best-val-rmse",
            175,
            "best-rmse.pth",
            "ab1f65573c7f18f115e711998f4f4fb252e816bc5b9ae3564d5b379f22f981e6",
        ),
        (
            "final-reference",
            800,
            "milestone-800.pth",
            "d13316a818ddaa2ed0da680718b234bfd21377934c7ef3b33fa074746c93c8f0",
        ),
    ]


def test_profile_freezes_test_access_and_closes_post_test_training() -> None:
    profile = load_b_final_test_profile(PROFILE)

    assert profile.test_split_id == "ucf-qnrf-official-test334-v1"
    assert profile.expected_samples == 334
    assert profile.test_access_approved is True
    assert profile.post_test_training_allowed is False
    assert profile.checkpoint_origin == "project_training"


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda p: p.update({"expected_samples": 333}), "334"),
        (
            lambda p: p["checkpoints"].reverse(),
            "checkpoint",
        ),
        (
            lambda p: p["checkpoints"][0].update({"sha256": "0" * 64}),
            "checkpoint",
        ),
        (lambda p: p.update({"test_access_approved": False}), "Test access"),
        (lambda p: p.update({"post_test_training_allowed": True}), "post-Test"),
        (lambda p: p.update({"checkpoint_origin": "research_checkpoint"}), "project_training"),
    ],
)
def test_profile_rejects_changes_to_the_preregistered_contract(mutation, message: str) -> None:
    payload = copy.deepcopy(_payload())
    mutation(payload)

    with pytest.raises((PermissionError, ValueError), match=message):
        validate_b_final_test_profile(payload)


def test_protocol_is_an_explicit_verified_final_test() -> None:
    profile = load_b_final_test_profile(PROFILE)
    rights_path = REPO_ROOT / "configs/training/steerer_ucf_qnrf_imagenet.home5090.json"

    protocol = build_b_final_test_protocol(
        profile,
        checkpoint=profile.checkpoints[0],
        rights_path=rights_path,
    )

    assert protocol.split_role == "test"
    assert protocol.expected_samples == 334
    assert protocol.sealed_test_access_approved is True
    assert protocol.leakage_free is True
    assert protocol.checkpoint_training_split_status == "VERIFIED_DISJOINT"
    assert protocol.comparison_scope == "held_out_performance"
