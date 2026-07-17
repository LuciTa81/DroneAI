from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from droneai.pet_smoke import (
    PET_CANDIDATE_ID,
    build_pet_protocol,
    load_pet_smoke_config,
    validate_pet_rights_decision,
)


ROOT = Path(__file__).parents[1]
CONFIG = ROOT / "configs/evaluation/pet_ucf_qnrf_smoke.json"
RIGHTS = ROOT / "results/pet/pet-rights-99bfc15/rights-decision.json"
MANIFEST = ROOT / "configs/candidates/pet_ucf_qnrf.candidate.json"


def test_pet_config_changes_only_point_spatial_contract() -> None:
    config = load_pet_smoke_config(CONFIG)
    assert config["model_id"] == "pet-official-ucf-qnrf"
    assert config["candidate_id"] == PET_CANDIDATE_ID
    assert config["expected_samples"] == 36
    assert config["split_role"] == "validation"
    assert config["sealed_test_access_approved"] is False
    assert config["required_action"] == "research_checkpoint_evaluation"
    assert config["targets"]["spatial_metric_name"] == "localization_f1"
    assert config["targets"]["spatial_direction"] == "maximize"


def test_pet_rights_accept_research_comparison_but_not_commercial_actions() -> None:
    decision = validate_pet_rights_decision(
        RIGHTS, manifest_path=MANIFEST, expected_candidate_id=PET_CANDIDATE_ID
    )
    assert decision["status"] == "PASS_RESEARCH_ONLY"
    assert "research_checkpoint_evaluation" in decision["allowed_actions"]
    assert not {
        "commercial_training",
        "deployment",
        "weight_reuse",
    }.intersection(decision["allowed_actions"])


def test_pet_rights_reject_commercial_status_or_action(tmp_path: Path) -> None:
    decision = json.loads(RIGHTS.read_text(encoding="utf-8"))
    decision["status"] = "PASS_COMMERCIAL_CANDIDATE"
    path = tmp_path / "decision.json"
    path.write_text(json.dumps(decision), encoding="utf-8")
    with pytest.raises(PermissionError, match="PASS_RESEARCH_ONLY"):
        validate_pet_rights_decision(
            path, manifest_path=MANIFEST, expected_candidate_id=PET_CANDIDATE_ID
        )


def test_pet_protocol_uses_localization_f1_and_hashes_rights() -> None:
    protocol = build_pet_protocol(
        load_pet_smoke_config(CONFIG),
        rights_decision_path=RIGHTS,
        rights_manifest_path=MANIFEST,
        split_verified=True,
    )
    assert protocol.expected_samples == 36
    assert protocol.split_role == "validation"
    assert protocol.spatial_metric_name == "localization_f1"
    assert protocol.spatial_direction == "maximize"
    assert protocol.spatial_target == 0.0


def test_pet_runner_requires_frozen_inputs() -> None:
    result = subprocess.run(
        [sys.executable, "scripts/run_pet_ucf_qnrf_smoke.py", "--help"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    for option in (
        "--config",
        "--train-root",
        "--upstream-dir",
        "--split-upstream-dir",
        "--train-list",
        "--validation-list",
        "--checkpoint",
        "--checkpoint-sha256",
        "--rights-decision",
        "--rights-manifest",
        "--output-dir",
        "--device",
    ):
        assert option in result.stdout
