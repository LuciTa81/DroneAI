from __future__ import annotations

import importlib
import json
from pathlib import Path

import pytest

from droneai.stage3c import manifest_semantic_sha256


ROOT = Path(__file__).parents[1]
CONFIG = ROOT / "configs/evaluation/csrnet_ucf_qnrf_smoke.json"
MANIFEST = ROOT / "configs/candidates/csrnet_shha_to_ucf_qnrf.candidate.json"


def _smoke():
    return importlib.import_module("droneai.csrnet_smoke")


def _rights(tmp_path: Path, *, status: str = "PASS_COMMERCIAL_CANDIDATE") -> Path:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    decision = {
        "status": status,
        "allowed_actions": ["research_checkpoint_evaluation"],
        "candidate_id": manifest["candidate_id"],
        "manifest_semantic_sha256": manifest_semantic_sha256(manifest),
        "component_ids": {
            component["component_type"]: component["component_id"]
            for component in manifest["components"]
        },
    }
    path = tmp_path / "rights-decision.json"
    path.write_text(json.dumps(decision), encoding="utf-8")
    return path


def test_csrnet_config_preserves_frozen_validation_contract() -> None:
    smoke = _smoke()
    config = smoke.load_csrnet_smoke_config(CONFIG)

    assert config["model_id"] == "csrnet-independent-shha-to-ucf-qnrf"
    assert config["candidate_id"] == smoke.CSRNET_CANDIDATE_ID
    assert config["expected_samples"] == 36
    assert config["split_role"] == "validation"
    assert config["seed"] == 20260715
    assert config["checkpoint_training_split_status"] == "VERIFIED_DISJOINT"
    assert config["comparison_scope"] == "compatibility_smoke"
    assert config["sealed_test_access_approved"] is False
    assert config["required_action"] == "research_checkpoint_evaluation"


def test_csrnet_rights_bind_candidate_and_research_action(tmp_path: Path) -> None:
    smoke = _smoke()
    decision = smoke.validate_csrnet_rights_decision(
        _rights(tmp_path),
        manifest_path=MANIFEST,
        expected_candidate_id=smoke.CSRNET_CANDIDATE_ID,
    )

    assert decision["status"] == "PASS_COMMERCIAL_CANDIDATE"
    assert decision["allowed_actions"] == ["research_checkpoint_evaluation"]


def test_csrnet_rights_reject_research_only_decision(tmp_path: Path) -> None:
    smoke = _smoke()
    with pytest.raises(PermissionError, match="PASS_COMMERCIAL_CANDIDATE"):
        smoke.validate_csrnet_rights_decision(
            _rights(tmp_path, status="PASS_RESEARCH_ONLY"),
            manifest_path=MANIFEST,
            expected_candidate_id=smoke.CSRNET_CANDIDATE_ID,
        )
