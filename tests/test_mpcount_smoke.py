from __future__ import annotations

import importlib
import json
from pathlib import Path

import pytest

from droneai.stage3c import manifest_semantic_sha256


ROOT = Path(__file__).parents[1]
CONFIG = ROOT / "configs/evaluation/mpcount_ucf_qnrf_smoke.json"
MANIFEST = ROOT / "configs/candidates/mpcount_sta_to_ucf_qnrf.candidate.json"


def _smoke():
    return importlib.import_module("droneai.mpcount_smoke")


def _rights(tmp_path: Path, *, status: str = "PASS_COMMERCIAL_CANDIDATE") -> Path:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    components = manifest["components"]
    component_ids = {
        component["component_type"]: component["component_id"]
        for component in components
    }
    decision = {
        "status": status,
        "allowed_actions": ["frozen_checkpoint_evaluation"],
        "candidate_id": manifest["candidate_id"],
        "manifest_semantic_sha256": manifest_semantic_sha256(manifest),
        "component_ids": component_ids,
    }
    path = tmp_path / "rights-decision.json"
    path.write_text(json.dumps(decision), encoding="utf-8")
    return path


def test_mpcount_config_preserves_frozen_validation_contract() -> None:
    smoke = _smoke()
    config = smoke.load_mpcount_smoke_config(CONFIG)
    assert config["model_id"] == "mpcount-official-sta-to-ucf-qnrf"
    assert config["candidate_id"] == smoke.MPCOUNT_CANDIDATE_ID
    assert config["expected_samples"] == 36
    assert config["split_role"] == "validation"
    assert config["seed"] == 20260715
    assert config["checkpoint_training_split_status"] == "VERIFIED_DISJOINT"
    assert config["comparison_scope"] == "compatibility_smoke"
    assert config["sealed_test_access_approved"] is False
    assert config["required_action"] == "frozen_checkpoint_evaluation"
    assert config["targets"]["spatial_metric_name"] == "game_l1"


def test_mpcount_rights_bind_accepted_candidate_and_action(tmp_path: Path) -> None:
    smoke = _smoke()
    decision = smoke.validate_mpcount_rights_decision(
        _rights(tmp_path),
        manifest_path=MANIFEST,
        expected_candidate_id=smoke.MPCOUNT_CANDIDATE_ID,
    )
    assert decision["status"] == "PASS_COMMERCIAL_CANDIDATE"
    assert "frozen_checkpoint_evaluation" in decision["allowed_actions"]


def test_mpcount_rights_reject_wrong_status(tmp_path: Path) -> None:
    smoke = _smoke()
    with pytest.raises(PermissionError, match="PASS_COMMERCIAL_CANDIDATE"):
        smoke.validate_mpcount_rights_decision(
            _rights(tmp_path, status="PASS_RESEARCH_ONLY"),
            manifest_path=MANIFEST,
            expected_candidate_id=smoke.MPCOUNT_CANDIDATE_ID,
        )


def test_mpcount_protocol_is_validation_density_compatibility(tmp_path: Path) -> None:
    smoke = _smoke()
    protocol = smoke.build_mpcount_protocol(
        smoke.load_mpcount_smoke_config(CONFIG),
        rights_decision_path=_rights(tmp_path),
        rights_manifest_path=MANIFEST,
        split_verified=True,
    )
    assert protocol.expected_samples == 36
    assert protocol.split_role == "validation"
    assert protocol.checkpoint_training_split_status == "VERIFIED_DISJOINT"
    assert protocol.comparison_scope == "compatibility_smoke"
    assert protocol.spatial_metric_name == "game_l1"
    assert protocol.spatial_direction == "minimize"
