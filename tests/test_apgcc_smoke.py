from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from droneai.apgcc_smoke import (
    APGCC_CANDIDATE_ID,
    load_apgcc_smoke_config,
    validate_apgcc_rights_decision,
)
from droneai.stage3c import (
    REQUIRED_COMPONENTS,
    classify_rights,
    manifest_semantic_sha256,
)


ROOT = Path(__file__).parents[1]
CONFIG = ROOT / "configs/evaluation/apgcc_ucf_qnrf_smoke.json"
MANIFEST = ROOT / "configs/candidates/apgcc_shha_to_ucf_qnrf.candidate.json"


def _rights(tmp_path: Path) -> Path:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    decision = classify_rights(manifest).to_dict()
    decision.update(
        {
            "candidate_id": manifest["candidate_id"],
            "manifest_semantic_sha256": manifest_semantic_sha256(manifest),
            "component_ids": {
                component["component_type"]: component["component_id"]
                for component in manifest["components"]
            },
        }
    )
    path = tmp_path / "rights-decision.json"
    path.write_text(json.dumps(decision), encoding="utf-8")
    return path


def test_config_freezes_cross_domain_point_protocol() -> None:
    config = load_apgcc_smoke_config(CONFIG)
    assert config["model_id"] == "apgcc-official-shha-to-ucf-qnrf"
    assert config["candidate_id"] == APGCC_CANDIDATE_ID
    assert config["expected_samples"] == 36
    assert config["split_role"] == "validation"
    assert config["seed"] == 20260715
    assert config["checkpoint_training_split_status"] == "VERIFIED_DISJOINT"
    assert config["comparison_scope"] == "compatibility_smoke"
    assert config["required_action"] == "research_checkpoint_evaluation"
    assert config["targets"]["spatial_metric_name"] == "localization_f1"


def test_rights_allow_research_checkpoint_evaluation_only(tmp_path: Path) -> None:
    decision = validate_apgcc_rights_decision(
        _rights(tmp_path),
        manifest_path=MANIFEST,
        expected_candidate_id=APGCC_CANDIDATE_ID,
    )
    assert decision["status"] == "PASS_COMMERCIAL_CANDIDATE"
    assert "research_checkpoint_evaluation" in decision["allowed_actions"]
    assert not {
        "frozen_checkpoint_evaluation",
        "commercial_training",
        "deployment",
    }.intersection(decision["allowed_actions"])


def test_rights_binding_rejects_changed_candidate(tmp_path: Path) -> None:
    rights = _rights(tmp_path)
    payload = json.loads(rights.read_text(encoding="utf-8"))
    payload["candidate_id"] = "changed"
    rights.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(PermissionError, match="binding"):
        validate_apgcc_rights_decision(
            rights,
            manifest_path=MANIFEST,
            expected_candidate_id=APGCC_CANDIDATE_ID,
        )


def test_one_sample_runner_requires_frozen_inputs() -> None:
    result = subprocess.run(
        [sys.executable, "scripts/run_apgcc_one_sample.py", "--help"],
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
        "--sample-id",
        "--output-dir",
        "--device",
    ):
        assert option in result.stdout
