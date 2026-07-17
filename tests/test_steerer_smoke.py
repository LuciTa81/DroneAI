from __future__ import annotations

import json
from pathlib import Path

import pytest

from droneai import dm_count_smoke
from droneai.dm_count_smoke import PreparedSmoke, load_smoke_config
from droneai.stage3c import REQUIRED_COMPONENTS, manifest_semantic_sha256
from droneai.steerer_smoke import (
    build_steerer_protocol,
    load_steerer_smoke_config,
    prepare_smoke_samples,
    validate_official_split_paths,
    validate_steerer_rights_decision,
    write_split_source_manifest,
)


REPO_ROOT = Path(__file__).parents[1]
DM_CONFIG = REPO_ROOT / "configs" / "evaluation" / "dm_count_ucf_qnrf_smoke.json"
STEERER_CONFIG = REPO_ROOT / "configs" / "evaluation" / "steerer_ucf_qnrf_smoke.json"
CANDIDATE_ID = "steerer-official-ucf-qnrf-research-comparison"


def _rights(
    root: Path,
    *,
    actions: tuple[str, ...] = (
        "catalog",
        "license_due_diligence",
        "research_asset_download",
        "research_checkpoint_evaluation",
    ),
    status: str = "PASS_COMMERCIAL_CANDIDATE",
) -> tuple[Path, Path]:
    root.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema_version": 1,
        "candidate_id": CANDIDATE_ID,
        "intended_use": "commercial_product_rnd",
        "components": [
            {
                "component_type": component_type,
                "component_id": f"fixture-{component_type}",
            }
            for component_type in sorted(REQUIRED_COMPONENTS)
        ],
    }
    manifest_path = root / "candidate.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    decision = {
        "status": status,
        "candidate_id": CANDIDATE_ID,
        "manifest_semantic_sha256": manifest_semantic_sha256(manifest),
        "component_ids": {
            component["component_type"]: component["component_id"]
            for component in manifest["components"]
        },
        "allowed_actions": list(actions),
    }
    decision_path = root / "rights-decision.json"
    decision_path.write_text(json.dumps(decision), encoding="utf-8")
    return decision_path, manifest_path


def test_steerer_uses_the_exact_dm_count_validation_sample_identity() -> None:
    dm = load_smoke_config(DM_CONFIG)
    steerer = load_steerer_smoke_config(STEERER_CONFIG)

    for key in (
        "dataset_id",
        "split_id",
        "expected_samples",
        "seed",
        "train_list_sha256",
        "validation_list_sha256",
    ):
        assert steerer[key] == dm[key]
    assert steerer["expected_samples"] == 36
    assert steerer["split_role"] == "validation"
    assert steerer["sealed_test_access_approved"] is False
    assert steerer["required_action"] == "research_checkpoint_evaluation"
    assert steerer["localization_radius"] == 16.0
    assert steerer["require_clean_git"] is True


def test_steerer_reuses_dm_count_sample_and_split_path_helpers() -> None:
    assert prepare_smoke_samples is dm_count_smoke.prepare_smoke_samples
    assert validate_official_split_paths is dm_count_smoke.validate_official_split_paths


def test_steerer_config_copy_cannot_change_the_frozen_sample_identity(
    tmp_path: Path,
) -> None:
    config = json.loads(STEERER_CONFIG.read_text(encoding="utf-8"))
    config["seed"] += 1
    copied_config = tmp_path / "steerer-smoke.json"
    copied_config.write_text(json.dumps(config), encoding="utf-8")

    with pytest.raises(ValueError, match="preserve the frozen DM-Count validation"):
        load_steerer_smoke_config(copied_config)


def test_research_decision_cannot_be_misreported_as_commercial(tmp_path: Path) -> None:
    decision_path, manifest_path = _rights(tmp_path)

    decision = validate_steerer_rights_decision(
        decision_path,
        manifest_path=manifest_path,
        expected_candidate_id=CANDIDATE_ID,
    )

    assert decision["status"] == "PASS_COMMERCIAL_CANDIDATE"
    assert {
        "research_asset_download",
        "research_checkpoint_evaluation",
    }.issubset(decision["allowed_actions"])
    assert "frozen_checkpoint_evaluation" not in decision["allowed_actions"]
    assert "deployment" not in decision["allowed_actions"]


def test_research_decision_requires_checkpoint_evaluation_action(tmp_path: Path) -> None:
    decision_path, manifest_path = _rights(
        tmp_path,
        actions=("catalog", "research_asset_download"),
    )

    with pytest.raises(PermissionError, match="research_checkpoint_evaluation"):
        validate_steerer_rights_decision(
            decision_path,
            manifest_path=manifest_path,
            expected_candidate_id=CANDIDATE_ID,
        )


@pytest.mark.parametrize("forbidden_action", ["frozen_checkpoint_evaluation", "deployment"])
def test_research_decision_rejects_commercial_or_deployment_actions(
    tmp_path: Path,
    forbidden_action: str,
) -> None:
    decision_path, manifest_path = _rights(
        tmp_path,
        actions=(
            "catalog",
            "research_asset_download",
            "research_checkpoint_evaluation",
            forbidden_action,
        ),
    )

    with pytest.raises(PermissionError, match=forbidden_action):
        validate_steerer_rights_decision(
            decision_path,
            manifest_path=manifest_path,
            expected_candidate_id=CANDIDATE_ID,
        )


def test_split_manifest_records_research_comparison_scope(tmp_path: Path) -> None:
    prepared = PreparedSmoke(
        samples=(),
        selected_records=(),
        source_train_count=1201,
        train_count=1081,
        validation_count=120,
        split_verified=True,
        train_list_sha256="a" * 64,
        validation_list_sha256="b" * 64,
    )

    manifest_path = write_split_source_manifest(
        tmp_path / "split-source-manifest.json",
        prepared=prepared,
        train_list_path=tmp_path / "qnrf_train.txt",
        validation_list_path=tmp_path / "qnrf_val.txt",
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert manifest["evaluation_scope"] == "research_comparison_only"
    assert manifest["test_root_argument_exposed"] is False
    assert manifest["test_data_access"] == "not_possible_through_adapter_api"


def test_protocol_references_research_decision_without_test_approval(
    tmp_path: Path,
) -> None:
    config = load_steerer_smoke_config(STEERER_CONFIG)
    decision_path, manifest_path = _rights(tmp_path)

    protocol = build_steerer_protocol(
        config,
        rights_decision_path=decision_path,
        rights_manifest_path=manifest_path,
        split_verified=True,
    )

    assert protocol.dataset_id == load_smoke_config(DM_CONFIG)["dataset_id"]
    assert protocol.split_role == "validation"
    assert protocol.expected_samples == 36
    assert protocol.sealed_test_access_approved is False
    assert protocol.localization_radius == 16.0
    assert len(protocol.rights_decision_sha256) == 64
