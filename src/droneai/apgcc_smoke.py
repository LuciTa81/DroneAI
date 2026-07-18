"""APGCC wrapper around the frozen DM-Count QNRF validation split."""

from __future__ import annotations

import json
from pathlib import Path

from droneai.dm_count_adapter import _git_head, _git_status
from droneai.dm_count_smoke import (
    PreparedSmoke,
    load_smoke_config,
    prepare_smoke_samples,
    validate_official_split_paths,
    write_split_source_manifest as _write_dm_split_manifest,
)
from droneai.stage3c import REQUIRED_COMPONENTS, manifest_semantic_sha256


APGCC_CANDIDATE_ID = "apgcc-official-shha-to-ucf-qnrf-research-comparison"
APGCC_PINNED_COMMIT = "e3e997bf592a70fd34233a432a974200d8c0c847"
SPLIT_PINNED_COMMIT = "cc5f2132e0d1328909f31b6d665b8e0b15c30467"
REQUIRED_ACTION = "research_checkpoint_evaluation"
_ROOT = Path(__file__).resolve().parents[2]
_DM_CONFIG = _ROOT / "configs/evaluation/dm_count_ucf_qnrf_smoke.json"
_CONFIG_OVERRIDES: dict[str, object] = {
    "run_id": "apgcc-ucf-qnrf-validation-smoke",
    "protocol_id": "apgcc-official-shha-to-qnrf-val-smoke-v1",
    "model_id": "apgcc-official-shha-to-ucf-qnrf",
    "upstream_commit": APGCC_PINNED_COMMIT,
    "split_upstream_commit": SPLIT_PINNED_COMMIT,
    "candidate_id": APGCC_CANDIDATE_ID,
    "required_action": REQUIRED_ACTION,
    "localization_radius": 16.0,
    "checkpoint_training_split_status": "VERIFIED_DISJOINT",
    "comparison_scope": "compatibility_smoke",
    "checkpoint_split_evidence": (
        "The official checkpoint is trained on ShanghaiTech Part A, so no "
        "UCF-QNRF validation sample belongs to checkpoint training; the result "
        "is cross-domain compatibility evidence only."
    ),
}
_FORBIDDEN = frozenset(
    {
        "asset_download",
        "frozen_checkpoint_evaluation",
        "commercial_training",
        "derived_weight_use",
        "weight_reuse",
        "deployment",
    }
)


def _object(path: str | Path, label: str) -> dict[str, object]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must be an object")
    return payload


def _expected_config() -> dict[str, object]:
    expected = dict(load_smoke_config(_DM_CONFIG))
    expected.update(_CONFIG_OVERRIDES)
    targets = dict(expected["targets"])
    targets.update(
        {
            "spatial_metric_name": "localization_f1",
            "spatial_direction": "maximize",
            "spatial_target": 0.0,
        }
    )
    expected["targets"] = targets
    return expected


def load_apgcc_smoke_config(path: str | Path) -> dict[str, object]:
    payload = _object(path, "APGCC smoke config")
    expected = _expected_config()
    if payload != expected:
        changed = sorted(
            key
            for key in set(payload) | set(expected)
            if payload.get(key) != expected.get(key)
        )
        raise ValueError(
            "APGCC config must preserve the frozen validation protocol; "
            f"mismatched={changed}"
        )
    return payload


def _component_ids(manifest: dict[str, object]) -> dict[str, str]:
    components = manifest.get("components")
    if not isinstance(components, list) or len(components) != len(REQUIRED_COMPONENTS):
        raise PermissionError("rights manifest must contain exactly five components")
    result: dict[str, str] = {}
    for component in components:
        if not isinstance(component, dict):
            raise PermissionError("rights manifest component must be an object")
        kind = component.get("component_type")
        identity = component.get("component_id")
        if (
            not isinstance(kind, str)
            or kind not in REQUIRED_COMPONENTS
            or kind in result
            or not isinstance(identity, str)
            or not identity.strip()
        ):
            raise PermissionError("rights manifest component identity is invalid")
        result[kind] = identity
    if set(result) != REQUIRED_COMPONENTS:
        raise PermissionError("rights manifest component set is incomplete")
    return result


def validate_apgcc_rights_decision(
    path: str | Path,
    *,
    manifest_path: str | Path,
    expected_candidate_id: str = APGCC_CANDIDATE_ID,
) -> dict[str, object]:
    decision = _object(path, "rights decision")
    manifest = _object(manifest_path, "rights manifest")
    if (
        expected_candidate_id != APGCC_CANDIDATE_ID
        or decision.get("candidate_id") != expected_candidate_id
        or manifest.get("candidate_id") != expected_candidate_id
        or decision.get("manifest_semantic_sha256")
        != manifest_semantic_sha256(manifest)
        or decision.get("component_ids") != _component_ids(manifest)
    ):
        raise PermissionError("APGCC rights decision candidate or manifest binding failed")
    if decision.get("status") != "PASS_COMMERCIAL_CANDIDATE":
        raise PermissionError("APGCC evaluation requires PASS_COMMERCIAL_CANDIDATE")
    actions = decision.get("allowed_actions")
    if not isinstance(actions, list) or any(not isinstance(item, str) for item in actions):
        raise PermissionError("APGCC rights actions must be a string list")
    if REQUIRED_ACTION not in actions:
        raise PermissionError(f"APGCC rights must authorize {REQUIRED_ACTION}")
    forbidden = sorted(_FORBIDDEN.intersection(actions))
    if forbidden:
        raise PermissionError(
            "APGCC research lane cannot authorize " + ", ".join(forbidden)
        )
    return decision


def validate_apgcc_split_upstream(
    *,
    split_upstream_dir: str | Path,
    train_list_path: str | Path,
    validation_list_path: str | Path,
    expected_commit: str,
) -> None:
    root = Path(split_upstream_dir).resolve()
    if expected_commit != SPLIT_PINNED_COMMIT:
        raise ValueError("APGCC split upstream must use the pinned DM-Count commit")
    validate_official_split_paths(
        upstream_dir=root,
        train_list_path=train_list_path,
        validation_list_path=validation_list_path,
    )
    if _git_head(root) != expected_commit or _git_status(root):
        raise ValueError("APGCC split upstream must be clean at the pinned commit")


def write_split_source_manifest(
    path: str | Path,
    *,
    prepared: PreparedSmoke,
    train_list_path: str | Path,
    validation_list_path: str | Path,
    model_upstream_commit: str,
    split_upstream_commit: str,
) -> Path:
    if model_upstream_commit != APGCC_PINNED_COMMIT:
        raise ValueError("APGCC model upstream commit is not pinned")
    if split_upstream_commit != SPLIT_PINNED_COMMIT:
        raise ValueError("APGCC split upstream commit is not pinned")
    target = _write_dm_split_manifest(
        path,
        prepared=prepared,
        train_list_path=train_list_path,
        validation_list_path=validation_list_path,
    )
    payload = _object(target, "split source manifest")
    payload.update(
        {
            "evaluation_scope": "research_comparison_only",
            "model_upstream_commit": model_upstream_commit,
            "split_upstream_commit": split_upstream_commit,
        }
    )
    target.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return target


__all__ = [
    "APGCC_CANDIDATE_ID",
    "load_apgcc_smoke_config",
    "prepare_smoke_samples",
    "validate_apgcc_rights_decision",
    "validate_apgcc_split_upstream",
    "write_split_source_manifest",
]
