"""MPCount wrapper around the frozen DM-Count QNRF validation split."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal, cast

from droneai.dm_count_adapter import _git_head, _git_status
from droneai.dm_count_smoke import (
    PreparedSmoke,
    load_smoke_config,
    prepare_smoke_samples,
    validate_official_split_paths,
    write_split_source_manifest as _write_dm_split_manifest,
)
from droneai.evaluation_runner import EvaluationProtocol
from droneai.integrity import sha256_file
from droneai.stage3c import REQUIRED_COMPONENTS, manifest_semantic_sha256


MPCOUNT_CANDIDATE_ID = "mpcount-official-sta-to-ucf-qnrf-apache"
MPCOUNT_PINNED_COMMIT = "6eb06772bcf7dfb771c43a14d67146fce767f103"
SPLIT_PINNED_COMMIT = "cc5f2132e0d1328909f31b6d665b8e0b15c30467"
REQUIRED_ACTION = "frozen_checkpoint_evaluation"
_ROOT = Path(__file__).resolve().parents[2]
_DM_CONFIG = _ROOT / "configs/evaluation/dm_count_ucf_qnrf_smoke.json"
_CONFIG_OVERRIDES: dict[str, object] = {
    "run_id": "mpcount-ucf-qnrf-validation-smoke",
    "protocol_id": "mpcount-official-sta-to-qnrf-val-smoke-v1",
    "model_id": "mpcount-official-sta-to-ucf-qnrf",
    "upstream_commit": MPCOUNT_PINNED_COMMIT,
    "split_upstream_commit": SPLIT_PINNED_COMMIT,
    "candidate_id": MPCOUNT_CANDIDATE_ID,
    "required_action": REQUIRED_ACTION,
    "checkpoint_training_split_status": "VERIFIED_DISJOINT",
    "comparison_scope": "compatibility_smoke",
    "checkpoint_split_evidence": (
        "The deterministic checkpoint is trained on ShanghaiTech-A and evaluated "
        "on UCF-QNRF, so no UCF-QNRF validation sample belongs to checkpoint training."
    ),
}
_FORBIDDEN = frozenset({"commercial_training", "derived_weight_use", "deployment"})


def _object(path: str | Path, label: str) -> dict[str, object]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must be an object")
    return payload


def _expected_config() -> dict[str, object]:
    expected = dict(load_smoke_config(_DM_CONFIG))
    expected.update(_CONFIG_OVERRIDES)
    return expected


def load_mpcount_smoke_config(path: str | Path) -> dict[str, object]:
    payload = _object(path, "MPCount smoke config")
    expected = _expected_config()
    if payload != expected:
        changed = sorted(
            key
            for key in set(payload) | set(expected)
            if payload.get(key) != expected.get(key)
        )
        raise ValueError(
            "MPCount config must preserve the frozen validation protocol; "
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


def validate_mpcount_rights_decision(
    path: str | Path,
    *,
    manifest_path: str | Path,
    expected_candidate_id: str = MPCOUNT_CANDIDATE_ID,
) -> dict[str, object]:
    decision = _object(path, "rights decision")
    manifest = _object(manifest_path, "rights manifest")
    component_ids = _component_ids(manifest)
    if (
        expected_candidate_id != MPCOUNT_CANDIDATE_ID
        or decision.get("candidate_id") != expected_candidate_id
        or manifest.get("candidate_id") != expected_candidate_id
        or decision.get("manifest_semantic_sha256")
        != manifest_semantic_sha256(manifest)
        or decision.get("component_ids") != component_ids
    ):
        raise PermissionError(
            "MPCount rights decision candidate or manifest binding failed"
        )
    if decision.get("status") != "PASS_COMMERCIAL_CANDIDATE":
        raise PermissionError("MPCount evaluation requires PASS_COMMERCIAL_CANDIDATE")
    actions = decision.get("allowed_actions")
    if not isinstance(actions, list) or any(not isinstance(item, str) for item in actions):
        raise PermissionError("MPCount rights actions must be a string list")
    if REQUIRED_ACTION not in actions:
        raise PermissionError(f"MPCount rights must authorize {REQUIRED_ACTION}")
    forbidden = sorted(_FORBIDDEN.intersection(actions))
    if forbidden:
        raise PermissionError(
            "MPCount evaluation lane cannot authorize " + ", ".join(forbidden)
        )
    return decision


def validate_mpcount_split_upstream(
    *,
    split_upstream_dir: str | Path,
    train_list_path: str | Path,
    validation_list_path: str | Path,
    expected_commit: str,
) -> None:
    root = Path(split_upstream_dir).resolve()
    if expected_commit != SPLIT_PINNED_COMMIT:
        raise ValueError("MPCount split upstream must use the pinned DM-Count commit")
    validate_official_split_paths(
        upstream_dir=root,
        train_list_path=train_list_path,
        validation_list_path=validation_list_path,
    )
    if _git_head(root) != expected_commit or _git_status(root):
        raise ValueError("MPCount split upstream must be clean at the pinned commit")


def write_split_source_manifest(
    path: str | Path,
    *,
    prepared: PreparedSmoke,
    train_list_path: str | Path,
    validation_list_path: str | Path,
    model_upstream_commit: str,
    split_upstream_commit: str,
) -> Path:
    if model_upstream_commit != MPCOUNT_PINNED_COMMIT:
        raise ValueError("MPCount model upstream commit is not pinned")
    if split_upstream_commit != SPLIT_PINNED_COMMIT:
        raise ValueError("MPCount split upstream commit is not pinned")
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


def _strings(value: object) -> tuple[str, ...]:
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise ValueError("expected a string list")
    return tuple(value)


def build_mpcount_protocol(
    config: dict[str, object],
    *,
    rights_decision_path: str | Path,
    rights_manifest_path: str | Path,
    split_verified: bool,
) -> EvaluationProtocol:
    if config != _expected_config():
        raise ValueError("MPCount protocol config is not frozen")
    rights = Path(rights_decision_path).resolve()
    validate_mpcount_rights_decision(
        rights,
        manifest_path=rights_manifest_path,
        expected_candidate_id=str(config["candidate_id"]),
    )
    targets = cast(dict[str, object], config["targets"])
    return EvaluationProtocol(
        run_id=str(config["run_id"]),
        protocol_id=str(config["protocol_id"]),
        dataset_id=str(config["dataset_id"]),
        split_id=str(config["split_id"]),
        split_role=cast(Literal["train", "validation", "test", "smoke"], config["split_role"]),
        expected_samples=int(config["expected_samples"]),
        split_verified=split_verified,
        leakage_free=split_verified,
        checkpoint_training_split_status=str(
            config["checkpoint_training_split_status"]
        ),
        comparison_scope=str(config["comparison_scope"]),
        checkpoint_split_evidence=str(config["checkpoint_split_evidence"]),
        sealed_test_access_approved=False,
        require_clean_git=bool(config["require_clean_git"]),
        rights_decision_path=str(rights),
        rights_decision_sha256=sha256_file(rights),
        localization_radius=float(config["localization_radius"]),
        mae_max=float(targets["mae_max"]),
        rmse_max=float(targets["rmse_max"]),
        bias_max=float(targets["bias_max"]),
        band_bias_max=float(targets["band_bias_max"]),
        condition_bias_max=float(targets["condition_bias_max"]),
        latency_max_ms=float(targets["latency_max_ms"]),
        vram_max_mb=float(targets["vram_max_mb"]),
        spatial_metric_name=str(targets["spatial_metric_name"]),
        spatial_direction=cast(Literal["minimize", "maximize"], targets["spatial_direction"]),
        spatial_target=float(targets["spatial_target"]),
        required_density_bands=_strings(targets["required_density_bands"]),
        density_band_rules=_strings(targets["density_band_rules"]),
        required_condition_keys=_strings(targets["required_condition_keys"]),
        zone_warning_count=float(targets["zone_warning_count"]),
        zone_critical_count=float(targets["zone_critical_count"]),
    )


__all__ = [
    "MPCOUNT_CANDIDATE_ID",
    "build_mpcount_protocol",
    "load_mpcount_smoke_config",
    "prepare_smoke_samples",
    "validate_mpcount_rights_decision",
    "validate_mpcount_split_upstream",
    "write_split_source_manifest",
]
