"""Research-only STEERER wrapper around the frozen DM-Count QNRF smoke."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal, cast

from droneai.dm_count_smoke import (
    PreparedSmoke,
    load_smoke_config,
    prepare_smoke_samples,
    validate_official_split_paths,
    write_split_source_manifest as _write_dm_count_split_source_manifest,
)
from droneai.dm_count_adapter import _git_head, _git_status
from droneai.evaluation_runner import EvaluationProtocol
from droneai.integrity import sha256_file
from droneai.stage3c import REQUIRED_COMPONENTS, manifest_semantic_sha256


STEERER_CANDIDATE_ID = "steerer-official-ucf-qnrf-research-comparison"
STEERER_PINNED_COMMIT = "5b1854dbc2d280f2326d67c65515d8baf9083810"
SPLIT_UPSTREAM_PINNED_COMMIT = "cc5f2132e0d1328909f31b6d665b8e0b15c30467"
RESEARCH_CHECKPOINT_ACTION = "research_checkpoint_evaluation"
EVALUATION_SCOPE = "research_comparison_only"
_DM_COUNT_CONFIG = (
    Path(__file__).resolve().parents[2]
    / "configs"
    / "evaluation"
    / "dm_count_ucf_qnrf_smoke.json"
)

_STEERER_CONFIG_OVERRIDES: dict[str, object] = {
    "run_id": "steerer-ucf-qnrf-validation-smoke",
    "protocol_id": "steerer-official-qnrf-val-smoke-v1",
    "model_id": "steerer-official-ucf-qnrf",
    "upstream_commit": STEERER_PINNED_COMMIT,
    "split_upstream_commit": SPLIT_UPSTREAM_PINNED_COMMIT,
    "candidate_id": STEERER_CANDIDATE_ID,
    "required_action": RESEARCH_CHECKPOINT_ACTION,
    "localization_radius": 16.0,
}
_FORBIDDEN_RESEARCH_ACTIONS = frozenset(
    {
        "asset_download",
        "frozen_checkpoint_evaluation",
        "commercial_training",
        "derived_weight_use",
        "weight_reuse",
        "deployment",
    }
)


def _validate_steerer_smoke_config(config: dict[str, object]) -> None:
    dm_count = load_smoke_config(_DM_COUNT_CONFIG)
    expected = dict(dm_count)
    expected.update(_STEERER_CONFIG_OVERRIDES)
    if config != expected:
        changed_keys = sorted(
            key
            for key in set(config) | set(expected)
            if config.get(key) != expected.get(key)
        )
        raise ValueError(
            "STEERER smoke config must preserve the frozen DM-Count validation "
            f"protocol except for approved overrides; mismatched={changed_keys}"
        )


def load_steerer_smoke_config(path: str | Path) -> dict[str, object]:
    """Load a STEERER config that differs from the frozen DM-Count config only by scope."""

    target = Path(path)
    payload = json.loads(target.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("STEERER smoke config must be an object")
    _validate_steerer_smoke_config(payload)
    return payload


def validate_steerer_split_upstream(
    *,
    split_upstream_dir: str | Path,
    train_list_path: str | Path,
    validation_list_path: str | Path,
    expected_commit: str,
) -> None:
    """Validate the clean DM-Count checkout used only as frozen split provenance."""

    root = Path(split_upstream_dir).resolve()
    if expected_commit != SPLIT_UPSTREAM_PINNED_COMMIT:
        raise ValueError("STEERER split upstream must use the pinned DM-Count commit")
    if not root.is_dir():
        raise FileNotFoundError(f"split upstream directory is missing: {root}")
    validate_official_split_paths(
        upstream_dir=root,
        train_list_path=train_list_path,
        validation_list_path=validation_list_path,
    )
    observed_commit = _git_head(root)
    if observed_commit != expected_commit:
        raise ValueError(
            "split upstream commit mismatch: "
            f"expected={expected_commit} observed={observed_commit}"
        )
    if _git_status(root):
        raise ValueError("split upstream Git checkout must be clean")


def _load_object(path: str | Path, *, label: str) -> dict[str, object]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must be an object")
    return payload


def _validated_component_ids(manifest: dict[str, object]) -> dict[str, str]:
    components = manifest.get("components")
    if not isinstance(components, list) or len(components) != len(
        REQUIRED_COMPONENTS
    ):
        raise PermissionError(
            "rights manifest components must contain exactly five objects"
        )
    component_ids: dict[str, str] = {}
    for component in components:
        if not isinstance(component, dict):
            raise PermissionError("rights manifest components must all be objects")
        component_type = component.get("component_type")
        component_id = component.get("component_id")
        if not isinstance(component_type, str) or component_type not in REQUIRED_COMPONENTS:
            raise PermissionError(
                "rights manifest components require a known component_type"
            )
        if component_type in component_ids:
            raise PermissionError(
                "rights manifest components must contain each component_type exactly once"
            )
        if not isinstance(component_id, str) or not component_id.strip():
            raise PermissionError(
                "rights manifest components require non-empty string component_id values"
            )
        component_ids[component_type] = component_id
    if set(component_ids) != REQUIRED_COMPONENTS:
        raise PermissionError(
            "rights manifest components must contain each required component_type"
        )
    return component_ids


def validate_steerer_rights_decision(
    path: str | Path,
    *,
    manifest_path: str | Path,
    expected_candidate_id: str = STEERER_CANDIDATE_ID,
) -> dict[str, object]:
    """Require manifest-bound research comparison rights without production actions."""

    payload = _load_object(path, label="rights decision")
    manifest = _load_object(manifest_path, label="rights manifest")
    component_ids = _validated_component_ids(manifest)
    identity_matches = (
        expected_candidate_id == STEERER_CANDIDATE_ID
        and expected_candidate_id == manifest.get("candidate_id")
        and expected_candidate_id == payload.get("candidate_id")
        and payload.get("manifest_semantic_sha256")
        == manifest_semantic_sha256(manifest)
        and payload.get("component_ids") == component_ids
        and set(component_ids) == REQUIRED_COMPONENTS
    )
    if not identity_matches:
        raise PermissionError("rights decision candidate or manifest binding failed")
    if payload.get("status") != "PASS_COMMERCIAL_CANDIDATE":
        raise PermissionError(
            "STEERER research comparison requires PASS_COMMERCIAL_CANDIDATE status"
        )
    actions = payload.get("allowed_actions")
    if not isinstance(actions, list) or any(
        not isinstance(action, str) for action in actions
    ):
        raise PermissionError("rights decision allowed_actions must be a string list")
    if RESEARCH_CHECKPOINT_ACTION not in actions:
        raise PermissionError(
            f"rights decision must authorize {RESEARCH_CHECKPOINT_ACTION}"
        )
    forbidden = sorted(_FORBIDDEN_RESEARCH_ACTIONS.intersection(actions))
    if forbidden:
        raise PermissionError(
            "STEERER research comparison cannot authorize " + ", ".join(forbidden)
        )
    return payload


def write_split_source_manifest(
    path: str | Path,
    *,
    prepared: PreparedSmoke,
    train_list_path: str | Path,
    validation_list_path: str | Path,
    model_upstream_commit: str,
    split_upstream_commit: str,
) -> Path:
    """Write the shared split provenance plus the STEERER research-only scope."""

    if model_upstream_commit != STEERER_PINNED_COMMIT:
        raise ValueError("split manifest model upstream commit is not pinned STEERER")
    if split_upstream_commit != SPLIT_UPSTREAM_PINNED_COMMIT:
        raise ValueError("split manifest split upstream commit is not pinned DM-Count")
    target = _write_dm_count_split_source_manifest(
        path,
        prepared=prepared,
        train_list_path=train_list_path,
        validation_list_path=validation_list_path,
    )
    payload = _load_object(target, label="split source manifest")
    payload["evaluation_scope"] = EVALUATION_SCOPE
    payload["model_upstream_commit"] = model_upstream_commit
    payload["split_upstream_commit"] = split_upstream_commit
    target.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return target


def _string_tuple(values: object) -> tuple[str, ...]:
    if not isinstance(values, list):
        raise ValueError("expected a string list")
    return tuple(str(value) for value in values)


def build_steerer_protocol(
    config: dict[str, object],
    *,
    rights_decision_path: str | Path,
    rights_manifest_path: str | Path,
    split_verified: bool,
) -> EvaluationProtocol:
    """Build a common evaluation protocol after research-only rights validation."""

    _validate_steerer_smoke_config(config)
    rights_path = Path(rights_decision_path).resolve()
    validate_steerer_rights_decision(
        rights_path,
        manifest_path=rights_manifest_path,
        expected_candidate_id=str(config["candidate_id"]),
    )
    targets = cast(dict[str, object], config["targets"])
    return EvaluationProtocol(
        run_id=str(config["run_id"]),
        protocol_id=str(config["protocol_id"]),
        dataset_id=str(config["dataset_id"]),
        split_id=str(config["split_id"]),
        split_role=cast(
            Literal["train", "validation", "test", "smoke"],
            config["split_role"],
        ),
        expected_samples=int(config["expected_samples"]),
        split_verified=split_verified,
        leakage_free=split_verified,
        sealed_test_access_approved=False,
        require_clean_git=bool(config["require_clean_git"]),
        rights_decision_path=str(rights_path),
        rights_decision_sha256=sha256_file(rights_path),
        localization_radius=float(config["localization_radius"]),
        mae_max=float(targets["mae_max"]),
        rmse_max=float(targets["rmse_max"]),
        bias_max=float(targets["bias_max"]),
        band_bias_max=float(targets["band_bias_max"]),
        condition_bias_max=float(targets["condition_bias_max"]),
        latency_max_ms=float(targets["latency_max_ms"]),
        vram_max_mb=float(targets["vram_max_mb"]),
        spatial_metric_name=str(targets["spatial_metric_name"]),
        spatial_direction=cast(
            Literal["minimize", "maximize"],
            targets["spatial_direction"],
        ),
        spatial_target=float(targets["spatial_target"]),
        required_density_bands=_string_tuple(targets["required_density_bands"]),
        density_band_rules=_string_tuple(targets["density_band_rules"]),
        required_condition_keys=_string_tuple(targets["required_condition_keys"]),
        zone_warning_count=float(targets["zone_warning_count"]),
        zone_critical_count=float(targets["zone_critical_count"]),
    )
