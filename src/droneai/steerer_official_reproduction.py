"""Fail-closed contract for the STEERER UCF-QNRF official-code reproduction."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Iterator, Mapping, cast

from droneai.evaluation_runner import EvaluationProtocol
from droneai.integrity import is_sha256, sha256_file
from droneai.steerer_constants import UPSTREAM_COMMIT


OFFICIAL_CHECKPOINT_SHA256 = (
    "85f71763595527e419257d459288e4a784766f41687ff0b859dcff65a151bbec"
)
OFFICIAL_CHECKPOINT_SIZE = 259607784
OFFICIAL_CHECKPOINT_FILENAME = "QNRF_mae_77.8_mse_138.0_85f71763.pth"
OFFICIAL_CHECKPOINT_ARTIFACT_ID = "steerer-official-ucf-qnrf-checkpoint"
ROOT_PREFIX = "steerer-official-code-reproduction"

_ROOT_KEYS = {
    "schema_version",
    "run_id",
    "protocol_id",
    "dataset_id",
    "train_split_id",
    "test_split_id",
    "train_samples",
    "test_samples",
    "upstream_commit",
    "checkpoint_origin",
    "checkpoint",
    "result_label",
    "selection_basis",
    "test_access_approved",
    "official_weight_scope",
    "roots",
    "repository_claim",
    "compatibility_tolerance",
    "targets",
}
_TARGET_KEYS = {
    "mae_max",
    "rmse_max",
    "bias_max",
    "band_bias_max",
    "condition_bias_max",
    "spatial_metric_name",
    "spatial_direction",
    "spatial_target",
    "required_density_bands",
    "density_band_rules",
    "required_condition_keys",
    "zone_warning_count",
    "zone_critical_count",
    "latency_max_ms",
    "vram_max_mb",
}


@dataclass(frozen=True)
class OfficialCheckpoint:
    artifact_id: str
    filename: str
    sha256: str
    size_bytes: int


@dataclass(frozen=True)
class AStorageRoots:
    processed: str
    checkpoints: str
    results: str

    def __iter__(self) -> Iterator[str]:
        return iter((self.processed, self.checkpoints, self.results))


@dataclass(frozen=True)
class OfficialReproductionProfile:
    run_id: str
    protocol_id: str
    dataset_id: str
    train_split_id: str
    test_split_id: str
    train_samples: int
    test_samples: int
    upstream_commit: str
    checkpoint_origin: str
    checkpoint: OfficialCheckpoint
    result_label: str
    selection_basis: str
    test_access_approved: bool
    official_weight_scope: str
    roots: AStorageRoots
    repository_mae: float
    repository_rmse: float
    mae_tolerance: float
    rmse_tolerance: float
    targets: Mapping[str, object]

    @property
    def g1_prepared_test_root(self) -> str:
        return f"{self.roots.processed.rstrip('/')}/g1-test-v1"


def _exact_mapping(value: Any, *, keys: set[str], name: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != keys:
        raise ValueError(f"{name} keys do not match the frozen contract")
    return value


def _nonempty(value: Any, *, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _positive_number(value: Any, *, name: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value <= 0
    ):
        raise ValueError(f"{name} must be finite and positive")
    return float(value)


def _string_list(value: object, *, name: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not value or any(
        not isinstance(item, str) or not item.strip() for item in value
    ):
        raise ValueError(f"{name} must be a non-empty string list")
    return tuple(value)


def validate_official_reproduction_profile(
    payload: Any,
) -> OfficialReproductionProfile:
    root = _exact_mapping(payload, keys=_ROOT_KEYS, name="A reproduction profile")
    if root["schema_version"] != 1:
        raise ValueError("schema_version must be 1")
    if root["dataset_id"] != "ucf-qnrf-kaggle-apache":
        raise ValueError("dataset_id is not approved")
    if root["train_split_id"] != "ucf-qnrf-official-train1201-v1":
        raise ValueError("official Train split ID is not approved")
    if root["test_split_id"] != "ucf-qnrf-official-test334-v1":
        raise ValueError("official Test split ID is not approved")
    if root["train_samples"] != 1201:
        raise ValueError("official-code reproduction requires exactly 1,201 Train samples")
    if root["test_samples"] != 334:
        raise ValueError("official Test evaluation requires exactly 334 samples")
    if root["upstream_commit"] != UPSTREAM_COMMIT:
        raise ValueError("upstream commit is not pinned STEERER")
    if root["checkpoint_origin"] != "research_checkpoint":
        raise ValueError("G1 checkpoint origin must be research_checkpoint")
    if root["result_label"] != "official Test, test-selected":
        raise ValueError("result label must disclose test-selected evaluation")
    _nonempty(root["selection_basis"], name="selection_basis")
    if root["test_access_approved"] is not True:
        raise PermissionError("official Test access must be explicitly approved")
    if root["official_weight_scope"] != "PASS_RESEARCH_ONLY":
        raise PermissionError("official checkpoint is restricted to research evaluation")

    raw_checkpoint = _exact_mapping(
        root["checkpoint"],
        keys={"artifact_id", "filename", "sha256", "size_bytes"},
        name="checkpoint",
    )
    checkpoint = OfficialCheckpoint(
        artifact_id=_nonempty(raw_checkpoint["artifact_id"], name="artifact_id"),
        filename=_nonempty(raw_checkpoint["filename"], name="checkpoint filename"),
        sha256=_nonempty(raw_checkpoint["sha256"], name="checkpoint SHA-256").lower(),
        size_bytes=raw_checkpoint["size_bytes"],
    )
    if (
        checkpoint.artifact_id != OFFICIAL_CHECKPOINT_ARTIFACT_ID
        or checkpoint.filename != OFFICIAL_CHECKPOINT_FILENAME
        or checkpoint.sha256 != OFFICIAL_CHECKPOINT_SHA256
        or not is_sha256(checkpoint.sha256)
        or checkpoint.size_bytes != OFFICIAL_CHECKPOINT_SIZE
    ):
        raise ValueError("checkpoint identity does not match the frozen official artifact")

    raw_roots = _exact_mapping(
        root["roots"],
        keys={"processed", "checkpoints", "results"},
        name="roots",
    )
    roots = AStorageRoots(
        processed=_nonempty(raw_roots["processed"], name="processed root"),
        checkpoints=_nonempty(raw_roots["checkpoints"], name="checkpoint root"),
        results=_nonempty(raw_roots["results"], name="result root"),
    )
    if any(ROOT_PREFIX not in path.replace("\\", "/") for path in roots):
        raise ValueError("A roots must use the isolated official-code reproduction lane")

    claim = _exact_mapping(
        root["repository_claim"], keys={"mae", "rmse"}, name="repository claim"
    )
    tolerance = _exact_mapping(
        root["compatibility_tolerance"],
        keys={"mae_absolute", "rmse_absolute"},
        name="compatibility tolerance",
    )
    targets = _exact_mapping(root["targets"], keys=_TARGET_KEYS, name="targets")
    if targets["spatial_direction"] not in {"minimize", "maximize"}:
        raise ValueError("spatial_direction is invalid")
    _string_list(targets["required_density_bands"], name="required_density_bands")
    _string_list(targets["density_band_rules"], name="density_band_rules")
    _string_list(targets["required_condition_keys"], name="required_condition_keys")

    for name in (
        "run_id",
        "protocol_id",
        "dataset_id",
        "train_split_id",
        "test_split_id",
    ):
        _nonempty(root[name], name=name)
    for name in (
        "mae_max",
        "rmse_max",
        "bias_max",
        "band_bias_max",
        "condition_bias_max",
        "spatial_target",
        "zone_warning_count",
        "zone_critical_count",
        "latency_max_ms",
        "vram_max_mb",
    ):
        _positive_number(targets[name], name=name)

    return OfficialReproductionProfile(
        run_id=root["run_id"],
        protocol_id=root["protocol_id"],
        dataset_id=root["dataset_id"],
        train_split_id=root["train_split_id"],
        test_split_id=root["test_split_id"],
        train_samples=1201,
        test_samples=334,
        upstream_commit=root["upstream_commit"],
        checkpoint_origin=root["checkpoint_origin"],
        checkpoint=checkpoint,
        result_label=root["result_label"],
        selection_basis=root["selection_basis"],
        test_access_approved=True,
        official_weight_scope=root["official_weight_scope"],
        roots=roots,
        repository_mae=_positive_number(claim["mae"], name="repository MAE"),
        repository_rmse=_positive_number(claim["rmse"], name="repository RMSE"),
        mae_tolerance=_positive_number(tolerance["mae_absolute"], name="MAE tolerance"),
        rmse_tolerance=_positive_number(tolerance["rmse_absolute"], name="RMSE tolerance"),
        targets=MappingProxyType(dict(targets)),
    )


def load_official_reproduction_profile(
    path: str | Path,
) -> OfficialReproductionProfile:
    return validate_official_reproduction_profile(
        json.loads(Path(path).read_text(encoding="utf-8"))
    )


def build_g1_protocol(
    profile: OfficialReproductionProfile,
    *,
    rights_path: str | Path,
) -> EvaluationProtocol:
    rights = Path(rights_path).resolve()
    if not rights.is_file():
        raise FileNotFoundError(f"rights decision is missing: {rights}")
    targets = profile.targets
    return EvaluationProtocol(
        run_id=profile.run_id,
        protocol_id=profile.protocol_id,
        dataset_id=profile.dataset_id,
        split_id=profile.test_split_id,
        split_role="test",
        expected_samples=profile.test_samples,
        split_verified=True,
        leakage_free=True,
        checkpoint_training_split_status="UNKNOWN",
        comparison_scope="compatibility_smoke",
        checkpoint_split_evidence=(
            "The official checkpoint training and selection membership is not "
            "independently verified; this is a reproduction compatibility gate."
        ),
        sealed_test_access_approved=profile.test_access_approved,
        require_clean_git=True,
        rights_decision_path=str(rights),
        rights_decision_sha256=sha256_file(rights),
        localization_radius=16.0,
        mae_max=float(targets["mae_max"]),
        rmse_max=float(targets["rmse_max"]),
        bias_max=float(targets["bias_max"]),
        band_bias_max=float(targets["band_bias_max"]),
        condition_bias_max=float(targets["condition_bias_max"]),
        latency_max_ms=float(targets["latency_max_ms"]),
        vram_max_mb=float(targets["vram_max_mb"]),
        spatial_metric_name=str(targets["spatial_metric_name"]),
        spatial_direction=cast(str, targets["spatial_direction"]),
        spatial_target=float(targets["spatial_target"]),
        required_density_bands=_string_list(
            targets["required_density_bands"], name="required_density_bands"
        ),
        density_band_rules=_string_list(
            targets["density_band_rules"], name="density_band_rules"
        ),
        required_condition_keys=_string_list(
            targets["required_condition_keys"], name="required_condition_keys"
        ),
        zone_warning_count=float(targets["zone_warning_count"]),
        zone_critical_count=float(targets["zone_critical_count"]),
    )


def validate_official_checkpoint_manifest(
    profile: OfficialReproductionProfile,
    path: str | Path,
) -> dict[str, object]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("official checkpoint artifact manifest must be an object")
    if payload.get("allowed_scope") != "research_comparison_only":
        raise PermissionError("official checkpoint manifest must remain research-only")
    expected = {
        "schema_version": 1,
        "artifact_id": profile.checkpoint.artifact_id,
        "byte_count": profile.checkpoint.size_bytes,
        "filename": profile.checkpoint.filename,
        "sha256": profile.checkpoint.sha256,
        "upstream_commit": profile.upstream_commit,
    }
    mismatches = [key for key, value in expected.items() if payload.get(key) != value]
    if mismatches:
        raise ValueError(
            "official checkpoint artifact manifest differs from frozen identity: "
            + ", ".join(mismatches)
        )
    return payload


def verify_official_checkpoint_file(
    profile: OfficialReproductionProfile,
    path: str | Path,
) -> Path:
    checkpoint = Path(path).resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"official checkpoint is missing: {checkpoint}")
    if checkpoint.name != profile.checkpoint.filename:
        raise ValueError("official checkpoint filename differs from frozen identity")
    if checkpoint.stat().st_size != profile.checkpoint.size_bytes:
        raise ValueError("official checkpoint byte count differs from frozen identity")
    observed = sha256_file(checkpoint)
    if observed != profile.checkpoint.sha256:
        raise ValueError("official checkpoint SHA-256 differs from frozen identity")
    return checkpoint


def validate_g1_rights_decision(path: str | Path) -> dict[str, object]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise PermissionError("G1 rights decision must be an object")
    if payload.get("candidate_id") != "steerer-official-ucf-qnrf-research-comparison":
        raise PermissionError("G1 rights decision candidate does not match STEERER")
    actions = payload.get("allowed_actions")
    if not isinstance(actions, list) or "research_checkpoint_evaluation" not in actions:
        raise PermissionError("G1 rights decision must authorize research checkpoint evaluation")
    if any(action in actions for action in ("deployment", "weight_reuse", "commercial_training")):
        raise PermissionError("G1 research rights cannot authorize deployment or weight reuse")
    return payload


def evaluate_g1_compatibility(
    profile: OfficialReproductionProfile,
    metrics: Mapping[str, object],
) -> dict[str, object]:
    expected = profile.test_samples
    accounting_complete = (
        metrics.get("accounting_complete") is True
        and metrics.get("expected_samples") == expected
        and metrics.get("recorded_samples") == expected
        and metrics.get("successful_samples") == expected
        and metrics.get("explicit_failures") == 0
    )
    mae = float(metrics.get("mae", math.nan))
    rmse = float(metrics.get("rmse", math.nan))
    finite = math.isfinite(mae) and math.isfinite(rmse)
    mae_difference = abs(mae - profile.repository_mae) if finite else math.inf
    rmse_difference = abs(rmse - profile.repository_rmse) if finite else math.inf
    compatible = (
        accounting_complete
        and finite
        and mae_difference <= profile.mae_tolerance
        and rmse_difference <= profile.rmse_tolerance
    )
    return {
        "schema_version": 1,
        "gate_id": "steerer-a-g1-official-checkpoint-compatibility",
        "status": "PASS_RESEARCH_ONLY" if compatible else "BLOCKED",
        "training_authorized": compatible,
        "result_label": profile.result_label,
        "accounting_complete": accounting_complete,
        "observed": {"mae": mae, "rmse": rmse},
        "repository_claim": {
            "mae": profile.repository_mae,
            "rmse": profile.repository_rmse,
        },
        "absolute_difference": {
            "mae": round(mae_difference, 6),
            "rmse": round(rmse_difference, 6),
        },
        "frozen_tolerance": {
            "mae_absolute": profile.mae_tolerance,
            "rmse_absolute": profile.rmse_tolerance,
        },
        "decision": (
            "G1 compatible; G2 data preparation may begin"
            if compatible
            else "G1 mismatch; long training remains blocked pending evaluator audit"
        ),
    }
