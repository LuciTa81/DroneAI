"""Fail-closed contract for the one-time STEERER B-lane final Test evaluation."""

from __future__ import annotations

import json
import csv
import math
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, cast

from droneai.evaluation_runner import EvaluationProtocol
from droneai.integrity import is_sha256, sha256_file


_UPSTREAM_COMMIT = "5b1854dbc2d280f2326d67c65515d8baf9083810"
_EXPECTED_CHECKPOINTS = (
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
)
_ROOT_KEYS = {
    "schema_version",
    "run_id",
    "protocol_id",
    "dataset_id",
    "test_split_id",
    "expected_samples",
    "seed",
    "upstream_commit",
    "training_run_id",
    "checkpoint_origin",
    "selection_basis",
    "result_label",
    "checkpoints",
    "test_access_approved",
    "post_test_training_allowed",
    "rights_scope",
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
class FrozenCheckpointSelection:
    role: str
    epoch: int
    filename: str
    sha256: str


@dataclass(frozen=True)
class BFinalTestProfile:
    run_id: str
    protocol_id: str
    dataset_id: str
    test_split_id: str
    expected_samples: int
    seed: int
    upstream_commit: str
    training_run_id: str
    checkpoint_origin: str
    selection_basis: str
    result_label: str
    checkpoints: tuple[FrozenCheckpointSelection, ...]
    test_access_approved: bool
    post_test_training_allowed: bool
    rights_scope: str
    targets: Mapping[str, object]


def _exact_mapping(value: Any, *, keys: set[str], name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be an object")
    if set(value) != keys:
        raise ValueError(f"{name} keys do not match the frozen contract")
    return value


def _nonempty_string(value: Any, *, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value


def validate_b_final_test_profile(payload: Any) -> BFinalTestProfile:
    root = _exact_mapping(payload, keys=_ROOT_KEYS, name="B final-Test profile")
    if root["schema_version"] != 1:
        raise ValueError("schema_version must be 1")
    if root["dataset_id"] != "ucf-qnrf-kaggle-apache":
        raise ValueError("dataset_id is not approved")
    if root["test_split_id"] != "ucf-qnrf-official-test334-v1":
        raise ValueError("official Test split ID is not approved")
    if root["expected_samples"] != 334:
        raise ValueError("official Test evaluation requires exactly 334 samples")
    if root["seed"] != 3035:
        raise ValueError("seed must preserve the B training lineage")
    if root["upstream_commit"] != _UPSTREAM_COMMIT:
        raise ValueError("upstream commit is not pinned STEERER")
    if root["training_run_id"] != "steerer-qnrf-imagenet-20260806-t1-e":
        raise ValueError("training run ID is not the completed B lane")
    if root["checkpoint_origin"] != "project_training":
        raise ValueError("checkpoint_origin must be project_training")
    if root["selection_basis"] != (
        "B-lane Train-derived validation only; frozen before official Test access"
    ):
        raise ValueError("checkpoint selection basis is not preregistered")
    if root["result_label"] != "official Test, validation-selected":
        raise ValueError("result label must disclose validation selection")
    if root["test_access_approved"] is not True:
        raise PermissionError("official Test access must be explicitly approved")
    if root["post_test_training_allowed"] is not False:
        raise PermissionError("post-Test B training is forbidden")
    if root["rights_scope"] != "PASS_COMMERCIAL_CANDIDATE":
        raise PermissionError("B final Test rights scope is not approved")

    raw_checkpoints = root["checkpoints"]
    if not isinstance(raw_checkpoints, list) or len(raw_checkpoints) != 3:
        raise ValueError("checkpoint selection must contain exactly three entries")
    checkpoints: list[FrozenCheckpointSelection] = []
    for raw in raw_checkpoints:
        row = _exact_mapping(
            raw,
            keys={"role", "epoch", "filename", "sha256"},
            name="checkpoint selection",
        )
        checkpoints.append(
            FrozenCheckpointSelection(
                role=_nonempty_string(row["role"], name="checkpoint role"),
                epoch=row["epoch"],
                filename=_nonempty_string(row["filename"], name="checkpoint filename"),
                sha256=_nonempty_string(row["sha256"], name="checkpoint SHA-256").lower(),
            )
        )
    observed = tuple(
        (row.role, row.epoch, row.filename, row.sha256) for row in checkpoints
    )
    if observed != _EXPECTED_CHECKPOINTS or any(
        not isinstance(row.epoch, int)
        or isinstance(row.epoch, bool)
        or not is_sha256(row.sha256)
        for row in checkpoints
    ):
        raise ValueError("checkpoint identities do not match the frozen selection")

    targets = _exact_mapping(root["targets"], keys=_TARGET_KEYS, name="targets")
    string_fields = ("run_id", "protocol_id")
    for field in string_fields:
        _nonempty_string(root[field], name=field)
    return BFinalTestProfile(
        run_id=root["run_id"],
        protocol_id=root["protocol_id"],
        dataset_id=root["dataset_id"],
        test_split_id=root["test_split_id"],
        expected_samples=root["expected_samples"],
        seed=root["seed"],
        upstream_commit=root["upstream_commit"],
        training_run_id=root["training_run_id"],
        checkpoint_origin=root["checkpoint_origin"],
        selection_basis=root["selection_basis"],
        result_label=root["result_label"],
        checkpoints=tuple(checkpoints),
        test_access_approved=True,
        post_test_training_allowed=False,
        rights_scope=root["rights_scope"],
        targets=MappingProxyType(dict(targets)),
    )


def load_b_final_test_profile(path: str | Path) -> BFinalTestProfile:
    return validate_b_final_test_profile(
        json.loads(Path(path).read_text(encoding="utf-8"))
    )


def _string_tuple(value: object, *, name: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not value or any(
        not isinstance(item, str) or not item.strip() for item in value
    ):
        raise ValueError(f"{name} must be a non-empty string list")
    return tuple(value)


def build_b_final_test_protocol(
    profile: BFinalTestProfile,
    *,
    checkpoint: FrozenCheckpointSelection,
    rights_path: str | Path,
) -> EvaluationProtocol:
    if checkpoint not in profile.checkpoints:
        raise ValueError("checkpoint is not part of the frozen B selection")
    rights = Path(rights_path).resolve()
    if not rights.is_file():
        raise FileNotFoundError(f"training rights profile is missing: {rights}")
    targets = profile.targets
    return EvaluationProtocol(
        run_id=f"{profile.run_id}-{checkpoint.role}",
        protocol_id=f"{profile.protocol_id}-{checkpoint.role}",
        dataset_id=profile.dataset_id,
        split_id=profile.test_split_id,
        split_role="test",
        expected_samples=profile.expected_samples,
        split_verified=True,
        leakage_free=True,
        checkpoint_training_split_status="VERIFIED_DISJOINT",
        comparison_scope="held_out_performance",
        checkpoint_split_evidence=(
            "The project checkpoint used only the official UCF-QNRF Train "
            "partition; this evaluation uses the disjoint official Test partition."
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
        required_density_bands=_string_tuple(
            targets["required_density_bands"], name="required_density_bands"
        ),
        density_band_rules=_string_tuple(
            targets["density_band_rules"], name="density_band_rules"
        ),
        required_condition_keys=_string_tuple(
            targets["required_condition_keys"], name="required_condition_keys"
        ),
        zone_warning_count=float(targets["zone_warning_count"]),
        zone_critical_count=float(targets["zone_critical_count"]),
    )


def validate_checkpoint_manifest(
    profile: BFinalTestProfile, path: str | Path
) -> tuple[FrozenCheckpointSelection, ...]:
    """Bind the preregistered roles to existing training-manifest entries."""

    source = Path(path)
    payload = json.loads(source.read_text(encoding="utf-8"))
    if (
        not isinstance(payload, dict)
        or payload.get("schema_version") != 1
        or payload.get("run_id") != profile.training_run_id
        or not isinstance(payload.get("checkpoints"), list)
    ):
        raise ValueError("checkpoint manifest does not match the frozen B run")
    entries = payload["checkpoints"]
    for checkpoint in profile.checkpoints:
        if not any(
            isinstance(entry, dict)
            and entry.get("epoch") == checkpoint.epoch
            and entry.get("filename") == checkpoint.filename
            and entry.get("sha256") == checkpoint.sha256
            for entry in entries
        ):
            raise ValueError(
                "checkpoint manifest is missing frozen identity: "
                f"{checkpoint.role}@{checkpoint.epoch}"
            )
    return profile.checkpoints


def verify_checkpoint_files(
    profile: BFinalTestProfile, checkpoint_root: str | Path
) -> tuple[Path, ...]:
    root = Path(checkpoint_root).resolve()
    paths: list[Path] = []
    for checkpoint in profile.checkpoints:
        target = (root / checkpoint.filename).resolve()
        try:
            target.relative_to(root)
        except ValueError as error:
            raise ValueError("checkpoint filename escapes the checkpoint root") from error
        if not target.is_file():
            raise FileNotFoundError(f"frozen checkpoint is missing: {target}")
        observed = sha256_file(target)
        if observed != checkpoint.sha256:
            raise ValueError(
                "frozen checkpoint SHA-256 mismatch: "
                f"role={checkpoint.role} expected={checkpoint.sha256} observed={observed}"
            )
        paths.append(target)
    return tuple(paths)


def _atomic_json(path: Path, payload: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"result artifact already exists: {path}")
    content = (
        json.dumps(
            payload,
            indent=2,
            ensure_ascii=False,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return path


def _finite_float(row: Mapping[str, object], key: str) -> float:
    try:
        value = float(row[key])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"result metric is missing or invalid: {key}") from error
    if not math.isfinite(value):
        raise ValueError(f"result metric is non-finite: {key}")
    return value


def _mean_prediction_field(rows: list[dict[str, str]], key: str) -> float:
    values = [_finite_float(row, key) for row in rows]
    return float(sum(values) / len(values))


def _role_result(
    role_root: Path,
    *,
    checkpoint: FrozenCheckpointSelection,
    expected_samples: int,
) -> dict[str, object]:
    metrics_path = role_root / "metrics.json"
    score_path = role_root / "score.json"
    predictions_path = role_root / "predictions.csv"
    if not all(path.is_file() for path in (metrics_path, score_path, predictions_path)):
        raise FileNotFoundError(f"incomplete final-Test role bundle: {role_root}")
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    score = json.loads(score_path.read_text(encoding="utf-8"))
    if not isinstance(metrics, dict) or not isinstance(score, dict):
        raise ValueError("final-Test metrics and score must be objects")
    if (
        metrics.get("accounting_complete") is not True
        or metrics.get("expected_samples") != expected_samples
        or metrics.get("recorded_samples") != expected_samples
        or metrics.get("successful_samples") != expected_samples
        or metrics.get("explicit_failures") != 0
    ):
        raise ValueError(f"final-Test accounting is incomplete for {checkpoint.role}")
    with predictions_path.open("r", encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    sample_ids = [row.get("sample_id", "") for row in rows]
    if (
        len(rows) != expected_samples
        or any(not sample_id for sample_id in sample_ids)
        or len(set(sample_ids)) != expected_samples
    ):
        raise ValueError(f"final-Test predictions are incomplete for {checkpoint.role}")
    percent_errors = [
        _finite_float(row, "absolute_error")
        / _finite_float(row, "ground_truth_count")
        * 100.0
        for row in rows
        if _finite_float(row, "ground_truth_count") > 0.0
    ]
    if not percent_errors:
        raise ValueError("reference MAPE requires at least one positive ground truth")
    return {
        "role": checkpoint.role,
        "epoch": checkpoint.epoch,
        "filename": checkpoint.filename,
        "checkpoint_sha256": checkpoint.sha256,
        "sample_count": expected_samples,
        "mae": _finite_float(metrics, "mae"),
        "rmse": _finite_float(metrics, "rmse"),
        "signed_bias": _finite_float(metrics, "signed_bias"),
        "reference_mape_percent": float(sum(percent_errors) / len(percent_errors)),
        "localization_precision": _mean_prediction_field(rows, "localization_precision"),
        "localization_recall": _mean_prediction_field(rows, "localization_recall"),
        "localization_f1": _mean_prediction_field(rows, "localization_f1"),
        "localization_mean_distance_px": _mean_prediction_field(
            rows, "localization_mean_distance"
        ),
        "density_zone_mae": _mean_prediction_field(rows, "density_zone_mae"),
        "density_psnr": _mean_prediction_field(rows, "density_psnr"),
        "density_ssim": _mean_prediction_field(rows, "density_ssim"),
        "median_latency_ms": _finite_float(metrics, "median_latency_ms"),
        "throughput_fps_batch1": _finite_float(metrics, "throughput_fps_batch1"),
        "peak_vram_mb": _finite_float(metrics, "peak_vram_mb"),
        "technical_score": score.get("score"),
        "technical_status": score.get("status"),
        "metrics_sha256": sha256_file(metrics_path),
        "predictions_sha256": sha256_file(predictions_path),
        "score_sha256": sha256_file(score_path),
    }


def finalize_b_final_test(
    output_root: str | Path, *, profile: BFinalTestProfile
) -> tuple[Path, Path]:
    """Aggregate complete role bundles and atomically close the B lane."""

    root = Path(output_root)
    closure_path = root / "b-lane-closed.json"
    if closure_path.exists():
        raise FileExistsError(f"B lane is already closed: {closure_path}")
    comparison_path = root / "comparison.json"
    if comparison_path.exists():
        raise FileExistsError(f"partial B finalization requires audit: {comparison_path}")
    results = [
        _role_result(
            root / checkpoint.role,
            checkpoint=checkpoint,
            expected_samples=profile.expected_samples,
        )
        for checkpoint in profile.checkpoints
    ]
    comparison = {
        "schema_version": 1,
        "run_id": profile.run_id,
        "dataset_id": profile.dataset_id,
        "split_id": profile.test_split_id,
        "split_role": "test",
        "result_label": profile.result_label,
        "selection_basis": profile.selection_basis,
        "checkpoint_selection_remains_frozen": True,
        "results": results,
    }
    _atomic_json(comparison_path, comparison)
    closure = {
        "schema_version": 1,
        "run_id": profile.run_id,
        "status": "CLOSED_AFTER_OFFICIAL_TEST",
        "test_sample_count": profile.expected_samples,
        "result_label": profile.result_label,
        "checkpoint_selection_remains_frozen": True,
        "post_test_training_allowed": profile.post_test_training_allowed,
        "comparison_sha256": sha256_file(comparison_path),
    }
    _atomic_json(closure_path, closure)
    return comparison_path, closure_path


__all__ = [
    "BFinalTestProfile",
    "FrozenCheckpointSelection",
    "build_b_final_test_protocol",
    "finalize_b_final_test",
    "load_b_final_test_profile",
    "validate_checkpoint_manifest",
    "validate_b_final_test_profile",
    "verify_checkpoint_files",
]
