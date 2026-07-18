"""Frozen configuration and input preparation for the DM-Count QNRF smoke."""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Sequence, cast

from droneai.comparison_claim import validate_comparison_claim
from droneai.evaluation_contract import EvaluationSample
from droneai.evaluation_runner import EvaluationProtocol
from droneai.integrity import is_sha256, sha256_file
from droneai.stage3c import REQUIRED_COMPONENTS, manifest_semantic_sha256
from droneai.ucf_qnrf import (
    UCFQNRFRecord,
    apply_official_train_validation_split,
    index_ucf_qnrf_train,
    prepare_evaluation_sample,
    select_stratified_smoke,
)


@dataclass(frozen=True)
class PreparedSmoke:
    samples: tuple[EvaluationSample, ...]
    selected_records: tuple[UCFQNRFRecord, ...]
    source_train_count: int
    train_count: int
    validation_count: int
    split_verified: bool
    train_list_sha256: str
    validation_list_sha256: str


def _positive_integer(config: dict[str, object], key: str) -> int:
    value = config.get(key)
    if type(value) is not int or value <= 0:
        raise ValueError(f"smoke {key} must be a positive integer")
    return value


def _nonnegative_number(config: dict[str, object], key: str) -> float:
    value = config.get(key)
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value < 0
    ):
        raise ValueError(f"smoke {key} must be finite and non-negative")
    return float(value)


def load_smoke_config(path: str | Path) -> dict[str, object]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise ValueError("DM-Count smoke config requires schema version 1")
    for key in (
        "run_id",
        "protocol_id",
        "model_id",
        "dataset_id",
        "split_id",
        "upstream_commit",
        "required_action",
        "candidate_id",
    ):
        if not isinstance(payload.get(key), str) or not str(payload[key]).strip():
            raise ValueError(f"smoke {key} must be a non-empty string")
    if payload["split_role"] != "validation":
        raise ValueError("DM-Count smoke must use the official validation role")
    if payload.get("sealed_test_access_approved") is not False:
        raise ValueError("DM-Count smoke cannot approve sealed test access")
    if payload.get("required_action") != "frozen_checkpoint_evaluation":
        raise ValueError("DM-Count smoke requires frozen-checkpoint evaluation rights")
    if re.fullmatch(r"[0-9a-f]{40}", str(payload["upstream_commit"])) is None:
        raise ValueError("DM-Count smoke requires a pinned upstream commit")
    for key in ("train_list_sha256", "validation_list_sha256"):
        if not is_sha256(payload.get(key)):
            raise ValueError(f"DM-Count smoke requires a frozen {key}")
    counts = tuple(
        _positive_integer(payload, key)
        for key in (
            "expected_samples",
            "source_train_images",
            "official_train_samples",
            "official_validation_samples",
        )
    )
    if counts[0] != 36 or counts[1] != counts[2] + counts[3]:
        raise ValueError("DM-Count smoke requires 36 samples and a complete official split")
    if type(payload.get("seed")) is not int or int(payload["seed"]) < 0:
        raise ValueError("DM-Count smoke seed must be a non-negative integer")
    for key in ("split_verified", "leakage_free", "require_clean_git"):
        if type(payload.get(key)) is not bool:
            raise ValueError(f"smoke {key} must be boolean")
    validate_comparison_claim(
        str(payload.get("checkpoint_training_split_status", "")),
        str(payload.get("comparison_scope", "")),
        payload.get("checkpoint_split_evidence"),
    )
    _nonnegative_number(payload, "localization_radius")
    targets = payload.get("targets")
    if not isinstance(targets, dict):
        raise ValueError("DM-Count smoke targets must be an object")
    for key in (
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
        _nonnegative_number(targets, key)
    for key in (
        "required_density_bands",
        "density_band_rules",
        "required_condition_keys",
    ):
        values = targets.get(key)
        if not isinstance(values, list) or not values or any(
            not isinstance(value, str) or not value.strip() for value in values
        ):
            raise ValueError(f"smoke target {key} must be a non-empty string list")
    if targets.get("spatial_direction") not in {"minimize", "maximize"}:
        raise ValueError("smoke spatial direction is invalid")
    if not isinstance(targets.get("spatial_metric_name"), str):
        raise ValueError("smoke spatial metric name is required")
    return payload


def validate_rights_decision(
    path: str | Path,
    *,
    manifest_path: str | Path,
    expected_candidate_id: str,
) -> dict[str, object]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("rights decision must be an object")
    if not isinstance(manifest, dict):
        raise ValueError("rights manifest must be an object")
    component_ids = {
        str(component.get("component_type")): component.get("component_id")
        for component in manifest.get("components", [])
        if isinstance(component, dict)
    }
    identity_matches = (
        expected_candidate_id == manifest.get("candidate_id")
        and expected_candidate_id == payload.get("candidate_id")
        and payload.get("manifest_semantic_sha256")
        == manifest_semantic_sha256(manifest)
        and payload.get("component_ids") == component_ids
        and set(component_ids) == REQUIRED_COMPONENTS
    )
    if not identity_matches:
        raise PermissionError("rights decision candidate or manifest binding failed")
    actions = payload.get("allowed_actions")
    if not isinstance(actions, list) or "frozen_checkpoint_evaluation" not in actions:
        raise PermissionError(
            "rights decision must authorize frozen_checkpoint_evaluation"
        )
    if payload.get("status") not in {
        "PASS_COMMERCIAL_CANDIDATE",
        "PRODUCTION_APPROVED",
    }:
        raise PermissionError("rights status does not permit the approved benchmark lane")
    return payload


def validate_official_split_paths(
    *,
    upstream_dir: str | Path,
    train_list_path: str | Path,
    validation_list_path: str | Path,
) -> None:
    preprocess = Path(upstream_dir).resolve() / "preprocess"
    expected = (
        (preprocess / "qnrf_train.txt").resolve(),
        (preprocess / "qnrf_val.txt").resolve(),
    )
    observed = (Path(train_list_path).resolve(), Path(validation_list_path).resolve())
    if observed != expected:
        raise ValueError(
            "official QNRF split lists must be the pinned upstream preprocess files"
        )


def prepare_smoke_samples(
    *,
    config: dict[str, object],
    train_root: str | Path,
    train_list_path: str | Path,
    validation_list_path: str | Path,
) -> PreparedSmoke:
    train_list_hash = sha256_file(train_list_path)
    validation_list_hash = sha256_file(validation_list_path)
    if train_list_hash != config["train_list_sha256"]:
        raise ValueError("official train split-list SHA-256 mismatch")
    if validation_list_hash != config["validation_list_sha256"]:
        raise ValueError("official validation split-list SHA-256 mismatch")
    records = index_ucf_qnrf_train(train_root)
    official_train, official_validation = apply_official_train_validation_split(
        records,
        train_list_path=train_list_path,
        validation_list_path=validation_list_path,
    )
    expected_counts = (
        int(config["source_train_images"]),
        int(config["official_train_samples"]),
        int(config["official_validation_samples"]),
    )
    observed_counts = (len(records), len(official_train), len(official_validation))
    if observed_counts != expected_counts:
        raise ValueError(
            f"official UCF-QNRF split cardinality mismatch: expected={expected_counts} "
            f"observed={observed_counts}"
        )
    selected = select_stratified_smoke(
        official_validation,
        sample_count=int(config["expected_samples"]),
        seed=int(config["seed"]),
    )
    samples = tuple(
        prepare_evaluation_sample(
            record,
            split_id=str(config["split_id"]),
            dataset_id=str(config["dataset_id"]),
            source_partition="official_validation",
        )
        for record in selected
    )
    return PreparedSmoke(
        samples=samples,
        selected_records=selected,
        source_train_count=len(records),
        train_count=len(official_train),
        validation_count=len(official_validation),
        split_verified=True,
        train_list_sha256=train_list_hash,
        validation_list_sha256=validation_list_hash,
    )


def write_split_source_manifest(
    path: str | Path,
    *,
    prepared: PreparedSmoke,
    train_list_path: str | Path,
    validation_list_path: str | Path,
) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "source_train_images": prepared.source_train_count,
                "official_train_samples": prepared.train_count,
                "official_validation_samples": prepared.validation_count,
                "test_root_argument_exposed": False,
                "test_data_access": "not_possible_through_adapter_api",
                "split_verified": prepared.split_verified,
                "train_list_sha256": prepared.train_list_sha256,
                "validation_list_sha256": prepared.validation_list_sha256,
                "selected_samples": [
                    {
                        "sample_id": record.sample_id,
                        "image_path": str(record.image_path),
                        "image_sha256": record.image_sha256,
                        "annotation_path": str(record.annotation_path),
                        "annotation_sha256": record.annotation_sha256,
                        "count": record.count,
                        "density_band": record.density_band,
                        "normalization_corrections": record.normalization_corrections,
                    }
                    for record in prepared.selected_records
                ],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return target


def _string_tuple(values: object) -> tuple[str, ...]:
    if not isinstance(values, list):
        raise ValueError("expected a string list")
    return tuple(str(value) for value in values)


def build_protocol(
    config: dict[str, object],
    *,
    rights_decision_path: str | Path,
    rights_manifest_path: str | Path,
    split_verified: bool,
) -> EvaluationProtocol:
    rights_path = Path(rights_decision_path).resolve()
    validate_rights_decision(
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
        spatial_direction=cast(Literal["minimize", "maximize"], targets["spatial_direction"]),
        spatial_target=float(targets["spatial_target"]),
        required_density_bands=_string_tuple(targets["required_density_bands"]),
        density_band_rules=_string_tuple(targets["density_band_rules"]),
        required_condition_keys=_string_tuple(targets["required_condition_keys"]),
        zone_warning_count=float(targets["zone_warning_count"]),
        zone_critical_count=float(targets["zone_critical_count"]),
    )
