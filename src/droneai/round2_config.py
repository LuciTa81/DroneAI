"""Immutable configuration contract for the Round 2 reference benchmark."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


LEGACY_MODEL_IDS = ("steerer", "dm-count", "mpcount")
POINT_MODEL_IDS = ("steerer", "pet", "apgcc")
APPROVED_MODEL_MATRICES = (LEGACY_MODEL_IDS, POINT_MODEL_IDS)
SUPPORTED_MODEL_IDS = tuple(
    dict.fromkeys(LEGACY_MODEL_IDS + POINT_MODEL_IDS)
)
# Backward-compatible alias for historical callers that verify the legacy
# Round 2 evidence.
APPROVED_MODEL_IDS = LEGACY_MODEL_IDS
APPROVED_DATASET_MATRIX = (
    ("ucf-qnrf-kaggle-apache", "test", 334),
    ("jhu-crowd-plus-v2", "val", 500),
    ("up-count-v1", "val_test", 166),
)


@dataclass(frozen=True)
class DatasetLane:
    dataset_id: str
    partition: str
    samples: int
    rights_scope: str
    manifest_path: Path


@dataclass(frozen=True)
class ModelLane:
    model_id: str
    family: str
    rights_scope: str


@dataclass(frozen=True)
class Round2Config:
    round_id: str
    action: str
    models: tuple[ModelLane, ...]
    datasets: tuple[DatasetLane, ...]
    total_samples_per_model: int
    combined_rights_scope: str
    fine_tuning: bool


def _object(value: object, *, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value


def _repository_root(config_path: Path) -> Path:
    try:
        return config_path.resolve().parents[2]
    except IndexError as exc:
        raise ValueError("Round 2 config must live under configs/evaluation") from exc


def _load_dataset_lane(row: object, *, repository_root: Path) -> DatasetLane:
    value = _object(row, label="dataset lane")
    required = ("dataset_id", "partition", "samples", "rights_scope", "manifest_path")
    if any(key not in value for key in required):
        raise ValueError("dataset lane is missing required fields")
    raw_manifest = Path(str(value["manifest_path"]))
    manifest_path = (
        raw_manifest if raw_manifest.is_absolute() else repository_root / raw_manifest
    ).resolve()
    if not manifest_path.is_file():
        raise FileNotFoundError(f"dataset manifest missing: {manifest_path}")
    manifest = _object(
        json.loads(manifest_path.read_text(encoding="utf-8")),
        label="dataset manifest",
    )
    dataset_id = str(value["dataset_id"])
    if manifest.get("dataset_id") != dataset_id:
        raise ValueError(f"dataset manifest identity mismatch: {dataset_id}")
    rights = _object(manifest.get("rights"), label="dataset rights")
    scope = str(value["rights_scope"])
    if scope == "PASS_RESEARCH_ONLY" and rights.get("research_use") is not True:
        raise ValueError(f"research rights are not verified: {dataset_id}")
    if scope == "PASS_COMMERCIAL_CANDIDATE" and rights.get("commercial_use") is not True:
        raise ValueError(f"commercial-candidate rights are not recorded: {dataset_id}")
    samples = value["samples"]
    if isinstance(samples, bool) or not isinstance(samples, int) or samples <= 0:
        raise ValueError("dataset sample count must be a positive integer")
    return DatasetLane(
        dataset_id=dataset_id,
        partition=str(value["partition"]),
        samples=samples,
        rights_scope=scope,
        manifest_path=manifest_path,
    )


def load_round2_config(path: str | Path) -> Round2Config:
    config_path = Path(path).resolve()
    payload = _object(
        json.loads(config_path.read_text(encoding="utf-8")), label="Round 2 config"
    )
    if payload.get("schema_version") != 1:
        raise ValueError("Round 2 config requires schema_version=1")
    if payload.get("action") != "frozen_internal_reference_evaluation":
        raise ValueError("action must be frozen_internal_reference_evaluation")
    models = tuple(
        ModelLane(
            model_id=str(_object(row, label="model lane")["model_id"]),
            family=str(_object(row, label="model lane")["family"]),
            rights_scope=str(_object(row, label="model lane")["rights_scope"]),
        )
        for row in payload.get("models", ())
    )
    model_ids = tuple(model.model_id for model in models)
    if model_ids not in APPROVED_MODEL_MATRICES:
        raise ValueError(
            "model order differs from an approved ordered model shortlist"
        )
    repository_root = _repository_root(config_path)
    datasets = tuple(
        _load_dataset_lane(row, repository_root=repository_root)
        for row in payload.get("datasets", ())
    )
    observed_matrix = tuple(
        (lane.dataset_id, lane.partition, lane.samples) for lane in datasets
    )
    if observed_matrix != APPROVED_DATASET_MATRIX:
        raise ValueError("dataset matrix differs from the approved Round 2 scope")
    total = payload.get("total_samples_per_model")
    if total != 1000 or sum(lane.samples for lane in datasets) != 1000:
        raise ValueError("Round 2 requires exactly 1,000 samples per model")
    if payload.get("combined_rights_scope") != "PASS_RESEARCH_ONLY":
        raise ValueError("combined Round 2 scope must be PASS_RESEARCH_ONLY")
    if payload.get("fine_tuning") is not False:
        raise ValueError("Round 2 requires fine_tuning=false")
    return Round2Config(
        round_id=str(payload.get("round_id") or ""),
        action=str(payload["action"]),
        models=models,
        datasets=datasets,
        total_samples_per_model=total,
        combined_rights_scope=str(payload["combined_rights_scope"]),
        fine_tuning=False,
    )
