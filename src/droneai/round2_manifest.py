"""Build and verify the shared Round 2 sample manifest."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import Mapping

from droneai.evaluation_contract import EvaluationSample
from droneai.integrity import sha256_file
from droneai.round2_config import Round2Config
from droneai.round2_samples import load_round2_samples


def _canonical_bytes(payload: object) -> bytes:
    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _payload_sha256(payload: object) -> str:
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


def _config_identity(config: Round2Config) -> dict[str, object]:
    return {
        "round_id": config.round_id,
        "action": config.action,
        "models": [asdict(model) for model in config.models],
        "datasets": [
            {
                "dataset_id": lane.dataset_id,
                "partition": lane.partition,
                "samples": lane.samples,
                "rights_scope": lane.rights_scope,
                "manifest_sha256": sha256_file(lane.manifest_path),
            }
            for lane in config.datasets
        ],
        "total_samples_per_model": config.total_samples_per_model,
        "combined_rights_scope": config.combined_rights_scope,
        "fine_tuning": config.fine_tuning,
    }


def _split_id(config: Round2Config, dataset_id: str, partition: str) -> str:
    safe_dataset = dataset_id.replace("_", "-")
    safe_partition = partition.replace("_", "-")
    return f"{config.round_id}-{safe_dataset}-{safe_partition}"


def _materialize(
    config: Round2Config,
    dataset_roots: Mapping[str, str | Path],
) -> tuple[list[dict[str, object]], tuple[EvaluationSample, ...], dict[str, int]]:
    rows: list[dict[str, object]] = []
    samples: list[EvaluationSample] = []
    counts: dict[str, int] = {}
    for lane in config.datasets:
        if lane.dataset_id not in dataset_roots:
            raise KeyError(f"dataset root missing: {lane.dataset_id}")
        split_id = _split_id(config, lane.dataset_id, lane.partition)
        lane_samples = load_round2_samples(
            dataset_id=lane.dataset_id,
            dataset_root=dataset_roots[lane.dataset_id],
            partition=lane.partition,
            expected_samples=lane.samples,
            split_id=split_id,
        )
        if len(lane_samples) != lane.samples:
            raise ValueError(f"dataset sample count mismatch: {lane.dataset_id}")
        counts[lane.dataset_id] = len(lane_samples)
        for sample in sorted(lane_samples, key=lambda value: value.sample_id):
            global_sample_id = f"{lane.dataset_id}:{sample.sample_id}"
            rows.append(
                {
                    "global_sample_id": global_sample_id,
                    "sample_id": sample.sample_id,
                    "dataset_id": lane.dataset_id,
                    "partition": lane.partition,
                    "split_id": split_id,
                    "rights_scope": lane.rights_scope,
                    "source_sha256": sample.source_sha256,
                    "annotation_sha256": sample.annotation_sha256,
                    "width": sample.width,
                    "height": sample.height,
                    "ground_truth_count": sample.ground_truth_count,
                    "condition_tags": dict(sorted(sample.condition_tags.items())),
                }
            )
            samples.append(sample)
    global_ids = [str(row["global_sample_id"]) for row in rows]
    if len(global_ids) != len(set(global_ids)):
        raise ValueError("Round 2 global sample IDs must be unique")
    if len(rows) != config.total_samples_per_model:
        raise ValueError(
            "Round 2 materialized count differs from total_samples_per_model"
        )
    return rows, tuple(samples), counts


def build_round2_manifest(
    config: Round2Config,
    dataset_roots: Mapping[str, str | Path],
) -> dict[str, object]:
    rows, _, counts = _materialize(config, dataset_roots)
    identity = _config_identity(config)
    return {
        "schema_version": 1,
        "round_id": config.round_id,
        "action": config.action,
        "combined_rights_scope": config.combined_rights_scope,
        "fine_tuning": config.fine_tuning,
        "config_identity": identity,
        "config_identity_sha256": _payload_sha256(identity),
        "counts": counts,
        "expected_samples_per_model": config.total_samples_per_model,
        "samples": rows,
        "sample_manifest_sha256": _payload_sha256(rows),
    }


def verify_round2_manifest(
    payload: dict[str, object],
    config: Round2Config,
    dataset_roots: Mapping[str, str | Path],
) -> tuple[EvaluationSample, ...]:
    if payload.get("schema_version") != 1:
        raise ValueError("Round 2 manifest requires schema_version=1")
    saved_rows = payload.get("samples")
    if not isinstance(saved_rows, list):
        raise ValueError("Round 2 manifest requires a samples list")
    if payload.get("sample_manifest_sha256") != _payload_sha256(saved_rows):
        raise ValueError("sample manifest SHA-256 does not match its rows")
    rebuilt_rows, samples, rebuilt_counts = _materialize(config, dataset_roots)
    saved_by_id = {
        str(row.get("global_sample_id")): row
        for row in saved_rows
        if isinstance(row, dict)
    }
    rebuilt_by_id = {
        str(row["global_sample_id"]): row for row in rebuilt_rows
    }
    if set(saved_by_id) != set(rebuilt_by_id):
        raise ValueError("Round 2 manifest sample identities changed")
    for global_id, rebuilt in rebuilt_by_id.items():
        saved = saved_by_id[global_id]
        if saved.get("source_sha256") != rebuilt["source_sha256"]:
            raise ValueError(f"source SHA-256 mismatch: {global_id}")
        if saved.get("annotation_sha256") != rebuilt["annotation_sha256"]:
            raise ValueError(f"annotation SHA-256 mismatch: {global_id}")
        if saved != rebuilt:
            raise ValueError(f"sample metadata mismatch: {global_id}")
    expected = build_round2_manifest(config, dataset_roots)
    if payload.get("config_identity_sha256") != expected["config_identity_sha256"]:
        raise ValueError("Round 2 config identity changed")
    if payload.get("counts") != rebuilt_counts:
        raise ValueError("Round 2 dataset counts changed")
    if payload != expected:
        raise ValueError("Round 2 manifest differs from deterministic rebuild")
    return samples


def write_round2_manifest(
    path: str | Path,
    *,
    config: Round2Config,
    dataset_roots: Mapping[str, str | Path],
) -> str:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(f"Round 2 manifest already exists: {output}")
    temporary = output.with_suffix(output.suffix + ".tmp")
    if temporary.exists():
        raise FileExistsError(f"Round 2 manifest staging file exists: {temporary}")
    payload = build_round2_manifest(config, dataset_roots)
    verify_round2_manifest(payload, config, dataset_roots)
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(output)
    return sha256_file(output)
