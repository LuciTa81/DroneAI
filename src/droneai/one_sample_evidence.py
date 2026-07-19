"""Shared fail-closed helpers for report-only one-sample evidence."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, TypeVar

from .evaluation_contract import EvaluationSample
from .integrity import sha256_file


T = TypeVar("T")


def select_sample(samples: Iterable[T], sample_id: str) -> T:
    matches = [sample for sample in samples if getattr(sample, "sample_id", None) == sample_id]
    if len(matches) != 1:
        raise ValueError(f"sample-id is not in the frozen split: {sample_id}")
    return matches[0]


def _reference(path: Path, expected_sha256: str, *, label: str) -> dict[str, str]:
    target = path.resolve()
    if not target.is_file():
        raise ValueError(f"one-sample {label} artifact is missing: {target}")
    observed = sha256_file(target)
    if observed != expected_sha256:
        raise ValueError(f"one-sample {label} SHA-256 mismatch: {target}")
    return {"path": target.as_posix(), "sha256": observed}


def build_artifact_bindings(
    *,
    sample: EvaluationSample,
    split_manifest_path: str | Path,
    checkpoint_path: str | Path,
    checkpoint_sha256: str,
    rights_decision_path: str | Path,
    environment_path: str | Path,
    panel_path: str | Path,
) -> dict[str, dict[str, str]]:
    """Recompute every external artifact hash used by a report sample."""

    split_manifest = Path(split_manifest_path).resolve()
    try:
        payload: Any = json.loads(split_manifest.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("one-sample split manifest is invalid") from exc
    selected = payload.get("selected_samples") if isinstance(payload, dict) else None
    if not isinstance(selected, list):
        raise ValueError("one-sample split manifest has no selected_samples")
    matches = [
        item
        for item in selected
        if isinstance(item, dict) and item.get("sample_id") == sample.sample_id
    ]
    if len(matches) != 1:
        raise ValueError("one-sample split manifest does not bind the selected sample")
    record = matches[0]
    annotation_hash = sample.annotation_sha256
    if annotation_hash is None:
        raise ValueError("one-sample report evidence requires an annotation SHA-256")
    if record.get("image_sha256") != sample.source_sha256:
        raise ValueError("one-sample split image hash differs from the sample")
    if record.get("annotation_sha256") != annotation_hash:
        raise ValueError("one-sample split annotation hash differs from the sample")

    return {
        "source_image": _reference(
            Path(str(record.get("image_path", ""))), sample.source_sha256, label="source image"
        ),
        "annotation": _reference(
            Path(str(record.get("annotation_path", ""))), annotation_hash, label="annotation"
        ),
        "checkpoint": _reference(
            Path(checkpoint_path), checkpoint_sha256, label="checkpoint"
        ),
        "rights_decision": _reference(
            Path(rights_decision_path), sha256_file(rights_decision_path), label="rights decision"
        ),
        "split_manifest": _reference(
            split_manifest, sha256_file(split_manifest), label="split manifest"
        ),
        "environment": _reference(
            Path(environment_path), sha256_file(environment_path), label="environment"
        ),
        "panel": _reference(Path(panel_path), sha256_file(panel_path), label="panel"),
    }


__all__ = ["build_artifact_bindings", "select_sample"]
