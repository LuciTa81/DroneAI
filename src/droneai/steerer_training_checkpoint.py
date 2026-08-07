"""Atomic, hash-bound checkpoints for staged STEERER training runs."""

from __future__ import annotations

import json
import math
import os
import random
import re
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np

from droneai.integrity import is_sha256, sha256_file


CHECKPOINT_SCHEMA_VERSION = 1
CHECKPOINT_PAYLOAD_KEYS = frozenset(
    {
        "schema_version",
        "run_id",
        "stage",
        "epoch",
        "global_step",
        "model",
        "optimizer",
        "scheduler",
        "scaler",
        "best_mae",
        "best_rmse",
        "config_sha256",
        "split_sha256s",
        "dataset_inventory_sha256",
        "backbone_sha256",
        "upstream_commit",
        "rng",
        "environment_manifest_sha256",
    }
)
_STAGES = frozenset({"T0", "T1", "T5", "T50", "T800"})
_MILESTONE_EPOCHS = {"T1": 1, "T5": 5, "T50": 50}
_MANIFEST_FILENAME = "checkpoint-manifest.json"


@dataclass(frozen=True)
class CheckpointArtifact:
    """The immutable identity of one checkpoint byte stream."""

    path: Path
    sha256: str
    byte_size: int


@dataclass(frozen=True)
class CheckpointPolicyResult:
    """Artifacts created for one epoch and their atomically replaced manifest."""

    artifacts: dict[str, CheckpointArtifact]
    manifest_path: Path


def _optional_torch() -> Any | None:
    try:
        import torch
    except ImportError:
        return None
    return torch


def _require_torch(torch_module: Any | None) -> Any:
    torch_module = _optional_torch() if torch_module is None else torch_module
    if torch_module is None:
        raise RuntimeError("PyTorch is required for checkpoint serialization")
    return torch_module


def _require_rng_mapping(rng: object) -> Mapping[str, object]:
    if not isinstance(rng, Mapping) or set(rng) != {
        "seed", "python", "numpy", "torch_cpu", "torch_cuda"
    }:
        raise ValueError("checkpoint RNG state must contain the complete RNG state")
    seed = rng["seed"]
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise ValueError("checkpoint RNG seed must be an integer")
    return rng


def capture_rng_state(seed: int, *, torch_module: Any | None = None) -> dict[str, object]:
    """Capture all pseudo-random streams needed for an exact resumed epoch."""

    if not isinstance(seed, int) or isinstance(seed, bool):
        raise ValueError("checkpoint RNG seed must be an integer")
    torch_module = _require_torch(torch_module)
    cuda = getattr(torch_module, "cuda", None)
    return {
        "seed": seed,
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch_module.get_rng_state(),
        "torch_cuda": (
            cuda.get_rng_state_all()
            if cuda is not None and hasattr(cuda, "get_rng_state_all")
            else []
        ),
    }


def restore_rng_state(rng: Mapping[str, object], *, torch_module: Any | None = None) -> None:
    """Restore all pseudo-random streams captured by :func:`capture_rng_state`."""

    _require_rng_mapping(rng)
    torch_module = _require_torch(torch_module)
    cuda = getattr(torch_module, "cuda", None)
    cuda_state = rng["torch_cuda"]
    if cuda_state and (cuda is None or not hasattr(cuda, "set_rng_state_all")):
        raise RuntimeError("PyTorch CUDA RNG state access is required for checkpoints")
    random.setstate(rng["python"])  # type: ignore[arg-type]
    np.random.set_state(rng["numpy"])  # type: ignore[arg-type]
    torch_module.set_rng_state(rng["torch_cpu"])
    if cuda_state:
        cuda.set_rng_state_all(cuda_state)


def _validate_checkpoint_payload(payload: Mapping[str, object]) -> dict[str, object]:
    actual_keys = set(payload)
    if actual_keys != CHECKPOINT_PAYLOAD_KEYS:
        missing = sorted(CHECKPOINT_PAYLOAD_KEYS.difference(actual_keys))
        extra = sorted(actual_keys.difference(CHECKPOINT_PAYLOAD_KEYS))
        raise ValueError(f"checkpoint payload keys are invalid: missing={missing} extra={extra}")

    state = dict(payload)
    if state["schema_version"] != CHECKPOINT_SCHEMA_VERSION:
        raise ValueError("checkpoint schema version is unsupported")
    if not isinstance(state["run_id"], str) or not state["run_id"]:
        raise ValueError("checkpoint run ID is required")
    if state["stage"] not in _STAGES:
        raise ValueError("checkpoint stage is invalid")
    for field in ("epoch", "global_step"):
        value = state[field]
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise ValueError(f"checkpoint {field} must be a non-negative integer")
    for field in ("model", "optimizer", "scheduler", "scaler"):
        if not isinstance(state[field], Mapping):
            raise ValueError(f"checkpoint {field} must be a state mapping")
    for field in ("best_mae", "best_rmse"):
        value = state[field]
        if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value):
            raise ValueError(f"checkpoint {field} must be finite")
    for field in (
        "config_sha256",
        "dataset_inventory_sha256",
        "backbone_sha256",
        "environment_manifest_sha256",
    ):
        if not is_sha256(state[field]):
            raise ValueError(f"checkpoint {field} must be a SHA-256")
    split_sha256s = state["split_sha256s"]
    if (
        not isinstance(split_sha256s, Mapping)
        or not split_sha256s
        or not all(isinstance(name, str) and name and is_sha256(value) for name, value in split_sha256s.items())
    ):
        raise ValueError("checkpoint split SHA-256s must be a non-empty hash mapping")
    upstream_commit = state["upstream_commit"]
    if not isinstance(upstream_commit, str) or re.fullmatch(r"[0-9a-fA-F]{40}", upstream_commit) is None:
        raise ValueError("checkpoint upstream commit must be a full Git commit")
    _require_rng_mapping(state["rng"])
    return state


@dataclass(frozen=True)
class ResumeExpectations:
    """Immutable provenance that must match before a run may resume."""

    run_id: str
    config_sha256: str
    split_sha256s: Mapping[str, str]
    dataset_inventory_sha256: str
    backbone_sha256: str
    upstream_commit: str
    environment_manifest_sha256: str

    def __post_init__(self) -> None:
        if not isinstance(self.run_id, str) or not self.run_id:
            raise ValueError("resume run ID is required")
        for field in (
            "config_sha256",
            "dataset_inventory_sha256",
            "backbone_sha256",
            "environment_manifest_sha256",
        ):
            value = getattr(self, field)
            if not is_sha256(value):
                raise ValueError(f"resume {field} must be a SHA-256")
            object.__setattr__(self, field, value.lower())
        if not isinstance(self.upstream_commit, str) or re.fullmatch(
            r"[0-9a-fA-F]{40}", self.upstream_commit
        ) is None:
            raise ValueError("resume upstream_commit must be a full Git commit")
        object.__setattr__(self, "upstream_commit", self.upstream_commit.lower())
        if (
            not isinstance(self.split_sha256s, Mapping)
            or not self.split_sha256s
            or not all(
                isinstance(name, str) and name and is_sha256(value)
                for name, value in self.split_sha256s.items()
            )
        ):
            raise ValueError("resume split_sha256s must be a non-empty hash mapping")
        object.__setattr__(
            self,
            "split_sha256s",
            MappingProxyType(
                {name: value.lower() for name, value in self.split_sha256s.items()}
            ),
        )


def _artifact(path: Path) -> CheckpointArtifact:
    if not path.is_file():
        raise FileNotFoundError(f"checkpoint missing: {path}")
    return CheckpointArtifact(
        path=path.resolve(), sha256=sha256_file(path), byte_size=path.stat().st_size
    )


def atomic_torch_save(
    payload: Mapping[str, object], path: str | Path, *, torch_module: Any | None = None
) -> CheckpointArtifact:
    """Serialize a complete checkpoint, fsync it, then atomically replace ``path``."""

    state = _validate_checkpoint_payload(payload)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    torch_module = _require_torch(torch_module)
    try:
        torch_module.save(state, temporary)
        with temporary.open("r+b") as stream:
            os.fsync(stream.fileno())
        os.replace(temporary, target)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return _artifact(target)


def save_training_checkpoint(
    path: str | Path, state: Mapping[str, object], *, torch_module: Any | None = None
) -> CheckpointArtifact:
    """Save one complete, hash-addressable training state atomically."""

    return atomic_torch_save(state, path, torch_module=torch_module)


def _torch_load(torch_module: Any, path: Path) -> object:
    try:
        return torch_module.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch_module.load(path, map_location="cpu")


def load_training_checkpoint(
    path: str | Path, *, expected_sha256: str, torch_module: Any | None = None
) -> dict[str, object]:
    """Verify checkpoint bytes before deserializing its complete state."""

    target = Path(path)
    if not is_sha256(expected_sha256):
        raise ValueError("expected checkpoint SHA-256 is invalid")
    observed_sha256 = sha256_file(target)
    if observed_sha256.lower() != expected_sha256.lower():
        raise ValueError(
            "checkpoint SHA-256 mismatch: "
            f"expected={expected_sha256} observed={observed_sha256}"
        )
    payload = _torch_load(_require_torch(torch_module), target)
    if not isinstance(payload, Mapping):
        raise ValueError("checkpoint payload must be a mapping")
    return _validate_checkpoint_payload(payload)


def _checkpoint_path(checkpoint: CheckpointArtifact | str | Path) -> Path:
    if isinstance(checkpoint, CheckpointArtifact):
        return checkpoint.path
    return Path(checkpoint)


def verify_resume(
    checkpoint: CheckpointArtifact | str | Path,
    *,
    expected_sha256: str,
    expectations: ResumeExpectations,
    torch_module: Any | None = None,
) -> dict[str, object]:
    """Load only a hash-verified checkpoint from the expected run lineage."""

    if not isinstance(expectations, ResumeExpectations):
        raise TypeError("complete ResumeExpectations are required for checkpoint resume")
    state = load_training_checkpoint(
        _checkpoint_path(checkpoint),
        expected_sha256=expected_sha256,
        torch_module=torch_module,
    )
    if state["run_id"] != expectations.run_id:
        raise ValueError(
            "checkpoint lineage mismatch: "
            f"expected run ID {expectations.run_id}, got {state['run_id']}"
        )
    for field in (
        "config_sha256",
        "dataset_inventory_sha256",
        "backbone_sha256",
        "upstream_commit",
        "environment_manifest_sha256",
    ):
        if state[field].lower() != getattr(expectations, field):
            raise ValueError(f"checkpoint provenance mismatch: {field}")
    observed_split_sha256s = {
        name: value.lower() for name, value in state["split_sha256s"].items()
    }
    if observed_split_sha256s != dict(expectations.split_sha256s):
        raise ValueError("checkpoint provenance mismatch: split_sha256s")
    return state


def _atomic_write_manifest(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
    try:
        with temporary.open("xb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _load_manifest(path: Path, *, run_id: str) -> list[dict[str, object]]:
    if not path.exists():
        return []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("checkpoint manifest is unreadable") from exc
    if (
        not isinstance(payload, dict)
        or payload.get("schema_version") != CHECKPOINT_SCHEMA_VERSION
        or payload.get("run_id") != run_id
        or not isinstance(payload.get("checkpoints"), list)
        or not all(isinstance(entry, dict) for entry in payload["checkpoints"])
    ):
        raise ValueError("checkpoint manifest lineage is invalid")
    return [dict(entry) for entry in payload["checkpoints"]]


def _manifest_entry(
    artifact: CheckpointArtifact,
    *,
    epoch: int,
    reason: str,
    parent_checkpoint_sha256: str | None,
    environment_manifest_sha256: str,
) -> dict[str, object]:
    return {
        "filename": artifact.path.name,
        "sha256": artifact.sha256,
        "byte_count": artifact.byte_size,
        "epoch": epoch,
        "metric_reason": reason,
        "parent_checkpoint_sha256": parent_checkpoint_sha256,
        "environment_manifest_sha256": environment_manifest_sha256,
    }


def save_checkpoint_with_policy(
    checkpoint_dir: str | Path,
    state: Mapping[str, object],
    *,
    current_mae: float,
    current_rmse: float,
    previous_best_mae: float | None = None,
    previous_best_rmse: float | None = None,
    parent_checkpoint_sha256: str | None = None,
    preserve_milestone: bool | None = None,
    torch_module: Any | None = None,
) -> CheckpointPolicyResult:
    """Persist ``last``, improved bests, and the stage milestone with an atomic manifest."""

    validated = _validate_checkpoint_payload(state)
    if parent_checkpoint_sha256 is not None and not is_sha256(parent_checkpoint_sha256):
        raise ValueError("parent checkpoint SHA-256 is invalid")
    stage = str(validated["stage"])
    epoch = int(validated["epoch"])
    if preserve_milestone is None:
        if stage == "T800":
            raise ValueError("T800 requires an explicit milestone decision")
        should_preserve_milestone = _MILESTONE_EPOCHS.get(stage) == epoch
    else:
        if not isinstance(preserve_milestone, bool):
            raise TypeError("preserve_milestone must be boolean or null")
        if stage != "T800":
            raise ValueError("explicit milestone control is reserved for T800")
        if not 25 <= epoch <= 800 or epoch % 25 != 0:
            raise ValueError("T800 checkpoints must be 25-epoch validation boundaries")
        expected_milestone = epoch % 100 == 0
        if preserve_milestone is not expected_milestone:
            raise ValueError("T800 milestone decision must match the 100-epoch policy")
        should_preserve_milestone = preserve_milestone
    for name, value in (("current MAE", current_mae), ("current RMSE", current_rmse)):
        if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value):
            raise ValueError(f"{name} must be finite")
    previous_mae = (
        float(validated["best_mae"])
        if previous_best_mae is None
        else previous_best_mae
    )
    previous_rmse = (
        float(validated["best_rmse"])
        if previous_best_rmse is None
        else previous_best_rmse
    )
    for name, value in (
        ("previous best MAE", previous_mae),
        ("previous best RMSE", previous_rmse),
    ):
        if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value):
            raise ValueError(f"{name} must be finite")
    if previous_best_mae is not None and float(validated["best_mae"]) != min(
        float(previous_mae), float(current_mae)
    ):
        raise ValueError("checkpoint best_mae must contain the updated best metric")
    if previous_best_rmse is not None and float(validated["best_rmse"]) != min(
        float(previous_rmse), float(current_rmse)
    ):
        raise ValueError("checkpoint best_rmse must contain the updated best metric")

    directory = Path(checkpoint_dir)
    manifest_path = directory / _MANIFEST_FILENAME
    records = _load_manifest(manifest_path, run_id=str(validated["run_id"]))
    artifacts: dict[str, CheckpointArtifact] = {}
    artifacts["last"] = save_training_checkpoint(
        directory / "last.pth", validated, torch_module=torch_module
    )
    writes: list[tuple[str, CheckpointArtifact, str]] = [("last", artifacts["last"], "last")]
    if current_mae < float(previous_mae):
        artifact = save_training_checkpoint(
            directory / "best-mae.pth", validated, torch_module=torch_module
        )
        artifacts["best-mae"] = artifact
        writes.append(("best-mae", artifact, "best_mae"))
    if current_rmse < float(previous_rmse):
        artifact = save_training_checkpoint(
            directory / "best-rmse.pth", validated, torch_module=torch_module
        )
        artifacts["best-rmse"] = artifact
        writes.append(("best-rmse", artifact, "best_rmse"))
    if should_preserve_milestone:
        key = f"milestone-{epoch:03d}"
        artifact = save_training_checkpoint(
            directory / f"{key}.pth", validated, torch_module=torch_module
        )
        artifacts[key] = artifact
        writes.append((key, artifact, key))

    replacement_names = {artifact.path.name for _, artifact, _ in writes}
    records = [record for record in records if record.get("filename") not in replacement_names]
    records.extend(
        _manifest_entry(
            artifact,
            epoch=int(validated["epoch"]),
            reason=reason,
            parent_checkpoint_sha256=parent_checkpoint_sha256,
            environment_manifest_sha256=str(validated["environment_manifest_sha256"]),
        )
        for _, artifact, reason in writes
    )
    records.sort(key=lambda record: str(record["filename"]))
    _atomic_write_manifest(
        manifest_path,
        {
            "schema_version": CHECKPOINT_SCHEMA_VERSION,
            "run_id": validated["run_id"],
            "checkpoints": records,
        },
    )
    return CheckpointPolicyResult(artifacts=artifacts, manifest_path=manifest_path)


__all__ = [
    "CHECKPOINT_PAYLOAD_KEYS",
    "CHECKPOINT_SCHEMA_VERSION",
    "CheckpointArtifact",
    "CheckpointPolicyResult",
    "ResumeExpectations",
    "atomic_torch_save",
    "capture_rng_state",
    "load_training_checkpoint",
    "restore_rng_state",
    "save_checkpoint_with_policy",
    "save_training_checkpoint",
    "verify_resume",
]
