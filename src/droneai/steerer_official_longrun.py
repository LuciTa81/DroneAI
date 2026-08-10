"""Pinned raw-upstream launch primitives for the STEERER A800 lane."""

from __future__ import annotations

from bisect import bisect_right
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
from typing import Mapping, Sequence

from droneai.integrity import is_sha256, sha256_file
from droneai.steerer_training_longrun import RunLock


OFFICIAL_END_EPOCH = 800
OFFICIAL_VAL_SPAN = (-800, -600, -400, -200, -200, -100, -100)
_OFFICIAL_CHECKPOINT_KEYS = frozenset(
    {"epoch", "best_MAE", "best_MSE", "state_dict", "optimizer"}
)


@dataclass(frozen=True)
class OfficialA800Lineage:
    project_commit: str
    upstream_commit: str
    upstream_config_sha256: str
    profile_sha256: str
    dataset_manifest_sha256: str
    dataset_content_sha256: str
    train_split_sha256: str
    test_split_sha256: str
    backbone_sha256: str
    container_image_digest: str

    def __post_init__(self) -> None:
        for field in (
            "upstream_config_sha256",
            "profile_sha256",
            "dataset_manifest_sha256",
            "dataset_content_sha256",
            "train_split_sha256",
            "test_split_sha256",
            "backbone_sha256",
        ):
            if not is_sha256(getattr(self, field)):
                raise ValueError(f"A800 lineage {field} must be a SHA-256")
        for field in ("project_commit", "upstream_commit"):
            value = getattr(self, field)
            if re.fullmatch(r"[0-9a-fA-F]{40}", value) is None:
                raise ValueError(f"A800 lineage {field} must be a full Git commit")
        digest = self.container_image_digest.removeprefix("sha256:")
        if not self.container_image_digest.startswith("sha256:") or not is_sha256(digest):
            raise ValueError("A800 lineage container image digest is invalid")


@dataclass(frozen=True)
class OfficialLaunchResult:
    return_code: int
    attempt: int
    status_path: Path
    attempt_manifest_path: Path


def official_validation_epochs() -> tuple[int, ...]:
    """Return the exact 1-based Test epochs selected by upstream train_cc.py."""

    return tuple(
        zero_based_epoch + 1
        for zero_based_epoch in range(OFFICIAL_END_EPOCH)
        if (zero_based_epoch + 1)
        % bisect_right(OFFICIAL_VAL_SPAN, -zero_based_epoch)
        == 0
    )


def build_official_a800_command(
    *,
    python_executable: str | Path,
    upstream_dir: str | Path,
    processed_root: str | Path,
    backbone_path: str | Path,
    log_root: str | Path,
    resume_run_dir: str | Path | None = None,
) -> tuple[str, ...]:
    """Build raw train_cc.py argv with path-only configuration overrides."""

    upstream = Path(upstream_dir).resolve(strict=True)
    trainer = (upstream / "tools" / "train_cc.py").resolve(strict=True)
    config = (upstream / "configs" / "QNRF_final.py").resolve(strict=True)
    processed = Path(processed_root).resolve(strict=True)
    backbone = Path(backbone_path).resolve(strict=True)
    output = Path(log_root).resolve(strict=False)
    command = (
        str(Path(python_executable).resolve(strict=False)),
        str(trainer),
        "--cfg",
        str(config),
        "--launcher",
        "none",
        "--cfg-options",
        f"dataset.root={processed.as_posix()}/",
        f"network.pretrained_backbone={backbone.as_posix()}",
        f"log_dir={output.as_posix()}",
    )
    if resume_run_dir is None:
        return command
    resume = Path(resume_run_dir).resolve(strict=True)
    (resume / "checkpoint.pth.tar").resolve(strict=True)
    return (*command, f"train.resume_path={resume.as_posix()}")


def _safe_run_id(run_id: str) -> str:
    if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", run_id or "") is None:
        raise ValueError("A800 run_id must be one safe path component")
    return run_id


def _json_bytes(payload: object) -> bytes:
    return (json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n").encode(
        "utf-8"
    )


def _atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        with temporary.open("xb") as stream:
            stream.write(_json_bytes(payload))
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _execute_locked_attempt(
    *,
    run_id: str,
    command: Sequence[str],
    working_directory: Path,
    result_root: Path,
    checkpoint_root: Path,
    lineage: OfficialA800Lineage,
    environment: Mapping[str, object],
    resume: bool,
) -> OfficialLaunchResult:
    attempt = len(tuple(result_root.glob("attempt-*-manifest.json"))) + 1
    attempt_manifest_path = result_root / f"attempt-{attempt:03d}-manifest.json"
    status_path = result_root / "launcher-status.json"
    started_at = _now()
    command_tuple = tuple(command)
    command_sha256 = hashlib.sha256(
        json.dumps(command_tuple, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    manifest = {
        "schema_version": 1,
        "run_id": run_id,
        "attempt": attempt,
        "execution": "pinned raw upstream tools/train_cc.py",
        "resume": resume,
        "command": list(command_tuple),
        "command_sha256": command_sha256,
        "cwd": str(working_directory),
        "result_root": str(result_root),
        "checkpoint_root": str(checkpoint_root),
        "official_validation_count": len(official_validation_epochs()),
        "official_validation_epochs": list(official_validation_epochs()),
        "lineage": asdict(lineage),
        "environment": dict(environment),
        "started_at": started_at,
    }
    _atomic_json(attempt_manifest_path, manifest)
    status = {
        "schema_version": 1,
        "run_id": run_id,
        "attempt": attempt,
        "state": "starting",
        "pid": None,
        "return_code": None,
        "attempt_manifest": attempt_manifest_path.name,
        "command_sha256": command_sha256,
        "started_at": started_at,
        "finished_at": None,
    }
    _atomic_json(status_path, status)
    child_environment = dict(os.environ)
    child_environment.update({"CUDA_VISIBLE_DEVICES": "0", "PYTHONUNBUFFERED": "1"})
    process = subprocess.Popen(
        command_tuple,
        cwd=working_directory,
        env=child_environment,
    )
    status.update({"state": "running", "pid": process.pid})
    _atomic_json(status_path, status)
    return_code = int(process.wait())
    status.update(
        {
            "state": "completed" if return_code == 0 else "failed",
            "return_code": return_code,
            "finished_at": _now(),
        }
    )
    _atomic_json(status_path, status)
    if return_code != 0:
        raise subprocess.CalledProcessError(return_code, command_tuple)
    return OfficialLaunchResult(
        return_code=return_code,
        attempt=attempt,
        status_path=status_path,
        attempt_manifest_path=attempt_manifest_path,
    )


def run_raw_official_training(
    *,
    run_id: str,
    command: Sequence[str],
    cwd: str | Path,
    result_dir: str | Path,
    checkpoint_dir: str | Path,
    lineage: OfficialA800Lineage,
    environment: Mapping[str, object],
    resume: bool,
) -> OfficialLaunchResult:
    """Run one locked raw-upstream attempt and record immutable launch evidence."""

    run_id = _safe_run_id(run_id)
    if not isinstance(lineage, OfficialA800Lineage):
        raise TypeError("validated A800 lineage is required")
    if not command or not all(isinstance(argument, str) and argument for argument in command):
        raise ValueError("A800 raw-upstream command must be a non-empty argv sequence")
    working_directory = Path(cwd).resolve(strict=True)
    result_root = Path(result_dir)
    checkpoint_root = Path(checkpoint_dir)
    if not resume:
        if result_root.exists() or checkpoint_root.exists():
            raise FileExistsError("fresh A800 result and checkpoint directories must be new")
        result_root.mkdir(parents=True)
        checkpoint_root.mkdir(parents=True)
    else:
        result_root = result_root.resolve(strict=True)
        checkpoint_root = checkpoint_root.resolve(strict=True)
        if not result_root.is_dir() or not checkpoint_root.is_dir():
            raise ValueError("A800 resume roots must be directories")

    with RunLock(result_root / "run.lock", run_id=run_id):
        return _execute_locked_attempt(
            run_id=run_id,
            command=command,
            working_directory=working_directory,
            result_root=result_root,
            checkpoint_root=checkpoint_root,
            lineage=lineage,
            environment=environment,
            resume=resume,
        )


def verify_official_resume_checkpoint(
    path: str | Path,
    *,
    expected_sha256: str,
    torch_module: object,
) -> dict[str, object]:
    """Hash-check and validate the exact raw train_cc.py resume payload."""

    checkpoint = Path(path).resolve(strict=True)
    if not is_sha256(expected_sha256):
        raise ValueError("expected checkpoint SHA-256 is invalid")
    observed_sha256 = sha256_file(checkpoint)
    if observed_sha256.lower() != expected_sha256.lower():
        raise ValueError(
            f"checkpoint SHA-256 mismatch: expected={expected_sha256} observed={observed_sha256}"
        )
    load = getattr(torch_module, "load", None)
    if not callable(load):
        raise TypeError("torch_module.load is required")
    try:
        payload = load(checkpoint, map_location="cpu", weights_only=False)
    except TypeError:
        payload = load(checkpoint, map_location="cpu")
    if not isinstance(payload, Mapping) or set(payload) != _OFFICIAL_CHECKPOINT_KEYS:
        raise ValueError("checkpoint must contain the exact raw-upstream schema")
    state = dict(payload)
    epoch = state["epoch"]
    if not isinstance(epoch, int) or isinstance(epoch, bool) or not 1 <= epoch <= 800:
        raise ValueError("raw-upstream checkpoint epoch must stay within 1..800")
    for field in ("best_MAE", "best_MSE"):
        value = state[field]
        if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value):
            raise ValueError(f"raw-upstream checkpoint {field} must be finite")
    if not isinstance(state["state_dict"], Mapping) or not isinstance(
        state["optimizer"], Mapping
    ):
        raise ValueError("raw-upstream checkpoint model and optimizer must be mappings")
    return state


__all__ = [
    "OFFICIAL_END_EPOCH",
    "OFFICIAL_VAL_SPAN",
    "OfficialA800Lineage",
    "OfficialLaunchResult",
    "build_official_a800_command",
    "official_validation_epochs",
    "run_raw_official_training",
    "verify_official_resume_checkpoint",
]
