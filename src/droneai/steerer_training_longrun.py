"""Crash-safe control primitives for the authoritative STEERER T800 run."""

from __future__ import annotations

import json
import errno
import math
import os
import re
import socket
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping

from droneai.integrity import is_sha256

try:
    import fcntl as _fcntl
except ImportError:  # pragma: no cover - exercised by the Windows test runner
    _fcntl = None


VALIDATION_INTERVAL = 25
MILESTONE_INTERVAL = 100
TARGET_EPOCH = 800
_STATUS_KEYS = frozenset(
    {
        "schema_version",
        "run_id",
        "stage",
        "state",
        "epoch",
        "global_step",
        "recent_loss",
        "validation_boundary",
        "checkpoint_sha256",
        "elapsed_seconds",
        "eta_seconds",
        "updated_at",
    }
)
_STATUS_STATES = frozenset(
    {"starting", "training", "validating", "validated", "completed", "failed"}
)


class RunLock:
    """Hold one Linux advisory lock for the lifetime of a T800 process."""

    def __init__(self, path: str | Path, *, run_id: str) -> None:
        if not isinstance(run_id, str) or re.fullmatch(
            r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", run_id
        ) is None:
            raise ValueError("long-run lock run_id is invalid")
        self.path = Path(path)
        self.run_id = run_id
        self._descriptor: int | None = None

    def acquire(self) -> "RunLock":
        if _fcntl is None:
            raise RuntimeError("T800 advisory locking requires Linux fcntl.flock")
        if self._descriptor is not None:
            raise RuntimeError("long-run lock is already acquired by this instance")
        parent = self.path.parent.resolve(strict=True)
        target = parent / self.path.name
        if target.exists() and (target.is_dir() or target.is_symlink()):
            raise ValueError("long-run lock path must be a regular non-symlink file")
        flags = os.O_RDWR | os.O_CREAT
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        descriptor = os.open(target, flags, 0o600)
        try:
            os.fchmod(descriptor, 0o600)
            try:
                _fcntl.flock(descriptor, _fcntl.LOCK_EX | _fcntl.LOCK_NB)
            except OSError as exc:
                if exc.errno in {errno.EACCES, errno.EAGAIN}:
                    raise FileExistsError(
                        f"T800 run is already active: {target}"
                    ) from exc
                raise
            metadata = (
                json.dumps(
                    {
                        "schema_version": 1,
                        "run_id": self.run_id,
                        "pid": os.getpid(),
                        "hostname": socket.gethostname(),
                        "created_at": datetime.now(timezone.utc).isoformat(),
                    },
                    indent=2,
                    sort_keys=True,
                    allow_nan=False,
                )
                + "\n"
            ).encode("utf-8")
            os.ftruncate(descriptor, 0)
            os.lseek(descriptor, 0, os.SEEK_SET)
            view = memoryview(metadata)
            while view:
                written = os.write(descriptor, view)
                if written <= 0:
                    raise OSError("long-run lock metadata write made no progress")
                view = view[written:]
            os.fsync(descriptor)
        except BaseException:
            try:
                _fcntl.flock(descriptor, _fcntl.LOCK_UN)
            finally:
                os.close(descriptor)
            raise
        self.path = target
        self._descriptor = descriptor
        return self

    def release(self) -> None:
        descriptor = self._descriptor
        if descriptor is None:
            return
        self._descriptor = None
        assert _fcntl is not None
        try:
            _fcntl.flock(descriptor, _fcntl.LOCK_UN)
        finally:
            os.close(descriptor)

    def __enter__(self) -> "RunLock":
        return self.acquire()

    def __exit__(self, _exc_type: object, _exc: object, _traceback: object) -> None:
        self.release()


def validation_boundaries(
    completed_epoch: int, target_epoch: int = TARGET_EPOCH
) -> tuple[int, ...]:
    """Return uncompleted 25-epoch validation ceilings through epoch 800."""

    if isinstance(completed_epoch, bool) or not isinstance(completed_epoch, int):
        raise TypeError("completed_epoch must be an integer")
    if isinstance(target_epoch, bool) or not isinstance(target_epoch, int):
        raise TypeError("target_epoch must be an integer")
    if target_epoch != TARGET_EPOCH:
        raise ValueError("T800 target_epoch must remain 800")
    if not 0 <= completed_epoch <= TARGET_EPOCH:
        raise ValueError("completed_epoch must stay within 0..800")
    return tuple(
        epoch
        for epoch in range(VALIDATION_INTERVAL, TARGET_EPOCH + 1, VALIDATION_INTERVAL)
        if epoch > completed_epoch
    )


def _finite_nonnegative(value: object, *, name: str, nullable: bool = False) -> None:
    if nullable and value is None:
        return
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(float(value))
        or float(value) < 0.0
    ):
        raise ValueError(f"{name} must be a finite non-negative number")


def _validated_status(payload: Mapping[str, object]) -> dict[str, object]:
    if not isinstance(payload, Mapping) or set(payload) != _STATUS_KEYS:
        raise ValueError("long-run status must contain the exact schema")
    status = dict(payload)
    if status["schema_version"] != 1:
        raise ValueError("long-run status schema version is unsupported")
    if not isinstance(status["run_id"], str) or re.fullmatch(
        r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", status["run_id"]
    ) is None:
        raise ValueError("long-run status run_id is invalid")
    if status["stage"] != "T800":
        raise ValueError("long-run status stage must be T800")
    if status["state"] not in _STATUS_STATES:
        raise ValueError("long-run status state is invalid")
    if (
        not isinstance(status["epoch"], int)
        or isinstance(status["epoch"], bool)
        or not 0 <= status["epoch"] <= TARGET_EPOCH
    ):
        raise ValueError("long-run status epoch must stay within 0..800")
    if (
        not isinstance(status["global_step"], int)
        or isinstance(status["global_step"], bool)
        or status["global_step"] < 0
    ):
        raise ValueError("long-run status global_step must be non-negative")
    _finite_nonnegative(status["recent_loss"], name="recent_loss", nullable=True)
    if not isinstance(status["validation_boundary"], bool):
        raise TypeError("long-run status validation_boundary must be boolean")
    checkpoint_sha256 = status["checkpoint_sha256"]
    if checkpoint_sha256 is not None and not is_sha256(checkpoint_sha256):
        raise ValueError("long-run status checkpoint_sha256 is invalid")
    _finite_nonnegative(status["elapsed_seconds"], name="elapsed_seconds")
    _finite_nonnegative(status["eta_seconds"], name="eta_seconds", nullable=True)
    if not isinstance(status["updated_at"], str):
        raise TypeError("long-run status updated_at must be a string")
    try:
        updated_at = datetime.fromisoformat(status["updated_at"])
    except ValueError as exc:
        raise ValueError("long-run status updated_at must be ISO-8601") from exc
    if updated_at.tzinfo is None:
        raise ValueError("long-run status updated_at must include a timezone")
    if status["state"] in {"validated", "completed"} and (
        status["validation_boundary"] is not True or checkpoint_sha256 is None
    ):
        raise ValueError("validated status requires a boundary checkpoint")
    return status


def write_longrun_status(path: str | Path, payload: Mapping[str, object]) -> None:
    """Atomically write one strict T800 operator-status snapshot."""

    status = _validated_status(payload)
    target = Path(path)
    parent = target.parent.resolve(strict=True)
    if target.exists() and (target.is_dir() or target.is_symlink()):
        raise ValueError("long-run status path must be a regular non-symlink file")
    encoded = (
        json.dumps(status, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)


__all__ = [
    "MILESTONE_INTERVAL",
    "TARGET_EPOCH",
    "VALIDATION_INTERVAL",
    "RunLock",
    "validation_boundaries",
    "write_longrun_status",
]
