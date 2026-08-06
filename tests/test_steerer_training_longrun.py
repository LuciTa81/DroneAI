from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from droneai import steerer_training_longrun as longrun
from droneai.steerer_training_longrun import (
    RunLock,
    validation_boundaries,
    write_longrun_status,
)


def _status(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "schema_version": 1,
        "run_id": "run-3035",
        "stage": "T800",
        "state": "validated",
        "epoch": 25,
        "global_step": 3000,
        "recent_loss": 0.5,
        "validation_boundary": True,
        "checkpoint_sha256": "a" * 64,
        "elapsed_seconds": 100.0,
        "eta_seconds": 3100.0,
        "updated_at": "2026-08-07T00:00:00+00:00",
    }
    payload.update(overrides)
    return payload


def test_validation_boundaries_resume_after_completed_epoch() -> None:
    assert validation_boundaries(5)[:3] == (25, 50, 75)
    assert validation_boundaries(50)[:2] == (75, 100)
    assert validation_boundaries(799) == (800,)
    assert validation_boundaries(800) == ()


@pytest.mark.parametrize("epoch", [-1, 801])
def test_validation_boundaries_reject_out_of_range_epoch(epoch: int) -> None:
    with pytest.raises(ValueError, match="0.*800"):
        validation_boundaries(epoch)


@pytest.mark.parametrize("epoch", [True, 5.0, "5"])
def test_validation_boundaries_reject_non_integer_epoch(epoch: object) -> None:
    with pytest.raises(TypeError, match="integer"):
        validation_boundaries(epoch)  # type: ignore[arg-type]


def test_status_write_is_strict_and_atomic(tmp_path: Path) -> None:
    path = tmp_path / "status.json"

    write_longrun_status(path, _status())

    assert json.loads(path.read_text(encoding="utf-8")) == _status()
    assert not list(tmp_path.glob(".status.json.*"))


@pytest.mark.parametrize(
    "mutation",
    [
        {"stage": "T50"},
        {"epoch": 801},
        {"recent_loss": float("nan")},
        {"checkpoint_sha256": "invalid"},
        {"unexpected": True},
    ],
)
def test_status_write_rejects_untrusted_payload(
    tmp_path: Path, mutation: dict[str, object]
) -> None:
    with pytest.raises((TypeError, ValueError)):
        write_longrun_status(tmp_path / "status.json", _status(**mutation))


def test_status_replace_failure_preserves_previous_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "status.json"
    write_longrun_status(path, _status(epoch=25))
    before = path.read_bytes()

    def fail_replace(_source: object, _target: object) -> None:
        raise OSError("injected replace failure")

    monkeypatch.setattr(longrun.os, "replace", fail_replace)

    with pytest.raises(OSError, match="injected"):
        write_longrun_status(path, _status(epoch=50))

    assert path.read_bytes() == before
    assert not list(tmp_path.glob(".status.json.*"))


@pytest.mark.skipif(os.name != "posix", reason="production lock uses Linux flock")
def test_run_lock_rejects_a_second_live_owner(tmp_path: Path) -> None:
    path = tmp_path / "run.lock"

    with RunLock(path, run_id="run-3035"):
        with pytest.raises(FileExistsError, match="already active"):
            RunLock(path, run_id="run-3035").acquire()

    assert json.loads(path.read_text(encoding="utf-8"))["run_id"] == "run-3035"


@pytest.mark.skipif(os.name != "posix", reason="production lock uses Linux flock")
def test_process_exit_releases_lock_even_when_metadata_remains(tmp_path: Path) -> None:
    path = tmp_path / "run.lock"
    source_root = Path(__file__).resolve().parents[1] / "src"
    script = (
        "from pathlib import Path; "
        "from droneai.steerer_training_longrun import RunLock; "
        "lock=RunLock(Path(r'" + str(path) + "'), run_id='run-3035'); "
        "lock.acquire(); print('locked', flush=True)"
    )
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(source_root)

    completed = subprocess.run(
        [sys.executable, "-c", script],
        check=True,
        capture_output=True,
        text=True,
        env=environment,
    )

    assert completed.stdout.strip() == "locked"
    assert path.exists()
    with RunLock(path, run_id="run-3035"):
        assert json.loads(path.read_text(encoding="utf-8"))["pid"] == os.getpid()
