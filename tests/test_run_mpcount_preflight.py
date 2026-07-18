from __future__ import annotations

import importlib
import subprocess
from pathlib import Path
from typing import Sequence

import pytest


UPSTREAM_COMMIT = "6eb06772bcf7dfb771c43a14d67146fce767f103"


def _preflight():
    return importlib.import_module("scripts.run_mpcount_preflight")


class GitRunner:
    def __init__(self, *, commit: str = UPSTREAM_COMMIT, dirty: str = "") -> None:
        self.commit = commit
        self.dirty = dirty

    def __call__(self, command: Sequence[str]) -> subprocess.CompletedProcess[str]:
        call = tuple(str(part) for part in command)
        if call[-2:] == ("rev-parse", "HEAD"):
            return subprocess.CompletedProcess(call, 0, self.commit + "\n", "")
        if call[-2:] == ("status", "--porcelain"):
            return subprocess.CompletedProcess(call, 0, self.dirty, "")
        raise AssertionError(f"unexpected command: {call}")


def _artifact(path: Path, payload: bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return path


def test_capture_preflight_hashes_all_frozen_identities(tmp_path: Path) -> None:
    preflight = _preflight()
    result = preflight.capture_preflight(
        environment={"status": "PASS", "cuda": {"cuda_available": True}},
        git_commit="d" * 40,
        upstream_path=tmp_path / "MPCount",
        expected_upstream_commit=UPSTREAM_COMMIT,
        checkpoint_path=_artifact(tmp_path / "checkpoint.pth", b"checkpoint"),
        dataset_config_path=_artifact(tmp_path / "dataset.json", b"dataset"),
        split_manifest_path=_artifact(tmp_path / "split.json", b"split"),
        sample_manifest_path=_artifact(tmp_path / "samples.json", b"samples"),
        runner=GitRunner(),
    )

    assert result["status"] == "PASS"
    assert result["upstream"]["commit"] == UPSTREAM_COMMIT
    assert result["checkpoint"]["sha256"] == preflight.sha256_file(
        tmp_path / "checkpoint.pth"
    )
    assert result["dataset_config"]["sha256"] == preflight.sha256_file(
        tmp_path / "dataset.json"
    )
    assert result["split_manifest"]["sha256"] == preflight.sha256_file(
        tmp_path / "split.json"
    )
    assert result["sample_manifest"]["sha256"] == preflight.sha256_file(
        tmp_path / "samples.json"
    )


@pytest.mark.parametrize(
    ("runner", "message"),
    [
        (GitRunner(dirty=" M models/models.py\n"), "dirty"),
        (GitRunner(commit="f" * 40), "commit mismatch"),
    ],
)
def test_capture_preflight_rejects_dirty_or_wrong_upstream(
    tmp_path: Path, runner: GitRunner, message: str
) -> None:
    preflight = _preflight()
    with pytest.raises(ValueError, match=message):
        preflight.capture_preflight(
            environment={"status": "PASS", "cuda": {"cuda_available": True}},
            git_commit="d" * 40,
            upstream_path=tmp_path / "MPCount",
            expected_upstream_commit=UPSTREAM_COMMIT,
            checkpoint_path=_artifact(tmp_path / "checkpoint.pth", b"checkpoint"),
            dataset_config_path=_artifact(tmp_path / "dataset.json", b"dataset"),
            split_manifest_path=_artifact(tmp_path / "split.json", b"split"),
            sample_manifest_path=_artifact(tmp_path / "samples.json", b"samples"),
            runner=runner,
        )
