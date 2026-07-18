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
    def __init__(
        self,
        *,
        upstream_commit: str = UPSTREAM_COMMIT,
        upstream_dirty: str = "",
        repo_commit: str = "d" * 40,
        repo_dirty: str = "",
    ) -> None:
        self.upstream_commit = upstream_commit
        self.upstream_dirty = upstream_dirty
        self.repo_commit = repo_commit
        self.repo_dirty = repo_dirty

    def __call__(self, command: Sequence[str]) -> subprocess.CompletedProcess[str]:
        call = tuple(str(part) for part in command)
        is_repo = Path(call[2]).name == "DroneAI"
        if call[-2:] == ("rev-parse", "HEAD"):
            commit = self.repo_commit if is_repo else self.upstream_commit
            return subprocess.CompletedProcess(call, 0, commit + "\n", "")
        if call[-2:] == ("status", "--porcelain"):
            dirty = self.repo_dirty if is_repo else self.upstream_dirty
            return subprocess.CompletedProcess(call, 0, dirty, "")
        raise AssertionError(f"unexpected command: {call}")


def _artifact(path: Path, payload: bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return path


def test_capture_preflight_hashes_all_frozen_identities(tmp_path: Path) -> None:
    preflight = _preflight()
    result = preflight.capture_preflight(
        environment={"status": "PASS", "cuda": {"cuda_available": True}},
        repo_path=tmp_path / "DroneAI",
        upstream_path=tmp_path / "MPCount",
        expected_upstream_commit=UPSTREAM_COMMIT,
        checkpoint_path=_artifact(tmp_path / "checkpoint.pth", b"checkpoint"),
        dataset_config_path=_artifact(tmp_path / "dataset.json", b"dataset"),
        split_manifest_path=_artifact(tmp_path / "split.json", b"split"),
        sample_manifest_path=_artifact(tmp_path / "samples.json", b"samples"),
        runner=GitRunner(),
    )

    assert result["status"] == "PASS"
    assert result["git"]["commit"] == "d" * 40
    assert result["git"]["clean"] is True
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
        (GitRunner(upstream_dirty=" M models/models.py\n"), "upstream repository is dirty"),
        (GitRunner(upstream_commit="f" * 40), "upstream commit mismatch"),
        (GitRunner(repo_dirty=" M scripts/run_mpcount_preflight.py\n"), "DroneAI repository is dirty"),
    ],
)
def test_capture_preflight_rejects_dirty_or_wrong_upstream(
    tmp_path: Path, runner: GitRunner, message: str
) -> None:
    preflight = _preflight()
    with pytest.raises(ValueError, match=message):
        preflight.capture_preflight(
            environment={"status": "PASS", "cuda": {"cuda_available": True}},
            repo_path=tmp_path / "DroneAI",
            upstream_path=tmp_path / "MPCount",
            expected_upstream_commit=UPSTREAM_COMMIT,
            checkpoint_path=_artifact(tmp_path / "checkpoint.pth", b"checkpoint"),
            dataset_config_path=_artifact(tmp_path / "dataset.json", b"dataset"),
            split_manifest_path=_artifact(tmp_path / "split.json", b"split"),
            sample_manifest_path=_artifact(tmp_path / "samples.json", b"samples"),
            runner=runner,
        )
