import json
from pathlib import Path

import pytest

from droneai.runtime_profile import load_runtime_profile


def _write_profile(path: Path, *, results_dir: str) -> Path:
    payload = {
        "schema_version": 1,
        "backend_id": "test_backend",
        "role": "test",
        "execution": {
            "repo_root": "/workspace",
            "container_name": "test-container",
            "container_image": "example/image:1",
            "container_image_digest": "example/image@sha256:" + "b" * 64,
        },
        "storage": {
            "root": "/workspace/data",
            "datasets_dir": "/workspace/data/datasets",
            "checkpoints_dir": "/workspace/data/checkpoints",
            "results_dir": results_dir,
        },
        "policy": {
            "license_scope": "research_only",
            "large_artifacts_in_git": False,
        },
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_home5090_profile_has_exact_container_paths() -> None:
    profile = load_runtime_profile("configs/runtime/home5090_docker.json")

    assert profile.backend_id == "home5090_docker"
    assert profile.role == "default"
    assert profile.repo_root == Path("/workspace")
    assert profile.storage.root == Path("/workspace/data")
    assert profile.storage.datasets_dir == Path("/workspace/data/datasets")
    assert profile.storage.checkpoints_dir == Path("/workspace/data/checkpoints")
    assert profile.storage.results_dir == Path("/workspace/data/results")
    assert profile.container_image == "nvcr.io/nvidia/pytorch:25.10-py3"
    assert profile.container_image_digest == (
        "nvcr.io/nvidia/pytorch@sha256:"
        "42263b2424fc237b34c4fc4a91c30d603c57eed36e37d31ff6d9a4f1f801edee"
    )
    assert profile.license_scope == "research_only"
    assert len(profile.source_sha256) == 64


def test_profile_rejects_storage_child_outside_root(tmp_path: Path) -> None:
    path = _write_profile(tmp_path / "bad.json", results_dir="/tmp/results")

    with pytest.raises(ValueError, match="results_dir must be inside storage.root"):
        load_runtime_profile(path)


def test_profile_rejects_relative_repo_root(tmp_path: Path) -> None:
    path = _write_profile(tmp_path / "relative.json", results_dir="/workspace/data/results")
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["execution"]["repo_root"] = "workspace"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="repo_root must be absolute"):
        load_runtime_profile(path)


def test_profile_rejects_parent_traversal_outside_storage_root(tmp_path: Path) -> None:
    path = _write_profile(
        tmp_path / "traversal.json",
        results_dir="/workspace/data/../outside-results",
    )

    with pytest.raises(ValueError, match="results_dir must be inside storage.root"):
        load_runtime_profile(path)
