"""Validated runtime and storage profiles for experiment backends."""

from __future__ import annotations

import hashlib
import json
import posixpath
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any


@dataclass(frozen=True)
class StoragePaths:
    root: Path
    datasets_dir: Path
    checkpoints_dir: Path
    results_dir: Path
    host_bind_source: Path | None = None


@dataclass(frozen=True)
class RuntimeProfile:
    schema_version: int
    backend_id: str
    role: str
    repo_root: Path
    container_name: str | None
    container_image: str | None
    container_image_digest: str | None
    storage: StoragePaths
    license_scope: str
    large_artifacts_in_git: bool
    source_path: Path
    source_sha256: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "backend_id": self.backend_id,
            "role": self.role,
            "execution": {
                "repo_root": str(self.repo_root).replace("\\", "/"),
                "container_name": self.container_name,
                "container_image": self.container_image,
                "container_image_digest": self.container_image_digest,
            },
            "storage": {
                "root": str(self.storage.root).replace("\\", "/"),
                "datasets_dir": str(self.storage.datasets_dir).replace("\\", "/"),
                "checkpoints_dir": str(self.storage.checkpoints_dir).replace("\\", "/"),
                "results_dir": str(self.storage.results_dir).replace("\\", "/"),
                "host_bind_source": (
                    str(self.storage.host_bind_source).replace("\\", "/")
                    if self.storage.host_bind_source is not None
                    else None
                ),
            },
            "policy": {
                "license_scope": self.license_scope,
                "large_artifacts_in_git": self.large_artifacts_in_git,
            },
            "profile_source": {
                "path": str(self.source_path),
                "sha256": self.source_sha256,
            },
        }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _required_mapping(payload: dict[str, Any], key: str) -> dict[str, Any]:
    value = payload.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"{key} must be an object")
    return value


def _absolute_posix_path(value: Any, name: str) -> Path:
    text = str(value or "")
    if not PurePosixPath(text).is_absolute():
        raise ValueError(f"{name} must be absolute")
    return Path(posixpath.normpath(text))


def _assert_child(root: Path, child: Path, name: str) -> None:
    root_posix = PurePosixPath(posixpath.normpath(str(root).replace("\\", "/")))
    child_posix = PurePosixPath(posixpath.normpath(str(child).replace("\\", "/")))
    try:
        child_posix.relative_to(root_posix)
    except ValueError as exc:
        raise ValueError(f"{name} must be inside storage.root") from exc


def load_runtime_profile(path: str | Path) -> RuntimeProfile:
    source_path = Path(path)
    payload = json.loads(source_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("runtime profile must be a JSON object")
    if payload.get("schema_version") != 1:
        raise ValueError("runtime profile schema_version must be 1")

    backend_id = str(payload.get("backend_id") or "")
    role = str(payload.get("role") or "")
    if not backend_id:
        raise ValueError("backend_id is required")
    if not role:
        raise ValueError("role is required")

    execution = _required_mapping(payload, "execution")
    storage_payload = _required_mapping(payload, "storage")
    policy = _required_mapping(payload, "policy")

    repo_root = _absolute_posix_path(execution.get("repo_root"), "repo_root")
    storage_root = _absolute_posix_path(storage_payload.get("root"), "storage.root")
    datasets_dir = _absolute_posix_path(
        storage_payload.get("datasets_dir"), "datasets_dir"
    )
    checkpoints_dir = _absolute_posix_path(
        storage_payload.get("checkpoints_dir"), "checkpoints_dir"
    )
    results_dir = _absolute_posix_path(storage_payload.get("results_dir"), "results_dir")
    for name, child in (
        ("datasets_dir", datasets_dir),
        ("checkpoints_dir", checkpoints_dir),
        ("results_dir", results_dir),
    ):
        _assert_child(storage_root, child, name)

    host_bind_value = storage_payload.get("host_bind_source")
    host_bind_source = (
        _absolute_posix_path(host_bind_value, "host_bind_source")
        if host_bind_value
        else None
    )
    license_scope = str(policy.get("license_scope") or "")
    if license_scope not in {"research_only", "commercial_candidate"}:
        raise ValueError("license_scope must be research_only or commercial_candidate")
    large_artifacts_in_git = policy.get("large_artifacts_in_git")
    if not isinstance(large_artifacts_in_git, bool):
        raise ValueError("large_artifacts_in_git must be boolean")

    return RuntimeProfile(
        schema_version=1,
        backend_id=backend_id,
        role=role,
        repo_root=repo_root,
        container_name=execution.get("container_name"),
        container_image=execution.get("container_image"),
        container_image_digest=execution.get("container_image_digest"),
        storage=StoragePaths(
            root=storage_root,
            datasets_dir=datasets_dir,
            checkpoints_dir=checkpoints_dir,
            results_dir=results_dir,
            host_bind_source=host_bind_source,
        ),
        license_scope=license_scope,
        large_artifacts_in_git=large_artifacts_in_git,
        source_path=source_path.resolve(),
        source_sha256=_sha256_file(source_path),
    )
