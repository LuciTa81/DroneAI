"""Shared integrity helpers for persisted experiment evidence."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any

SHA256_PATTERN = re.compile(r"^[0-9a-fA-F]{64}$")


def is_sha256(value: Any) -> bool:
    return isinstance(value, str) and bool(SHA256_PATTERN.fullmatch(value))


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_artifact_reference(
    reference: Any, *, base_dir: str | Path | None = None
) -> tuple[bool, str]:
    """Validate an artifact path/hash record and optionally recalculate the hash."""

    if not isinstance(reference, dict):
        return False, "artifact reference must be an object with path and sha256"
    raw_path = reference.get("path")
    expected_hash = reference.get("sha256")
    if not isinstance(raw_path, str) or not raw_path.strip():
        return False, "artifact path missing"
    if not is_sha256(expected_hash):
        return False, f"invalid SHA-256 for {raw_path}"
    if base_dir is None:
        return True, f"schema verified: {raw_path}"

    path = Path(raw_path)
    if not path.is_absolute():
        path = Path(base_dir) / path
    if not path.is_file():
        return False, f"artifact missing: {path}"
    observed_hash = sha256_file(path)
    if observed_hash.lower() != str(expected_hash).lower():
        return False, f"artifact hash mismatch: {path}; observed={observed_hash}"
    return True, f"verified: {path}; sha256={observed_hash}"
