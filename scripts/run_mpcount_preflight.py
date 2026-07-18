"""Capture MPCount preflight identities without executing model inference."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Sequence


Runner = Callable[[Sequence[str]], subprocess.CompletedProcess[str]]


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _run(command: Sequence[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [str(part) for part in command],
        check=False,
        capture_output=True,
        text=True,
    )


def _git_output(
    runner: Runner, upstream: Path, *arguments: str
) -> str:
    result = runner(("git", "-C", str(upstream), *arguments))
    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "no output").strip()
        raise RuntimeError(f"git {' '.join(arguments)} failed: {detail}")
    return result.stdout.strip()


def _identity(path: str | Path) -> dict[str, object]:
    resolved = Path(path)
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    return {
        "path": str(resolved),
        "sha256": sha256_file(resolved),
        "size_bytes": resolved.stat().st_size,
    }


def capture_preflight(
    *,
    environment: dict[str, object],
    repo_path: str | Path,
    upstream_path: str | Path,
    expected_upstream_commit: str,
    checkpoint_path: str | Path,
    dataset_config_path: str | Path,
    split_manifest_path: str | Path,
    sample_manifest_path: str | Path,
    runner: Runner | None = None,
) -> dict[str, object]:
    if environment.get("status") != "PASS":
        raise ValueError("environment verification must have status PASS")
    cuda = environment.get("cuda")
    if not isinstance(cuda, dict) or cuda.get("cuda_available") is not True:
        raise ValueError("environment verification must prove CUDA availability")
    command_runner = runner or _run
    repo = Path(repo_path)
    repo_dirty = _git_output(command_runner, repo, "status", "--porcelain")
    if repo_dirty:
        raise ValueError(f"DroneAI repository is dirty: {repo_dirty}")
    git_commit = _git_output(command_runner, repo, "rev-parse", "HEAD")
    if len(git_commit) != 40 or any(
        character not in "0123456789abcdef" for character in git_commit
    ):
        raise ValueError("DroneAI HEAD must be a lowercase 40-character SHA-1")

    upstream = Path(upstream_path)
    dirty = _git_output(command_runner, upstream, "status", "--porcelain")
    if dirty:
        raise ValueError(f"upstream repository is dirty: {dirty}")
    actual_commit = _git_output(command_runner, upstream, "rev-parse", "HEAD")
    if actual_commit != expected_upstream_commit:
        raise ValueError(
            "upstream commit mismatch: "
            f"expected {expected_upstream_commit}, found {actual_commit}"
        )

    return {
        "schema_version": 1,
        "status": "PASS",
        "gate": "preflight",
        "model_id": "mpcount",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "inference_executed": False,
        "git": {"path": str(repo), "commit": git_commit, "clean": True},
        "upstream": {"path": str(upstream), "commit": actual_commit, "clean": True},
        "checkpoint": _identity(checkpoint_path),
        "dataset_config": _identity(dataset_config_path),
        "split_manifest": _identity(split_manifest_path),
        "sample_manifest": _identity(sample_manifest_path),
        "environment": environment,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Capture MPCount preflight identities without inference."
    )
    parser.add_argument("--environment-json", required=True, type=Path)
    parser.add_argument("--repo", required=True, type=Path)
    parser.add_argument("--upstream", required=True, type=Path)
    parser.add_argument("--expected-upstream-commit", required=True)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--dataset-config", required=True, type=Path)
    parser.add_argument("--split-manifest", required=True, type=Path)
    parser.add_argument("--sample-manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        environment = json.loads(args.environment_json.read_text(encoding="utf-8"))
        payload = capture_preflight(
            environment=environment,
            repo_path=args.repo,
            upstream_path=args.upstream,
            expected_upstream_commit=args.expected_upstream_commit,
            checkpoint_path=args.checkpoint,
            dataset_config_path=args.dataset_config,
            split_manifest_path=args.split_manifest,
            sample_manifest_path=args.sample_manifest,
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
        return 0
    except Exception as exc:
        print(
            json.dumps(
                {
                    "schema_version": 1,
                    "status": "FAIL",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                },
                ensure_ascii=False,
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    sys.exit(main())
