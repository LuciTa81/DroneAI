#!/usr/bin/env python3
"""Run only explicitly bounded stages of the isolated STEERER A lane."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import platform
import re
import subprocess
from pathlib import Path

from droneai.integrity import sha256_file
from droneai.steerer_official_data import (
    load_official_training_data_profile,
    verify_prepared_official_train_test_lane,
)
from droneai.steerer_official_training import (
    A0Lineage,
    TorchOfficialA0Runtime,
    run_a0_gate,
    synthesize_a0_config,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="stage", required=True)
    a0 = subparsers.add_parser("A0", help="one Train-batch optimizer/checkpoint gate")
    a0.add_argument("--config", type=Path, required=True)
    a0.add_argument("--run-id", required=True)
    a0.add_argument("--processed-root", type=Path, required=True)
    a0.add_argument("--g2-manifest", type=Path, required=True)
    a0.add_argument("--upstream-dir", type=Path, required=True)
    a0.add_argument("--backbone", type=Path, required=True)
    a0.add_argument("--container-image-digest", required=True)
    a0.add_argument("--device", default="cuda:0")
    return parser


def _git(repo: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ["git", *arguments],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _project_identity(repo: Path) -> tuple[str, list[str]]:
    commit = _git(repo, "rev-parse", "HEAD")
    if re.fullmatch(r"[0-9a-f]{40}", commit) is None:
        raise RuntimeError("project commit is not a full Git SHA")
    tracked = _git(repo, "status", "--porcelain", "--untracked-files=no")
    if tracked:
        raise RuntimeError("tracked project files must be clean before A0")
    untracked = _git(repo, "ls-files", "--others", "--exclude-standard").splitlines()
    return commit, untracked


def _config_sha256(config: dict[str, object]) -> str:
    stable = copy.deepcopy(config)
    encoded = json.dumps(
        stable, sort_keys=True, separators=(",", ":"), default=str
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _runtime_environment(runtime: TorchOfficialA0Runtime) -> dict[str, object]:
    torch = runtime.torch_module
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": str(torch.__version__),
        "torch_cuda": str(getattr(torch.version, "cuda", None)),
        "cudnn": int(torch.backends.cudnn.version()),
        "gpu": str(torch.cuda.get_device_name(0)),
        "gpu_count": int(torch.cuda.device_count()),
        "fp32_training": True,
        "amp_enabled": False,
    }


def _run_a0(args: argparse.Namespace) -> int:
    profile = load_official_training_data_profile(args.config)
    processed = args.processed_root.resolve(strict=True)
    expected_processed = Path(str(profile.processed_root)).resolve(strict=True)
    if processed != expected_processed:
        raise ValueError("processed root differs from the isolated A profile")
    manifest = args.g2_manifest.resolve(strict=True)
    if manifest.parent != (processed / "manifests").resolve(strict=True):
        raise ValueError("G2 manifest must be inside the prepared A dataset root")
    verified = verify_prepared_official_train_test_lane(
        processed,
        manifest_path=manifest,
        expected_train_samples=profile.train_samples,
        expected_test_samples=profile.test_samples,
    )
    if verified.get("validation_samples") != 0 or verified.get("artifact_hashes_verified") is not True:
        raise ValueError("G2 data verification is incomplete")

    config, upstream_audit = synthesize_a0_config(
        profile,
        upstream_dir=args.upstream_dir,
        processed_root=processed,
        backbone_path=args.backbone,
        run_id=args.run_id,
    )
    project_root = Path(__file__).resolve().parents[1]
    project_commit, untracked = _project_identity(project_root)
    runtime = TorchOfficialA0Runtime(
        profile,
        config=config,
        upstream_dir=args.upstream_dir,
        backbone_path=args.backbone,
        device=args.device,
    )
    lineage = A0Lineage(
        profile_sha256=sha256_file(args.config),
        split_sha256s={
            "train": sha256_file(processed / "train.txt"),
            "test": sha256_file(processed / "test.txt"),
        },
        dataset_manifest_sha256=sha256_file(manifest),
        dataset_content_sha256=str(verified["dataset_content_sha256"]),
        backbone_sha256=sha256_file(args.backbone),
        upstream_commit=upstream_audit.commit,
        upstream_config_sha256=upstream_audit.config_sha256,
        project_commit=project_commit,
        container_image_digest=args.container_image_digest,
    )
    environment = _runtime_environment(runtime)
    environment.update(
        {
            "effective_config_sha256": _config_sha256(config),
            "g2_manifest_sha256": lineage.dataset_manifest_sha256,
            "g2_dataset_content_sha256": lineage.dataset_content_sha256,
            "project_untracked_files": untracked,
        }
    )
    result = run_a0_gate(
        profile,
        run_id=args.run_id,
        runtime=runtime,
        lineage=lineage,
        output_dir=Path(str(profile.result_root)) / args.run_id,
        checkpoint_dir=Path(str(profile.checkpoint_root)) / args.run_id,
        environment=environment,
    )
    print(
        json.dumps(
            {
                "status": result.report.status,
                "score": result.report.score,
                "manifest": str(result.manifest_path),
                "checkpoint": str(result.checkpoint_path),
                "checkpoint_sha256": result.checkpoint_sha256,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.stage == "A0":
        return _run_a0(args)
    raise AssertionError("unreachable stage")


if __name__ == "__main__":
    raise SystemExit(main())
