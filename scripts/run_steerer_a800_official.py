#!/usr/bin/env python3
"""Launch the pinned raw upstream STEERER QNRF 800-epoch reproduction."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import platform
from pathlib import Path
import re
import subprocess
import sys

from droneai.integrity import sha256_file
from droneai.steerer_official_data import (
    load_official_training_data_profile,
    verify_prepared_official_train_test_lane,
)
from droneai.steerer_official_longrun import (
    OfficialA800Lineage,
    build_official_a800_command,
    official_validation_epochs,
    run_raw_official_training,
    verify_official_resume_checkpoint,
)
from droneai.steerer_training_upstream import (
    load_audited_official_config,
    verify_imagenet_backbone,
)


def _add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--processed-root", type=Path, required=True)
    parser.add_argument("--g2-manifest", type=Path, required=True)
    parser.add_argument("--upstream-dir", type=Path, required=True)
    parser.add_argument("--backbone", type=Path, required=True)
    parser.add_argument("--container-image-digest", required=True)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="action", required=True)
    start = subparsers.add_parser("start", help="fresh raw-upstream A800 run")
    _add_common_arguments(start)
    preflight = subparsers.add_parser(
        "preflight", help="verify the exact fresh raw-upstream launch without writing"
    )
    _add_common_arguments(preflight)
    resume = subparsers.add_parser("resume", help="resume a verified upstream run")
    _add_common_arguments(resume)
    resume.add_argument("--resume-run-dir", type=Path, required=True)
    resume.add_argument("--checkpoint-sha256", required=True)
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
        raise RuntimeError("tracked project files must be clean before A800")
    untracked = _git(repo, "ls-files", "--others", "--exclude-standard").splitlines()
    return commit, untracked


def _prepare(args: argparse.Namespace) -> dict[str, object]:
    profile = load_official_training_data_profile(args.config)
    processed = args.processed_root.resolve(strict=True)
    if processed != Path(str(profile.processed_root)).resolve(strict=True):
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
    if (
        verified.get("validation_samples") != 0
        or verified.get("train_samples") != 1201
        or verified.get("test_samples") != 334
        or verified.get("artifact_hashes_verified") is not True
    ):
        raise ValueError("G2 full Train/Test data verification is incomplete")
    backbone = verify_imagenet_backbone(profile, args.backbone)
    _config, upstream_audit = load_audited_official_config(profile, args.upstream_dir)
    if len(official_validation_epochs()) != 399:
        raise RuntimeError("pinned upstream validation cadence must contain 399 evaluations")

    project_root = Path(__file__).resolve().parents[1]
    project_commit, untracked = _project_identity(project_root)
    lineage = OfficialA800Lineage(
        project_commit=project_commit,
        upstream_commit=upstream_audit.commit,
        upstream_config_sha256=upstream_audit.config_sha256,
        profile_sha256=sha256_file(args.config),
        dataset_manifest_sha256=sha256_file(manifest),
        dataset_content_sha256=str(verified["dataset_content_sha256"]),
        train_split_sha256=sha256_file(processed / "train.txt"),
        test_split_sha256=sha256_file(processed / "test.txt"),
        backbone_sha256=sha256_file(backbone),
        container_image_digest=args.container_image_digest,
    )
    result_root = Path(str(profile.result_root)) / args.run_id
    checkpoint_root = Path(str(profile.checkpoint_root)) / args.run_id
    log_root = checkpoint_root / "official-output"
    resume_run_dir = None
    if args.action == "resume":
        resume_run_dir = args.resume_run_dir.resolve(strict=True)
        allowed_root = log_root.resolve(strict=True)
        if not resume_run_dir.is_relative_to(allowed_root):
            raise ValueError("resume run directory must stay inside the A800 checkpoint root")
        checkpoint = resume_run_dir / "checkpoint.pth.tar"
        import torch

        verify_official_resume_checkpoint(
            checkpoint,
            expected_sha256=args.checkpoint_sha256,
            torch_module=torch,
        )
        prior_manifests = sorted(result_root.glob("attempt-*-manifest.json"))
        if not prior_manifests:
            raise ValueError("A800 resume requires a prior attempt manifest")
        prior = json.loads(prior_manifests[0].read_text(encoding="utf-8"))
        if prior.get("lineage") != asdict(lineage):
            raise ValueError("A800 resume lineage differs from the fresh attempt")
    elif result_root.exists() or checkpoint_root.exists():
        raise FileExistsError("fresh A800 result and checkpoint directories must be new")

    command = build_official_a800_command(
        python_executable=sys.executable,
        upstream_dir=args.upstream_dir,
        processed_root=processed,
        backbone_path=backbone,
        log_root=log_root,
        resume_run_dir=resume_run_dir,
    )
    import torch

    environment = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": str(torch.__version__),
        "torch_cuda": str(getattr(torch.version, "cuda", None)),
        "cudnn": int(torch.backends.cudnn.version()),
        "gpu": str(torch.cuda.get_device_name(0)),
        "gpu_count": int(torch.cuda.device_count()),
        "project_untracked_files": untracked,
        "raw_upstream_entrypoint": str(
            (Path(args.upstream_dir) / "tools" / "train_cc.py").resolve(strict=True)
        ),
        "path_only_overrides": list(command[7:]),
    }
    return {
        "profile": profile,
        "lineage": lineage,
        "result_root": result_root,
        "checkpoint_root": checkpoint_root,
        "command": command,
        "environment": environment,
        "upstream_dir": Path(args.upstream_dir).resolve(strict=True),
    }


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    prepared = _prepare(args)
    summary = {
        "action": args.action,
        "run_id": args.run_id,
        "execution": "pinned raw upstream tools/train_cc.py",
        "official_validation_count": len(official_validation_epochs()),
        "lineage": asdict(prepared["lineage"]),
        "result_root": str(prepared["result_root"]),
        "checkpoint_root": str(prepared["checkpoint_root"]),
        "command": list(prepared["command"]),
    }
    if args.action == "preflight":
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0
    result = run_raw_official_training(
        run_id=args.run_id,
        command=prepared["command"],
        cwd=prepared["upstream_dir"],
        result_dir=prepared["result_root"],
        checkpoint_dir=prepared["checkpoint_root"],
        lineage=prepared["lineage"],
        environment=prepared["environment"],
        resume=args.action == "resume",
    )
    print(
        json.dumps(
            {
                **summary,
                "return_code": result.return_code,
                "attempt": result.attempt,
                "status_path": str(result.status_path),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
