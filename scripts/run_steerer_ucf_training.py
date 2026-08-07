"""Run one explicitly approved, stage-bounded STEERER training segment."""

from __future__ import annotations

import argparse
import json
import re
import sys
from contextlib import nullcontext
from pathlib import Path
from typing import Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from droneai.steerer_training_profile import load_training_profile
from droneai.steerer_training_longrun import RunLock
from droneai.steerer_training_runner import (
    PinnedUpstreamTrainingEngine,
    run_training_stage,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the pinned STEERER UCF-QNRF training lane without Test access."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--stage", choices=("T0", "T1", "T5", "T50", "T800"), required=True
    )
    parser.add_argument("--approved-stage", choices=("T5", "T50", "T800"))
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--processed-root", type=Path, required=True)
    parser.add_argument("--upstream-dir", type=Path, required=True)
    parser.add_argument("--backbone", type=Path, required=True)
    parser.add_argument("--backbone-sha256", required=True)
    parser.add_argument("--container-image-digest", required=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--precision",
        choices=("auto", "fp32"),
        default="auto",
        help="Use fp32 only for an explicitly approved resumed T800 recovery.",
    )
    return parser


def _validate_stage_ceiling(args: argparse.Namespace) -> None:
    if args.precision == "fp32" and args.stage != "T800":
        raise ValueError("FP32 recovery is reserved for an approved resumed T800")
    if args.stage in {"T0", "T1"}:
        if args.resume is not None:
            raise ValueError("T0 and T1 cannot use --resume")
        if args.approved_stage is not None:
            raise ValueError("--approved-stage is reserved for T5, T50, and T800")
        return
    if args.resume is None:
        raise ValueError(f"{args.stage} requires --resume")
    if args.approved_stage != args.stage:
        raise PermissionError(
            f"{args.stage} requires an explicit matching --approved-stage"
        )


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        _validate_stage_ceiling(args)
        profile = load_training_profile(args.config)
        if args.processed_root != profile.processed_root:
            raise ValueError("--processed-root must match the authoritative profile")
        if args.backbone != profile.imagenet_backbone.path:
            raise ValueError("--backbone must match the authoritative profile")
        if args.backbone_sha256.lower() != profile.imagenet_backbone.sha256:
            raise ValueError(
                "--backbone-sha256 must confirm the authoritative profile hash"
            )
        if re.fullmatch(
            r"sha256:[0-9a-fA-F]{64}", args.container_image_digest
        ) is None:
            raise ValueError(
                "--container-image-digest must be sha256 plus 64 hexadecimal characters"
            )
        run_lock = (
            RunLock(
                profile.result_root / args.run_id / "run.lock",
                run_id=args.run_id,
            )
            if args.stage == "T800"
            else nullcontext()
        )
        with run_lock:
            engine = PinnedUpstreamTrainingEngine(
                profile=profile,
                stage=args.stage,
                run_id=args.run_id,
                processed_root=args.processed_root,
                upstream_dir=args.upstream_dir,
                backbone_path=args.backbone,
                device=args.device,
                resume=args.resume,
            )
            result = run_training_stage(
                profile,
                stage=args.stage,
                run_id=args.run_id,
                resume=args.resume,
                force_fp32=args.precision == "fp32",
                device=args.device,
                container_image_digest=args.container_image_digest,
                engine=engine,
            )
    except (
        FileExistsError,
        FileNotFoundError,
        FloatingPointError,
        PermissionError,
        RuntimeError,
        TypeError,
        ValueError,
    ) as error:
        print(
            json.dumps({"status": "gate_violation", "error": str(error)}),
            file=sys.stderr,
        )
        return 2
    print(
        json.dumps(
            {
                "status": "completed",
                "stage": result.stage,
                "completed_epoch": result.completed_epoch,
                "optimizer_steps": result.optimizer_steps,
                "validation_samples": result.validation_samples,
                "finite_loss": result.finite_loss,
                "checkpoint_round_trip": result.checkpoint_round_trip,
                "physical_batch": result.physical_batch,
                "accumulation_steps": result.accumulation_steps,
                "amp_enabled": result.amp_enabled,
                "cuda_oom_evidence": result.cuda_oom_evidence,
                "elapsed_seconds": result.elapsed_seconds,
                "metrics_path": str(result.metrics_path),
                "metrics_sha256": result.metrics_sha256,
                "environment_path": str(result.environment_path),
                "environment_sha256": result.environment_sha256,
                "checkpoint_path": str(result.checkpoint_path),
                "checkpoint_sha256": result.checkpoint_sha256,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
