"""Run a resumable, explicitly test-selected DM-Count faithful reproduction."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

PINNED_COMMIT = "cc5f2132e0d1328909f31b6d665b8e0b15c30467"
UPSTREAM_URL = "https://github.com/cvlab-stonybrook/DM-Count.git"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Stage 3A faithful DM-Count training")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--upstream-dir", type=Path, default=Path("/content/DM-Count"))
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--max-epoch", type=int, default=1000)
    parser.add_argument("--checkpoint-every", type=int, default=10)
    parser.add_argument("--device", default="0")
    parser.add_argument("--num-workers", type=int, default=3)
    parser.add_argument("--no-resume", action="store_true")
    return parser


def make_trainer_args(args: argparse.Namespace) -> SimpleNamespace:
    """Return the official ShanghaiTech Part A hyperparameters."""

    return SimpleNamespace(
        data_dir=str(args.data_dir),
        dataset="sha",
        lr=1e-5,
        weight_decay=1e-4,
        resume="",
        max_epoch=args.max_epoch,
        val_epoch=5,
        val_start=50,
        batch_size=10,
        device=args.device,
        num_workers=args.num_workers,
        crop_size=256,
        wot=0.1,
        wtv=0.01,
        reg=10.0,
        num_of_iter_in_ot=100,
        norm_cood=0,
    )


def checkpoint_due(epoch: int, maximum: int, interval: int) -> bool:
    if interval <= 0:
        raise ValueError("checkpoint interval must be positive")
    return epoch % interval == 0 or epoch == maximum


def ensure_upstream(path: Path) -> str:
    if not (path / ".git").is_dir():
        path.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", UPSTREAM_URL, str(path)], check=True)
    subprocess.run(["git", "-C", str(path), "checkout", PINNED_COMMIT], check=True)
    actual = subprocess.check_output(
        ["git", "-C", str(path), "rev-parse", "HEAD"], text=True
    ).strip()
    if actual != PINNED_COMMIT:
        raise RuntimeError(f"unexpected upstream commit: {actual}")
    return actual


def seed_everything(seed: int) -> None:
    import random

    import numpy as np
    import torch

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Faithful lane preserves the upstream cuDNN behavior. The clean lane will
    # use deterministic algorithms separately.
    torch.backends.cudnn.benchmark = True


def assert_drive_writable(run_dir: Path) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    probe = run_dir / ".write-probe"
    probe.write_text("stage3a", encoding="utf-8")
    if probe.read_text(encoding="utf-8") != "stage3a":
        raise RuntimeError(f"write verification failed: {run_dir}")
    probe.unlink()


def atomic_copy_with_sha256(source: Path, destination: Path) -> tuple[str, int]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    digest = hashlib.sha256()
    size = 0
    with source.open("rb") as src, temporary.open("wb") as dst:
        while block := src.read(1024 * 1024):
            dst.write(block)
            digest.update(block)
            size += len(block)
    os.replace(temporary, destination)
    return digest.hexdigest(), size


def write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def copy_if_changed(source: Path, destination: Path) -> None:
    if destination.exists() and destination.stat().st_size == source.stat().st_size:
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    shutil.copy2(source, temporary)
    os.replace(temporary, destination)


def load_previous_state(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def sync_checkpoint(
    *,
    trainer: Any,
    run_dir: Path,
    state_path: Path,
    seed: int,
    epoch: int,
    upstream_commit: str,
    trainer_args: SimpleNamespace,
) -> dict[str, Any]:
    save_dir = Path(trainer.save_dir)
    local_checkpoints = list(save_dir.glob("*_ckpt.tar"))
    if len(local_checkpoints) != 1:
        raise RuntimeError(f"expected one resumable checkpoint, found {local_checkpoints}")

    checkpoint_hash, checkpoint_bytes = atomic_copy_with_sha256(
        local_checkpoints[0], run_dir / "resume" / "last_ckpt.tar"
    )
    for model_path in save_dir.glob("best_model_*.pth"):
        copy_if_changed(model_path, run_dir / "best" / model_path.name)
    for log_path in save_dir.glob("train-*.log"):
        copy_if_changed(log_path, run_dir / "logs" / log_path.name)

    state = {
        "schema_version": 1,
        "stage": "stage-3a",
        "lane": "faithful",
        "checkpoint_selection": "test-selected",
        "seed": seed,
        "epoch": epoch,
        "maximum_epoch": trainer_args.max_epoch,
        "upstream_commit": upstream_commit,
        "dataset": "ShanghaiTech Part A",
        "data_dir": trainer_args.data_dir,
        "checkpoint": "resume/last_ckpt.tar",
        "checkpoint_sha256": checkpoint_hash,
        "checkpoint_bytes": checkpoint_bytes,
        "best_mae": None if trainer.best_mae == float("inf") else float(trainer.best_mae),
        "best_rmse": None if trainer.best_mse == float("inf") else float(trainer.best_mse),
        "best_count": int(trainer.best_count),
        "official_hyperparameters": vars(trainer_args),
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    write_json_atomic(state_path, state)
    return state


def restore_checkpoint(trainer: Any, resume_path: Path, state: dict[str, Any]) -> None:
    import torch

    checkpoint = torch.load(resume_path, map_location=trainer.device, weights_only=False)
    trainer.model.load_state_dict(checkpoint["model_state_dict"])
    trainer.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    trainer.start_epoch = int(checkpoint["epoch"]) + 1
    if state.get("best_mae") is not None:
        trainer.best_mae = float(state["best_mae"])
    if state.get("best_rmse") is not None:
        trainer.best_mse = float(state["best_rmse"])
    trainer.best_count = int(state.get("best_count", 0))
    trainer.logger.info("resumed faithful run from epoch %s", checkpoint["epoch"])


def run(args: argparse.Namespace) -> dict[str, Any]:
    if not args.data_dir.is_dir():
        raise FileNotFoundError(args.data_dir)
    assert_drive_writable(args.run_dir)
    upstream_commit = ensure_upstream(args.upstream_dir)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device.strip()
    seed_everything(args.seed)

    sys.path.insert(0, str(args.upstream_dir))
    from train_helper import Trainer

    args.work_dir.mkdir(parents=True, exist_ok=True)
    trainer_args = make_trainer_args(args)
    state_path = args.run_dir / "state.json"
    previous_state = load_previous_state(state_path)
    resume_path = args.run_dir / "resume" / "last_ckpt.tar"

    original_cwd = Path.cwd()
    started = time.time()
    try:
        os.chdir(args.work_dir)
        trainer = Trainer(trainer_args)
        trainer.setup()
        if resume_path.exists() and not args.no_resume:
            restore_checkpoint(trainer, resume_path, previous_state)

        if trainer.start_epoch > args.max_epoch:
            trainer.logger.info("run already reached epoch %s", trainer.start_epoch - 1)
            return previous_state

        last_state: dict[str, Any] = previous_state
        for epoch in range(trainer.start_epoch, args.max_epoch + 1):
            trainer.logger.info("-----Epoch %s/%s-----", epoch, args.max_epoch)
            trainer.epoch = epoch
            trainer.train_eopch()
            if epoch % trainer_args.val_epoch == 0 and epoch >= trainer_args.val_start:
                trainer.val_epoch()
            if checkpoint_due(epoch, args.max_epoch, args.checkpoint_every):
                last_state = sync_checkpoint(
                    trainer=trainer,
                    run_dir=args.run_dir,
                    state_path=state_path,
                    seed=args.seed,
                    epoch=epoch,
                    upstream_commit=upstream_commit,
                    trainer_args=trainer_args,
                )
                trainer.logger.info("synced resumable checkpoint at epoch %s", epoch)
        last_state["elapsed_seconds_this_process"] = time.time() - started
        write_json_atomic(state_path, last_state)
        return last_state
    finally:
        os.chdir(original_cwd)


def main() -> None:
    args = build_parser().parse_args()
    state = run(args)
    print(json.dumps(state, indent=2))


if __name__ == "__main__":
    main()
