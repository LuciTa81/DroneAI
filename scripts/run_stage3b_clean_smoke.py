"""Run a one-epoch, test-isolated DM-Count clean-protocol smoke on Colab."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import shutil
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from droneai.stage3b import (  # noqa: E402
    PINNED_COMMIT,
    build_clean_split,
    run_stage3b_gate,
    write_split_manifest,
)

UPSTREAM_URL = "https://github.com/cvlab-stonybrook/DM-Count.git"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Stage 3B clean DM-Count smoke")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--upstream-dir", type=Path, default=Path("/content/DM-Count"))
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--smoke-epochs", type=int, default=1)
    parser.add_argument("--device", default="0")
    parser.add_argument("--num-workers", type=int, default=0)
    return parser


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
    import numpy as np
    import torch

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def copy_artifact(source: Path, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    shutil.copy2(source, temporary)
    os.replace(temporary, destination)
    return destination


def trainer_args(args: argparse.Namespace) -> SimpleNamespace:
    return SimpleNamespace(
        data_dir=str(args.data_dir),
        dataset="sha",
        lr=1e-5,
        weight_decay=1e-4,
        resume="",
        max_epoch=args.smoke_epochs - 1,
        val_epoch=1,
        val_start=0,
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


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, Any]:
    import torch

    if args.smoke_epochs < 1:
        raise ValueError("smoke_epochs must be at least 1")
    if not args.data_dir.is_dir():
        raise FileNotFoundError(args.data_dir)
    args.run_dir.mkdir(parents=True, exist_ok=True)
    args.work_dir.mkdir(parents=True, exist_ok=True)
    upstream_commit = ensure_upstream(args.upstream_dir)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device.strip()
    seed_everything(args.seed)

    split = build_clean_split(args.data_dir, seed=args.seed)
    repeated = build_clean_split(args.data_dir, seed=args.seed)
    if split["split_hash"] != repeated["split_hash"]:
        raise RuntimeError("clean split is not reproducible")
    manifest_path = args.run_dir / "split_manifest.csv"
    split_path = args.run_dir / "split_manifest.json"
    write_split_manifest(split, manifest_path)
    write_json(split_path, split)

    train_records = [row for row in split["records"] if row["split"] == "train"]
    validation_records = [row for row in split["records"] if row["split"] == "validation"]
    train_paths = [str(args.data_dir / row["image"]) for row in train_records]
    validation_paths = [str(args.data_dir / row["image"]) for row in validation_records]
    if set(train_paths).intersection(validation_paths):
        raise RuntimeError("train/validation overlap")

    sys.path.insert(0, str(args.upstream_dir))
    import train_helper

    official_crowd_sh = train_helper.Crowd_sh
    actual_dataset_roots: list[str] = []

    def clean_crowd_sh(
        root_path: str, crop_size: int, downsample_ratio: int = 8, method: str = "train"
    ) -> Any:
        del root_path
        dataset = official_crowd_sh(
            str(args.data_dir / "train_data"), crop_size, downsample_ratio, method
        )
        dataset.im_list = train_paths if method == "train" else validation_paths
        actual_dataset_roots.append(str(Path(dataset.root_path).resolve()))
        return dataset

    train_helper.Crowd_sh = clean_crowd_sh
    config = trainer_args(args)
    config_snapshot = args.run_dir / "clean_config.json"
    write_json(
        config_snapshot,
        {
            "schema_version": 1,
            "stage": "stage-3b",
            "upstream_commit": upstream_commit,
            "split_hash": split["split_hash"],
            "split_seed": args.seed,
            "trainer_args": vars(config),
        },
    )

    original_cwd = Path.cwd()
    started = time.time()
    try:
        os.chdir(args.work_dir)
        trainer = train_helper.Trainer(config)
        trainer.setup()
        if len(trainer.datasets["train"]) != 240 or len(trainer.datasets["val"]) != 60:
            raise RuntimeError("trainer did not receive the frozen 240/60 split")
        for epoch in range(args.smoke_epochs):
            trainer.epoch = epoch
            trainer.logger.info("-----Clean smoke epoch %s/%s-----", epoch + 1, args.smoke_epochs)
            trainer.train_eopch()
            trainer.val_epoch()
    finally:
        os.chdir(original_cwd)
        train_helper.Crowd_sh = official_crowd_sh

    save_dir = args.work_dir / trainer.save_dir
    checkpoints = sorted(save_dir.glob("*_ckpt.tar"))
    best_models = sorted(save_dir.glob("best_model_*.pth"))
    if len(checkpoints) != 1 or not best_models:
        raise RuntimeError(f"unexpected smoke artifacts: checkpoints={checkpoints}, best={best_models}")
    checkpoint = copy_artifact(checkpoints[0], args.run_dir / "checkpoint" / checkpoints[0].name)
    best_model = copy_artifact(best_models[-1], args.run_dir / "best" / best_models[-1].name)
    for log_path in save_dir.glob("train-*.log"):
        copy_artifact(log_path, args.run_dir / "logs" / log_path.name)

    resolved_train_root = str((args.data_dir / "train_data").resolve())
    if not actual_dataset_roots or any(root != resolved_train_root for root in actual_dataset_roots):
        raise RuntimeError(f"unexpected dataset roots: {actual_dataset_roots}")

    evidence = {
        "schema_version": 1,
        "stage": "stage-3b",
        "status": "CLEAN_PROTOCOL_SMOKE_COMPLETE",
        "split": {
            "seed": split["seed"],
            "counts": split["counts"],
            "overlap": 0,
            "split_hash": split["split_hash"],
            "repeat_hash": repeated["split_hash"],
        },
        "test_isolation": {
            "validation_source": "training_partition_only",
            "dataset_roots": actual_dataset_roots,
            "test_paths_seen": 0,
            "test_evaluations": 0,
            "split_frozen_before_smoke": True,
        },
        "smoke": {
            "epochs_completed": args.smoke_epochs,
            "validation_samples": len(validation_paths),
            "validation_mae": float(trainer.best_mae),
            "validation_rmse": float(trainer.best_mse),
            "checkpoint_exists": checkpoint.is_file(),
            "best_model_exists": best_model.is_file(),
            "elapsed_seconds": time.time() - started,
        },
        "traceability": {
            "upstream_commit": upstream_commit,
            "torch": torch.__version__,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": sha256_file(checkpoint),
            "best_model": str(best_model),
            "best_model_sha256": sha256_file(best_model),
            "config_snapshot": str(config_snapshot),
        },
        "artifacts": {
            "manifest_csv": str(manifest_path),
            "split_json": str(split_path),
            "evidence_json": str(args.run_dir / "evidence.snapshot.json"),
            "score_json": str(args.run_dir / "score.json"),
            "score_md": str(args.run_dir / "score.md"),
        },
    }
    report = run_stage3b_gate(evidence=evidence, output_dir=args.run_dir)
    result = {"evidence": evidence, "score": report.to_dict()}
    write_json(args.run_dir / "result.json", result)
    print("STAGE3B_RESULT=" + json.dumps(result, separators=(",", ":")), flush=True)
    print(report.to_markdown())
    return result


def main() -> None:
    result = run(build_parser().parse_args())
    raise SystemExit(0 if result["score"]["status"] == "PASS" else 1)


if __name__ == "__main__":
    main()
