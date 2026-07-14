"""Evaluate and score a stopped Stage 3A faithful-training pilot.

The faithful DM-Count lane uses ShanghaiTech test_data during checkpoint
selection.  This evaluator therefore labels the selected model as
test-selected and also evaluates the last resumable checkpoint separately.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PAPER_MAE = 59.7
PAPER_RMSE = 95.7
EXPECTED_TEST_IMAGES = 182
BEST_MODEL_PATTERN = re.compile(r"best_model_(\d+)\.pth$")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate a Stage 3A DM-Count pilot")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--upstream-dir", type=Path, default=Path("/content/DM-Count"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--minimum-pilot-epoch", type=int, default=100)
    return parser


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def best_checkpoint_index(path: Path) -> int:
    match = BEST_MODEL_PATTERN.fullmatch(path.name)
    if not match:
        raise ValueError(f"not a DM-Count best checkpoint: {path}")
    return int(match.group(1))


def choose_test_selected_checkpoint(best_dir: Path) -> Path:
    checkpoints = list(best_dir.glob("best_model_*.pth"))
    if not checkpoints:
        raise FileNotFoundError(f"no test-selected checkpoint under {best_dir}")
    return max(checkpoints, key=best_checkpoint_index)


def load_state(run_dir: Path) -> dict[str, Any]:
    path = run_dir / "state.json"
    state = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "epoch",
        "seed",
        "upstream_commit",
        "checkpoint",
        "checkpoint_sha256",
        "checkpoint_bytes",
        "best_mae",
        "best_rmse",
    }
    missing = sorted(required.difference(state))
    if missing:
        raise ValueError(f"state.json is missing fields: {missing}")
    return state


def load_model_state(checkpoint: Path, device: Any) -> dict[str, Any]:
    import torch

    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    if isinstance(payload, dict) and "model_state_dict" in payload:
        payload = payload["model_state_dict"]
    if not isinstance(payload, dict):
        raise TypeError(f"unsupported checkpoint payload: {type(payload)!r}")
    return payload


def evaluate_checkpoint(
    *,
    checkpoint: Path,
    label: str,
    dataloader: Any,
    dataset_size: int,
    device: Any,
    model_factory: Any,
    output_dir: Path,
) -> dict[str, Any]:
    import numpy as np
    import torch

    model = model_factory().to(device)
    model.load_state_dict(load_model_state(checkpoint, device))
    model.eval()

    rows: list[dict[str, Any]] = []
    errors: list[float] = []
    started = time.time()
    with torch.no_grad():
        for index, (inputs, count, name) in enumerate(dataloader, 1):
            outputs, _ = model(inputs.to(device, non_blocking=True))
            ground_truth = float(count[0].item())
            prediction = float(outputs.sum().item())
            error = ground_truth - prediction
            sample = name[0] if isinstance(name, (list, tuple)) else str(name)
            rows.append(
                {
                    "index": index,
                    "name": str(sample),
                    "ground_truth": ground_truth,
                    "prediction": prediction,
                    "error_gt_minus_pred": error,
                    "absolute_error": abs(error),
                }
            )
            errors.append(error)
            if index == 1 or index % 50 == 0 or index == dataset_size:
                print(
                    f"[{label} {index}/{dataset_size}] {sample}: "
                    f"gt={ground_truth:.1f}, pred={prediction:.1f}, "
                    f"abs_err={abs(error):.1f}",
                    flush=True,
                )

    error_array = np.asarray(errors, dtype=np.float64)
    metrics = {
        "mae": float(np.mean(np.abs(error_array))),
        "rmse": float(np.sqrt(np.mean(np.square(error_array)))),
    }
    prediction_path = output_dir / f"{label}_predictions.csv"
    with prediction_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    return {
        "label": label,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "checkpoint_bytes": checkpoint.stat().st_size,
        "test_samples": len(rows),
        "metrics": metrics,
        "paper_relative_gap_pct": {
            "mae": (metrics["mae"] / PAPER_MAE - 1.0) * 100.0,
            "rmse": (metrics["rmse"] / PAPER_RMSE - 1.0) * 100.0,
        },
        "within_paper_5pct": {
            "mae": abs(metrics["mae"] / PAPER_MAE - 1.0) <= 0.05,
            "rmse": abs(metrics["rmse"] / PAPER_RMSE - 1.0) <= 0.05,
        },
        "elapsed_seconds": time.time() - started,
        "predictions": str(prediction_path),
    }


def score_pilot(
    *, state: dict[str, Any], selected: dict[str, Any], last: dict[str, Any], minimum_epoch: int
) -> dict[str, Any]:
    last_hash_ok = (
        last["checkpoint_sha256"] == state["checkpoint_sha256"]
        and last["checkpoint_bytes"] == state["checkpoint_bytes"]
    )
    evaluations_ok = all(
        item["test_samples"] == EXPECTED_TEST_IMAGES
        and all(math.isfinite(value) for value in item["metrics"].values())
        for item in (selected, last)
    )
    recorded_metric_match = (
        abs(selected["metrics"]["mae"] - float(state["best_mae"])) < 1e-4
        and abs(selected["metrics"]["rmse"] - float(state["best_rmse"])) < 1e-4
    )
    checks = [
        {"id": "pilot.epoch", "points": 25, "passed": int(state["epoch"]) >= minimum_epoch},
        {"id": "pilot.resume_integrity", "points": 25, "passed": last_hash_ok},
        {"id": "pilot.evaluation_coverage", "points": 20, "passed": evaluations_ok},
        {
            "id": "pilot.traceability",
            "points": 20,
            "passed": bool(state.get("upstream_commit")) and bool(state.get("official_hyperparameters")),
        },
        {"id": "pilot.recorded_metric_match", "points": 10, "passed": recorded_metric_match},
    ]
    score = sum(item["points"] for item in checks if item["passed"])
    return {
        "score": score,
        "maximum": 100,
        "review_threshold": 90,
        "review_passed": score >= 90,
        "checks": checks,
    }


def write_score_markdown(path: Path, result: dict[str, Any]) -> None:
    score = result["pilot_score"]
    selected = result["evaluations"]["test_selected"]
    last = result["evaluations"]["last_resumable"]
    lines = [
        "# Stage 3A faithful pilot score",
        "",
        f"- Status: **{result['status']}**",
        f"- Epoch reached: **{result['training']['epoch_reached']}**",
        f"- Pilot score: **{score['score']}/{score['maximum']}**",
        "- Full 1000-epoch reproduction: **deferred to a suitable GPU server**",
        "",
        "| Evaluation | MAE | RMSE | Test images | Selection |",
        "|---|---:|---:|---:|---|",
        f"| Test-selected checkpoint | {selected['metrics']['mae']:.4f} | {selected['metrics']['rmse']:.4f} | {selected['test_samples']} | ShanghaiTech test set |",
        f"| Last resumable checkpoint | {last['metrics']['mae']:.4f} | {last['metrics']['rmse']:.4f} | {last['test_samples']} | Stop epoch |",
        "",
        "This is a pilot result, not a Stage 3 pass and not a production-use claim.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, Any]:
    import torch
    from torch.utils.data import DataLoader

    if not torch.cuda.is_available() and args.device.startswith("cuda"):
        raise RuntimeError("CUDA is required for the Colab pilot evaluation")
    state = load_state(args.run_dir)
    last_checkpoint = args.run_dir / str(state["checkpoint"])
    selected_checkpoint = choose_test_selected_checkpoint(args.run_dir / "best")
    if not last_checkpoint.is_file():
        raise FileNotFoundError(last_checkpoint)

    sys.path.insert(0, str(args.upstream_dir))
    from datasets.crowd import Crowd_sh
    from models import vgg19

    test_dir = args.data_dir / "test_data"
    dataset = Crowd_sh(str(test_dir), 512, 8, method="val")
    if len(dataset) != EXPECTED_TEST_IMAGES:
        raise ValueError(f"expected {EXPECTED_TEST_IMAGES} test images, found {len(dataset)}")
    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    device = torch.device(args.device)
    output_dir = args.run_dir / "evaluation" / f"pilot-epoch-{state['epoch']}"
    output_dir.mkdir(parents=True, exist_ok=True)

    selected = evaluate_checkpoint(
        checkpoint=selected_checkpoint,
        label="test_selected",
        dataloader=dataloader,
        dataset_size=len(dataset),
        device=device,
        model_factory=vgg19,
        output_dir=output_dir,
    )
    last = evaluate_checkpoint(
        checkpoint=last_checkpoint,
        label="last_resumable",
        dataloader=dataloader,
        dataset_size=len(dataset),
        device=device,
        model_factory=vgg19,
        output_dir=output_dir,
    )

    result = {
        "schema_version": 1,
        "stage": "stage-3a",
        "status": "PILOT_COMPLETE_FULL_1000_EPOCH_DEFERRED",
        "lane": "faithful",
        "dataset": "ShanghaiTech Part A",
        "checkpoint_selection_warning": "The faithful lane selects its best checkpoint on test_data.",
        "training": {
            "seed": state["seed"],
            "epoch_reached": state["epoch"],
            "full_reproduction_epoch": 1000,
            "full_reproduction_deferred": True,
            "upstream_commit": state["upstream_commit"],
            "official_hyperparameters": state.get("official_hyperparameters"),
        },
        "evaluation_protocol": {
            "crop_size": 512,
            "downsample_ratio": 8,
            "batch_size": 1,
            "method": "val",
            "prediction": "sum of predicted density map",
        },
        "environment": {
            "torch": torch.__version__,
            "cuda": torch.cuda.is_available(),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        },
        "evaluations": {"test_selected": selected, "last_resumable": last},
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    result["pilot_score"] = score_pilot(
        state=state,
        selected=selected,
        last=last,
        minimum_epoch=args.minimum_pilot_epoch,
    )
    metrics_path = output_dir / "pilot_metrics.json"
    metrics_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    write_score_markdown(output_dir / "pilot_score.md", result)
    print("PILOT_RESULT=" + json.dumps(result, separators=(",", ":")), flush=True)
    print(json.dumps({"output_dir": str(output_dir), "score": result["pilot_score"]}, indent=2))
    return result


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
