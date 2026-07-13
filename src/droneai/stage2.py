"""Stage 2: CSRNet pipeline correctness and tiny-set learning gate."""

from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path
from typing import Any

import numpy as np

from droneai.density import block_sum_density, points_to_density
from droneai.scoring import CheckResult, StageReport, score_stage

DEFAULT_SEED = 2026
EXPECTED_ARCHITECTURE = {
    "convolution_layers": 17,
    "dilated_convolution_layers": 6,
    "output_stride": 8,
}


def set_deterministic(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    import torch

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)


def _state_hash(model: Any) -> str:
    digest = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        digest.update(name.encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _synthetic_batch(device: Any) -> tuple[Any, Any, list[int]]:
    import torch

    point_sets = [
        [(12, 12)],
        [(12, 12), (48, 48)],
        [(8, 48), (32, 16), (52, 44)],
        [(8, 8), (8, 52), (52, 8), (52, 52)],
    ]
    images: list[np.ndarray] = []
    densities: list[np.ndarray] = []
    for points in point_sets:
        density = points_to_density(points, width=64, height=64, output_stride=8, sigma=0.7)
        signal = np.repeat(np.repeat(density, 8, axis=0), 8, axis=1)
        image = np.stack([signal, signal * 0.5, signal * 0.25]).astype(np.float32)
        images.append(image)
        densities.append(density[None, ...])
    return (
        torch.from_numpy(np.stack(images)).to(device),
        torch.from_numpy(np.stack(densities)).to(device),
        [len(points) for points in point_sets],
    )


def run_tiny_overfit(*, device: Any, steps: int, learning_rate: float) -> dict[str, Any]:
    import torch

    from droneai.csrnet import CSRNet, architecture_summary

    set_deterministic(DEFAULT_SEED)
    model = CSRNet(pretrained_frontend=False).to(device)
    images, targets, counts = _synthetic_batch(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = torch.nn.MSELoss(reduction="sum")

    model.train()
    with torch.no_grad():
        initial_output = model(images)
        initial_loss = float(criterion(initial_output, targets).item())
    losses = []
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        output = model(images)
        loss = criterion(output, targets)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.item()))

    model.eval()
    with torch.no_grad():
        predictions = model(images)
        final_loss = float(criterion(predictions, targets).item())
        predicted_counts = predictions.sum(dim=(1, 2, 3)).cpu().tolist()
    count_mae = float(np.mean(np.abs(np.asarray(predicted_counts) - np.asarray(counts))))
    reduction = 1.0 - final_loss / initial_loss if initial_loss else 0.0

    set_deterministic(DEFAULT_SEED)
    reference_hash = _state_hash(CSRNet(pretrained_frontend=False))
    set_deterministic(DEFAULT_SEED)
    repeated_hash = _state_hash(CSRNet(pretrained_frontend=False))

    return {
        "architecture": architecture_summary(model),
        "output_shape": list(predictions.shape),
        "finite_output": bool(torch.isfinite(predictions).all().item()),
        "initial_loss": initial_loss,
        "final_loss": final_loss,
        "loss_reduction": reduction,
        "target_counts": counts,
        "predicted_counts": predicted_counts,
        "count_mae": count_mae,
        "steps": steps,
        "learning_rate": learning_rate,
        "initial_state_hash": reference_hash,
        "repeated_state_hash": repeated_hash,
        "trained_state_hash": _state_hash(model),
        "loss_tail": losses[-10:],
    }


def build_stage2_checks(
    *, metrics: dict[str, Any], density_sum_error: float, block_sum_error: float,
    config_written: bool, metrics_written: bool
) -> list[CheckResult]:
    architecture = metrics.get("architecture") or {}
    expected_shape = [4, 1, 8, 8]
    architecture_ok = all(
        architecture.get(key) == value for key, value in EXPECTED_ARCHITECTURE.items()
    )
    deterministic = metrics.get("initial_state_hash") == metrics.get("repeated_state_hash")
    return [
        CheckResult("pipeline.architecture", "pipeline correctness", "Published CSRNet layer contract is preserved", 10, architecture_ok, True, expected=str(EXPECTED_ARCHITECTURE), observed=str(architecture)),
        CheckResult("pipeline.shape", "pipeline correctness", "64x64 input produces an 8x8 density map", 10, metrics.get("output_shape") == expected_shape, True, expected=str(expected_shape), observed=str(metrics.get("output_shape"))),
        CheckResult("pipeline.finite", "pipeline correctness", "Model outputs and losses are finite", 10, bool(metrics.get("finite_output") and np.isfinite(metrics.get("final_loss", np.nan))), True, expected="finite", observed=f"output={metrics.get('finite_output')}; loss={metrics.get('final_loss')}"),
        CheckResult("count.point_density", "count preservation", "Point-to-density conversion preserves total count", 15, density_sum_error <= 1e-5, True, expected="absolute error <= 1e-5", observed=f"{density_sum_error:.8f}"),
        CheckResult("count.downsample", "count preservation", "Density downsampling preserves total mass", 10, block_sum_error <= 1e-5, True, expected="absolute error <= 1e-5", observed=f"{block_sum_error:.8f}"),
        CheckResult("overfit.loss", "tiny-set overfit", "Tiny-set loss falls by at least 90%", 15, metrics.get("loss_reduction", 0) >= 0.90, expected=">=0.90", observed=f"{metrics.get('loss_reduction', 0):.4f}"),
        CheckResult("overfit.count", "tiny-set overfit", "Tiny-set count MAE is at most 0.25", 5, metrics.get("count_mae", float("inf")) <= 0.25, expected="<=0.25", observed=f"{metrics.get('count_mae', float('inf')):.4f}"),
        CheckResult("determinism.initialization", "determinism", "Fixed seed reproduces initial model state", 10, deterministic, True, expected="matching SHA-256", observed=str(deterministic)),
        CheckResult("artifacts.config", "artifacts", "Stage configuration is snapshotted", 5, config_written, True, expected="stage2_config.json", observed=str(config_written)),
        CheckResult("artifacts.metrics", "artifacts", "Training metrics and predictions are persisted", 5, metrics_written, True, expected="metrics.json", observed=str(metrics_written)),
        CheckResult("artifacts.model_hash", "artifacts", "Trained model state is content-addressed", 5, len(str(metrics.get("trained_state_hash") or "")) == 64, expected="64-character SHA-256", observed=str(metrics.get("trained_state_hash"))),
    ]


def run_stage2(
    *, output_dir: str | Path, steps: int = 200, learning_rate: float = 1e-4,
    device_name: str | None = None
) -> StageReport:
    import torch

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(device_name or ("cuda" if torch.cuda.is_available() else "cpu"))
    config = {
        "stage_id": "stage-2", "threshold": 80, "seed": DEFAULT_SEED,
        "steps": steps, "learning_rate": learning_rate, "device": str(device),
        "model": "CSRNet paper-compatible architecture", "pretrained_frontend": False,
    }
    config_path = output_dir / "stage2_config.json"
    config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")

    points = [(0, 0), (31, 31), (63, 63), (8, 55)]
    density = points_to_density(points, width=64, height=64, output_stride=8, sigma=1.0)
    density_sum_error = abs(float(density.sum()) - len(points))
    full_density = points_to_density(points, width=64, height=64, output_stride=1, sigma=3.0)
    reduced = block_sum_density(full_density, 8)
    block_sum_error = abs(float(reduced.sum()) - float(full_density.sum()))

    metrics = run_tiny_overfit(device=device, steps=steps, learning_rate=learning_rate)
    metrics.update({
        "device": str(device), "density_sum_error": density_sum_error,
        "block_sum_error": block_sum_error,
    })
    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    checks = build_stage2_checks(
        metrics=metrics, density_sum_error=density_sum_error,
        block_sum_error=block_sum_error, config_written=config_path.is_file(),
        metrics_written=metrics_path.is_file(),
    )
    report = score_stage(
        stage_id="stage-2", stage_name="CSRNet pipeline smoke", threshold=80, checks=checks
    )
    (output_dir / "score.json").write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")
    (output_dir / "score.md").write_text(report.to_markdown(), encoding="utf-8")
    return report
