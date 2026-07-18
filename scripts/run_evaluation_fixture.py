from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
from pathlib import Path
from typing import Literal, Sequence, cast

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import numpy as np
from PIL import Image

from droneai.comparison_claim import validate_comparison_claim
from droneai.evaluation_contract import EvaluationSample, NativePrediction, ZoneBox
from droneai.evaluation_runner import EvaluationProtocol, run_evaluation
from droneai.integrity import sha256_file
from droneai.model_brief import ModelBrief


class FixtureAdapter:
    def __init__(self, brief: ModelBrief, device: str) -> None:
        self._brief = brief
        self._device = device
        self._torch = None
        self._first_runtime: dict[str, tuple[float, float]] = {}
        if device == "cuda":
            import torch

            if not torch.cuda.is_available():
                raise RuntimeError("CUDA fixture requested but CUDA is unavailable")
            self._torch = torch
            warmup = torch.ones((16, 16), device="cuda")
            warmup = warmup @ warmup
            torch.cuda.synchronize()

    def brief(self) -> ModelBrief:
        return self._brief

    def predict(
        self,
        sample: EvaluationSample,
        *,
        retain_native: bool,
    ) -> NativePrediction:
        density = np.asarray(sample.ground_truth_density, dtype=np.float32).copy()
        sample_index = int(sample.sample_id.rsplit("-", 1)[1])
        density.flat[np.flatnonzero(density)[0]] += (-0.003, 0.0, 0.003)[
            sample_index % 3
        ]
        latency_ms = 1.0
        peak_vram_mb = 0.0
        if self._torch is not None:
            torch = self._torch
            torch.cuda.reset_peak_memory_stats()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            tensor = torch.as_tensor(density, device="cuda")
            output = tensor.clone()
            end.record()
            torch.cuda.synchronize()
            measured = (
                max(float(start.elapsed_time(end)), 1e-6),
                float(torch.cuda.max_memory_allocated()) / (1024 * 1024),
            )
            density = output.cpu().numpy()
            if retain_native:
                latency_ms, peak_vram_mb = self._first_runtime[sample.sample_id]
            else:
                latency_ms, peak_vram_mb = measured
                self._first_runtime[sample.sample_id] = measured
        return NativePrediction(
            sample_id=sample.sample_id,
            output_type="density",
            predicted_count=float(density.sum()),
            latency_ms=latency_ms,
            peak_vram_mb=peak_vram_mb,
            density=density,
        )


def build_fixture_samples(
    output_dir: Path,
    *,
    dataset_id: str,
    split_id: str,
    seed: int,
) -> tuple[EvaluationSample, ...]:
    input_dir = output_dir / "fixture-input"
    input_dir.mkdir(parents=True, exist_ok=False)
    rng = np.random.default_rng(seed)
    samples: list[EvaluationSample] = []
    band_bases = {"low": 2.0, "medium": 20.0, "high": 100.0}
    for band_index, (band, base_count) in enumerate(band_bases.items()):
        for index in range(12):
            sample_id = f"{band}-{index:02d}"
            image_path = input_dir / f"{sample_id}.png"
            color = tuple(int(value) for value in rng.integers(32, 224, size=3))
            Image.new("RGB", (64, 48), color).save(image_path)
            count = base_count + float(index)
            density = np.zeros((12, 16), dtype=np.float32)
            density[(band_index * 3 + index) % 12, (index * 5) % 16] = count
            samples.append(
                EvaluationSample(
                    sample_id=sample_id,
                    dataset_id=dataset_id,
                    split_id=split_id,
                    image_path=image_path,
                    source_sha256=sha256_file(image_path),
                    width=64,
                    height=48,
                    ground_truth_count=count,
                    ground_truth_density=density,
                    condition_tags={"density_band": band, "source": "synthetic"},
                    zones=(
                        ZoneBox("left", 0, 0, 32, 48),
                        ZoneBox("right", 32, 0, 64, 48),
                    ),
                )
            )
    return tuple(samples)


def _string_tuple(targets: dict[str, object], key: str) -> tuple[str, ...]:
    values = targets.get(key)
    if not isinstance(values, list) or not values or any(
        not isinstance(value, str) or not value.strip() for value in values
    ):
        raise ValueError(f"fixture target {key} must be a non-empty string list")
    return tuple(values)


def _nonempty_string(values: dict[str, object], key: str) -> str:
    value = values.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"fixture {key} must be a non-empty string")
    return value


def _boolean(values: dict[str, object], key: str) -> bool:
    value = values.get(key)
    if type(value) is not bool:
        raise ValueError(f"fixture {key} must be a boolean")
    return value


def _nonnegative_number(values: dict[str, object], key: str) -> float:
    value = values.get(key)
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value < 0
    ):
        raise ValueError(
            f"fixture {key} must be a finite non-negative number"
        )
    return float(value)


def _validate_config(payload: dict[str, object]) -> None:
    for key in ("run_id", "protocol_id", "model_id", "dataset_id", "split_id"):
        _nonempty_string(payload, key)
    split_role = _nonempty_string(payload, "split_role")
    if split_role not in {"train", "validation", "test", "smoke"}:
        raise ValueError("fixture split_role is invalid")
    expected_samples = payload.get("expected_samples")
    if type(expected_samples) is not int or expected_samples != 36:
        raise ValueError("fixture expected_samples must be the integer 36")
    seed = payload.get("seed")
    if type(seed) is not int or seed < 0:
        raise ValueError("fixture seed must be a non-negative integer")
    for key in (
        "split_verified",
        "leakage_free",
        "sealed_test_access_approved",
        "require_clean_git",
    ):
        _boolean(payload, key)
    validate_comparison_claim(
        _nonempty_string(payload, "checkpoint_training_split_status"),
        _nonempty_string(payload, "comparison_scope"),
        _nonempty_string(payload, "checkpoint_split_evidence"),
    )
    _nonnegative_number(payload, "localization_radius")

    targets = payload.get("targets")
    if not isinstance(targets, dict):
        raise ValueError("fixture targets must be an object")
    for key in (
        "mae_max",
        "rmse_max",
        "bias_max",
        "band_bias_max",
        "condition_bias_max",
        "spatial_target",
        "zone_warning_count",
        "zone_critical_count",
        "latency_max_ms",
        "vram_max_mb",
    ):
        _nonnegative_number(targets, key)
    _nonempty_string(targets, "spatial_metric_name")
    spatial_direction = _nonempty_string(targets, "spatial_direction")
    if spatial_direction not in {"minimize", "maximize"}:
        raise ValueError("fixture spatial_direction is invalid")
    for key in (
        "required_density_bands",
        "density_band_rules",
        "required_condition_keys",
    ):
        _string_tuple(targets, key)
    if float(targets["zone_critical_count"]) < float(
        targets["zone_warning_count"]
    ):
        raise ValueError(
            "fixture zone_critical_count must be at least zone_warning_count"
        )


def build_protocol(
    config: dict[str, object],
    *,
    rights_decision_path: Path,
) -> EvaluationProtocol:
    targets = config.get("targets")
    if not isinstance(targets, dict):
        raise ValueError("fixture targets must be an object")
    return EvaluationProtocol(
        run_id=cast(str, config["run_id"]),
        protocol_id=cast(str, config["protocol_id"]),
        dataset_id=cast(str, config["dataset_id"]),
        split_id=cast(str, config["split_id"]),
        split_role=cast(
            Literal["train", "validation", "test", "smoke"],
            config["split_role"],
        ),
        expected_samples=cast(int, config["expected_samples"]),
        split_verified=cast(bool, config["split_verified"]),
        leakage_free=cast(bool, config["leakage_free"]),
        checkpoint_training_split_status=cast(
            str,
            config["checkpoint_training_split_status"],
        ),
        comparison_scope=cast(str, config["comparison_scope"]),
        checkpoint_split_evidence=cast(str, config["checkpoint_split_evidence"]),
        sealed_test_access_approved=cast(
            bool,
            config["sealed_test_access_approved"],
        ),
        require_clean_git=cast(bool, config["require_clean_git"]),
        rights_decision_path=str(rights_decision_path.resolve()),
        rights_decision_sha256=sha256_file(rights_decision_path),
        localization_radius=float(config["localization_radius"]),
        mae_max=float(targets["mae_max"]),
        rmse_max=float(targets["rmse_max"]),
        bias_max=float(targets["bias_max"]),
        band_bias_max=float(targets["band_bias_max"]),
        condition_bias_max=float(targets["condition_bias_max"]),
        latency_max_ms=float(targets["latency_max_ms"]),
        vram_max_mb=float(targets["vram_max_mb"]),
        spatial_metric_name=cast(str, targets["spatial_metric_name"]),
        spatial_direction=cast(
            Literal["minimize", "maximize"],
            targets["spatial_direction"],
        ),
        spatial_target=float(targets["spatial_target"]),
        required_density_bands=_string_tuple(targets, "required_density_bands"),
        density_band_rules=_string_tuple(targets, "density_band_rules"),
        required_condition_keys=_string_tuple(targets, "required_condition_keys"),
        zone_warning_count=float(targets["zone_warning_count"]),
        zone_critical_count=float(targets["zone_critical_count"]),
    )


def _load_config(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("fixture config must be a JSON object")
    schema_version = payload.get("schema_version")
    if type(schema_version) is not int or schema_version != 1:
        raise ValueError("fixture schema_version must be the integer 1")
    _validate_config(payload)
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run the deterministic DroneAI evaluation fixture.",
    )
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args(argv)

    config_path = args.config.resolve()
    output_dir = args.output_dir.resolve()
    config = _load_config(config_path)
    if output_dir.exists():
        if not output_dir.is_dir() or any(output_dir.iterdir()):
            raise FileExistsError(
                f"fixture output must be new or empty: {output_dir}"
            )
    else:
        output_dir.mkdir(parents=True)

    # The harness provenance probe is repository-scoped. Resolving user paths
    # first preserves caller-relative CLI behavior while making local and
    # /workspace container invocation independent of the caller's cwd.
    os.chdir(REPO_ROOT)

    samples = build_fixture_samples(
        output_dir,
        dataset_id=cast(str, config["dataset_id"]),
        split_id=cast(str, config["split_id"]),
        seed=cast(int, config["seed"]),
    )
    checkpoint_path = output_dir / "fixture-input" / "fixture-checkpoint.bin"
    checkpoint_path.write_bytes(b"fixture-no-learned-weights-v1\n")
    rights_decision_path = output_dir / "fixture-input" / "fixture-rights.json"
    rights_decision_path.write_text(
        json.dumps(
            {"scope": "research_only", "production_approved": False},
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    upstream_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        text=True,
    ).strip()
    brief = ModelBrief(
        model_id=cast(str, config["model_id"]),
        paper="Synthetic harness fixture; no research claim",
        role="common harness integration proof",
        family="density",
        backbone="identity density fixture",
        parameter_count=0,
        blocks=("fixture input", "deterministic density output"),
        feature_scales=("native 12x16 annotation grid",),
        input_contract="RGB image plus synthetic density annotation",
        preprocessing_policy="no resize, crop, normalization, or tiling",
        coordinate_transform=(
            "density grid is scaled to original 64x48 image coordinates"
        ),
        native_output="one-channel density map",
        count_derivation="sum of native density map",
        zone_derivation=(
            "integrate density inside proportionally scaled zone boxes"
        ),
        original_losses=("none; no training",),
        official_protocol=(
            "36 synthetic validation samples across three density bands"
        ),
        official_reported_metrics=("none; synthetic harness proof",),
        strengths=("deterministic near-count preservation",),
        failure_modes=("does not measure real model quality",),
        runtime_risks=("CUDA proof covers tensor execution only",),
        rights_status="PASS_RESEARCH_ONLY",
        code_rights_status="project synthetic fixture",
        dataset_rights_status="project synthetic fixture",
        checkpoint_rights_status="project synthetic fixture",
        deployment_rights_status="not evaluated",
        upstream_commit=upstream_commit,
        checkpoint_path=str(checkpoint_path.resolve()),
        checkpoint_sha256=sha256_file(checkpoint_path),
        reviewed_paths=(
            str((REPO_ROOT / "scripts" / "run_evaluation_fixture.py").resolve()),
            str((REPO_ROOT / "src" / "droneai" / "evaluation_runner.py").resolve()),
        ),
        review_status="approved",
    )
    report = run_evaluation(
        adapter=FixtureAdapter(brief, args.device),
        samples=samples,
        protocol=build_protocol(
            config,
            rights_decision_path=rights_decision_path,
        ),
        output_dir=output_dir,
    )
    print(report.to_markdown())
    return 0 if report.is_success else 1


if __name__ == "__main__":
    sys.exit(main())
