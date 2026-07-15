"""Run one dataset-free CUDA forward pass through official DM-Count weights."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import numpy as np
from PIL import Image

from droneai.dm_count_adapter import DMCountAdapter
from droneai.evaluation_contract import EvaluationSample
from droneai.integrity import sha256_file
from droneai.runtime_probe import collect_environment

UPSTREAM_COMMIT = "cc5f2132e0d1328909f31b6d665b8e0b15c30467"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Dataset-free official DM-Count model/checkpoint compatibility smoke."
    )
    parser.add_argument("--upstream-dir", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    output = args.output_dir.resolve()
    if output.exists():
        if not output.is_dir() or any(output.iterdir()):
            raise FileExistsError(f"output must be new or empty: {output}")
    else:
        output.mkdir(parents=True)

    input_dir = output / "synthetic-input"
    input_dir.mkdir()
    image_path = input_dir / "synthetic-512.png"
    Image.new("RGB", (512, 512), color=(96, 128, 160)).save(image_path)
    ground_truth_density = np.zeros((64, 64), dtype=np.float32)
    ground_truth_density[32, 32] = 1.0
    sample = EvaluationSample(
        sample_id="synthetic-512",
        dataset_id="synthetic-compatibility-v1",
        split_id="smoke",
        image_path=image_path,
        source_sha256=sha256_file(image_path),
        width=512,
        height=512,
        ground_truth_count=1.0,
        ground_truth_density=ground_truth_density,
        condition_tags={
            "density_band": "synthetic",
            "source_partition": "synthetic",
        },
    )
    adapter = DMCountAdapter(
        upstream_dir=args.upstream_dir.resolve(),
        expected_upstream_commit=UPSTREAM_COMMIT,
        checkpoint_path=args.checkpoint.resolve(),
        checkpoint_sha256=args.checkpoint_sha256,
        device=args.device,
    )
    prediction = adapter.predict(sample, retain_native=True)
    if prediction.failure_state or prediction.density is None:
        raise RuntimeError(f"DM-Count compatibility inference failed: {prediction.failure_state}")
    density_path = output / "predicted-density.npy"
    np.save(density_path, prediction.density)
    result = {
        "schema_version": 1,
        "status": "PASS_COMPATIBILITY_SMOKE",
        "dataset_accessed": False,
        "upstream_commit": UPSTREAM_COMMIT,
        "checkpoint_path": str(args.checkpoint.resolve()),
        "checkpoint_sha256": args.checkpoint_sha256.lower(),
        "device": args.device,
        "predicted_count": prediction.predicted_count,
        "density_shape": list(prediction.density.shape),
        "density_sha256": sha256_file(density_path),
        "latency_ms": prediction.latency_ms,
        "peak_vram_mb": prediction.peak_vram_mb,
        "environment": collect_environment(REPO_ROOT),
    }
    (output / "compatibility-smoke.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
