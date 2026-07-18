from __future__ import annotations

import importlib
import subprocess
import sys
from pathlib import Path

from droneai.evaluation_contract import NativePrediction


ROOT = Path(__file__).parents[1]


def _runner():
    return importlib.import_module("scripts.run_apgcc_one_sample")


def test_apgcc_one_sample_runner_exposes_no_test_path() -> None:
    result = subprocess.run(
        [sys.executable, "scripts/run_apgcc_one_sample.py", "--help"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    assert "--sample-id" in result.stdout
    assert "--rights-decision" in result.stdout
    assert "--rights-manifest" in result.stdout
    assert "--test-root" not in result.stdout


def test_compatibility_score_requires_valid_native_points() -> None:
    prediction = NativePrediction(
        sample_id="sample",
        output_type="points",
        predicted_count=2.0,
        latency_ms=12.0,
        peak_vram_mb=2048.0,
        points=((10.0, 20.0), (30.0, 40.0)),
        point_confidences=(0.9, 0.8),
        metadata={
            "original_width": 100,
            "original_height": 80,
            "missing_checkpoint_key_count": 0,
            "unexpected_checkpoint_key_count": 0,
        },
    )

    checks, score = _runner().compatibility_checks(prediction)

    assert all(checks.values())
    assert score == 100.0


def test_compatibility_score_is_zero_for_explicit_failure() -> None:
    prediction = NativePrediction(
        sample_id="sample",
        output_type="points",
        predicted_count=None,
        latency_ms=2.0,
        peak_vram_mb=0.0,
        failure_state="RuntimeError: failed",
    )

    checks, score = _runner().compatibility_checks(prediction)

    assert not any(checks.values())
    assert score == 0.0
