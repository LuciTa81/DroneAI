from __future__ import annotations

import importlib
import subprocess
import sys
from pathlib import Path

import numpy as np

from droneai.evaluation_contract import NativePrediction


ROOT = Path(__file__).parents[1]


def _runner():
    return importlib.import_module("scripts.run_mpcount_one_sample")


def test_mpcount_one_sample_runner_exposes_no_test_path() -> None:
    result = subprocess.run(
        [sys.executable, "scripts/run_mpcount_one_sample.py", "--help"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    for option in (
        "--config",
        "--train-root",
        "--upstream-dir",
        "--split-upstream-dir",
        "--train-list",
        "--validation-list",
        "--checkpoint",
        "--checkpoint-sha256",
        "--rights-decision",
        "--rights-manifest",
        "--sample-id",
        "--output-dir",
        "--device",
    ):
        assert option in result.stdout
    assert "--test-root" not in result.stdout


def test_compatibility_score_requires_all_native_density_checks() -> None:
    runner = _runner()
    prediction = NativePrediction(
        sample_id="sample",
        output_type="density",
        predicted_count=4.0,
        latency_ms=2.0,
        peak_vram_mb=32.0,
        density=np.ones((2, 2), dtype=np.float32),
        metadata={
            "checkpoint_missing_key_count": 0,
            "checkpoint_unexpected_key_count": 0,
        },
    )

    checks, score = runner.compatibility_checks(prediction)

    assert all(checks.values())
    assert score == 100.0


def test_compatibility_score_is_zero_for_explicit_failure() -> None:
    runner = _runner()
    prediction = NativePrediction(
        sample_id="sample",
        output_type="density",
        predicted_count=None,
        latency_ms=2.0,
        peak_vram_mb=0.0,
        failure_state="RuntimeError: failed",
    )

    checks, score = runner.compatibility_checks(prediction)

    assert not any(checks.values())
    assert score == 0.0
