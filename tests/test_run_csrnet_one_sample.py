from __future__ import annotations

import importlib
import subprocess
import sys
from pathlib import Path

import numpy as np

from droneai.evaluation_contract import NativePrediction


ROOT = Path(__file__).parents[1]


def _runner():
    return importlib.import_module("scripts.run_csrnet_one_sample")


def test_csrnet_one_sample_runner_exposes_validation_without_test_path() -> None:
    result = subprocess.run(
        [sys.executable, "scripts/run_csrnet_one_sample.py", "--help"],
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

    checks, score = _runner().compatibility_checks(prediction)

    assert all(checks.values())
    assert score == 100.0


def test_compatibility_score_preserves_forward_runtime_and_checkpoint_evidence_on_contract_failure() -> None:
    prediction = NativePrediction(
        sample_id="sample",
        output_type="density",
        predicted_count=None,
        latency_ms=3.5,
        peak_vram_mb=128.0,
        failure_state="ValueError: CSRNet density must be non-negative",
        metadata={
            "forward_completed": True,
            "checkpoint_missing_key_count": 0,
            "checkpoint_unexpected_key_count": 0,
        },
    )

    checks, score = _runner().compatibility_checks(prediction)

    assert checks == {
        "successful_forward": True,
        "native_density_present": False,
        "density_finite_nonnegative": False,
        "density_sum_matches_count": False,
        "checkpoint_loaded_strictly": True,
        "runtime_metrics_finite": True,
    }
    assert score == 50.0
