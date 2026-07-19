from __future__ import annotations

import importlib
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]


def test_dm_count_report_runner_exposes_validation_sample_only() -> None:
    result = subprocess.run(
        [sys.executable, "scripts/run_dm_count_one_sample.py", "--help"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    assert "--sample-id" in result.stdout
    assert "--validation-list" in result.stdout
    assert "--test-root" not in result.stdout
    assert "--train " not in result.stdout


def test_dm_count_unknown_sample_fails_before_adapter_construction() -> None:
    runner = importlib.import_module("scripts.run_dm_count_one_sample")
    samples = [SimpleNamespace(sample_id="img_0775"), SimpleNamespace(sample_id="img_0221")]
    with pytest.raises(ValueError, match="sample-id is not in the frozen split"):
        runner._select_sample(samples, "img_missing")

