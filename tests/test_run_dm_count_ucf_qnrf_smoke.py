from __future__ import annotations

import subprocess
import sys
from pathlib import Path


SCRIPT = Path("scripts/run_dm_count_ucf_qnrf_smoke.py").resolve()


def test_cli_exposes_explicit_non_test_inputs_only() -> None:
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--help"],
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "--train-root" in result.stdout
    assert "--train-list" in result.stdout
    assert "--validation-list" in result.stdout
    assert "--test-root" not in result.stdout
    assert "--checkpoint-sha256" in result.stdout
    assert "--rights-decision" in result.stdout
    assert "--rights-manifest" in result.stdout


def test_cli_refuses_nonempty_output_before_model_loading(tmp_path: Path) -> None:
    output = tmp_path / "output"
    output.mkdir()
    (output / "keep.txt").write_text("do not overwrite", encoding="utf-8")
    dummy = tmp_path / "dummy"
    dummy.write_text("fixture", encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--config",
            str(dummy),
            "--train-root",
            str(tmp_path),
            "--upstream-dir",
            str(tmp_path),
            "--train-list",
            str(dummy),
            "--validation-list",
            str(dummy),
            "--checkpoint",
            str(dummy),
            "--checkpoint-sha256",
            "0" * 64,
            "--rights-decision",
            str(dummy),
            "--rights-manifest",
            str(dummy),
            "--output-dir",
            str(output),
            "--device",
            "cpu",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode != 0
    assert "output must be new or empty" in result.stderr
    assert (output / "keep.txt").read_text(encoding="utf-8") == "do not overwrite"
