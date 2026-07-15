from __future__ import annotations

import subprocess
import sys
from pathlib import Path


SCRIPT = Path("scripts/run_dm_count_compatibility_smoke.py").resolve()


def test_compatibility_cli_requires_only_code_checkpoint_and_device() -> None:
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--help"],
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "--upstream-dir" in result.stdout
    assert "--checkpoint" in result.stdout
    assert "--checkpoint-sha256" in result.stdout
    assert "--output-dir" in result.stdout
    assert "--dataset" not in result.stdout


def test_compatibility_cli_refuses_nonempty_output_before_torch(tmp_path: Path) -> None:
    output = tmp_path / "output"
    output.mkdir()
    marker = output / "keep.txt"
    marker.write_text("keep", encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--upstream-dir",
            str(tmp_path),
            "--checkpoint",
            str(marker),
            "--checkpoint-sha256",
            "0" * 64,
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
    assert marker.read_text(encoding="utf-8") == "keep"
