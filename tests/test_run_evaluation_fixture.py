from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

import pytest


SCRIPT = Path("scripts/run_evaluation_fixture.py").resolve()
CONFIG = Path("configs/evaluation/fixture_density.json").resolve()


def _run_fixture(
    output: Path,
    *,
    config: Path = CONFIG,
    cwd: Path | None = None,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--config",
            str(config.resolve()),
            "--output-dir",
            str(output),
            "--device",
            "cpu",
        ],
        text=True,
        capture_output=True,
        check=False,
        cwd=cwd,
    )


def test_fixture_cli_writes_complete_small_review_bundle(tmp_path: Path) -> None:
    output = tmp_path / "run"

    result = _run_fixture(output)

    assert result.returncode == 0, result.stderr
    assert "PASS_RESEARCH_ONLY" in result.stdout
    score = json.loads((output / "score.json").read_text(encoding="utf-8"))
    assert score["status"] == "PASS_RESEARCH_ONLY"
    assert score["score"] >= 80
    with (output / "predictions.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 36
    assert {row["latency_ms"] for row in rows} == {"1.0"}
    assert {row["peak_vram_mb"] for row in rows} == {"0.0"}
    assert len(list((output / "figures").glob("*.png"))) == 12
    assert (output / "fixture-input").is_dir()
    assert not (output / "native-outputs").exists()
    assert sum(path.stat().st_size for path in output.rglob("*") if path.is_file()) <= 25 * 1024 * 1024


def test_fixture_is_deterministic_across_isolated_output_directories(
    tmp_path: Path,
) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    first_result = _run_fixture(first)
    second_result = _run_fixture(second, cwd=tmp_path)
    assert first_result.returncode == 0, first_result.stderr
    assert second_result.returncode == 0, second_result.stderr

    for name in (
        "sample-manifest.json",
        "predictions.csv",
        "selection_manifest.json",
        "metrics.json",
    ):
        assert (first / name).read_bytes() == (second / name).read_bytes()
    first_images = {
        path.name: path.read_bytes() for path in (first / "fixture-input").glob("*.png")
    }
    second_images = {
        path.name: path.read_bytes() for path in (second / "fixture-input").glob("*.png")
    }
    assert first_images == second_images
    first_panels = {
        path.name: path.read_bytes() for path in (first / "figures").glob("*.png")
    }
    second_panels = {
        path.name: path.read_bytes() for path in (second / "figures").glob("*.png")
    }
    assert first_panels == second_panels


def test_fixture_cli_refuses_to_overwrite_nonempty_output(tmp_path: Path) -> None:
    output = tmp_path / "existing"
    output.mkdir()
    sentinel = output / "keep.txt"
    sentinel.write_text("preserve me", encoding="utf-8")

    result = _run_fixture(output)

    assert result.returncode != 0
    assert "fixture output must be new or empty" in result.stderr
    assert sentinel.read_text(encoding="utf-8") == "preserve me"
    assert list(output.iterdir()) == [sentinel]


def test_fixture_cli_rejects_invalid_config_before_creating_output(
    tmp_path: Path,
) -> None:
    invalid_config = json.loads(CONFIG.read_text(encoding="utf-8"))
    invalid_config["expected_samples"] = True
    config_path = tmp_path / "invalid-config.json"
    config_path.write_text(
        json.dumps(invalid_config, sort_keys=True),
        encoding="utf-8",
    )
    output = tmp_path / "invalid-run"

    result = _run_fixture(output, config=config_path)

    assert result.returncode != 0
    assert "expected_samples must be the integer 36" in result.stderr
    assert not output.exists()


@pytest.mark.parametrize(
    ("config_updates", "target_updates", "expected_error"),
    (
        (
            {"schema_version": True},
            {},
            "schema_version must be the integer 1",
        ),
        (
            {"schema_version": 1.0},
            {},
            "schema_version must be the integer 1",
        ),
        ({"run_id": 7}, {}, "run_id must be a non-empty string"),
        ({"seed": -1}, {}, "seed must be a non-negative integer"),
        (
            {"split_verified": "false"},
            {},
            "split_verified must be a boolean",
        ),
        (
            {},
            {"mae_max": True},
            "mae_max must be a finite non-negative number",
        ),
    ),
)
def test_fixture_cli_strictly_validates_config_types_before_writes(
    tmp_path: Path,
    config_updates: dict[str, object],
    target_updates: dict[str, object],
    expected_error: str,
) -> None:
    invalid_config = json.loads(CONFIG.read_text(encoding="utf-8"))
    invalid_config.update(config_updates)
    invalid_config["targets"].update(target_updates)
    config_path = tmp_path / "invalid-types.json"
    config_path.write_text(
        json.dumps(invalid_config, sort_keys=True),
        encoding="utf-8",
    )
    output = tmp_path / "invalid-types-run"

    result = _run_fixture(output, config=config_path)

    assert result.returncode != 0
    assert expected_error in result.stderr
    assert not output.exists()


@pytest.mark.parametrize(
    ("config_updates", "target_updates", "expected_status"),
    (
        ({"split_verified": False}, {}, "BLOCKED"),
        (
            {},
            {
                "spatial_direction": "maximize",
                "spatial_target": 1.0,
                "latency_max_ms": 0.0,
            },
            "REVIEW",
        ),
    ),
)
def test_fixture_cli_returns_nonzero_for_unsuccessful_gate(
    tmp_path: Path,
    config_updates: dict[str, object],
    target_updates: dict[str, object],
    expected_status: str,
) -> None:
    blocked_config = json.loads(CONFIG.read_text(encoding="utf-8"))
    blocked_config.update(config_updates)
    blocked_config["targets"].update(target_updates)
    config_path = tmp_path / f"{expected_status.lower()}-config.json"
    config_path.write_text(
        json.dumps(blocked_config, sort_keys=True),
        encoding="utf-8",
    )
    output = tmp_path / expected_status.lower()

    result = _run_fixture(output, config=config_path)

    assert result.returncode != 0
    score = json.loads((output / "score.json").read_text(encoding="utf-8"))
    assert score["status"] == expected_status
    if expected_status == "BLOCKED":
        assert "data.split" in score["failed_blockers"]
