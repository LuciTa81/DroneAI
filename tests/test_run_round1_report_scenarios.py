from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.run_round1_report_scenarios import (
    MODEL_ORDER,
    build_commands,
    execute,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/reporting/round1_dual_scenarios.home5090.json"


def test_command_matrix_is_exactly_six_models_times_two_scenarios() -> None:
    commands = build_commands(CONFIG, Path("/workspace/data/results/round1-report-scenarios-v1"))
    assert len(commands) == 12
    assert {(row.model_id, row.sample_id) for row in commands} == {
        (model, sample)
        for model in MODEL_ORDER
        for sample in ("img_0775", "img_0221")
    }
    assert all("--sample-id" in row.argv for row in commands)
    assert all("--test-root" not in row.argv for row in commands)
    assert all("--train" not in row.argv for row in commands)


def test_existing_nonempty_result_blocks_execution(tmp_path: Path) -> None:
    commands = build_commands(CONFIG, tmp_path / "results")
    commands[0].output_dir.mkdir(parents=True)
    (commands[0].output_dir / "result.json").write_text("{}", encoding="utf-8")
    with pytest.raises(FileExistsError, match="new or empty"):
        execute(commands, dry_run=False)


def test_dry_run_creates_no_directories_or_subprocesses(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "results"
    commands = build_commands(CONFIG, output)

    def forbidden(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("dry-run must not launch subprocesses")

    monkeypatch.setattr("scripts.run_round1_report_scenarios.subprocess.run", forbidden)
    rows = execute(commands, dry_run=True)
    assert len(rows) == 12
    assert not output.exists()


def test_home_profile_pins_checkpoint_hashes_and_no_downloads() -> None:
    payload = json.loads(CONFIG.read_text(encoding="utf-8"))
    assert tuple(payload["model_order"]) == MODEL_ORDER
    for model in payload["models"].values():
        assert len(model["arguments"]["checkpoint-sha256"]) == 64
        assert not any(key in model["arguments"] for key in ("url", "download", "test-root"))

