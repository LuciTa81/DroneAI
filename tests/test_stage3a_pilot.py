import json
from pathlib import Path

import pytest

from scripts.evaluate_stage3a_pilot import (
    best_checkpoint_index,
    build_parser,
    choose_test_selected_checkpoint,
    load_state,
    score_pilot,
)


def test_pilot_parser_requires_explicit_upstream_dir() -> None:
    with pytest.raises(SystemExit):
        build_parser().parse_args(
            ["--data-dir", "/data/part_A", "--run-dir", "/results/seed-2026"]
        )


def test_choose_test_selected_checkpoint_uses_highest_index(tmp_path: Path) -> None:
    best_dir = tmp_path / "best"
    best_dir.mkdir()
    for index in (0, 6, 2):
        (best_dir / f"best_model_{index}.pth").write_bytes(b"checkpoint")
    assert choose_test_selected_checkpoint(best_dir).name == "best_model_6.pth"
    assert best_checkpoint_index(best_dir / "best_model_12.pth") == 12


def test_choose_test_selected_checkpoint_requires_checkpoint(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        choose_test_selected_checkpoint(tmp_path)


def test_load_state_requires_traceability_fields(tmp_path: Path) -> None:
    (tmp_path / "state.json").write_text(json.dumps({"epoch": 140}), encoding="utf-8")
    with pytest.raises(ValueError, match="missing fields"):
        load_state(tmp_path)


def test_pilot_score_is_separate_from_full_stage_gate() -> None:
    state = {
        "epoch": 140,
        "checkpoint_sha256": "abc",
        "checkpoint_bytes": 10,
        "upstream_commit": "pin",
        "official_hyperparameters": {"max_epoch": 1000},
        "best_mae": 71.0,
        "best_rmse": 113.0,
    }
    selected = {
        "checkpoint_sha256": "selected",
        "checkpoint_bytes": 5,
        "test_samples": 182,
        "metrics": {"mae": 71.0, "rmse": 113.0},
    }
    last = {
        "checkpoint_sha256": "abc",
        "checkpoint_bytes": 10,
        "test_samples": 182,
        "metrics": {"mae": 75.0, "rmse": 120.0},
    }
    report = score_pilot(state=state, selected=selected, last=last, minimum_epoch=100)
    assert report["score"] == 100
    assert report["review_passed"] is True
