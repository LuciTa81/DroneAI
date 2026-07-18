from __future__ import annotations

import json
from pathlib import Path

from scripts.build_round1_cctv_comparison import main


def test_cli_returns_structured_error_for_missing_queue(
    capsys, tmp_path: Path
) -> None:
    code = main(
        [
            "--queue",
            str(tmp_path / "missing.json"),
            "--results-root",
            str(tmp_path / "results"),
            "--scorecard",
            "configs/evaluation/round1_cctv_scorecard.json",
            "--output-dir",
            str(tmp_path / "out"),
        ]
    )

    error = json.loads(capsys.readouterr().err)
    assert code == 2
    assert error["status"] == "FAIL"
    assert error["error_type"] == "FileNotFoundError"
    assert not (tmp_path / "out").exists()


def test_cli_rejects_dirty_git_before_writing(
    capsys, monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        "scripts.build_round1_cctv_comparison.load_verified_round",
        lambda *_args: (),
    )
    monkeypatch.setattr(
        "scripts.build_round1_cctv_comparison._git_identity",
        lambda: (_ for _ in ()).throw(ValueError("clean Git worktree")),
    )
    code = main(
        [
            "--queue",
            str(tmp_path / "queue.json"),
            "--results-root",
            str(tmp_path / "results"),
            "--scorecard",
            "configs/evaluation/round1_cctv_scorecard.json",
            "--output-dir",
            str(tmp_path / "out"),
        ]
    )

    error = json.loads(capsys.readouterr().err)
    assert code == 2
    assert error["error_type"] == "ValueError"
    assert "clean Git worktree" in error["error"]
    assert not (tmp_path / "out").exists()
