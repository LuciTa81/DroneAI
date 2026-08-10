from __future__ import annotations

import importlib
import os
from pathlib import Path


def _subject():
    return importlib.import_module("droneai.steerer_official_retention")


def _snapshot(root: Path, epoch: int, mae: float, mse: float, *, age: int = 600) -> Path:
    path = root / f"Ep_{epoch}_mae_{mae}_mse_{mse}.pth"
    path.write_bytes(f"epoch={epoch}".encode())
    timestamp = 2_000_000_000 - age
    os.utime(path, (timestamp, timestamp))
    return path


def test_retention_keeps_only_best_mae_and_best_mse_snapshots(tmp_path: Path) -> None:
    run = tmp_path / "QNRF_final_run"
    run.mkdir()
    old = _snapshot(run, 6, 300.0, 450.0)
    best_mae = _snapshot(run, 13, 250.0, 470.0)
    best_mse = _snapshot(run, 20, 280.0, 400.0)

    result = _subject().prune_official_best_snapshots(
        tmp_path,
        min_age_seconds=120,
        now_timestamp=2_000_000_000,
    )

    assert result.deleted == (old,)
    assert set(result.retained) == {best_mae, best_mse}
    assert not old.exists()
    assert best_mae.exists()
    assert best_mse.exists()


def test_retention_never_deletes_recent_or_unrecognized_files(tmp_path: Path) -> None:
    run = tmp_path / "QNRF_final_run"
    run.mkdir()
    best = _snapshot(run, 6, 250.0, 400.0)
    recent = _snapshot(run, 13, 300.0, 450.0, age=30)
    unrelated = run / "final_state.pth"
    unrelated.write_bytes(b"final")

    result = _subject().prune_official_best_snapshots(
        tmp_path,
        min_age_seconds=120,
        now_timestamp=2_000_000_000,
    )

    assert result.deleted == ()
    assert best.exists()
    assert recent.exists()
    assert unrelated.exists()


def test_retention_dry_run_reports_without_deleting(tmp_path: Path) -> None:
    run = tmp_path / "QNRF_final_run"
    run.mkdir()
    old = _snapshot(run, 6, 300.0, 450.0)
    _snapshot(run, 13, 250.0, 400.0)

    result = _subject().prune_official_best_snapshots(
        tmp_path,
        min_age_seconds=0,
        now_timestamp=2_000_000_000,
        dry_run=True,
    )

    assert result.deleted == (old,)
    assert old.exists()
