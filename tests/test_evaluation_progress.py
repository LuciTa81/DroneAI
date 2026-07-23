from __future__ import annotations

from pathlib import Path

import pytest

from droneai.evaluation_contract import ScalarEvaluation
from droneai.evaluation_progress import ProgressLedger


def _record(sample_id: str) -> ScalarEvaluation:
    return ScalarEvaluation(
        sample_id=sample_id,
        ground_truth_count=10.0,
        predicted_count=11.0,
        signed_error=1.0,
        absolute_error=1.0,
        normalized_error=0.1,
        density_band="low",
        latency_ms=2.0,
        peak_vram_mb=3.0,
        output_type="density",
        failure_state=None,
        spatial_metric_name="game_l1",
        spatial_metric_value=4.0,
        extra_metrics={"zone_mae": 1.0},
        condition_values={"source": "fixture"},
    )


def test_progress_resume_reuses_immutable_prefix(tmp_path: Path) -> None:
    path = tmp_path / "progress.jsonl"
    ledger = ProgressLedger.open(
        path,
        identity_sha256="a" * 64,
        ordered_sample_ids=("a", "b"),
    )
    ledger.append(
        _record("a"),
        native_metadata={"shape": "1x1"},
        fingerprint="b" * 64,
    )

    resumed = ProgressLedger.open(
        path,
        identity_sha256="a" * 64,
        ordered_sample_ids=("a", "b"),
    )

    assert [row.sample_id for row in resumed.completed_records()] == ["a"]
    assert resumed.completed_entries()[0].native_metadata == {"shape": "1x1"}
    assert resumed.completed_entries()[0].fingerprint == "b" * 64


def test_progress_rejects_changed_run_identity(tmp_path: Path) -> None:
    path = tmp_path / "progress.jsonl"
    ProgressLedger.open(
        path,
        identity_sha256="a" * 64,
        ordered_sample_ids=("a",),
    )

    with pytest.raises(ValueError, match="identity"):
        ProgressLedger.open(
            path,
            identity_sha256="b" * 64,
            ordered_sample_ids=("a",),
        )


def test_progress_requires_canonical_next_sample(tmp_path: Path) -> None:
    ledger = ProgressLedger.open(
        tmp_path / "progress.jsonl",
        identity_sha256="a" * 64,
        ordered_sample_ids=("a", "b"),
    )

    with pytest.raises(ValueError, match="next sample"):
        ledger.append(
            _record("b"),
            native_metadata={},
            fingerprint="b" * 64,
        )
