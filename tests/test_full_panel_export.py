from __future__ import annotations

import csv
from pathlib import Path

import pytest

from droneai.evaluation_contract import ScalarEvaluation
from droneai.full_panel_export import (
    AcceptedPrediction,
    PanelExportIdentity,
    PanelProgressLedger,
    load_accepted_predictions,
    panel_filename,
    verify_fresh_record,
)


def _record(
    *,
    sample_id: str = "img_0001",
    predicted_count: float = 10.0,
    output_type: str = "density",
) -> ScalarEvaluation:
    return ScalarEvaluation(
        sample_id=sample_id,
        ground_truth_count=11.0,
        predicted_count=predicted_count,
        signed_error=predicted_count - 11.0,
        absolute_error=abs(predicted_count - 11.0),
        normalized_error=abs(predicted_count - 11.0) / 11.0,
        density_band="low",
        latency_ms=2.0,
        peak_vram_mb=3.0,
        output_type=output_type,  # type: ignore[arg-type]
        failure_state=None,
    )


def _identity(model_id: str = "steerer") -> PanelExportIdentity:
    return PanelExportIdentity(
        model_id=model_id,
        dataset_id="ucf-qnrf-kaggle-apache",
        git_commit="a" * 40,
        upstream_commit="b" * 40,
        checkpoint_sha256="c" * 64,
        shared_manifest_sha256="d" * 64,
    )


def _write_predictions(path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames = ["sample_id", "predicted_count", "output_type", "failure_state"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def test_load_accepted_predictions_requires_unique_finite_rows(
    tmp_path: Path,
) -> None:
    path = tmp_path / "predictions.csv"
    _write_predictions(
        path,
        [
            {
                "sample_id": "img_1",
                "predicted_count": "10.5",
                "output_type": "density",
                "failure_state": "",
            },
            {
                "sample_id": "img_2",
                "predicted_count": "4",
                "output_type": "points",
                "failure_state": "",
            },
        ],
    )

    accepted = load_accepted_predictions(path)

    assert accepted == {
        "img_1": AcceptedPrediction("img_1", 10.5, "density"),
        "img_2": AcceptedPrediction("img_2", 4.0, "points"),
    }


@pytest.mark.parametrize(
    "rows, message",
    [
        (
            [
                {
                    "sample_id": "same",
                    "predicted_count": "1",
                    "output_type": "density",
                    "failure_state": "",
                },
                {
                    "sample_id": "same",
                    "predicted_count": "1",
                    "output_type": "density",
                    "failure_state": "",
                },
            ],
            "duplicate",
        ),
        (
            [
                {
                    "sample_id": "img",
                    "predicted_count": "nan",
                    "output_type": "density",
                    "failure_state": "",
                }
            ],
            "finite",
        ),
        (
            [
                {
                    "sample_id": "img",
                    "predicted_count": "1",
                    "output_type": "density",
                    "failure_state": "oom",
                }
            ],
            "failure",
        ),
    ],
)
def test_load_accepted_predictions_rejects_invalid_rows(
    tmp_path: Path,
    rows: list[dict[str, object]],
    message: str,
) -> None:
    path = tmp_path / "predictions.csv"
    _write_predictions(path, rows)

    with pytest.raises(ValueError, match=message):
        load_accepted_predictions(path)


def test_verify_fresh_record_rejects_count_drift() -> None:
    accepted = AcceptedPrediction("img_0001", 10.0, "density")

    with pytest.raises(ValueError, match="accepted predicted count"):
        verify_fresh_record(accepted, _record(predicted_count=10.001))


def test_verify_fresh_record_accepts_micro_tolerance() -> None:
    accepted = AcceptedPrediction("img_0001", 10.0, "density")

    verify_fresh_record(accepted, _record(predicted_count=10.0000005))


@pytest.mark.parametrize("sample_id", ("../x", "a/b", "a\\b", "", "."))
def test_panel_filename_rejects_unsafe_sample_ids(sample_id: str) -> None:
    with pytest.raises(ValueError, match="sample"):
        panel_filename(sample_id)


def test_panel_filename_is_stable_and_keeps_readable_identity() -> None:
    name = panel_filename("img_0001")

    assert name.startswith("img_0001-")
    assert name.endswith(".png")
    assert name == panel_filename("img_0001")


def test_progress_resume_verifies_panel_hash_and_identity(tmp_path: Path) -> None:
    root = tmp_path / "lane"
    panel = root / "panels" / panel_filename("img_0001")
    panel.parent.mkdir(parents=True)
    panel.write_bytes(b"png")
    ledger = PanelProgressLedger.open(
        root / "progress.jsonl",
        identity=_identity(),
        output_root=root,
    )
    accepted = AcceptedPrediction("img_0001", 10.0, "density")
    fresh = _record()

    ledger.append_completed(
        source_sha256="e" * 64,
        accepted=accepted,
        fresh=fresh,
        panel_path=panel,
    )
    resumed = PanelProgressLedger.open(
        root / "progress.jsonl",
        identity=_identity(),
        output_root=root,
    )

    assert resumed.completed_sample_ids() == ("img_0001",)


def test_progress_rejects_changed_identity(tmp_path: Path) -> None:
    root = tmp_path / "lane"
    PanelProgressLedger.open(
        root / "progress.jsonl",
        identity=_identity(),
        output_root=root,
    )

    with pytest.raises(ValueError, match="identity"):
        PanelProgressLedger.open(
            root / "progress.jsonl",
            identity=_identity("pet"),
            output_root=root,
        )


def test_progress_rejects_tampered_panel(tmp_path: Path) -> None:
    root = tmp_path / "lane"
    panel = root / "panels" / panel_filename("img_0001")
    panel.parent.mkdir(parents=True)
    panel.write_bytes(b"png")
    ledger = PanelProgressLedger.open(
        root / "progress.jsonl",
        identity=_identity(),
        output_root=root,
    )
    ledger.append_completed(
        source_sha256="e" * 64,
        accepted=AcceptedPrediction("img_0001", 10.0, "density"),
        fresh=_record(),
        panel_path=panel,
    )
    panel.write_bytes(b"changed")

    with pytest.raises(ValueError, match="panel SHA-256"):
        PanelProgressLedger.open(
            root / "progress.jsonl",
            identity=_identity(),
            output_root=root,
        )


def test_progress_rejects_panel_outside_output_root(tmp_path: Path) -> None:
    root = tmp_path / "lane"
    outside = tmp_path / "outside.png"
    outside.write_bytes(b"png")
    ledger = PanelProgressLedger.open(
        root / "progress.jsonl",
        identity=_identity(),
        output_root=root,
    )

    with pytest.raises(ValueError, match="inside output root"):
        ledger.append_completed(
            source_sha256="e" * 64,
            accepted=AcceptedPrediction("img_0001", 10.0, "density"),
            fresh=_record(),
            panel_path=outside,
        )
