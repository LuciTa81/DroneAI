import csv
import json
from pathlib import Path

import pytest

from droneai.evaluation_artifacts import (
    artifact_reference,
    enforce_review_budget,
    write_json,
    write_predictions_csv,
)
from droneai.evaluation_contract import ScalarEvaluation
from droneai.integrity import verify_artifact_reference


def _record(sample_id: str = "s1") -> ScalarEvaluation:
    return ScalarEvaluation(
        sample_id,
        2.0,
        1.0,
        -1.0,
        1.0,
        0.5,
        "medium",
        5.0,
        100.0,
        "count",
        None,
        extra_metrics={"unavailable_metric": float("inf")},
        condition_values={"lighting": "night"},
    )


def test_json_artifact_reference_recalculates_sha256(tmp_path: Path) -> None:
    path = write_json(tmp_path / "metrics.json", {"mae": 1.25})
    reference = artifact_reference(path, base_dir=tmp_path)

    assert verify_artifact_reference(reference, base_dir=tmp_path)[0]
    assert json.loads(path.read_text(encoding="utf-8"))["mae"] == 1.25


def test_json_writer_converts_nested_non_finite_metrics_to_null(tmp_path: Path) -> None:
    path = write_json(
        tmp_path / "metrics.json",
        {"mae": float("inf"), "nested": [float("nan"), 1.0]},
    )

    assert json.loads(path.read_text(encoding="utf-8")) == {
        "mae": None,
        "nested": [None, 1.0],
    }


def test_writers_refuse_to_overwrite_existing_artifacts(tmp_path: Path) -> None:
    target = tmp_path / "evidence.json"
    target.write_text("preserve", encoding="utf-8")

    with pytest.raises(FileExistsError):
        write_json(target, {"replace": True})

    assert target.read_text(encoding="utf-8") == "preserve"


def test_predictions_csv_is_flat_deterministic_and_json_safe(tmp_path: Path) -> None:
    path = write_predictions_csv(tmp_path / "predictions.csv", [_record()])

    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)

    assert reader.fieldnames == sorted(reader.fieldnames or [])
    assert rows[0]["sample_id"] == "s1"
    assert rows[0]["condition_lighting"] == "night"
    assert rows[0]["unavailable_metric"] == ""


def test_artifact_reference_rejects_paths_outside_base(tmp_path: Path) -> None:
    outside = tmp_path.parent / "outside.bin"
    outside.write_bytes(b"outside")

    with pytest.raises(ValueError, match="inside base directory"):
        artifact_reference(outside, base_dir=tmp_path)


def test_review_budget_rejects_more_than_twenty_five_mib(tmp_path: Path) -> None:
    (tmp_path / "large.bin").write_bytes(b"0" * (25 * 1024 * 1024 + 1))

    with pytest.raises(ValueError, match="25 MiB"):
        enforce_review_budget(tmp_path)


def test_review_budget_allows_exact_limit(tmp_path: Path) -> None:
    (tmp_path / "exact.bin").write_bytes(b"0" * (25 * 1024 * 1024))

    assert enforce_review_budget(tmp_path) == 25 * 1024 * 1024
