from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from droneai.integrity import sha256_file
from droneai.steerer_training_gate import (
    REQUIRED_T1_METRICS,
    score_training_stage,
    weights_by_category,
)
from scripts import score_steerer_ucf_training as scoring_cli


def _artifact(tmp_path: Path, name: str, content: bytes = b"evidence") -> dict[str, str]:
    path = tmp_path / name
    path.write_bytes(content)
    return {"path": str(path), "sha256": sha256_file(path)}


def _evidence(tmp_path: Path) -> dict[str, object]:
    artifacts = {
        name: _artifact(tmp_path, f"{name}.bin", name.encode("utf-8"))
        for name in (
            "environment",
            "dataset_inventory",
            "train_split",
            "validation_split",
            "backbone",
            "input_checkpoint",
            "output_checkpoint",
            "config",
            "rights_decision",
        )
    }
    environment_bytes = b"immutable environment evidence"
    environment_sha256 = hashlib.sha256(environment_bytes).hexdigest()
    environment_path = tmp_path / f"environment.{environment_sha256}.json"
    environment_path.write_bytes(environment_bytes)
    artifacts["environment"] = {"path": str(environment_path), "sha256": environment_sha256}
    return {
        "schema_version": 1,
        "run_id": "steerer-qnrf-imagenet-20260806-a",
        "stage": "T1",
        "provenance": {
            "git_commit": "a" * 40,
            "upstream_commit": "5b1854dbc2d280f2326d67c65515d8baf9083810",
            "container_image_digest": "sha256:" + "c" * 64,
            "python": "3.13.0",
            "torch": "2.8.0",
            "cuda": "12.8",
            "gpu": "NVIDIA GeForce RTX 5090",
            "upstream_clean": True,
        },
        "split": {
            "dataset_population": 1201,
            "train_count": 961,
            "validation_count": 240,
            "overlap": 0,
            "complete": True,
            "official_test_accessed": False,
        },
        "initialization": {"model_checkpoint_loaded": False},
        "stability": {"finite": True, "checkpoint_round_trip": True},
        "metrics": {name: 1.0 for name in REQUIRED_T1_METRICS},
        "artifacts": artifacts,
    }


def test_t1_score_uses_approved_weights_and_threshold(tmp_path: Path) -> None:
    report = score_training_stage(_evidence(tmp_path), stage="T1")

    assert report.threshold == 75
    assert sum(check.weight for check in report.checks) == 100
    assert weights_by_category(report) == {
        "provenance": 25,
        "split": 20,
        "stability": 20,
        "validation": 20,
        "artifacts": 15,
    }
    assert report.status == "PASS_COMMERCIAL_CANDIDATE"
    assert report.decision_scope == "commercial_candidate"


def test_leakage_or_official_checkpoint_is_a_hard_blocker(tmp_path: Path) -> None:
    evidence = _evidence(tmp_path)
    evidence["split"]["overlap"] = 1  # type: ignore[index]
    evidence["initialization"]["model_checkpoint_loaded"] = True  # type: ignore[index]

    report = score_training_stage(evidence, stage="T1")

    assert report.status == "BLOCKED"
    assert {check.check_id for check in report.failed_blockers} == {
        "split.no_overlap",
        "provenance.no_official_model_checkpoint",
    }


def test_t1_requires_every_validation_metric_to_be_present_and_finite(
    tmp_path: Path,
) -> None:
    evidence = _evidence(tmp_path)
    del evidence["metrics"]["mae"]  # type: ignore[index]
    evidence["metrics"]["rmse"] = float("nan")  # type: ignore[index]

    report = score_training_stage(evidence, stage="T1")

    assert report.status == "BLOCKED"
    assert "stability.finite_metrics" in {
        check.check_id for check in report.failed_blockers
    }


def test_finite_integer_metric_does_not_crash_the_gate(tmp_path: Path) -> None:
    evidence = _evidence(tmp_path)
    evidence["metrics"]["mae"] = 10**1000  # type: ignore[index]

    report = score_training_stage(evidence, stage="T1")

    assert report.status == "PASS_COMMERCIAL_CANDIDATE"


def test_t0_explicitly_does_not_require_full_validation(tmp_path: Path) -> None:
    evidence = _evidence(tmp_path)
    evidence["stage"] = "T0"
    evidence["metrics"] = {"train_loss": 1.0}

    report = score_training_stage(evidence, stage="T0")

    assert report.threshold == 70
    assert report.status == "PASS_COMMERCIAL_CANDIDATE"


def test_cli_verifies_artifacts_before_creating_empty_result_directory(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    evidence = _evidence(tmp_path)
    evidence_path = tmp_path / "evidence.json"
    evidence_path.write_text(json.dumps(evidence), encoding="utf-8")
    output = tmp_path / "result"
    Path(evidence["artifacts"]["backbone"]["path"]).write_bytes(b"tampered")  # type: ignore[index]

    assert scoring_cli.main(
        ["--evidence", str(evidence_path), "--stage", "T1", "--output-dir", str(output)]
    ) == 2

    assert not output.exists()
    assert "hash mismatch" in capsys.readouterr().err


def test_cli_writes_only_the_required_immutable_review_bundle(tmp_path: Path) -> None:
    evidence = _evidence(tmp_path)
    evidence_path = tmp_path / "evidence.json"
    evidence_path.write_text(json.dumps(evidence), encoding="utf-8")
    output = tmp_path / "result"

    assert scoring_cli.main(
        ["--evidence", str(evidence_path), "--stage", "T1", "--output-dir", str(output)]
    ) == 0

    assert {path.name for path in output.iterdir()} == {
        "score.json", "score.md", "metrics.json", "environment.json", "lineage.json", "README.md"
    }
    lineage = json.loads((output / "lineage.json").read_text(encoding="utf-8"))
    assert lineage["output_checkpoint"]["sha256"] == evidence["artifacts"]["output_checkpoint"]["sha256"]  # type: ignore[index]
    assert json.loads((output / "score.json").read_text(encoding="utf-8"))["status"] == "PASS_COMMERCIAL_CANDIDATE"


def test_cli_requires_a_new_output_directory_before_writing(tmp_path: Path) -> None:
    evidence = _evidence(tmp_path)
    evidence_path = tmp_path / "evidence.json"
    evidence_path.write_text(json.dumps(evidence), encoding="utf-8")
    output = tmp_path / "result"
    output.mkdir()
    with pytest.raises(FileExistsError, match="new and empty"):
        scoring_cli.write_stage_result(evidence, stage="T1", output_dir=output)


def test_bundle_write_failure_does_not_expose_a_partial_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    evidence = _evidence(tmp_path)
    output = tmp_path / "nested" / "result"
    original_write = scoring_cli._write_new
    writes = 0

    def fail_during_bundle(path: Path, content: str | bytes) -> None:
        nonlocal writes
        writes += 1
        if writes == 3:
            raise OSError("injected bundle write failure")
        original_write(path, content)

    monkeypatch.setattr(scoring_cli, "_write_new", fail_during_bundle)

    with pytest.raises(OSError, match="injected bundle write failure"):
        scoring_cli.write_stage_result(evidence, stage="T1", output_dir=output)

    assert not output.exists()
    assert not list(output.parent.glob(".result.*"))
