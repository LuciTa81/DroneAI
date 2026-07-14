import hashlib
import json
from pathlib import Path

from droneai.scoring import score_stage
from droneai.stage3 import build_stage3_checks, run_stage3_gate


def _passing_evidence() -> dict:
    return {
        "upstream": {"commit": "cc5f2132e0d1328909f31b6d665b8e0b15c30467", "license": "MIT"},
        "dataset": {
            "id": "shanghaitech-part-a",
            "train_images": 300,
            "test_images": 182,
            "split_hash": "a" * 64,
            "license_status": "verified",
            "commercial_use": False,
        },
        "faithful_runs": [
            {"seed": 2026, "mae": 59.5, "rmse": 95.5, "prediction_count": 182},
            {"seed": 2027, "mae": 59.7, "rmse": 95.7, "prediction_count": 182},
            {"seed": 2028, "mae": 60.0, "rmse": 96.0, "prediction_count": 182},
        ],
        "clean_protocol": {
            "validation_source": "training_partition_only",
            "test_evaluations_per_seed": 1,
            "seeds": [2026, 2027, 2028],
            "split_frozen_before_test": True,
        },
        "artifacts": {
            "config_snapshot": {"path": "config.json", "sha256": "c" * 64},
            "predictions": {"path": "predictions.jsonl", "sha256": "c" * 64},
            "checkpoint_hashes": {"path": "checkpoints.json", "sha256": "c" * 64},
            "environment": {"path": "environment.json", "sha256": "c" * 64},
        },
        "runtime": {"gpu": "Tesla T4", "median_seconds_per_image": 0.12},
    }


def test_stage3_weights_total_one_hundred_and_pass() -> None:
    checks = build_stage3_checks(_passing_evidence())
    report = score_stage(
        stage_id="stage-3", stage_name="DM-Count official reproduction", threshold=85, checks=checks
    )
    assert sum(check.weight for check in checks) == 100
    assert report.status == "PASS"


def test_stage3_blocks_test_leakage() -> None:
    evidence = _passing_evidence()
    evidence["clean_protocol"]["validation_source"] = "test_partition"
    report = score_stage(
        stage_id="stage-3",
        stage_name="DM-Count official reproduction",
        threshold=85,
        checks=build_stage3_checks(evidence),
    )
    assert report.status == "BLOCKED"
    assert "evaluation.test_isolation" in report.to_dict()["failed_blockers"]


def test_stage3_blocks_paper_gap() -> None:
    evidence = _passing_evidence()
    for run in evidence["faithful_runs"]:
        run.update({"mae": 70.0, "rmse": 115.0})
    report = score_stage(
        stage_id="stage-3",
        stage_name="DM-Count official reproduction",
        threshold=85,
        checks=build_stage3_checks(evidence),
    )
    assert report.status == "BLOCKED"
    assert "reproduction.paper_gap" in report.to_dict()["failed_blockers"]


def test_stage3_blocks_short_or_non_hex_split_hash() -> None:
    for invalid_hash in ("x", "z" * 64):
        evidence = _passing_evidence()
        evidence["dataset"]["split_hash"] = invalid_hash
        report = score_stage(
            stage_id="stage-3",
            stage_name="DM-Count official reproduction",
            threshold=85,
            checks=build_stage3_checks(evidence),
        )
        assert report.status == "BLOCKED"
        assert "dataset.contract" in report.to_dict()["failed_blockers"]


def test_stage3_run_recalculates_artifact_hashes_and_is_research_only(tmp_path: Path) -> None:
    evidence = _passing_evidence()
    for key, filename in {
        "config_snapshot": "config.json",
        "predictions": "predictions.jsonl",
        "checkpoint_hashes": "checkpoints.json",
        "environment": "environment.json",
    }.items():
        path = tmp_path / filename
        path.write_text(key, encoding="utf-8")
        evidence["artifacts"][key] = {
            "path": filename,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    evidence_path = tmp_path / "evidence.json"
    evidence_path.write_text(json.dumps(evidence), encoding="utf-8")

    report = run_stage3_gate(evidence_path=evidence_path, output_dir=tmp_path / "run")
    assert report.status == "PASS_RESEARCH_ONLY"

    evidence["artifacts"]["environment"]["sha256"] = "0" * 64
    evidence_path.write_text(json.dumps(evidence), encoding="utf-8")
    blocked = run_stage3_gate(evidence_path=evidence_path, output_dir=tmp_path / "bad-run")
    assert blocked.status == "BLOCKED"
    assert "artifacts.bundle" in blocked.to_dict()["failed_blockers"]
