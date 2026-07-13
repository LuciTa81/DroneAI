from droneai.scoring import score_stage
from droneai.stage3 import build_stage3_checks


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
            "config_snapshot": "config.json",
            "predictions": "predictions.jsonl",
            "checkpoint_hashes": "checkpoints.json",
            "environment": "environment.json",
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
