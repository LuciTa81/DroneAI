from pathlib import Path

from droneai.stage0 import build_stage0_checks
from droneai.scoring import score_stage


def test_stage0_weights_total_one_hundred(tmp_path: Path) -> None:
    drive_root = tmp_path / "drive"
    for name in ("datasets", "checkpoints", "runs"):
        (drive_root / name).mkdir(parents=True)

    environment = {
        "captured_at": "2026-01-01T00:00:00+00:00",
        "python": "3.12",
        "platform": "test",
        "git_commit": "a" * 40,
        "git_dirty": False,
        "torch": {
            "version": "2.6.0",
            "cuda_available": False,
            "device_names": [],
        },
    }
    checks = build_stage0_checks(
        environment=environment,
        drive_root=drive_root,
        seeds=(17, 42, 2026),
        unit_tests_passed=True,
        unit_test_evidence="tests passed",
        environment_report_written=True,
        config_snapshot_written=True,
    )
    report = score_stage(
        stage_id="stage-0",
        stage_name="Experiment foundation readiness",
        threshold=85,
        checks=checks,
    )
    assert sum(check.weight for check in checks) == 100
    assert report.status == "BLOCKED"
    assert "runtime.cuda" in report.to_dict()["failed_blockers"]
