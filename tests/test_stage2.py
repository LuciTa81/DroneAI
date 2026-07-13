from droneai.scoring import score_stage
from droneai.stage2 import build_stage2_checks


def _metrics() -> dict:
    return {
        "architecture": {
            "convolution_layers": 17,
            "dilated_convolution_layers": 6,
            "output_stride": 8,
            "parameters": 16_263_489,
        },
        "output_shape": [4, 1, 8, 8],
        "finite_output": True,
        "final_loss": 0.001,
        "loss_reduction": 0.95,
        "count_mae": 0.1,
        "initial_state_hash": "a" * 64,
        "repeated_state_hash": "a" * 64,
        "trained_state_hash": "b" * 64,
    }


def test_stage2_weights_total_one_hundred_and_pass() -> None:
    checks = build_stage2_checks(
        metrics=_metrics(), density_sum_error=0.0, block_sum_error=0.0,
        config_written=True, metrics_written=True,
    )
    report = score_stage(
        stage_id="stage-2", stage_name="CSRNet pipeline smoke", threshold=80, checks=checks
    )
    assert sum(check.weight for check in checks) == 100
    assert report.status == "PASS"


def test_stage2_blocks_count_mass_error() -> None:
    checks = build_stage2_checks(
        metrics=_metrics(), density_sum_error=0.01, block_sum_error=0.0,
        config_written=True, metrics_written=True,
    )
    report = score_stage(
        stage_id="stage-2", stage_name="CSRNet pipeline smoke", threshold=80, checks=checks
    )
    assert report.status == "BLOCKED"
    assert "count.point_density" in report.to_dict()["failed_blockers"]
