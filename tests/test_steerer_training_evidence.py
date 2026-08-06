from __future__ import annotations

import math
from types import SimpleNamespace

import pytest

from droneai.steerer_training_evidence import (
    ValidationSampleObservation,
    build_t0_metrics,
    build_t1_metrics,
    strict_metrics_payload,
)
from droneai.steerer_training_gate import REQUIRED_T1_METRICS
from droneai.steerer_training_runner import UpdateObservation


def test_t0_metrics_use_actual_optimizer_update_loss() -> None:
    """Using the AMP probe loss would misrepresent the authorized optimizer update."""

    update = UpdateObservation(
        loss=1.25,
        density_values=(10.0,),
        gradient_norm=0.5,
        learning_rate=1.0e-4,
    )

    assert build_t0_metrics(update) == {"train_loss": 1.25}


def test_t1_metrics_match_hand_calculated_stage_observations() -> None:
    """A formula or aggregation regression must change at least one literal result."""

    samples = (
        ValidationSampleObservation(
            sample_id="sample-a",
            target_count=100.0,
            predicted_count=90.0,
            game_l1=4.0,
            quadrant_zone_mae=1.0,
            localization_tp=8,
            localization_fp=2,
            localization_fn=2,
            latency_ms=10.0,
            peak_vram_mb=800.0,
            density_sum_count_difference=0.25,
        ),
        ValidationSampleObservation(
            sample_id="sample-b",
            target_count=50.0,
            predicted_count=70.0,
            game_l1=8.0,
            quadrant_zone_mae=3.0,
            localization_tp=5,
            localization_fp=5,
            localization_fn=5,
            latency_ms=30.0,
            peak_vram_mb=1200.0,
            density_sum_count_difference=0.75,
        ),
    )
    epoch = SimpleNamespace(losses=(10.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0, 0.0))
    validation = SimpleNamespace(samples=samples, loss=2.5, sample_count=2)

    metrics = build_t1_metrics(epoch, validation, expected_samples=2)

    assert set(metrics) == REQUIRED_T1_METRICS
    assert metrics == pytest.approx(
        {
            "mae": 15.0,
            "rmse": math.sqrt(250.0),
            "signed_bias": 5.0,
            "reference_mape": 25.0,
            "game_l1": 6.0,
            "quadrant_zone_mae": 2.0,
            "localization_precision_16px": 0.65,
            "localization_recall_16px": 0.65,
            "localization_f1_16px": 0.65,
            "median_latency_ms": 20.0,
            "p95_latency_ms": 29.0,
            "fps": 50.0,
            "peak_vram_mb": 1200.0,
            "density_sum_count_difference": 0.5,
            "train_loss": 4.6,
            "validation_loss": 2.5,
            "training_window_improvement": 10.0,
        }
    )


def test_t1_metrics_require_complete_unique_validation_observations() -> None:
    """A partial or duplicate validation set must never become scoreable evidence."""

    sample = ValidationSampleObservation(
        sample_id="duplicate",
        target_count=1.0,
        predicted_count=1.0,
        game_l1=0.0,
        quadrant_zone_mae=0.0,
        localization_tp=1,
        localization_fp=0,
        localization_fn=0,
        latency_ms=1.0,
        peak_vram_mb=1.0,
        density_sum_count_difference=0.0,
    )
    epoch = SimpleNamespace(losses=(1.0,))
    validation = SimpleNamespace(samples=(sample, sample), loss=1.0, sample_count=2)

    with pytest.raises(ValueError, match="unique"):
        build_t1_metrics(epoch, validation, expected_samples=2)

    validation = SimpleNamespace(samples=(sample,), loss=1.0, sample_count=1)
    with pytest.raises(ValueError, match="complete"):
        build_t1_metrics(epoch, validation, expected_samples=2)


@pytest.mark.parametrize("bad_loss", (float("nan"), float("inf"), -float("inf")))
def test_metrics_payload_rejects_non_finite_or_wrong_stage_schema(bad_loss: float) -> None:
    """Non-standard JSON numbers and incomplete stage schemas cannot be hashed as evidence."""

    with pytest.raises(ValueError, match="finite"):
        strict_metrics_payload(
            run_id="run-3035",
            stage="T0",
            metrics={"train_loss": bad_loss},
        )

    with pytest.raises(ValueError, match="exact"):
        strict_metrics_payload(
            run_id="run-3035",
            stage="T0",
            metrics={"train_loss": 1.0, "probe_loss": 0.5},
        )
