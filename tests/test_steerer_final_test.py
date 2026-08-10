from __future__ import annotations

import copy
import csv
import json
from pathlib import Path

import pytest

import scripts.run_steerer_b_final_test as final_cli

from droneai.steerer_final_test import (
    build_b_final_test_protocol,
    finalize_b_final_test,
    load_b_final_test_profile,
    validate_checkpoint_manifest,
    validate_b_final_test_profile,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
PROFILE = REPO_ROOT / "configs/evaluation/steerer_b_ucf_qnrf_final_test.json"


def _payload() -> dict[str, object]:
    return json.loads(PROFILE.read_text(encoding="utf-8"))


def test_profile_freezes_three_validation_selected_checkpoints() -> None:
    profile = load_b_final_test_profile(PROFILE)

    assert [(row.role, row.epoch, row.filename, row.sha256) for row in profile.checkpoints] == [
        (
            "primary-best-val-mae",
            625,
            "best-mae.pth",
            "e41df787f546a5195e078486ec6d1bd62ffcb2e3e77010436f74b65045667ad2",
        ),
        (
            "secondary-best-val-rmse",
            175,
            "best-rmse.pth",
            "ab1f65573c7f18f115e711998f4f4fb252e816bc5b9ae3564d5b379f22f981e6",
        ),
        (
            "final-reference",
            800,
            "milestone-800.pth",
            "d13316a818ddaa2ed0da680718b234bfd21377934c7ef3b33fa074746c93c8f0",
        ),
    ]


def test_profile_freezes_test_access_and_closes_post_test_training() -> None:
    profile = load_b_final_test_profile(PROFILE)

    assert profile.test_split_id == "ucf-qnrf-official-test334-v1"
    assert profile.expected_samples == 334
    assert profile.test_access_approved is True
    assert profile.post_test_training_allowed is False
    assert profile.checkpoint_origin == "project_training"


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda p: p.update({"expected_samples": 333}), "334"),
        (
            lambda p: p["checkpoints"].reverse(),
            "checkpoint",
        ),
        (
            lambda p: p["checkpoints"][0].update({"sha256": "0" * 64}),
            "checkpoint",
        ),
        (lambda p: p.update({"test_access_approved": False}), "Test access"),
        (lambda p: p.update({"post_test_training_allowed": True}), "post-Test"),
        (lambda p: p.update({"checkpoint_origin": "research_checkpoint"}), "project_training"),
    ],
)
def test_profile_rejects_changes_to_the_preregistered_contract(mutation, message: str) -> None:
    payload = copy.deepcopy(_payload())
    mutation(payload)

    with pytest.raises((PermissionError, ValueError), match=message):
        validate_b_final_test_profile(payload)


def test_protocol_is_an_explicit_verified_final_test() -> None:
    profile = load_b_final_test_profile(PROFILE)
    rights_path = REPO_ROOT / "configs/training/steerer_ucf_qnrf_imagenet.home5090.json"

    protocol = build_b_final_test_protocol(
        profile,
        checkpoint=profile.checkpoints[0],
        rights_path=rights_path,
    )

    assert protocol.split_role == "test"
    assert protocol.expected_samples == 334
    assert protocol.sealed_test_access_approved is True
    assert protocol.leakage_free is True
    assert protocol.checkpoint_training_split_status == "VERIFIED_DISJOINT"
    assert protocol.comparison_scope == "held_out_performance"


def test_cli_exposes_only_the_preregistered_final_test_inputs() -> None:
    options = {
        option
        for action in final_cli._parser()._actions
        for option in action.option_strings
    }

    assert {
        "--config",
        "--test-root",
        "--upstream-dir",
        "--checkpoint-root",
        "--checkpoint-manifest",
        "--training-profile",
        "--output-dir",
        "--device",
    }.issubset(options)
    assert {"--train-root", "--resume", "--checkpoint", "--checkpoint-sha256"}.isdisjoint(
        options
    )


def test_checkpoint_manifest_must_contain_the_frozen_epoch_hash_pairs(
    tmp_path: Path,
) -> None:
    profile = load_b_final_test_profile(PROFILE)
    manifest = tmp_path / "checkpoint-manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "run_id": profile.training_run_id,
                "checkpoints": [
                    {
                        "epoch": row.epoch,
                        "filename": row.filename,
                        "sha256": row.sha256,
                    }
                    for row in profile.checkpoints
                ],
            }
        ),
        encoding="utf-8",
    )

    verified = validate_checkpoint_manifest(profile, manifest)

    assert [row.role for row in verified] == [row.role for row in profile.checkpoints]

    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["checkpoints"][0]["epoch"] = 626
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="manifest"):
        validate_checkpoint_manifest(profile, manifest)


def _write_role_bundle(root: Path, role: str, *, offset: float) -> None:
    role_root = root / role
    role_root.mkdir(parents=True)
    (role_root / "metrics.json").write_text(
        json.dumps(
            {
                "accounting_complete": True,
                "expected_samples": 334,
                "recorded_samples": 334,
                "successful_samples": 334,
                "explicit_failures": 0,
                "mae": 100.0 + offset,
                "rmse": 300.0 + offset,
                "signed_bias": -20.0 + offset,
                "median_latency_ms": 210.0 + offset,
                "throughput_fps_batch1": 4.7,
                "peak_vram_mb": 5300.0,
            }
        ),
        encoding="utf-8",
    )
    (role_root / "score.json").write_text(
        json.dumps({"score": 100, "status": "PASS_COMMERCIAL_CANDIDATE"}),
        encoding="utf-8",
    )
    with (role_root / "predictions.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=(
                "sample_id",
                "ground_truth_count",
                "predicted_count",
                "absolute_error",
                "localization_precision",
                "localization_recall",
                "localization_f1",
                "localization_mean_distance",
                "density_zone_mae",
                "density_psnr",
                "density_ssim",
            ),
        )
        writer.writeheader()
        for index in range(334):
            writer.writerow(
                {
                    "sample_id": f"test_img_{index + 1:04d}",
                    "ground_truth_count": 100,
                    "predicted_count": 90,
                    "absolute_error": 10,
                    "localization_precision": 0.8,
                    "localization_recall": 0.7,
                    "localization_f1": 0.7466666667,
                    "localization_mean_distance": 8.0,
                    "density_zone_mae": 3.0,
                    "density_psnr": 30.0,
                    "density_ssim": 0.9,
                }
            )


def test_finalize_writes_comparison_and_irreversible_b_lane_closure(tmp_path: Path) -> None:
    profile = load_b_final_test_profile(PROFILE)
    for index, checkpoint in enumerate(profile.checkpoints):
        _write_role_bundle(tmp_path, checkpoint.role, offset=float(index))

    comparison_path, closure_path = finalize_b_final_test(tmp_path, profile=profile)

    comparison = json.loads(comparison_path.read_text(encoding="utf-8"))
    closure = json.loads(closure_path.read_text(encoding="utf-8"))
    assert [row["role"] for row in comparison["results"]] == [
        row.role for row in profile.checkpoints
    ]
    assert comparison["results"][0]["reference_mape_percent"] == 10.0
    assert comparison["results"][0]["localization_f1"] == pytest.approx(0.7466666667)
    assert closure["test_sample_count"] == 334
    assert closure["post_test_training_allowed"] is False
    assert closure["checkpoint_selection_remains_frozen"] is True

    with pytest.raises(FileExistsError, match="already closed"):
        finalize_b_final_test(tmp_path, profile=profile)
