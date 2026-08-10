from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
from PIL import Image
from scipy.io import savemat

import scripts.prepare_steerer_a_data as a_data_cli

from droneai.steerer_official_data import (
    convert_official_test_sample,
    load_official_training_data_profile,
    load_prepared_official_test,
    official_qnrf_prepared_size,
    prepare_official_train_test_lane,
    prepare_official_test_partition,
    score_g2_prepared_data,
    validate_official_training_data_profile,
    verify_prepared_official_train_test_lane,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
A_TRAINING_PROFILE = (
    REPO_ROOT
    / "configs/training/steerer_ucf_qnrf_official_code_reproduction.home5090.json"
)


def test_data_preparation_import_does_not_require_evaluation_dependencies() -> None:
    command = """
import importlib.abc
import sys

class BlockSkimage(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'skimage' or fullname.startswith('skimage.'):
            raise ModuleNotFoundError('skimage intentionally unavailable')
        return None

sys.meta_path.insert(0, BlockSkimage())
import droneai.steerer_official_data
"""
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(REPO_ROOT / "src")

    completed = subprocess.run(
        [sys.executable, "-c", command],
        cwd=REPO_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr


def _raw_sample(
    root: Path,
    *,
    sample_number: int = 1,
    points: list[list[float]],
    color: tuple[int, int, int] = (20, 40, 60),
) -> None:
    root.mkdir(parents=True, exist_ok=True)
    stem = f"img_{sample_number:04d}"
    Image.new("RGB", (640, 480), color=(20, 40, 60)).save(
        root / f"{stem}.jpg", quality=90
    )
    if color != (20, 40, 60):
        Image.new("RGB", (640, 480), color=color).save(
            root / f"{stem}.jpg", quality=90
        )
    savemat(root / f"{stem}_ann.mat", {"annPoints": points})


def test_official_qnrf_prepared_size_matches_pinned_reference() -> None:
    assert official_qnrf_prepared_size(2500, 1875) == (2512, 1888)
    assert official_qnrf_prepared_size(1024, 768) == (1040, 784)
    assert official_qnrf_prepared_size(640, 480) == (1024, 784)


def test_conversion_preserves_count_and_clips_edge_points(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    _raw_sample(raw, points=[[10.0, 20.0], [641.0, 482.0]])
    output = tmp_path / "converted"

    converted = convert_official_test_sample(
        raw / "img_0001.jpg",
        raw / "img_0001_ann.mat",
        prepared_id="1202",
        source_sample_id="test_img_0001",
        output_root=output,
    )

    assert converted.count == 2
    assert converted.width == 1024
    assert converted.height == 784
    assert converted.boundary_corrections == 1
    annotation = json.loads(converted.annotation_path.read_text(encoding="utf-8"))
    assert len(annotation["points"]) == 2
    assert annotation["points"][1] == [1023.0, 783.0]
    with Image.open(converted.image_path) as image:
        assert image.size == (1024, 784)


def test_partition_manifest_is_hash_verified_and_loadable(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    _raw_sample(raw, points=[[10.0, 20.0], [30.0, 40.0]])
    output = tmp_path / "prepared"

    manifest = prepare_official_test_partition(
        raw,
        output_root=output,
        expected_samples=1,
    )
    records = load_prepared_official_test(
        output,
        manifest_path=manifest,
        expected_samples=1,
    )

    assert len(records) == 1
    assert records[0].sample_id == "test_img_0001"
    assert records[0].count == 2
    assert records[0].image_sha256
    assert records[0].annotation_sha256


def test_full_lane_writes_disjoint_train_and_test_without_validation(
    tmp_path: Path,
) -> None:
    raw_train = tmp_path / "raw" / "Train"
    raw_test = tmp_path / "raw" / "Test"
    _raw_sample(raw_train, points=[[10.0, 20.0]], color=(10, 20, 30))
    _raw_sample(raw_test, points=[[30.0, 40.0], [50.0, 60.0]], color=(70, 80, 90))
    output = tmp_path / "prepared"

    manifest = prepare_official_train_test_lane(
        raw_train,
        raw_test,
        output_root=output,
        expected_train_samples=1,
        expected_test_samples=1,
    )
    verified = verify_prepared_official_train_test_lane(
        output,
        manifest_path=manifest,
        expected_train_samples=1,
        expected_test_samples=1,
    )

    assert (output / "train.txt").read_text(encoding="utf-8") == "0001\n"
    assert (output / "test.txt").read_text(encoding="utf-8") == "0002\n"
    assert not (output / "val.txt").exists()
    assert verified["train_samples"] == 1
    assert verified["test_samples"] == 1
    assert verified["validation_samples"] == 0
    assert verified["split_overlap"] == []
    assert verified["artifact_hashes_verified"] is True


def test_full_lane_rejects_cross_partition_duplicate_images(tmp_path: Path) -> None:
    raw_train = tmp_path / "raw" / "Train"
    raw_test = tmp_path / "raw" / "Test"
    _raw_sample(raw_train, points=[[10.0, 20.0]])
    _raw_sample(raw_test, points=[[30.0, 40.0]])

    with pytest.raises(ValueError, match="duplicate|overlap"):
        prepare_official_train_test_lane(
            raw_train,
            raw_test,
            output_root=tmp_path / "prepared",
            expected_train_samples=1,
            expected_test_samples=1,
        )


def test_full_lane_rejects_missing_annotation_and_nonempty_output(
    tmp_path: Path,
) -> None:
    raw_train = tmp_path / "raw" / "Train"
    raw_test = tmp_path / "raw" / "Test"
    _raw_sample(raw_train, points=[[10.0, 20.0]], color=(10, 20, 30))
    _raw_sample(raw_test, points=[[30.0, 40.0]], color=(70, 80, 90))
    (raw_test / "img_0001_ann.mat").unlink()

    with pytest.raises(FileNotFoundError, match="annotation"):
        prepare_official_train_test_lane(
            raw_train,
            raw_test,
            output_root=tmp_path / "prepared",
            expected_train_samples=1,
            expected_test_samples=1,
        )

    _raw_sample(raw_test, points=[[30.0, 40.0]], color=(70, 80, 90))
    output = tmp_path / "nonempty"
    output.mkdir()
    (output / "keep.txt").write_text("owned by user", encoding="utf-8")
    with pytest.raises(FileExistsError, match="new or empty"):
        prepare_official_train_test_lane(
            raw_train,
            raw_test,
            output_root=output,
            expected_train_samples=1,
            expected_test_samples=1,
        )


def test_full_lane_fresh_verification_detects_artifact_tampering(
    tmp_path: Path,
) -> None:
    raw_train = tmp_path / "raw" / "Train"
    raw_test = tmp_path / "raw" / "Test"
    _raw_sample(raw_train, points=[[10.0, 20.0]], color=(10, 20, 30))
    _raw_sample(raw_test, points=[[30.0, 40.0]], color=(70, 80, 90))
    output = tmp_path / "prepared"
    manifest = prepare_official_train_test_lane(
        raw_train,
        raw_test,
        output_root=output,
        expected_train_samples=1,
        expected_test_samples=1,
    )
    (output / "jsons" / "0001.json").write_text("{}", encoding="utf-8")

    with pytest.raises(ValueError, match="hash"):
        verify_prepared_official_train_test_lane(
            output,
            manifest_path=manifest,
            expected_train_samples=1,
            expected_test_samples=1,
        )


def test_full_lane_fresh_verification_detects_source_tampering(
    tmp_path: Path,
) -> None:
    raw_train = tmp_path / "raw" / "Train"
    raw_test = tmp_path / "raw" / "Test"
    _raw_sample(raw_train, points=[[10.0, 20.0]], color=(10, 20, 30))
    _raw_sample(raw_test, points=[[30.0, 40.0]], color=(70, 80, 90))
    output = tmp_path / "prepared"
    manifest = prepare_official_train_test_lane(
        raw_train,
        raw_test,
        output_root=output,
        expected_train_samples=1,
        expected_test_samples=1,
    )
    (raw_train / "img_0001_ann.mat").write_bytes(b"tampered source")

    with pytest.raises(ValueError, match="source.*hash"):
        verify_prepared_official_train_test_lane(
            output,
            manifest_path=manifest,
            expected_train_samples=1,
            expected_test_samples=1,
        )


def test_a_training_data_profile_freezes_full_train_test_protocol() -> None:
    profile = load_official_training_data_profile(A_TRAINING_PROFILE)

    assert profile.train_samples == 1201
    assert profile.validation_samples == 0
    assert profile.test_samples == 334
    assert profile.seed == 3035
    assert profile.test_selection == "test_selected"
    assert profile.test_access_approved is True
    assert profile.full_train_user_approved is True
    assert profile.initialization_mode == "imagenet_backbone_only"
    assert profile.model_checkpoint is None
    assert profile.success_scope == "PASS_COMMERCIAL_CANDIDATE"
    assert profile.production_approved is False
    assert profile.official_checkpoint_training_allowed is False
    assert str(profile.processed_root).endswith(
        "/steerer-official-code-reproduction-v1/full-train-test-v1"
    )
    assert profile.training_settings["optimizer"] == "AdamW"
    assert profile.training_settings["end_epoch"] == 800


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda payload: payload["split"].update({"train": 1200}), "1,201"),
        (lambda payload: payload["split"].update({"validation": 1}), "validation"),
        (
            lambda payload: payload["split"].update({"test_selection": "validation"}),
            "test-selected",
        ),
        (
            lambda payload: payload["initialization"].update(
                {"model_checkpoint": "/weights/QNRF_mae_77.8.pth"}
            ),
            "checkpoint",
        ),
        (
            lambda payload: payload["storage"].update(
                {
                    "processed_root": (
                        "/workspace/data/datasets/ucf-qnrf-kaggle-apache/"
                        "processed/steerer-training-v1"
                    )
                }
            ),
            "isolated",
        ),
        (
            lambda payload: payload["rights"].update({"production_approved": True}),
            "production",
        ),
    ],
)
def test_a_training_data_profile_rejects_protocol_drift(
    mutation, message: str
) -> None:
    payload = json.loads(A_TRAINING_PROFILE.read_text(encoding="utf-8"))
    mutation(payload)

    with pytest.raises((PermissionError, ValueError), match=message):
        validate_official_training_data_profile(payload)


def test_g2_score_is_independent_of_model_accuracy() -> None:
    profile = load_official_training_data_profile(A_TRAINING_PROFILE)
    report = score_g2_prepared_data(
        profile,
        {
            "status": "verified",
            "train_samples": 1201,
            "validation_samples": 0,
            "test_samples": 334,
            "split_overlap": [],
            "artifact_hashes_verified": True,
            "dataset_content_sha256": "a" * 64,
            "manifest_sha256": "b" * 64,
        },
        evidence_path="manifests/dataset-preparation.json",
    )

    assert report.score == 100
    assert report.status == "PASS_COMMERCIAL_CANDIDATE"
    assert all("mae" not in check.check_id for check in report.checks)


def test_a_data_cli_has_no_validation_or_checkpoint_override() -> None:
    parser = a_data_cli._parser()

    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "prepare",
                "--config",
                str(A_TRAINING_PROFILE),
                "--raw-train-root",
                "/tmp/Train",
                "--raw-test-root",
                "/tmp/Test",
                "--output-dir",
                "/tmp/output",
                "--validation-count",
                "240",
            ]
        )
    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "verify",
                "--config",
                str(A_TRAINING_PROFILE),
                "--output-dir",
                "/tmp/output",
                "--manifest",
                "/tmp/output/manifests/dataset-preparation.json",
                "--checkpoint",
                "/tmp/model.pth",
            ]
        )
