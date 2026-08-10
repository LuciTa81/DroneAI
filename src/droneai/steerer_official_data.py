"""Exact, isolated QNRF image preparation for the pinned STEERER code path."""

from __future__ import annotations

import json
import hashlib
import math
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from pathlib import PurePosixPath
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np
from PIL import Image, __version__ as PILLOW_VERSION

from droneai.integrity import is_sha256, sha256_file
from droneai.scoring import CheckResult, StageReport, score_stage
from droneai.steerer_constants import UPSTREAM_COMMIT
from droneai.steerer_training_profile import (
    ArtifactReference,
    ModelUpstreamReference,
)
from droneai.ucf_qnrf import (
    UCFQNRFRecord,
    density_band,
    read_ucf_qnrf_points,
)


@dataclass(frozen=True)
class ConvertedOfficialSample:
    sample_id: str
    prepared_id: str
    image_path: Path
    annotation_path: Path
    source_image_sha256: str
    source_annotation_sha256: str
    source_image_name: str
    source_annotation_name: str
    image_sha256: str
    annotation_sha256: str
    width: int
    height: int
    points: tuple[tuple[float, float], ...]
    boundary_corrections: int

    @property
    def count(self) -> int:
        return len(self.points)


@dataclass(frozen=True)
class OfficialTrainingDataProfile:
    run_family: str
    model_upstream: ModelUpstreamReference
    upstream_commit: str
    train_samples: int
    validation_samples: int
    test_samples: int
    seed: int
    test_selection: str
    test_access_approved: bool
    full_train_user_approved: bool
    initialization_mode: str
    model_checkpoint: None
    processed_root: PurePosixPath
    checkpoint_root: PurePosixPath
    result_root: PurePosixPath
    success_scope: str
    production_approved: bool
    official_checkpoint_training_allowed: bool
    training_settings: Mapping[str, object]
    imagenet_backbone: ArtifactReference
    dataset_license_basis: str
    code_license: str

    @property
    def initialization(self) -> str:
        """Shared initialization contract used by the audited model loader."""

        return self.initialization_mode


_A_PROCESSED_ROOT = PurePosixPath(
    "/workspace/data/datasets/ucf-qnrf-kaggle-apache/processed/"
    "steerer-official-code-reproduction-v1/full-train-test-v1"
)
_A_CHECKPOINT_ROOT = PurePosixPath(
    "/workspace/data/checkpoints/steerer-official-code-reproduction"
)
_A_RESULT_ROOT = PurePosixPath(
    "/workspace/data/results/steerer-official-code-reproduction"
)
_PINNED_BACKBONE = {
    "path": "/workspace/data/checkpoints/backbones/hrnetv2_w48_imagenet_pretrained.pth",
    "filename": "hrnetv2_w48_imagenet_pretrained.pth",
    "source_url": (
        "https://github.com/hsfzxjy/models.storage/releases/download/"
        "openseg-pytorch-pretrained/hrnetv2_w48_imagenet_pretrained.pth"
    ),
    "provenance_url": (
        "https://onedrive.live.com/?action=locate&authkey=%21AKvqI6pBZlifgJk&"
        "cid=F7FD0B7F26543CEB&id=F7FD0B7F26543CEB%21116&"
        "parId=F7FD0B7F26543CEB%21105"
    ),
    "sha256": "0efec102d97f2ef58f0e258b2c3076b3704b93ffc2b73f64c8da5462c0037ef8",
    "byte_size": 310643500,
}
_PINNED_TRAINING_SETTINGS = {
    "architecture": "STEERER-HRNet-W48-withMOE",
    "backbone": "MocHRBackbone",
    "sub_arch": "hrnet48",
    "counter_type": "withMOE",
    "resolution_num": [0, 1, 2, 3],
    "loss_weight": [1.0, 0.5, 0.25, 0.125],
    "sigma": [4],
    "gaussian_kernel_size": 15,
    "density_factor": 100,
    "crop": [768, 768],
    "route": [256, 256],
    "scale_range": [0.5, 2.0],
    "flip": True,
    "optimizer": "AdamW",
    "learning_rate": 0.0001,
    "betas": [0.9, 0.999],
    "weight_decay": 0.0001,
    "epsilon": 1e-8,
    "warmup_epochs": 10,
    "warmup_learning_rate": 5e-7,
    "scheduler": "cosine",
    "minimum_learning_rate": 1e-7,
    "end_epoch": 800,
    "batch_size_per_gpu": 8,
    "effective_batch": 8,
    "test_long_side": 3072,
    "test_batch_size": 1,
    "patch_batch_size": 16,
    "val_span": [-800, -600, -400, -200, -200, -100, -100],
}


def _exact_object(value: Any, *, keys: set[str], name: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != keys:
        raise ValueError(f"{name} keys do not match the frozen A contract")
    return value


def _workspace_data_path(value: Any, *, name: str) -> PurePosixPath:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty path")
    path = PurePosixPath(value)
    try:
        path.relative_to(PurePosixPath("/workspace/data"))
    except ValueError as error:
        raise ValueError(f"{name} must stay inside /workspace/data") from error
    return path


def validate_official_training_data_profile(
    payload: Any,
) -> OfficialTrainingDataProfile:
    root = _exact_object(
        payload,
        keys={
            "schema_version",
            "run_family",
            "runtime_backend",
            "model_upstream",
            "imagenet_backbone",
            "dataset",
            "split",
            "initialization",
            "training",
            "stage_epochs",
            "storage",
            "rights",
        },
        name="official-code training profile",
    )
    if root["schema_version"] != 1:
        raise ValueError("schema_version must be 1")
    if root["run_family"] != "steerer-ucf-qnrf-official-code-reproduction-a-home5090":
        raise ValueError("run_family is not the isolated A lane")
    if root["runtime_backend"] != "home5090_docker":
        raise ValueError("runtime backend must be home5090_docker")

    upstream = _exact_object(
        root["model_upstream"],
        keys={"url", "commit", "license", "license_sha256", "config_path"},
        name="model_upstream",
    )
    if upstream != {
        "url": "https://github.com/taohan10200/STEERER.git",
        "commit": UPSTREAM_COMMIT,
        "license": "MIT",
        "license_sha256": (
            "5c3649a9ac14d2839d2580710c10bdbc9c70cb6a79c07c06a3858952223b6733"
        ),
        "config_path": "configs/QNRF_final.py",
    }:
        raise ValueError("model_upstream does not match the pinned public code")

    backbone = _exact_object(
        root["imagenet_backbone"], keys=set(_PINNED_BACKBONE), name="imagenet_backbone"
    )
    if backbone != _PINNED_BACKBONE or not is_sha256(backbone.get("sha256")):
        raise PermissionError("ImageNet backbone does not match the pinned artifact")

    dataset = _exact_object(
        root["dataset"],
        keys={"id", "raw_root", "license_basis", "usage_scope"},
        name="dataset",
    )
    if (
        dataset["id"] != "ucf-qnrf-kaggle-apache"
        or dataset["raw_root"]
        != "/workspace/data/datasets/ucf-qnrf-kaggle-apache/raw/UCF-QNRF_ECCV18"
        or dataset["license_basis"]
        != "Kaggle Apache-2.0 listing accepted by user"
        or dataset["usage_scope"]
        != "research comparison and new-weight training candidate"
    ):
        raise PermissionError("dataset identity or accepted license basis is invalid")

    split = _exact_object(
        root["split"],
        keys={
            "seed",
            "train_split_id",
            "train_role",
            "train",
            "validation_role",
            "validation",
            "test_split_id",
            "test_role",
            "test",
            "test_selection",
            "full_train_user_approved",
            "test_access_approved",
        },
        name="split",
    )
    if split["train"] != 1201:
        raise ValueError("A reproduction requires exactly 1,201 Train samples")
    if split["validation"] != 0 or split["validation_role"] != "not_used":
        raise ValueError("A reproduction validation split must remain unused")
    if split["test"] != 334:
        raise ValueError("A reproduction requires exactly 334 Test samples")
    if (
        split["seed"] != 3035
        or split["train_split_id"] != "ucf-qnrf-official-train1201-v1"
        or split["train_role"] != "train"
        or split["test_split_id"] != "ucf-qnrf-official-test334-v1"
        or split["test_role"] != "test"
    ):
        raise ValueError("A reproduction split identity is invalid")
    if split["test_selection"] != "test_selected":
        raise ValueError("A reproduction must disclose official Test as test-selected")
    if split["full_train_user_approved"] is not True:
        raise PermissionError("full Train1201 use must be user-approved")
    if split["test_access_approved"] is not True:
        raise PermissionError("official Test access must be approved")

    initialization = _exact_object(
        root["initialization"],
        keys={"mode", "model_checkpoint", "official_steerer_checkpoint_allowed"},
        name="initialization",
    )
    if initialization["mode"] != "imagenet_backbone_only":
        raise PermissionError("initialization must use the ImageNet backbone only")
    if initialization["model_checkpoint"] is not None:
        raise PermissionError("full STEERER model checkpoint loading is forbidden")
    if initialization["official_steerer_checkpoint_allowed"] is not False:
        raise PermissionError("official STEERER checkpoint is forbidden for training")

    training = _exact_object(
        root["training"], keys=set(_PINNED_TRAINING_SETTINGS), name="training"
    )
    if training != _PINNED_TRAINING_SETTINGS:
        raise ValueError("training settings differ from pinned QNRF_final.py settings")
    stages = _exact_object(
        root["stage_epochs"], keys={"A0", "A1", "A800"}, name="stage_epochs"
    )
    if stages != {"A0": 0, "A1": 1, "A800": 800}:
        raise ValueError("A training gate epochs are invalid")

    storage = _exact_object(
        root["storage"],
        keys={"processed_root", "checkpoint_root", "result_root"},
        name="storage",
    )
    processed_root = _workspace_data_path(
        storage["processed_root"], name="storage.processed_root"
    )
    checkpoint_root = _workspace_data_path(
        storage["checkpoint_root"], name="storage.checkpoint_root"
    )
    result_root = _workspace_data_path(storage["result_root"], name="storage.result_root")
    if (processed_root, checkpoint_root, result_root) != (
        _A_PROCESSED_ROOT,
        _A_CHECKPOINT_ROOT,
        _A_RESULT_ROOT,
    ):
        raise ValueError("storage roots must use the isolated official-code A lane")

    rights = _exact_object(
        root["rights"],
        keys={
            "code_license",
            "dataset_license_basis",
            "success_scope",
            "production_approved",
            "official_checkpoint_training_allowed",
        },
        name="rights",
    )
    if rights["code_license"] != "MIT":
        raise PermissionError("model code license must remain MIT")
    if rights["dataset_license_basis"] != dataset["license_basis"]:
        raise PermissionError("dataset license basis is inconsistent")
    if rights["success_scope"] != "PASS_COMMERCIAL_CANDIDATE":
        raise PermissionError("success scope must remain PASS_COMMERCIAL_CANDIDATE")
    if rights["production_approved"] is not False:
        raise PermissionError("production approval is not granted")
    if rights["official_checkpoint_training_allowed"] is not False:
        raise PermissionError("official checkpoint training use is forbidden")

    return OfficialTrainingDataProfile(
        run_family=root["run_family"],
        model_upstream=ModelUpstreamReference(
            url=upstream["url"],
            commit=upstream["commit"],
            license_sha256=upstream["license_sha256"],
            config_path=Path(upstream["config_path"]),
        ),
        upstream_commit=upstream["commit"],
        train_samples=1201,
        validation_samples=0,
        test_samples=334,
        seed=3035,
        test_selection="test_selected",
        test_access_approved=True,
        full_train_user_approved=True,
        initialization_mode="imagenet_backbone_only",
        model_checkpoint=None,
        processed_root=processed_root,
        checkpoint_root=checkpoint_root,
        result_root=result_root,
        success_scope=rights["success_scope"],
        production_approved=False,
        official_checkpoint_training_allowed=False,
        training_settings=MappingProxyType(dict(training)),
        imagenet_backbone=ArtifactReference(
            path=Path(backbone["path"]),
            filename=backbone["filename"],
            source_url=backbone["source_url"],
            provenance_url=backbone["provenance_url"],
            sha256=backbone["sha256"],
            byte_size=backbone["byte_size"],
        ),
        dataset_license_basis=dataset["license_basis"],
        code_license=rights["code_license"],
    )


def load_official_training_data_profile(
    path: str | Path,
) -> OfficialTrainingDataProfile:
    return validate_official_training_data_profile(
        json.loads(Path(path).read_text(encoding="utf-8"))
    )


def official_qnrf_prepared_size(width: int, height: int) -> tuple[int, int]:
    """Match ``prepare_QNRF.py::resize_images`` including its next-16 behavior."""

    if width <= 0 or height <= 0:
        raise ValueError("official QNRF preparation requires positive dimensions")
    width_ratio = width / 1024.0
    height_ratio = height / 768.0
    if width_ratio < 1.0 or height_ratio < 1.0:
        if width_ratio > height_ratio:
            prepared_height = 768
            prepared_width = int(width / height_ratio)
            prepared_width = (prepared_width // 16 + 1) * 16
        else:
            prepared_height = int(height / width_ratio)
            prepared_height = (prepared_height // 16 + 1) * 16
            prepared_width = 1024
    else:
        prepared_width = (width // 16 + 1) * 16
        prepared_height = (height // 16 + 1) * 16
    return prepared_width, prepared_height


def _official_points(
    raw_points: tuple[tuple[float, float], ...],
    *,
    source_size: tuple[int, int],
    prepared_size: tuple[int, int],
) -> tuple[tuple[tuple[float, float], ...], int]:
    source_width, source_height = source_size
    prepared_width, prepared_height = prepared_size
    points = np.asarray(raw_points, dtype=np.float64).reshape((-1, 2))
    if not np.isfinite(points).all():
        raise ValueError("official QNRF points must be finite")
    if len(points):
        points *= np.asarray(
            (prepared_width / source_width, prepared_height / source_height),
            dtype=np.float64,
        )
        points = points.astype(np.int64).astype(np.float64)
        before = points.copy()
        points[:, 0] = np.clip(points[:, 0], 0, prepared_width - 1)
        points[:, 1] = np.clip(points[:, 1], 0, prepared_height - 1)
        corrections = int(np.any(points != before, axis=1).sum())
    else:
        corrections = 0
    return tuple((float(x), float(y)) for x, y in points), corrections


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _convert_official_sample(
    image_path: str | Path,
    annotation_path: str | Path,
    *,
    prepared_id: str,
    source_sample_id: str,
    source_partition: str,
    output_root: str | Path,
) -> ConvertedOfficialSample:
    source_image = Path(image_path).resolve()
    source_annotation = Path(annotation_path).resolve()
    if not source_image.is_file() or not source_annotation.is_file():
        raise FileNotFoundError("official QNRF source image and annotation are required")
    if not prepared_id.isdigit() or len(prepared_id) != 4:
        raise ValueError("prepared QNRF identity must be a four-digit number")
    expected_prefix = {"official_train": "train_img_", "official_test": "test_img_"}
    if source_partition not in expected_prefix:
        raise ValueError("source partition must be official_train or official_test")
    if not source_sample_id.startswith(expected_prefix[source_partition]):
        raise ValueError("prepared sample must preserve its source partition identity")

    root = Path(output_root)
    destination_image = root / "images" / f"{prepared_id}.jpg"
    destination_annotation = root / "jsons" / f"{prepared_id}.json"
    if destination_image.exists() or destination_annotation.exists():
        raise FileExistsError("prepared QNRF sample destination already exists")
    destination_image.parent.mkdir(parents=True, exist_ok=True)
    destination_annotation.parent.mkdir(parents=True, exist_ok=True)

    with Image.open(source_image) as opened:
        image = opened.convert("RGB")
        source_width, source_height = image.size
        prepared_width, prepared_height = official_qnrf_prepared_size(
            source_width, source_height
        )
        prepared = image.resize(
            (prepared_width, prepared_height), Image.Resampling.BILINEAR
        )
        prepared.save(destination_image, quality=95)

    raw_points = read_ucf_qnrf_points(source_annotation)
    points, corrections = _official_points(
        raw_points,
        source_size=(source_width, source_height),
        prepared_size=(prepared_width, prepared_height),
    )
    _write_json(
        destination_annotation,
        {
            "human_num": len(raw_points),
            "img_id": f"{prepared_id}.jpg",
            "points": [list(point) for point in points],
            "prepared_size": [prepared_width, prepared_height],
            "preprocessing_reference": (
                "lib/datasets/prepare/prepare_QNRF.py::resize_images/writer_jsons"
            ),
            "source_annotation_sha256": sha256_file(source_annotation),
            "source_image_sha256": sha256_file(source_image),
            "source_partition": source_partition,
            "source_sample_id": source_sample_id,
        },
    )
    return ConvertedOfficialSample(
        sample_id=source_sample_id,
        prepared_id=prepared_id,
        image_path=destination_image.resolve(),
        annotation_path=destination_annotation.resolve(),
        source_image_sha256=sha256_file(source_image),
        source_annotation_sha256=sha256_file(source_annotation),
        source_image_name=source_image.name,
        source_annotation_name=source_annotation.name,
        image_sha256=sha256_file(destination_image),
        annotation_sha256=sha256_file(destination_annotation),
        width=prepared_width,
        height=prepared_height,
        points=points,
        boundary_corrections=corrections,
    )


def convert_official_test_sample(
    image_path: str | Path,
    annotation_path: str | Path,
    *,
    prepared_id: str,
    source_sample_id: str,
    output_root: str | Path,
) -> ConvertedOfficialSample:
    return _convert_official_sample(
        image_path,
        annotation_path,
        prepared_id=prepared_id,
        source_sample_id=source_sample_id,
        source_partition="official_test",
        output_root=output_root,
    )


def _source_pairs(
    raw_root: Path, *, partition_name: str
) -> tuple[tuple[Path, Path], ...]:
    images = sorted(raw_root.glob("img_*.jpg"))
    pairs: list[tuple[Path, Path]] = []
    for image in images:
        annotation = raw_root / f"{image.stem}_ann.mat"
        if not annotation.is_file():
            raise FileNotFoundError(
                f"official {partition_name} annotation is missing: {annotation}"
            )
        pairs.append((image, annotation))
    if not pairs:
        raise ValueError(f"official {partition_name} preparation found no source images")
    expected_annotations = {annotation.resolve() for _, annotation in pairs}
    observed_annotations = {path.resolve() for path in raw_root.glob("img_*_ann.mat")}
    if observed_annotations != expected_annotations:
        raise ValueError(
            f"official {partition_name} images and annotations are not one-to-one"
        )
    return tuple(pairs)


def prepare_official_test_partition(
    raw_test_root: str | Path,
    *,
    output_root: str | Path,
    expected_samples: int = 334,
) -> Path:
    raw_root = Path(raw_test_root).resolve()
    destination = Path(output_root).resolve()
    if not raw_root.is_dir():
        raise FileNotFoundError(f"official Test source root is missing: {raw_root}")
    if isinstance(expected_samples, bool) or expected_samples <= 0:
        raise ValueError("expected Test sample count must be positive")
    pairs = _source_pairs(raw_root, partition_name="Test")
    if len(pairs) != expected_samples:
        raise ValueError(
            f"official Test preparation requires {expected_samples} samples; "
            f"observed={len(pairs)}"
        )
    if destination.exists() and (not destination.is_dir() or any(destination.iterdir())):
        raise FileExistsError(f"prepared Test destination must be new or empty: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging_parent = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent)
    )
    staging = staging_parent / "dataset"
    try:
        converted: list[ConvertedOfficialSample] = []
        for index, (image, annotation) in enumerate(pairs, start=1):
            converted.append(
                convert_official_test_sample(
                    image,
                    annotation,
                    prepared_id=f"{1201 + index:04d}",
                    source_sample_id=f"test_{image.stem}",
                    output_root=staging,
                )
            )
        (staging / "test.txt").write_text(
            "".join(f"{row.prepared_id}\n" for row in converted),
            encoding="utf-8",
        )
        samples = [
            {
                "annotation_path": row.annotation_path.relative_to(staging).as_posix(),
                "annotation_sha256": row.annotation_sha256,
                "boundary_corrections": row.boundary_corrections,
                "count": row.count,
                "height": row.height,
                "image_path": row.image_path.relative_to(staging).as_posix(),
                "image_sha256": row.image_sha256,
                "prepared_id": row.prepared_id,
                "sample_id": row.sample_id,
                "source_annotation_sha256": row.source_annotation_sha256,
                "source_image_sha256": row.source_image_sha256,
                "width": row.width,
            }
            for row in converted
        ]
        manifest_path = staging / "manifests" / "test-preparation.json"
        _write_json(
            manifest_path,
            {
                "schema_version": 1,
                "dataset_id": "ucf-qnrf-kaggle-apache",
                "partition": "official_test",
                "sample_count": len(samples),
                "source_root": str(raw_root),
                "upstream_commit": UPSTREAM_COMMIT,
                "preprocessing_reference": (
                    "lib/datasets/prepare/prepare_QNRF.py::resize_images/writer_jsons"
                ),
                "image_encoding": "Pillow bilinear JPEG quality=95",
                "pillow_version": PILLOW_VERSION,
                "test_list_sha256": sha256_file(staging / "test.txt"),
                "samples": samples,
            },
        )
        if destination.exists():
            destination.rmdir()
        os.replace(staging, destination)
        return destination / "manifests" / "test-preparation.json"
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    finally:
        staging_parent.rmdir()


def _manifest_sample(row: ConvertedOfficialSample, root: Path) -> dict[str, object]:
    return {
        "annotation_path": row.annotation_path.relative_to(root).as_posix(),
        "annotation_sha256": row.annotation_sha256,
        "boundary_corrections": row.boundary_corrections,
        "count": row.count,
        "height": row.height,
        "image_path": row.image_path.relative_to(root).as_posix(),
        "image_sha256": row.image_sha256,
        "prepared_id": row.prepared_id,
        "sample_id": row.sample_id,
        "source_annotation_sha256": row.source_annotation_sha256,
        "source_annotation_name": row.source_annotation_name,
        "source_image_sha256": row.source_image_sha256,
        "source_image_name": row.source_image_name,
        "width": row.width,
    }


def _write_partition_inventory(
    root: Path,
    *,
    partition: str,
    split_role: str,
    list_name: str,
    rows: list[ConvertedOfficialSample],
) -> Path:
    list_path = root / list_name
    samples = [_manifest_sample(row, root) for row in rows]
    inventory = root / "manifests" / f"{split_role}-inventory.json"
    _write_json(
        inventory,
        {
            "schema_version": 1,
            "dataset_id": "ucf-qnrf-kaggle-apache",
            "partition": partition,
            "split_role": split_role,
            "sample_count": len(samples),
            "list_path": list_name,
            "list_sha256": sha256_file(list_path),
            "samples": samples,
        },
    )
    return inventory


def _dataset_content_sha256(parts: list[str]) -> str:
    return hashlib.sha256("\n".join(parts).encode("utf-8")).hexdigest()


def prepare_official_train_test_lane(
    raw_train_root: str | Path,
    raw_test_root: str | Path,
    *,
    output_root: str | Path,
    expected_train_samples: int = 1201,
    expected_test_samples: int = 334,
) -> Path:
    """Prepare the user-approved full-Train/Test A lane without a validation split."""

    raw_train = Path(raw_train_root).resolve()
    raw_test = Path(raw_test_root).resolve()
    destination = Path(output_root).resolve()
    if raw_train == raw_test:
        raise ValueError("official Train and Test roots must be disjoint")
    if not raw_train.is_dir() or not raw_test.is_dir():
        raise FileNotFoundError("official Train and Test source roots are required")
    if (
        isinstance(expected_train_samples, bool)
        or isinstance(expected_test_samples, bool)
        or expected_train_samples <= 0
        or expected_test_samples <= 0
    ):
        raise ValueError("expected Train and Test counts must be positive")
    if destination.exists() and (not destination.is_dir() or any(destination.iterdir())):
        raise FileExistsError(f"prepared lane destination must be new or empty: {destination}")

    train_pairs = _source_pairs(raw_train, partition_name="Train")
    test_pairs = _source_pairs(raw_test, partition_name="Test")
    if len(train_pairs) != expected_train_samples:
        raise ValueError(
            f"official Train preparation requires {expected_train_samples} samples; "
            f"observed={len(train_pairs)}"
        )
    if len(test_pairs) != expected_test_samples:
        raise ValueError(
            f"official Test preparation requires {expected_test_samples} samples; "
            f"observed={len(test_pairs)}"
        )

    source_hashes: dict[str, str] = {}
    for partition, pairs in (("Train", train_pairs), ("Test", test_pairs)):
        for image, _ in pairs:
            digest = sha256_file(image)
            if digest in source_hashes:
                raise ValueError(
                    "source image duplicate or split overlap: "
                    f"{source_hashes[digest]} and {partition}/{image.name}"
                )
            source_hashes[digest] = f"{partition}/{image.name}"

    destination.parent.mkdir(parents=True, exist_ok=True)
    staging_parent = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent)
    )
    staging = staging_parent / "dataset"
    try:
        train_rows: list[ConvertedOfficialSample] = []
        test_rows: list[ConvertedOfficialSample] = []
        for index, (image, annotation) in enumerate(train_pairs, start=1):
            train_rows.append(
                _convert_official_sample(
                    image,
                    annotation,
                    prepared_id=f"{index:04d}",
                    source_sample_id=f"train_{image.stem}",
                    source_partition="official_train",
                    output_root=staging,
                )
            )
        for index, (image, annotation) in enumerate(test_pairs, start=1):
            test_rows.append(
                _convert_official_sample(
                    image,
                    annotation,
                    prepared_id=f"{expected_train_samples + index:04d}",
                    source_sample_id=f"test_{image.stem}",
                    source_partition="official_test",
                    output_root=staging,
                )
            )

        (staging / "train.txt").write_text(
            "".join(f"{row.prepared_id}\n" for row in train_rows), encoding="utf-8"
        )
        (staging / "test.txt").write_text(
            "".join(f"{row.prepared_id}\n" for row in test_rows), encoding="utf-8"
        )
        train_inventory = _write_partition_inventory(
            staging,
            partition="official_train",
            split_role="train",
            list_name="train.txt",
            rows=train_rows,
        )
        test_inventory = _write_partition_inventory(
            staging,
            partition="official_test",
            split_role="test",
            list_name="test.txt",
            rows=test_rows,
        )
        train_inventory_hash = sha256_file(train_inventory)
        test_inventory_hash = sha256_file(test_inventory)
        train_list_hash = sha256_file(staging / "train.txt")
        test_list_hash = sha256_file(staging / "test.txt")
        manifest = staging / "manifests" / "dataset-preparation.json"
        _write_json(
            manifest,
            {
                "schema_version": 2,
                "dataset_id": "ucf-qnrf-kaggle-apache",
                "protocol_id": "steerer-qnrf-official-code-reproduction-a-v1",
                "upstream_commit": UPSTREAM_COMMIT,
                "preprocessing_reference": (
                    "lib/datasets/prepare/prepare_QNRF.py::resize_images/writer_jsons"
                ),
                "protocol_note": (
                    "user-approved full official Train1201; no validation split; "
                    "official Test334 is test-selected"
                ),
                "image_encoding": "Pillow bilinear JPEG quality=95",
                "pillow_version": PILLOW_VERSION,
                "source_roots": {"train": str(raw_train), "test": str(raw_test)},
                "total_samples": len(train_rows) + len(test_rows),
                "train": {
                    "partition": "official_train",
                    "split_role": "train",
                    "sample_count": len(train_rows),
                    "list_path": "train.txt",
                    "list_sha256": train_list_hash,
                    "inventory_path": "manifests/train-inventory.json",
                    "inventory_sha256": train_inventory_hash,
                },
                "validation": {
                    "split_role": "not_used",
                    "sample_count": 0,
                    "list_path": None,
                },
                "test": {
                    "partition": "official_test",
                    "split_role": "test",
                    "selection": "test_selected",
                    "sample_count": len(test_rows),
                    "list_path": "test.txt",
                    "list_sha256": test_list_hash,
                    "inventory_path": "manifests/test-inventory.json",
                    "inventory_sha256": test_inventory_hash,
                },
                "split_overlap": [],
                "dataset_content_sha256": _dataset_content_sha256(
                    [
                        train_inventory_hash,
                        test_inventory_hash,
                        train_list_hash,
                        test_list_hash,
                    ]
                ),
            },
        )
        if destination.exists():
            destination.rmdir()
        os.replace(staging, destination)
        return destination / "manifests" / "dataset-preparation.json"
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    finally:
        staging_parent.rmdir()


def _verify_partition_inventory(
    root: Path,
    *,
    partition_manifest: dict[str, object],
    source_root: Path,
    expected_partition: str,
    expected_role: str,
    expected_ids: tuple[str, ...],
) -> tuple[dict[str, object], ...]:
    inventory = _contained_file(
        root, partition_manifest.get("inventory_path"), label="inventory"
    )
    if sha256_file(inventory) != partition_manifest.get("inventory_sha256"):
        raise ValueError("prepared inventory hash verification failed")
    payload = json.loads(inventory.read_text(encoding="utf-8"))
    if (
        not isinstance(payload, dict)
        or payload.get("schema_version") != 1
        or payload.get("dataset_id") != "ucf-qnrf-kaggle-apache"
        or payload.get("partition") != expected_partition
        or payload.get("split_role") != expected_role
        or payload.get("sample_count") != len(expected_ids)
    ):
        raise ValueError("prepared partition inventory identity is invalid")
    list_path = _contained_file(root, payload.get("list_path"), label="split list")
    if (
        sha256_file(list_path) != payload.get("list_sha256")
        or sha256_file(list_path) != partition_manifest.get("list_sha256")
        or tuple(list_path.read_text(encoding="utf-8").split()) != expected_ids
    ):
        raise ValueError("prepared split list hash or identities are invalid")
    raw_samples = payload.get("samples")
    if not isinstance(raw_samples, list) or len(raw_samples) != len(expected_ids):
        raise ValueError("prepared partition inventory is incomplete")

    samples: list[dict[str, object]] = []
    for expected_id, raw in zip(expected_ids, raw_samples):
        if not isinstance(raw, dict) or raw.get("prepared_id") != expected_id:
            raise ValueError("prepared sample identity is invalid")
        image = _contained_file(root, raw.get("image_path"), label="image")
        annotation = _contained_file(
            root, raw.get("annotation_path"), label="annotation"
        )
        if (
            sha256_file(image) != raw.get("image_sha256")
            or sha256_file(annotation) != raw.get("annotation_sha256")
        ):
            raise ValueError("prepared artifact hash verification failed")
        source_image = _contained_file(
            source_root, raw.get("source_image_name"), label="source image"
        )
        source_annotation = _contained_file(
            source_root,
            raw.get("source_annotation_name"),
            label="source annotation",
        )
        if (
            sha256_file(source_image) != raw.get("source_image_sha256")
            or sha256_file(source_annotation) != raw.get("source_annotation_sha256")
        ):
            raise ValueError("prepared source artifact hash verification failed")
        annotation_payload = json.loads(annotation.read_text(encoding="utf-8"))
        points = annotation_payload.get("points")
        width, height = raw.get("width"), raw.get("height")
        if (
            not isinstance(points, list)
            or len(points) != raw.get("count")
            or annotation_payload.get("human_num") != raw.get("count")
            or annotation_payload.get("img_id") != f"{expected_id}.jpg"
            or isinstance(width, bool)
            or not isinstance(width, int)
            or isinstance(height, bool)
            or not isinstance(height, int)
            or width <= 0
            or height <= 0
            or any(
                not isinstance(point, list)
                or len(point) != 2
                or not (0 <= float(point[0]) < width and 0 <= float(point[1]) < height)
                for point in points
            )
        ):
            raise ValueError("prepared annotation content is invalid")
        samples.append(raw)
    return tuple(samples)


def verify_prepared_official_train_test_lane(
    prepared_root: str | Path,
    *,
    manifest_path: str | Path,
    expected_train_samples: int = 1201,
    expected_test_samples: int = 334,
) -> dict[str, object]:
    root = Path(prepared_root).resolve()
    manifest = Path(manifest_path).resolve()
    if not root.is_dir() or not manifest.is_file():
        raise FileNotFoundError("prepared Train/Test root and manifest are required")
    try:
        manifest.relative_to(root)
    except ValueError as error:
        raise ValueError("prepared lane manifest must stay inside its dataset root") from error
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    if (
        not isinstance(payload, dict)
        or payload.get("schema_version") != 2
        or payload.get("dataset_id") != "ucf-qnrf-kaggle-apache"
        or payload.get("protocol_id")
        != "steerer-qnrf-official-code-reproduction-a-v1"
        or payload.get("upstream_commit") != UPSTREAM_COMMIT
    ):
        raise ValueError("prepared lane manifest identity is invalid")
    train = payload.get("train")
    validation = payload.get("validation")
    test = payload.get("test")
    if not all(isinstance(item, dict) for item in (train, validation, test)):
        raise ValueError("prepared lane split manifests are invalid")
    assert isinstance(train, dict) and isinstance(validation, dict) and isinstance(test, dict)
    if (
        train.get("sample_count") != expected_train_samples
        or train.get("split_role") != "train"
        or test.get("sample_count") != expected_test_samples
        or test.get("split_role") != "test"
        or test.get("selection") != "test_selected"
        or validation
        != {"split_role": "not_used", "sample_count": 0, "list_path": None}
        or payload.get("total_samples") != expected_train_samples + expected_test_samples
    ):
        raise ValueError("prepared lane split counts or roles are invalid")
    if any((root / name).exists() for name in ("val.txt", "validation.txt", "train_val.txt")):
        raise ValueError("prepared A lane must not contain a validation split")
    source_roots = payload.get("source_roots")
    if not isinstance(source_roots, dict):
        raise ValueError("prepared source roots are missing")
    raw_train = Path(str(source_roots.get("train"))).resolve()
    raw_test = Path(str(source_roots.get("test"))).resolve()
    if not raw_train.is_dir() or not raw_test.is_dir() or raw_train == raw_test:
        raise ValueError("prepared source roots are invalid or overlapping")

    train_ids = tuple(f"{index:04d}" for index in range(1, expected_train_samples + 1))
    test_ids = tuple(
        f"{index:04d}"
        for index in range(
            expected_train_samples + 1,
            expected_train_samples + expected_test_samples + 1,
        )
    )
    train_samples = _verify_partition_inventory(
        root,
        partition_manifest=train,
        source_root=raw_train,
        expected_partition="official_train",
        expected_role="train",
        expected_ids=train_ids,
    )
    test_samples = _verify_partition_inventory(
        root,
        partition_manifest=test,
        source_root=raw_test,
        expected_partition="official_test",
        expected_role="test",
        expected_ids=test_ids,
    )
    overlap = sorted(set(train_ids) & set(test_ids))
    sample_ids = [str(row.get("sample_id")) for row in (*train_samples, *test_samples)]
    source_hashes = [
        str(row.get("source_image_sha256")) for row in (*train_samples, *test_samples)
    ]
    if overlap or len(sample_ids) != len(set(sample_ids)):
        raise ValueError("prepared Train/Test sample identities overlap")
    if len(source_hashes) != len(set(source_hashes)):
        raise ValueError("prepared Train/Test source image hashes overlap")
    content_hash = _dataset_content_sha256(
        [
            str(train.get("inventory_sha256")),
            str(test.get("inventory_sha256")),
            str(train.get("list_sha256")),
            str(test.get("list_sha256")),
        ]
    )
    if payload.get("dataset_content_sha256") != content_hash:
        raise ValueError("prepared dataset content hash verification failed")
    return {
        "schema_version": 1,
        "status": "verified",
        "dataset_id": "ucf-qnrf-kaggle-apache",
        "protocol_id": payload["protocol_id"],
        "train_samples": len(train_samples),
        "validation_samples": 0,
        "test_samples": len(test_samples),
        "split_overlap": overlap,
        "artifact_hashes_verified": True,
        "dataset_content_sha256": content_hash,
        "manifest_sha256": sha256_file(manifest),
    }


def score_g2_prepared_data(
    profile: OfficialTrainingDataProfile,
    verification: Mapping[str, object],
    *,
    evidence_path: str,
) -> StageReport:
    """Score experiment integrity for G2 independently of model accuracy."""

    exact_counts = (
        verification.get("train_samples") == profile.train_samples
        and verification.get("validation_samples") == profile.validation_samples
        and verification.get("test_samples") == profile.test_samples
    )
    no_overlap = verification.get("split_overlap") == []
    artifact_hashes = verification.get("artifact_hashes_verified") is True
    content_hash = verification.get("dataset_content_sha256")
    manifest_hash = verification.get("manifest_sha256")
    hashes_valid = is_sha256(content_hash) and is_sha256(manifest_hash)
    rights_ready = (
        profile.code_license == "MIT"
        and profile.dataset_license_basis
        == "Kaggle Apache-2.0 listing accepted by user"
        and profile.success_scope == "PASS_COMMERCIAL_CANDIDATE"
        and not profile.production_approved
        and not profile.official_checkpoint_training_allowed
        and profile.initialization_mode == "imagenet_backbone_only"
        and profile.model_checkpoint is None
    )
    return score_stage(
        stage_id="steerer-a-g2-official-train-test-data",
        stage_name="STEERER A full official Train/Test preparation",
        threshold=100,
        success_status="PASS_COMMERCIAL_CANDIDATE",
        checks=(
            CheckResult(
                check_id="exact-split-counts",
                category="split",
                description="Train1201, validation0, Test334 are exact",
                weight=20,
                passed=exact_counts,
                blocker=True,
                expected="1201/0/334",
                observed=(
                    f"{verification.get('train_samples')}/"
                    f"{verification.get('validation_samples')}/"
                    f"{verification.get('test_samples')}"
                ),
                evidence=evidence_path,
            ),
            CheckResult(
                check_id="split-disjointness",
                category="split",
                description="Train and Test identities/content are disjoint",
                weight=20,
                passed=no_overlap,
                blocker=True,
                expected="no overlap",
                observed=str(verification.get("split_overlap")),
                evidence=evidence_path,
            ),
            CheckResult(
                check_id="artifact-hash-verification",
                category="integrity",
                description="Every prepared image and annotation hash was recalculated",
                weight=20,
                passed=artifact_hashes,
                blocker=True,
                expected="all verified",
                observed=str(verification.get("artifact_hashes_verified")),
                evidence=evidence_path,
            ),
            CheckResult(
                check_id="content-addressed-inventories",
                category="integrity",
                description="Dataset and manifest SHA-256 identities are valid",
                weight=20,
                passed=hashes_valid,
                blocker=True,
                expected="two valid SHA-256 values",
                observed=f"dataset={content_hash}; manifest={manifest_hash}",
                evidence=evidence_path,
            ),
            CheckResult(
                check_id="rights-and-initialization-scope",
                category="rights",
                description=(
                    "MIT code, user-accepted dataset basis, ImageNet-only initialization, "
                    "and no production approval"
                ),
                weight=20,
                passed=rights_ready,
                blocker=True,
                expected="PASS_COMMERCIAL_CANDIDATE without production approval",
                observed=(
                    f"{profile.success_scope}; production={profile.production_approved}; "
                    "official checkpoint forbidden"
                ),
                evidence=evidence_path,
            ),
        ),
    )


def _contained_file(root: Path, relative: Any, *, label: str) -> Path:
    if not isinstance(relative, str) or not relative.strip():
        raise ValueError(f"prepared {label} path is missing")
    target = (root / relative).resolve()
    try:
        target.relative_to(root)
    except ValueError as error:
        raise ValueError(f"prepared {label} path escapes the dataset root") from error
    if not target.is_file():
        raise FileNotFoundError(f"prepared {label} is missing: {target}")
    return target


def load_prepared_official_test(
    prepared_root: str | Path,
    *,
    manifest_path: str | Path,
    expected_samples: int = 334,
) -> tuple[UCFQNRFRecord, ...]:
    root = Path(prepared_root).resolve()
    manifest_file = Path(manifest_path).resolve()
    if not root.is_dir() or not manifest_file.is_file():
        raise FileNotFoundError("prepared Test root and manifest are required")
    try:
        manifest_file.relative_to(root)
    except ValueError as error:
        raise ValueError("prepared Test manifest must stay inside its dataset root") from error
    payload = json.loads(manifest_file.read_text(encoding="utf-8"))
    if (
        not isinstance(payload, dict)
        or payload.get("schema_version") != 1
        or payload.get("dataset_id") != "ucf-qnrf-kaggle-apache"
        or payload.get("partition") != "official_test"
        or payload.get("upstream_commit") != UPSTREAM_COMMIT
        or payload.get("sample_count") != expected_samples
    ):
        raise ValueError("prepared Test manifest identity or sample count is invalid")
    raw_samples = payload.get("samples")
    if not isinstance(raw_samples, list) or len(raw_samples) != expected_samples:
        raise ValueError("prepared Test manifest has incomplete samples")
    records: list[UCFQNRFRecord] = []
    for raw in raw_samples:
        if not isinstance(raw, dict):
            raise ValueError("prepared Test sample manifest rows must be objects")
        image = _contained_file(root, raw.get("image_path"), label="image")
        annotation = _contained_file(
            root, raw.get("annotation_path"), label="annotation"
        )
        if (
            sha256_file(image) != raw.get("image_sha256")
            or sha256_file(annotation) != raw.get("annotation_sha256")
        ):
            raise ValueError("prepared Test artifact hash verification failed")
        annotation_payload = json.loads(annotation.read_text(encoding="utf-8"))
        points = tuple(
            (float(point[0]), float(point[1]))
            for point in annotation_payload.get("points", [])
        )
        width = raw.get("width")
        height = raw.get("height")
        if (
            isinstance(width, bool)
            or not isinstance(width, int)
            or isinstance(height, bool)
            or not isinstance(height, int)
            or width <= 0
            or height <= 0
            or len(points) != raw.get("count")
            or any(not (0 <= x < width and 0 <= y < height) for x, y in points)
        ):
            raise ValueError("prepared Test dimensions, points, or count are invalid")
        records.append(
            UCFQNRFRecord(
                sample_id=str(raw.get("sample_id")),
                image_path=image,
                annotation_path=annotation,
                image_sha256=str(raw.get("image_sha256")),
                annotation_sha256=str(raw.get("annotation_sha256")),
                width=width,
                height=height,
                points=points,
                density_band=density_band(len(points)),
                normalization_corrections=int(raw.get("boundary_corrections", 0)),
            )
        )
    if len({record.sample_id for record in records}) != expected_samples:
        raise ValueError("prepared Test sample identities must be unique")
    return tuple(records)
