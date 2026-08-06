"""Policy gate for the approved STEERER UCF-QNRF training lane."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Any, Mapping

from droneai.integrity import is_sha256


_STORAGE_ROOT = PurePosixPath("/workspace/data")
_PROCESSED_ROOT = Path(
    "/workspace/data/datasets/ucf-qnrf-kaggle-apache/processed/steerer-training-v1"
)
_CHECKPOINT_ROOT = Path("/workspace/data/checkpoints/steerer-ucf-training")
_RESULT_ROOT = Path("/workspace/data/results/steerer-ucf-training")
_MODEL_URL = "https://github.com/taohan10200/STEERER.git"
_MODEL_COMMIT = "5b1854dbc2d280f2326d67c65515d8baf9083810"
_MODEL_LICENSE_SHA256 = "5c3649a9ac14d2839d2580710c10bdbc9c70cb6a79c07c06a3858952223b6733"
_MODEL_CONFIG_PATH = Path("configs/QNRF_final.py")
_SUCCESS_SCOPE = "PASS_COMMERCIAL_CANDIDATE"


@dataclass(frozen=True)
class ModelUpstreamReference:
    url: str
    commit: str
    license_sha256: str
    config_path: Path


@dataclass(frozen=True)
class ArtifactReference:
    path: Path
    filename: str
    source_url: str
    provenance_url: str
    sha256: str
    byte_size: int


_PINNED_IMAGENET_BACKBONE = ArtifactReference(
    path=Path("/workspace/data/checkpoints/backbones/hrnetv2_w48_imagenet_pretrained.pth"),
    filename="hrnetv2_w48_imagenet_pretrained.pth",
    source_url=(
        "https://github.com/hsfzxjy/models.storage/releases/download/"
        "openseg-pytorch-pretrained/hrnetv2_w48_imagenet_pretrained.pth"
    ),
    provenance_url=(
        "https://onedrive.live.com/?action=locate&authkey=%21AKvqI6pBZlifgJk&"
        "cid=F7FD0B7F26543CEB&id=F7FD0B7F26543CEB%21116&"
        "parId=F7FD0B7F26543CEB%21105"
    ),
    sha256="0efec102d97f2ef58f0e258b2c3076b3704b93ffc2b73f64c8da5462c0037ef8",
    byte_size=310643500,
)


@dataclass(frozen=True)
class SteererTrainingProfile:
    model_upstream: ModelUpstreamReference
    dataset_population: int
    train_count: int
    validation_count: int
    seed: int
    initialization: str
    sealed_test_access: bool
    success_scope: str
    stage_epochs: Mapping[str, int]
    imagenet_backbone: ArtifactReference
    processed_root: Path
    checkpoint_root: Path
    result_root: Path

    @property
    def model_upstream_commit(self) -> str:
        """Backward-compatible access to the authoritative upstream reference."""

        return self.model_upstream.commit


def _mapping(payload: Any, *, name: str, keys: set[str]) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError(f"{name} must be an object")
    unknown = set(payload).difference(keys)
    if unknown:
        raise ValueError(f"{name} has unknown key: {sorted(unknown)[0]}")
    missing = keys.difference(payload)
    if missing:
        raise ValueError(f"{name} is missing required key: {sorted(missing)[0]}")
    return payload


def _string(value: Any, *, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _integer(value: Any, *, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{name} must be an integer")
    return value


def _boolean(value: Any, *, name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be boolean")
    return value


def _storage_path(value: Any, *, name: str) -> Path:
    raw_path = _string(value, name=name)
    path = PurePosixPath(raw_path)
    if not path.is_absolute():
        raise ValueError(f"{name} must be an absolute home5090 path")
    if ".." in path.parts:
        raise ValueError(f"{name} must be inside /workspace/data")
    try:
        path.relative_to(_STORAGE_ROOT)
    except ValueError as exc:
        raise ValueError(f"{name} must be inside /workspace/data") from exc
    return Path(str(path))


def validate_initialization(
    *, backbone: ArtifactReference, model_checkpoint: Path | None
) -> None:
    if model_checkpoint is not None:
        raise PermissionError("model checkpoint loading is forbidden")
    if backbone != _PINNED_IMAGENET_BACKBONE:
        raise PermissionError("ImageNet backbone must match the pinned artifact")


def _artifact_reference(payload: Any) -> ArtifactReference:
    artifact = _mapping(
        payload,
        name="imagenet_backbone",
        keys={"path", "filename", "source_url", "provenance_url", "sha256", "byte_size"},
    )
    reference = ArtifactReference(
        path=_storage_path(artifact["path"], name="imagenet_backbone.path"),
        filename=_string(artifact["filename"], name="imagenet_backbone.filename"),
        source_url=_string(artifact["source_url"], name="imagenet_backbone.source_url"),
        provenance_url=_string(
            artifact["provenance_url"], name="imagenet_backbone.provenance_url"
        ),
        sha256=_string(artifact["sha256"], name="imagenet_backbone.sha256"),
        byte_size=_integer(artifact["byte_size"], name="imagenet_backbone.byte_size"),
    )
    if not reference.source_url.startswith("https://") or not is_sha256(reference.sha256):
        raise PermissionError("ImageNet backbone source URL and SHA-256 are required")
    return reference


def validate_training_profile(payload: Any) -> SteererTrainingProfile:
    root = _mapping(
        payload,
        name="training profile",
        keys={
            "schema_version", "run_family", "runtime_backend", "model_upstream", "imagenet_backbone",
            "dataset_id", "dataset_population", "split", "initialization", "training",
            "stage_epochs", "storage", "rights",
        },
    )
    if root["schema_version"] != 1:
        raise ValueError("schema_version must be 1")
    if root["run_family"] != "steerer-ucf-qnrf-imagenet-home5090":
        raise ValueError("run_family is not approved")
    if root["runtime_backend"] != "home5090_docker":
        raise ValueError("runtime_backend must be home5090_docker")
    if root["dataset_id"] != "ucf-qnrf-kaggle-apache":
        raise ValueError("dataset_id is not approved")

    upstream = _mapping(
        root["model_upstream"],
        name="model_upstream",
        keys={"url", "commit", "license_sha256", "config_path"},
    )
    source_url = _string(upstream["url"], name="model_upstream.url")
    commit = _string(upstream["commit"], name="model_upstream.commit")
    license_sha256 = _string(upstream["license_sha256"], name="model_upstream.license_sha256")
    config_path = Path(_string(upstream["config_path"], name="model_upstream.config_path"))
    model_upstream = ModelUpstreamReference(
        url=source_url,
        commit=commit,
        license_sha256=license_sha256,
        config_path=config_path,
    )
    if model_upstream != ModelUpstreamReference(
        url=_MODEL_URL,
        commit=_MODEL_COMMIT,
        license_sha256=_MODEL_LICENSE_SHA256,
        config_path=_MODEL_CONFIG_PATH,
    ):
        raise ValueError("model_upstream does not match the approved pinned source")

    backbone = _artifact_reference(root["imagenet_backbone"])

    population = _integer(root["dataset_population"], name="dataset_population")
    if population != 1201:
        raise ValueError("dataset population must be 1201")

    split = _mapping(root["split"], name="split", keys={"seed", "train", "validation", "test_role"})
    seed = _integer(split["seed"], name="split.seed")
    train_count = _integer(split["train"], name="split.train")
    validation_count = _integer(split["validation"], name="split.validation")
    if (seed, train_count, validation_count, split["test_role"]) != (3035, 961, 240, "sealed"):
        raise ValueError("split must be the approved 3035/961/240 sealed split")

    initialization = _mapping(root["initialization"], name="initialization", keys={"mode", "model_checkpoint"})
    initialization_mode = _string(initialization["mode"], name="initialization.mode")
    if initialization_mode != "imagenet_backbone_only":
        raise ValueError("initialization.mode must be imagenet_backbone_only")
    checkpoint = initialization["model_checkpoint"]
    if checkpoint is not None and not isinstance(checkpoint, str):
        raise ValueError("initialization.model_checkpoint must be null or a path")
    validate_initialization(
        backbone=backbone,
        model_checkpoint=Path(checkpoint) if checkpoint is not None else None,
    )

    training = _mapping(root["training"], name="training", keys={"crop", "scale_range", "flip", "density_factor", "optimizer", "learning_rate", "weight_decay", "warmup_epochs", "scheduler", "schedule_horizon_epochs", "effective_batch", "validation_long_side"})
    if training != {"crop": [768, 768], "scale_range": [0.5, 2.0], "flip": True, "density_factor": 100, "optimizer": "AdamW", "learning_rate": 0.0001, "weight_decay": 0.0001, "warmup_epochs": 10, "scheduler": "cosine", "schedule_horizon_epochs": 800, "effective_batch": 8, "validation_long_side": 3072}:
        raise ValueError("training settings are not approved")
    stage_epochs = _mapping(
        root["stage_epochs"],
        name="stage_epochs",
        keys={"T0", "T1", "T5", "T50", "T800"},
    )
    if stage_epochs != {"T0": 0, "T1": 1, "T5": 5, "T50": 50, "T800": 800}:
        raise ValueError("stage epochs are not approved")

    storage = _mapping(root["storage"], name="storage", keys={"processed_root", "checkpoint_root", "result_root"})
    processed_root = _storage_path(storage["processed_root"], name="storage.processed_root")
    checkpoint_root = _storage_path(storage["checkpoint_root"], name="storage.checkpoint_root")
    result_root = _storage_path(storage["result_root"], name="storage.result_root")
    if (processed_root, checkpoint_root, result_root) != (
        _PROCESSED_ROOT,
        _CHECKPOINT_ROOT,
        _RESULT_ROOT,
    ):
        raise ValueError("storage roots do not match the approved training lane")

    rights = _mapping(root["rights"], name="rights", keys={"success_scope", "production_approved", "sealed_test_access"})
    success_scope = _string(rights["success_scope"], name="rights.success_scope")
    if success_scope != _SUCCESS_SCOPE:
        raise ValueError("success scope must be PASS_COMMERCIAL_CANDIDATE")
    if _boolean(rights["production_approved"], name="rights.production_approved"):
        raise PermissionError("production approval is forbidden for this training profile")
    sealed_test_access = _boolean(rights["sealed_test_access"], name="rights.sealed_test_access")
    if sealed_test_access:
        raise PermissionError("sealed test access is forbidden for this training profile")

    return SteererTrainingProfile(
        model_upstream=model_upstream,
        dataset_population=population,
        train_count=train_count,
        validation_count=validation_count,
        seed=seed,
        initialization=initialization_mode,
        sealed_test_access=sealed_test_access,
        success_scope=success_scope,
        stage_epochs=MappingProxyType(dict(stage_epochs)),
        imagenet_backbone=backbone,
        processed_root=processed_root,
        checkpoint_root=checkpoint_root,
        result_root=result_root,
    )


def load_training_profile(path: str | Path) -> SteererTrainingProfile:
    source_path = Path(path)
    payload = json.loads(source_path.read_text(encoding="utf-8"))
    return validate_training_profile(payload)
