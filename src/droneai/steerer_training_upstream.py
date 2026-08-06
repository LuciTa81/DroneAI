"""Pinned STEERER training-source and backbone-only initialization gates."""

from __future__ import annotations

import copy
import hashlib
import random
import re
import runpy
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Iterator, Literal, Mapping

import numpy as np

from droneai.dm_count_adapter import _git_head, _git_status
from droneai.integrity import is_sha256, sha256_file
from droneai.steerer_training_profile import SteererTrainingProfile


Stage = Literal["T0", "T1", "T5", "T50"]

_STAGE_STOP_EPOCH = {"T0": 0, "T1": 1, "T5": 5, "T50": 50}
_SCHEDULE_HORIZON = 800
_TRAIN_WORKERS = 6
_OFFICIAL_NETWORK = {
    "backbone": "MocHRBackbone",
    "sub_arch": "hrnet48",
    "counter_type": "withMOE",
    "resolution_num": [0, 1, 2, 3],
    "loss_weight": [1.0, 0.5, 0.25, 0.125],
    "sigma": [4],
    "gau_kernel_size": 15,
    "baseline_loss": False,
    "head": {
        "type": "CountingHead",
        "fuse_method": "cat",
        "in_channels": 96,
        "stages_channel": [384, 192, 96, 48],
        "inter_layer": [64, 32, 16],
        "out_channels": 1,
    },
}
_OFFICIAL_OPTIMIZER = {
    "NAME": "adamw",
    "BASE_LR": 1e-4,
    "BETAS": (0.9, 0.999),
    "WEIGHT_DECAY": 1e-4,
    "EPS": 1.0e-8,
    "MOMENTUM": 0.9,
    "AMSGRAD": False,
    "NESTEROV": True,
}
_OFFICIAL_LR_CONFIG = {
    "NAME": "cosine",
    "WARMUP_METHOD": "linear",
    "DECAY_EPOCHS": 250,
    "DECAY_RATE": 0.1,
    "WARMUP_EPOCHS": 10,
    "WARMUP_LR": 5.0e-7,
    "MIN_LR": 1.0e-7,
}


@dataclass(frozen=True)
class UpstreamAudit:
    commit: str
    clean: bool
    license_sha256: str
    config_sha256: str


@dataclass(frozen=True)
class ModelInitializationAudit:
    loaded_scope: str
    model_checkpoint_loaded: bool
    loaded_backbone_keys: tuple[str, ...]
    unmatched_backbone_keys: tuple[str, ...]
    randomly_initialized_non_backbone_keys: tuple[str, ...]
    random_head_parameter_count: int

    def to_json_dict(self) -> dict[str, object]:
        """Return the JSON-ready initialization evidence."""

        return asdict(self)


def audit_upstream(
    upstream_dir: str | Path,
    *,
    expected_commit: str,
    expected_license_sha256: str,
) -> UpstreamAudit:
    """Require the exact clean checkout and record reviewed source hashes."""

    root = Path(upstream_dir)
    if not root.is_dir():
        raise ValueError("pinned STEERER upstream directory is required")
    expected_commit = expected_commit.lower()
    if len(expected_commit) != 40 or any(c not in "0123456789abcdef" for c in expected_commit):
        raise ValueError("pinned STEERER commit must be a 40-character Git hash")
    if not is_sha256(expected_license_sha256):
        raise ValueError("pinned STEERER LICENSE SHA-256 is required")

    observed_commit = _git_head(root)
    if observed_commit != expected_commit:
        raise ValueError(
            f"pinned STEERER commit mismatch: expected {expected_commit}, got {observed_commit}"
        )
    if _git_status(root):
        raise ValueError("pinned STEERER working tree must be clean")

    license_path = root / "LICENSE"
    config_path = root / "configs" / "QNRF_final.py"
    if not license_path.is_file() or not config_path.is_file():
        raise ValueError("pinned STEERER LICENSE and configs/QNRF_final.py are required")
    license_sha256 = sha256_file(license_path)
    if license_sha256 != expected_license_sha256.lower():
        raise ValueError("pinned STEERER LICENSE SHA-256 mismatch")
    return UpstreamAudit(
        commit=observed_commit,
        clean=True,
        license_sha256=license_sha256,
        config_sha256=sha256_file(config_path),
    )


def _official_config(upstream_dir: Path) -> dict[str, object]:
    config_path = upstream_dir / "configs" / "QNRF_final.py"
    if not config_path.is_file():
        raise ValueError("official STEERER configs/QNRF_final.py is required")
    namespace = runpy.run_path(str(config_path))
    return {
        key: copy.deepcopy(value)
        for key, value in namespace.items()
        if not key.startswith("__")
    }


def _require_mapping(config: Mapping[str, object], key: str) -> dict[str, Any]:
    value = config.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"official STEERER {key} config must be a mapping")
    return value


def _assert_official_contract(config: Mapping[str, object]) -> None:
    network = _require_mapping(config, "network")
    architecture = {key: network.get(key) for key in _OFFICIAL_NETWORK}
    if architecture != _OFFICIAL_NETWORK:
        raise ValueError("official STEERER network architecture does not match the approved profile")

    dataset = _require_mapping(config, "dataset")
    train = _require_mapping(config, "train")
    optimizer = _require_mapping(config, "optimizer")
    lr_config = _require_mapping(config, "lr_config")
    checks = (
        (config.get("seed") == 3035, "seed"),
        (dataset.get("name") == "QNRF", "dataset architecture"),
        (dataset.get("den_factor") == 100, "density factor"),
        (optimizer == _OFFICIAL_OPTIMIZER, "optimizer"),
        (lr_config == _OFFICIAL_LR_CONFIG, "scheduler"),
        (train.get("image_size") == (768, 768), "crop"),
        (train.get("flip") is True, "flip"),
        (train.get("multi_scale") is True, "multi-scale augmentation"),
        (train.get("scale_factor") == (0.5, 2.0), "scale range"),
        (train.get("end_epoch") == _SCHEDULE_HORIZON, "end_epoch schedule horizon"),
    )
    for accepted, name in checks:
        if not accepted:
            raise ValueError(f"official STEERER {name} does not match the approved profile")


def _safe_run_id(run_id: str) -> str:
    if (
        not isinstance(run_id, str)
        or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", run_id) is None
        or run_id in {".", ".."}
    ):
        raise ValueError("run_id must be one safe, non-empty path component")
    return run_id


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.resolve(strict=False).relative_to(root.resolve(strict=False))
    except ValueError:
        return False
    return True


def synthesize_official_config(
    *,
    profile: SteererTrainingProfile,
    upstream_dir: Path,
    processed_root: Path,
    backbone_path: Path,
    stage: Stage,
    physical_batch: int,
    accumulation_steps: int,
    run_id: str,
    resume_path: Path | None = None,
) -> dict[str, object]:
    """Validate then narrowly adapt the pinned official QNRF MMCV config."""

    if not isinstance(stage, str) or stage not in _STAGE_STOP_EPOCH:
        raise ValueError("stage must be one of T0, T1, T5, or T50")
    run_id = _safe_run_id(run_id)
    if (
        not isinstance(physical_batch, int)
        or isinstance(physical_batch, bool)
        or not isinstance(accumulation_steps, int)
        or isinstance(accumulation_steps, bool)
        or physical_batch <= 0
        or accumulation_steps <= 0
        or physical_batch * accumulation_steps != 8
        or (physical_batch, accumulation_steps) not in {(8, 1), (4, 2)}
    ):
        raise ValueError("physical batch and accumulation must be approved 8x1 or 4x2")

    config = _official_config(Path(upstream_dir))
    _assert_official_contract(config)

    result_dir = profile.result_root / run_id
    checkpoint_run_dir = profile.checkpoint_root / run_id
    resolved_resume: Path | None = None
    if resume_path is not None:
        if stage in {"T0", "T1"}:
            raise ValueError("T0 and T1 cannot use a resume checkpoint")
        resolved_resume = Path(resume_path).resolve(strict=False)
        if not _is_within(resolved_resume, checkpoint_run_dir):
            raise ValueError("resume checkpoint must be inside the current approved run directory")

    network = _require_mapping(config, "network")
    dataset = _require_mapping(config, "dataset")
    train = _require_mapping(config, "train")
    network["pretrained_backbone"] = str(Path(backbone_path))
    dataset["root"] = str(Path(processed_root))
    dataset["train_set"] = "train.txt"
    dataset["test_set"] = "val.txt"
    config["gpus"] = (0,)
    config["workers"] = _TRAIN_WORKERS
    config["log_dir"] = str(result_dir)
    train["batch_size_per_gpu"] = physical_batch
    train["resume_path"] = str(resolved_resume) if resolved_resume is not None else None

    if "test" in str(dataset["test_set"]).lower():
        raise ValueError("sealed Test data cannot be selected for training validation")
    if train.get("end_epoch") != _SCHEDULE_HORIZON:
        raise ValueError("official STEERER end_epoch schedule horizon must remain 800")

    config["droneai"] = {
        "stage": stage,
        "stage_stop_epoch": _STAGE_STOP_EPOCH[stage],
        "schedule_horizon_epochs": _SCHEDULE_HORIZON,
        "physical_batch": physical_batch,
        "accumulation_steps": accumulation_steps,
        "effective_batch": physical_batch * accumulation_steps,
        "result_dir": str(result_dir),
        "checkpoint_run_dir": str(checkpoint_run_dir),
        "model_checkpoint_loaded": False,
    }
    return config


def _optional_torch() -> Any | None:
    try:
        import torch
    except ImportError:
        return None
    return torch


def _reset_rngs(seed: int, torch_module: Any | None) -> None:
    random.seed(seed)
    np.random.seed(seed)
    if torch_module is None:
        return
    torch_module.manual_seed(seed)
    cuda = getattr(torch_module, "cuda", None)
    if cuda is not None and hasattr(cuda, "manual_seed_all"):
        cuda.manual_seed_all(seed)


def _resolved_weight_path(source: object) -> Path:
    if not isinstance(source, (str, bytes, Path)):
        raise PermissionError("STEERER constructor attempted to read a non-path weight source")
    return Path(source).resolve(strict=False)


@contextmanager
def _weight_read_gate(
    torch_module: Any | None, approved_backbone: Path
) -> Iterator[list[Path]]:
    observed: list[Path] = []
    if torch_module is None or not hasattr(torch_module, "load"):
        yield observed
        return
    original_load = torch_module.load

    def audited_load(source: object, *args: object, **kwargs: object) -> object:
        path = _resolved_weight_path(source)
        observed.append(path)
        if path != approved_backbone:
            raise PermissionError("STEERER constructor attempted to read another weight")
        return original_load(source, *args, **kwargs)

    torch_module.load = audited_load
    try:
        yield observed
    finally:
        torch_module.load = original_load


def _parameter_bytes(parameter: object) -> bytes:
    value: Any = parameter
    for method_name in ("detach", "cpu", "contiguous"):
        method = getattr(value, method_name, None)
        if callable(method):
            value = method()
    numpy_method = getattr(value, "numpy", None)
    if callable(numpy_method):
        value = numpy_method()
    array = np.asarray(value)
    return array.tobytes(order="C")


def _parameter_digests(model: object) -> dict[str, str]:
    named_parameters = getattr(model, "named_parameters", None)
    if not callable(named_parameters):
        raise TypeError("STEERER model must expose named_parameters()")
    digests: dict[str, str] = {}
    for name, parameter in named_parameters():
        if not isinstance(name, str) or not name:
            raise ValueError("STEERER model returned an invalid parameter name")
        digests[name] = hashlib.sha256(_parameter_bytes(parameter)).hexdigest()
    if not digests:
        raise ValueError("STEERER model has no auditable parameters")
    return digests


def _declared_weight_paths(model: object) -> tuple[Path, ...]:
    raw = getattr(model, "loaded_weight_paths", ())
    if raw is None:
        return ()
    if isinstance(raw, (str, bytes, Path)):
        raw = (raw,)
    try:
        return tuple(_resolved_weight_path(path) for path in raw)
    except TypeError as exc:
        raise ValueError("loaded_weight_paths must be an iterable of paths") from exc


def initialize_steerer_model(
    model_factory: Callable[..., object],
    backbone_path: str | Path,
    *,
    expected_sha256: str,
    seed: int = 3035,
    torch_module: Any | None = None,
) -> ModelInitializationAudit:
    """Prove that constructor initialization changes only backbone parameters."""

    backbone = Path(backbone_path).resolve(strict=False)
    if not backbone.is_file():
        raise ValueError("verified ImageNet backbone file is required")
    if not is_sha256(expected_sha256) or sha256_file(backbone) != expected_sha256.lower():
        raise ValueError("ImageNet backbone SHA-256 mismatch")
    if not callable(model_factory):
        raise TypeError("model_factory must be callable")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed must be an integer")

    torch_module = _optional_torch() if torch_module is None else torch_module
    with _weight_read_gate(torch_module, backbone) as observed_paths:
        _reset_rngs(seed, torch_module)
        reference = model_factory(pretrained_backbone=None)
        reference_declared = _declared_weight_paths(reference)
        if reference_declared:
            raise PermissionError("random reference constructor read another weight")
        reference_digests = _parameter_digests(reference)

        _reset_rngs(seed, torch_module)
        training_model = model_factory(pretrained_backbone=backbone)
        training_digests = _parameter_digests(training_model)
        declared_paths = _declared_weight_paths(training_model)

    all_observed = tuple(observed_paths) + declared_paths
    if any(path != backbone for path in all_observed):
        raise PermissionError("STEERER constructor attempted to read another weight than the backbone")
    if set(reference_digests) != set(training_digests):
        raise ValueError("reference and backbone STEERER parameter keys differ")

    backbone_keys = tuple(sorted(key for key in reference_digests if key.startswith("backbone.")))
    non_backbone_keys = tuple(sorted(set(reference_digests).difference(backbone_keys)))
    if not backbone_keys or not non_backbone_keys:
        raise ValueError("STEERER model must expose backbone and non-backbone head parameters")
    loaded_backbone_keys = tuple(
        key for key in backbone_keys if reference_digests[key] != training_digests[key]
    )
    unmatched_backbone_keys = tuple(
        key for key in backbone_keys if reference_digests[key] == training_digests[key]
    )
    changed_non_backbone = tuple(
        key for key in non_backbone_keys if reference_digests[key] != training_digests[key]
    )
    if not loaded_backbone_keys:
        raise ValueError("verified ImageNet backbone changed no backbone parameters")
    if changed_non_backbone:
        raise ValueError(
            "backbone-only initialization changed non-backbone head parameters: "
            + ", ".join(changed_non_backbone)
        )

    return ModelInitializationAudit(
        loaded_scope="backbone_only",
        model_checkpoint_loaded=False,
        loaded_backbone_keys=loaded_backbone_keys,
        unmatched_backbone_keys=unmatched_backbone_keys,
        randomly_initialized_non_backbone_keys=non_backbone_keys,
        random_head_parameter_count=len(non_backbone_keys),
    )
