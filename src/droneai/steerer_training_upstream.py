"""Pinned STEERER training-source and backbone-only initialization gates."""

from __future__ import annotations

import copy
import hashlib
import random
import re
import subprocess
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterator, Literal, Mapping, Protocol
from urllib.parse import urlsplit, urlunsplit

import numpy as np

from droneai.dm_count_adapter import _git_head, _git_status
from droneai.integrity import sha256_file
from droneai.steerer_training_profile import SteererTrainingProfile


Stage = Literal["T0", "T1", "T5", "T50", "T800"]

_STAGE_STOP_EPOCH = {"T0": 0, "T1": 1, "T5": 5, "T50": 50, "T800": 800}
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
    origin_url: str
    commit: str
    clean: bool
    license_sha256: str
    config_path: Path
    config_sha256: str
    _config_bytes: bytes = field(repr=False, compare=False)


@dataclass(frozen=True)
class ModelInitializationAudit:
    loaded_scope: str
    model_checkpoint_loaded: bool
    loaded_backbone_keys: tuple[str, ...]
    unmatched_backbone_keys: tuple[str, ...]
    randomly_initialized_non_backbone_keys: tuple[str, ...]
    random_head_parameter_count: int
    observed_weight_load_paths: tuple[Path, ...]
    observed_weight_load_count: int

    def to_json_dict(self) -> dict[str, object]:
        """Return the JSON-ready initialization evidence."""

        payload = asdict(self)
        payload["observed_weight_load_paths"] = [
            str(path) for path in self.observed_weight_load_paths
        ]
        return payload


class TrainingSourceProfile(Protocol):
    """Minimum validated profile surface shared by the B and isolated A lanes."""

    model_upstream: object
    imagenet_backbone: object
    seed: int
    initialization: str


def _require_training_source_profile(profile: object) -> TrainingSourceProfile:
    upstream = getattr(profile, "model_upstream", None)
    backbone = getattr(profile, "imagenet_backbone", None)
    if (
        not all(hasattr(upstream, name) for name in ("url", "commit", "license_sha256", "config_path"))
        or not all(
            hasattr(backbone, name)
            for name in ("path", "filename", "sha256", "byte_size")
        )
        or not isinstance(getattr(profile, "seed", None), int)
        or not isinstance(getattr(profile, "initialization", None), str)
    ):
        raise TypeError("validated STEERER training source profile is required")
    return profile  # type: ignore[return-value]


def _git_remote_origin(upstream_dir: Path) -> str:
    completed = subprocess.run(
        ["git", "remote", "get-url", "origin"],
        cwd=upstream_dir,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _normalized_git_url(value: str) -> str:
    raw = value.strip().replace("\\", "/")
    scp_match = re.fullmatch(r"(?:git@)?([^/:]+):(.+)", raw)
    if scp_match and "://" not in raw:
        raw = f"https://{scp_match.group(1)}/{scp_match.group(2)}"
    parts = urlsplit(raw)
    if not parts.scheme or not parts.netloc:
        raise ValueError("pinned STEERER origin must be an absolute Git URL")
    scheme = parts.scheme.lower()
    hostname = (parts.hostname or "").lower()
    if scheme in {"ssh", "git+ssh"} and parts.username in {None, "git"}:
        scheme = "https"
    port = f":{parts.port}" if parts.port is not None else ""
    normalized_path = "/" + parts.path.strip("/")
    return urlunsplit((scheme, hostname + port, normalized_path, "", ""))


def _audited_file(root: Path, relative_path: Path, *, name: str) -> Path:
    if relative_path.is_absolute() or ".." in relative_path.parts:
        raise ValueError(f"pinned STEERER {name} path must stay inside the audited root")
    candidate = root / relative_path
    if candidate.is_symlink():
        raise ValueError(f"pinned STEERER {name} path cannot be a symlink")
    try:
        resolved = candidate.resolve(strict=True)
        resolved.relative_to(root)
    except (FileNotFoundError, ValueError) as exc:
        raise ValueError(
            f"pinned STEERER {name} path must be a file inside the audited root"
        ) from exc
    if not resolved.is_file():
        raise ValueError(f"pinned STEERER {name} path must be a regular file")
    return resolved


def audit_upstream(
    profile: TrainingSourceProfile,
    upstream_dir: str | Path,
) -> UpstreamAudit:
    """Require the profile-pinned origin, checkout, and immutable config snapshot."""

    profile = _require_training_source_profile(profile)
    try:
        root = Path(upstream_dir).resolve(strict=True)
    except FileNotFoundError as exc:
        raise ValueError("pinned STEERER upstream directory is required") from exc
    if not root.is_dir():
        raise ValueError("pinned STEERER upstream directory is required")

    expected_origin = _normalized_git_url(profile.model_upstream.url)
    observed_origin = _normalized_git_url(_git_remote_origin(root))
    if observed_origin != expected_origin:
        raise ValueError(
            f"pinned STEERER origin mismatch: expected {expected_origin}, got {observed_origin}"
        )
    observed_commit = _git_head(root)
    if observed_commit != profile.model_upstream.commit.lower():
        raise ValueError(
            "pinned STEERER commit mismatch: expected "
            f"{profile.model_upstream.commit}, got {observed_commit}"
        )
    if _git_status(root):
        raise ValueError("pinned STEERER working tree must be clean")

    license_path = _audited_file(root, Path("LICENSE"), name="LICENSE")
    config_path = _audited_file(
        root, profile.model_upstream.config_path, name="config"
    )
    license_bytes = license_path.read_bytes()
    config_bytes = config_path.read_bytes()
    license_sha256 = hashlib.sha256(license_bytes).hexdigest()
    if license_sha256 != profile.model_upstream.license_sha256.lower():
        raise ValueError("pinned STEERER LICENSE SHA-256 mismatch")
    return UpstreamAudit(
        origin_url=observed_origin,
        commit=observed_commit,
        clean=True,
        license_sha256=license_sha256,
        config_path=config_path,
        config_sha256=hashlib.sha256(config_bytes).hexdigest(),
        _config_bytes=config_bytes,
    )


def _official_config(audit: UpstreamAudit) -> dict[str, object]:
    namespace: dict[str, object] = {
        "__file__": str(audit.config_path),
        "__name__": "droneai_steerer_audited_config",
    }
    exec(
        compile(audit._config_bytes, str(audit.config_path), "exec"),
        namespace,
    )
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


def verify_imagenet_backbone(
    profile: TrainingSourceProfile, backbone_path: str | Path
) -> Path:
    """Verify the canonical ImageNet-only artifact for either approved lane."""

    profile = _require_training_source_profile(profile)
    if profile.initialization != "imagenet_backbone_only":
        raise PermissionError("official model checkpoint initialization is forbidden")
    reference = profile.imagenet_backbone
    if reference.path.name != reference.filename:
        raise ValueError("ImageNet backbone filename does not match its canonical path")
    try:
        supplied = Path(backbone_path).resolve(strict=True)
        canonical = reference.path.resolve(strict=True)
    except FileNotFoundError as exc:
        raise ValueError("canonical ImageNet backbone file is required") from exc
    if supplied != canonical:
        raise ValueError("ImageNet backbone must use the profile canonical path")
    if supplied.is_symlink() or not supplied.is_file():
        raise ValueError("canonical ImageNet backbone must be a regular non-symlink file")
    if supplied.name != reference.filename:
        raise ValueError("ImageNet backbone filename does not match the profile")
    if supplied.stat().st_size != reference.byte_size:
        raise ValueError("ImageNet backbone byte size does not match the profile")
    if sha256_file(supplied) != reference.sha256.lower():
        raise ValueError("ImageNet backbone SHA-256 does not match the profile")
    return supplied


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
        raise ValueError("stage must be one of T0, T1, T5, T50, or T800")
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

    upstream_audit = audit_upstream(profile, upstream_dir)
    backbone = verify_imagenet_backbone(profile, backbone_path)
    config = _official_config(upstream_audit)
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
    network["pretrained_backbone"] = str(backbone)
    dataset["root"] = str(Path(processed_root)).rstrip("/\\") + "/"
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
        "upstream_origin_url": upstream_audit.origin_url,
        "upstream_commit": upstream_audit.commit,
        "upstream_license_sha256": upstream_audit.license_sha256,
        "upstream_config_sha256": upstream_audit.config_sha256,
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
    if not isinstance(source, (str, Path)):
        raise PermissionError("STEERER constructor attempted to read a non-path weight source")
    return Path(source).resolve(strict=False)


@contextmanager
def _torch_load_gate(
    torch_module: Any, approved_backbone: Path | None
) -> Iterator[list[Path]]:
    observed: list[Path] = []
    if not hasattr(torch_module, "load"):
        raise RuntimeError("PyTorch torch.load is required for initialization audit")
    original_load = torch_module.load

    def audited_load(source: object, *args: object, **kwargs: object) -> object:
        path = _resolved_weight_path(source)
        observed.append(path)
        if approved_backbone is None:
            raise PermissionError(
                "random reference constructor attempted to read a checkpoint weight"
            )
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


def initialize_steerer_model(
    profile: TrainingSourceProfile,
    model_factory: Callable[..., object],
    backbone_path: str | Path,
    *,
    torch_module: Any | None = None,
) -> ModelInitializationAudit:
    """Prove that constructor initialization changes only backbone parameters."""

    profile = _require_training_source_profile(profile)
    backbone = verify_imagenet_backbone(profile, backbone_path)
    if not callable(model_factory):
        raise TypeError("model_factory must be callable")

    torch_module = _optional_torch() if torch_module is None else torch_module
    if torch_module is None:
        raise RuntimeError("PyTorch is required for STEERER initialization audit")
    with _torch_load_gate(torch_module, None):
        _reset_rngs(profile.seed, torch_module)
        reference = model_factory(pretrained_backbone=None)
        reference_digests = _parameter_digests(reference)

    with _torch_load_gate(torch_module, backbone) as observed_paths:
        _reset_rngs(profile.seed, torch_module)
        training_model = model_factory(pretrained_backbone=backbone)
        training_digests = _parameter_digests(training_model)

    if not observed_paths:
        raise ValueError("STEERER constructor did not read the approved backbone with torch.load")
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
        observed_weight_load_paths=tuple(observed_paths),
        observed_weight_load_count=len(observed_paths),
    )


def load_audited_official_config(
    profile: TrainingSourceProfile, upstream_dir: str | Path
) -> tuple[dict[str, object], UpstreamAudit]:
    """Load the exact pinned config bytes after Git/license/contract verification."""

    audited = audit_upstream(profile, upstream_dir)
    config = _official_config(audited)
    _assert_official_contract(config)
    return config, audited
