"""Isolated A-lane training gates for the official STEERER QNRF code path."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence

import numpy as np

from droneai.integrity import is_sha256, sha256_file
from droneai.scoring import CheckResult, StageReport, score_stage
from droneai.steerer_official_data import OfficialTrainingDataProfile
from droneai.steerer_training_checkpoint import (
    CHECKPOINT_SCHEMA_VERSION,
    capture_rng_state,
    load_training_checkpoint,
    restore_rng_state,
    save_training_checkpoint,
)
from droneai.steerer_training_upstream import (
    ModelInitializationAudit,
    UpstreamAudit,
    initialize_steerer_model,
    load_audited_official_config,
    verify_imagenet_backbone,
)


_SEED = 3035
_PHYSICAL_BATCH = 8
_ACCUMULATION_STEPS = 1
_SCHEDULE_HORIZON = 800


@dataclass(frozen=True)
class A0Lineage:
    profile_sha256: str
    split_sha256s: Mapping[str, str]
    dataset_manifest_sha256: str
    dataset_content_sha256: str
    backbone_sha256: str
    upstream_commit: str
    upstream_config_sha256: str
    project_commit: str
    container_image_digest: str

    def __post_init__(self) -> None:
        for field in (
            "profile_sha256",
            "dataset_manifest_sha256",
            "dataset_content_sha256",
            "backbone_sha256",
            "upstream_config_sha256",
        ):
            if not is_sha256(getattr(self, field)):
                raise ValueError(f"A0 lineage {field} must be a SHA-256")
        if set(self.split_sha256s) != {"train", "test"} or not all(
            is_sha256(value) for value in self.split_sha256s.values()
        ):
            raise ValueError("A0 lineage requires exact Train and Test split hashes")
        for field in ("upstream_commit", "project_commit"):
            value = getattr(self, field)
            if not isinstance(value, str) or len(value) != 40 or any(
                char not in "0123456789abcdefABCDEF" for char in value
            ):
                raise ValueError(f"A0 lineage {field} must be a full Git commit")
        if (
            not isinstance(self.container_image_digest, str)
            or not self.container_image_digest.startswith("sha256:")
            or not is_sha256(self.container_image_digest.removeprefix("sha256:"))
        ):
            raise ValueError("A0 lineage container image digest is invalid")


@dataclass(frozen=True)
class A0UpdateObservation:
    optimizer_steps: int
    loss: float
    density_values: tuple[float, ...]
    gradient_norm: float
    learning_rate: float
    elapsed_seconds: float
    peak_vram_mb: float
    train_sample_tokens: tuple[str, ...]


@dataclass(frozen=True)
class A0GateResult:
    report: StageReport
    manifest_path: Path
    checkpoint_path: Path
    checkpoint_sha256: str
    metrics_path: Path
    environment_path: Path


class A0Runtime(Protocol):
    torch_module: object
    global_step: int
    test_access_count: int
    initialization_audit: Mapping[str, object]
    cuda_evidence: Mapping[str, object]

    def run_update(self) -> A0UpdateObservation: ...

    def checkpoint_state(self) -> dict[str, object]: ...

    def restore_checkpoint_state(self, state: dict[str, object]) -> None: ...

    def state_sha256s(self) -> dict[str, str]: ...


def _hash_value(digest: Any, value: object) -> None:
    if value is None:
        digest.update(b"N")
    elif isinstance(value, bool):
        digest.update(b"B1" if value else b"B0")
    elif isinstance(value, int):
        digest.update(b"I" + str(value).encode("ascii"))
    elif isinstance(value, float):
        digest.update(b"F" + value.hex().encode("ascii"))
    elif isinstance(value, str):
        encoded = value.encode("utf-8")
        digest.update(b"S" + len(encoded).to_bytes(8, "big") + encoded)
    elif isinstance(value, (bytes, bytearray, memoryview)):
        encoded = bytes(value)
        digest.update(b"Y" + len(encoded).to_bytes(8, "big") + encoded)
    elif isinstance(value, Mapping):
        digest.update(b"M")
        for key in sorted(value, key=lambda item: repr(item)):
            _hash_value(digest, key)
            _hash_value(digest, value[key])
    elif isinstance(value, (list, tuple)):
        digest.update(b"L" if isinstance(value, list) else b"T")
        for item in value:
            _hash_value(digest, item)
    else:
        tensor = value
        for method_name in ("detach", "cpu", "contiguous"):
            method = getattr(tensor, method_name, None)
            if callable(method):
                tensor = method()
        numpy_method = getattr(tensor, "numpy", None)
        if callable(numpy_method):
            tensor = numpy_method()
        try:
            array = np.asarray(tensor)
        except Exception as exc:  # pragma: no cover - defensive boundary
            raise TypeError(f"unsupported checkpoint state value: {type(value)!r}") from exc
        if array.dtype == object:
            raise TypeError(f"unsupported object checkpoint state value: {type(value)!r}")
        digest.update(b"A")
        _hash_value(digest, str(array.dtype))
        _hash_value(digest, tuple(int(part) for part in array.shape))
        encoded = array.tobytes(order="C")
        digest.update(len(encoded).to_bytes(8, "big") + encoded)


def state_sha256(value: object) -> str:
    """Hash nested optimizer/model/RNG state without relying on pickle identity."""

    digest = hashlib.sha256()
    _hash_value(digest, value)
    return digest.hexdigest()


def initialize_cuda_memory_stats(torch_module: object, device: object) -> int:
    """Initialize the CUDA context before the NGC nightly memory-stat reset."""

    cuda = getattr(torch_module, "cuda", None)
    if cuda is None:
        raise RuntimeError("CUDA memory statistics are unavailable")
    current_device = int(cuda.current_device())
    cuda.reset_peak_memory_stats(device)
    return current_device


def _safe_run_id(run_id: str) -> str:
    if (
        not isinstance(run_id, str)
        or not run_id
        or run_id in {".", ".."}
        or any(char not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-" for char in run_id)
    ):
        raise ValueError("A0 run_id must be one safe path component")
    return run_id


def _atomic_bytes(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        with temporary.open("xb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _json_bytes(payload: object) -> bytes:
    return (json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n").encode(
        "utf-8"
    )


def _content_addressed_json(directory: Path, stem: str, payload: object) -> tuple[Path, str]:
    encoded = _json_bytes(payload)
    digest = hashlib.sha256(encoded).hexdigest()
    path = directory / f"{stem}.{digest}.json"
    _atomic_bytes(path, encoded)
    if sha256_file(path) != digest:
        raise RuntimeError(f"{stem} evidence hash mismatch")
    return path, digest


def _validate_observation(observation: A0UpdateObservation, *, global_step: int) -> None:
    if observation.optimizer_steps != 1 or global_step != 1:
        raise RuntimeError("A0 must perform exactly one optimizer update")
    values = (
        observation.loss,
        observation.gradient_norm,
        observation.learning_rate,
        observation.elapsed_seconds,
        observation.peak_vram_mb,
        *observation.density_values,
    )
    if not observation.density_values or not all(math.isfinite(float(value)) for value in values):
        raise FloatingPointError("A0 loss, gradients, learning rate, and densities must be finite")
    if min(
        observation.loss,
        observation.gradient_norm,
        observation.learning_rate,
        observation.elapsed_seconds,
        observation.peak_vram_mb,
    ) < 0:
        raise ValueError("A0 observations must be non-negative")
    if not observation.train_sample_tokens:
        raise ValueError("A0 must record the Train samples used by its one update")


def _score_a0(
    *,
    profile: OfficialTrainingDataProfile,
    runtime: A0Runtime,
    observation: A0UpdateObservation,
    checkpoint_round_trip: bool,
    rng_round_trip: bool,
    lineage: A0Lineage,
    manifest_path: Path,
) -> StageReport:
    initialization = dict(runtime.initialization_audit)
    cuda = dict(runtime.cuda_evidence)
    return score_stage(
        stage_id="G3-A0",
        stage_name="STEERER official-code one-update smoke",
        threshold=100,
        success_status=profile.success_scope,
        checks=(
            CheckResult(
                check_id="a0.identity",
                category="provenance",
                description="Pinned source, dataset, config, backbone, and container identities",
                weight=20,
                passed=(
                    lineage.upstream_commit == profile.upstream_commit
                    and lineage.backbone_sha256 == profile.imagenet_backbone.sha256
                ),
                blocker=True,
                expected="all identities pinned by SHA/commit",
                observed=f"upstream={lineage.upstream_commit} backbone={lineage.backbone_sha256}",
                evidence=str(manifest_path),
            ),
            CheckResult(
                check_id="a0.cuda",
                category="runtime",
                description="CUDA import and finite matrix multiplication",
                weight=20,
                passed=cuda.get("available") is True and cuda.get("matmul_finite") is True,
                blocker=True,
                expected="CUDA available and matmul finite",
                observed=json.dumps(cuda, sort_keys=True),
                evidence=str(manifest_path),
            ),
            CheckResult(
                check_id="a0.update",
                category="training",
                description="Exactly one finite optimizer update from Train1201",
                weight=20,
                passed=(observation.optimizer_steps == 1 and runtime.global_step == 1),
                blocker=True,
                expected="one optimizer update; finite loss/density/gradient",
                observed=f"steps={observation.optimizer_steps} loss={observation.loss}",
                evidence=str(manifest_path),
            ),
            CheckResult(
                check_id="a0.resume_round_trip",
                category="checkpoint",
                description="Hash-verified model/optimizer/scheduler/RNG checkpoint reload",
                weight=20,
                passed=checkpoint_round_trip and rng_round_trip,
                blocker=True,
                expected="all project state and RNG reload exactly",
                observed=f"state={checkpoint_round_trip} rng={rng_round_trip}",
                evidence=str(manifest_path),
            ),
            CheckResult(
                check_id="a0.rights_and_separation",
                category="rights",
                description="ImageNet-only initialization, no official model weight, no Test access",
                weight=20,
                passed=(
                    initialization.get("loaded_scope") == "backbone_only"
                    and initialization.get("model_checkpoint_loaded") is False
                    and runtime.test_access_count == 0
                    and profile.production_approved is False
                ),
                blocker=True,
                expected="backbone-only; model checkpoint false; Test access zero",
                observed=(
                    f"scope={initialization.get('loaded_scope')} "
                    f"model_checkpoint={initialization.get('model_checkpoint_loaded')} "
                    f"test_access={runtime.test_access_count}"
                ),
                evidence=str(manifest_path),
            ),
        ),
    )


def _advance_rngs(torch_module: object) -> None:
    random.random()
    np.random.random()
    rand = getattr(torch_module, "rand", None)
    if callable(rand):
        rand(1)
        cuda = getattr(torch_module, "cuda", None)
        if cuda is not None and bool(getattr(cuda, "is_available", lambda: False)()):
            rand(1, device="cuda")


def run_a0_gate(
    profile: OfficialTrainingDataProfile,
    *,
    run_id: str,
    runtime: A0Runtime,
    lineage: A0Lineage,
    output_dir: str | Path,
    checkpoint_dir: str | Path,
    environment: Mapping[str, object],
) -> A0GateResult:
    """Run and persist only the approved one-update A0 compatibility gate."""

    if not isinstance(profile, OfficialTrainingDataProfile):
        raise TypeError("validated official-code A profile is required")
    run_id = _safe_run_id(run_id)
    result_root = Path(output_dir)
    checkpoint_root = Path(checkpoint_dir)
    if result_root.exists() or checkpoint_root.exists():
        raise FileExistsError("A0 result and checkpoint directories must be new")

    observation = runtime.run_update()
    _validate_observation(observation, global_step=runtime.global_step)
    if runtime.test_access_count != 0:
        raise PermissionError("A0 must not access validation or Test data")

    metrics_payload = {
        "schema_version": 1,
        "run_id": run_id,
        "stage": "A0",
        **asdict(observation),
    }
    metrics_path, metrics_sha256 = _content_addressed_json(
        result_root, "metrics", metrics_payload
    )
    environment_payload = {
        "schema_version": 1,
        "run_id": run_id,
        "stage": "A0",
        "runtime": dict(environment),
        "cuda": dict(runtime.cuda_evidence),
        "lineage": {**asdict(lineage), "split_sha256s": dict(lineage.split_sha256s)},
        "metrics_sha256": metrics_sha256,
    }
    environment_path, environment_sha256 = _content_addressed_json(
        result_root, "environment", environment_payload
    )

    component_state = runtime.checkpoint_state()
    if set(component_state) != {"model", "optimizer", "scheduler", "scaler"}:
        raise ValueError("A0 checkpoint state must contain model, optimizer, scheduler, and scaler")
    before_sha256s = runtime.state_sha256s()
    rng_state = capture_rng_state(_SEED, torch_module=runtime.torch_module)
    rng_before_sha256 = state_sha256(rng_state)
    payload: dict[str, object] = {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "run_id": run_id,
        "stage": "A0",
        "epoch": 0,
        "global_step": 1,
        **component_state,
        "best_mae": 1.0e20,
        "best_rmse": 1.0e20,
        "config_sha256": lineage.profile_sha256,
        "split_sha256s": dict(lineage.split_sha256s),
        "dataset_inventory_sha256": lineage.dataset_content_sha256,
        "backbone_sha256": lineage.backbone_sha256,
        "upstream_commit": lineage.upstream_commit,
        "rng": rng_state,
        "environment_manifest_sha256": environment_sha256,
    }
    checkpoint = save_training_checkpoint(
        checkpoint_root / "last.pth", payload, torch_module=runtime.torch_module
    )
    restored = load_training_checkpoint(
        checkpoint.path,
        expected_sha256=checkpoint.sha256,
        torch_module=runtime.torch_module,
    )
    runtime.restore_checkpoint_state(
        {name: restored[name] for name in ("model", "optimizer", "scheduler", "scaler")}
    )
    after_sha256s = runtime.state_sha256s()
    checkpoint_round_trip = before_sha256s == after_sha256s
    _advance_rngs(runtime.torch_module)
    restore_rng_state(restored["rng"], torch_module=runtime.torch_module)  # type: ignore[arg-type]
    rng_after_sha256 = state_sha256(
        capture_rng_state(_SEED, torch_module=runtime.torch_module)
    )
    rng_round_trip = rng_before_sha256 == rng_after_sha256
    if not checkpoint_round_trip or not rng_round_trip:
        raise RuntimeError("A0 checkpoint state or RNG round-trip failed")

    checkpoint_manifest = {
        "schema_version": 1,
        "run_id": run_id,
        "stage": "A0",
        "checkpoint": {
            "filename": checkpoint.path.name,
            "sha256": checkpoint.sha256,
            "byte_count": checkpoint.byte_size,
            "environment_sha256": environment_sha256,
        },
    }
    _atomic_bytes(
        checkpoint_root / "checkpoint-manifest.json", _json_bytes(checkpoint_manifest)
    )

    manifest_path = result_root / "a0-manifest.json"
    manifest_payload = {
        "schema_version": 1,
        "run_id": run_id,
        "stage": "A0",
        "status": "verified",
        "optimizer_steps": observation.optimizer_steps,
        "global_step": runtime.global_step,
        "finite_loss": math.isfinite(observation.loss),
        "checkpoint_round_trip": checkpoint_round_trip,
        "rng_round_trip": rng_round_trip,
        "test_access_count": runtime.test_access_count,
        "training_settings": dict(profile.training_settings),
        "initialization": dict(runtime.initialization_audit),
        "cuda": dict(runtime.cuda_evidence),
        "lineage": {**asdict(lineage), "split_sha256s": dict(lineage.split_sha256s)},
        "metrics": {"path": metrics_path.name, "sha256": metrics_sha256},
        "environment": {"path": environment_path.name, "sha256": environment_sha256},
        "checkpoint": {
            "path": str(checkpoint.path),
            "sha256": checkpoint.sha256,
            "byte_count": checkpoint.byte_size,
            "component_sha256s": before_sha256s,
        },
        "rights": {
            "success_scope": profile.success_scope,
            "production_approved": profile.production_approved,
            "dataset_license_basis": profile.dataset_license_basis,
            "code_license": profile.code_license,
        },
    }
    _atomic_bytes(manifest_path, _json_bytes(manifest_payload))
    report = _score_a0(
        profile=profile,
        runtime=runtime,
        observation=observation,
        checkpoint_round_trip=checkpoint_round_trip,
        rng_round_trip=rng_round_trip,
        lineage=lineage,
        manifest_path=manifest_path,
    )
    _atomic_bytes(result_root / "stage-score.json", _json_bytes(report.to_dict()))
    _atomic_bytes(
        result_root / "stage-score.md", (report.to_markdown() + "\n").encode("utf-8")
    )
    if not report.is_success:
        raise RuntimeError(f"A0 gate did not pass: {report.status} {report.score}/100")
    return A0GateResult(
        report=report,
        manifest_path=manifest_path,
        checkpoint_path=checkpoint.path,
        checkpoint_sha256=checkpoint.sha256,
        metrics_path=metrics_path,
        environment_path=environment_path,
    )


def synthesize_a0_config(
    profile: OfficialTrainingDataProfile,
    *,
    upstream_dir: str | Path,
    processed_root: str | Path,
    backbone_path: str | Path,
    run_id: str,
) -> tuple[dict[str, object], UpstreamAudit]:
    """Adapt only paths/runtime fields; do not create a validation/Test loader."""

    if not isinstance(profile, OfficialTrainingDataProfile):
        raise TypeError("validated official-code A profile is required")
    run_id = _safe_run_id(run_id)
    root = Path(processed_root).resolve(strict=True)
    if root != Path(str(profile.processed_root)).resolve(strict=True):
        raise ValueError("A0 processed root must match the isolated A profile")
    backbone = verify_imagenet_backbone(profile, backbone_path)
    config, audit = load_audited_official_config(profile, upstream_dir)
    dataset = config["dataset"]
    train = config["train"]
    network = config["network"]
    if not all(isinstance(item, dict) for item in (dataset, train, network)):
        raise ValueError("official STEERER config sections are invalid")
    dataset["root"] = str(root).rstrip("/\\") + "/"  # type: ignore[index]
    dataset["train_set"] = "train.txt"  # type: ignore[index]
    dataset["test_set"] = "__A0_TEST_NOT_ACCESSED__"  # type: ignore[index]
    network["pretrained_backbone"] = str(backbone)  # type: ignore[index]
    train["batch_size_per_gpu"] = _PHYSICAL_BATCH  # type: ignore[index]
    train["resume_path"] = None  # type: ignore[index]
    config["gpus"] = (0,)
    config["workers"] = 6
    config["log_dir"] = str(Path(str(profile.result_root)) / run_id)
    config["droneai"] = {
        "stage": "A0",
        "stage_stop_epoch": 0,
        "schedule_horizon_epochs": _SCHEDULE_HORIZON,
        "physical_batch": _PHYSICAL_BATCH,
        "accumulation_steps": _ACCUMULATION_STEPS,
        "effective_batch": _PHYSICAL_BATCH,
        "validation_access": False,
        "test_access": False,
        "model_checkpoint_loaded": False,
    }
    return config, audit


class TorchOfficialA0Runtime:
    """One-CUDA official STEERER runtime that never constructs a Test loader."""

    def __init__(
        self,
        profile: OfficialTrainingDataProfile,
        *,
        config: Mapping[str, object],
        upstream_dir: str | Path,
        backbone_path: str | Path,
        device: str = "cuda:0",
    ) -> None:
        try:
            import torch
        except ImportError as exc:  # pragma: no cover - home5090 only
            raise RuntimeError("PyTorch is required for A0") from exc
        from droneai.steerer_adapter import _official_namespace_scope

        self.torch_module = torch
        self.global_step = 0
        self.test_access_count = 0
        self._scope = _official_namespace_scope
        self._upstream_dir = Path(upstream_dir).resolve(strict=True)
        self._device = torch.device(device)
        if self._device.type != "cuda" or not torch.cuda.is_available():
            raise RuntimeError("A0 requires an available CUDA device")
        random.seed(_SEED)
        np.random.seed(_SEED)
        torch.manual_seed(_SEED)
        torch.cuda.manual_seed_all(_SEED)

        cuda_index = initialize_cuda_memory_stats(torch, self._device)
        left = torch.randn((256, 256), device=self._device)
        right = torch.randn((256, 256), device=self._device)
        product = left @ right
        torch.cuda.synchronize(self._device)
        matmul_finite = bool(torch.isfinite(product).all().item())
        self.cuda_evidence = {
            "available": True,
            "device": str(self._device),
            "device_index": cuda_index,
            "device_name": torch.cuda.get_device_name(self._device),
            "matmul_shape": [256, 256],
            "matmul_finite": matmul_finite,
        }
        if not matmul_finite:
            raise FloatingPointError("A0 CUDA matmul was non-finite")
        del left, right, product

        mapping = copy.deepcopy(dict(config))
        with self._scope(self._upstream_dir):
            try:
                from mmcv import Config
                from lib.datasets.qnrf import QNRF
                from lib.datasets.utils.collate import default_collate
                from lib.models.build_counter import Baseline_Counter
                from lib.solver.build import build_optimizer_cls
                from lib.solver.lr_scheduler_cls import build_scheduler
            except ImportError as exc:  # pragma: no cover - home5090 only
                raise RuntimeError("Pinned STEERER/MMCV dependencies are required") from exc

            config_object = Config(copy.deepcopy(mapping))
            models: list[object] = []

            def model_factory(*, pretrained_backbone: Path | None) -> object:
                candidate = Config(copy.deepcopy(mapping))
                candidate.network.pretrained_backbone = (
                    "" if pretrained_backbone is None else str(pretrained_backbone)
                )
                model = Baseline_Counter(
                    candidate.network,
                    candidate.dataset.den_factor,
                    candidate.train.route_size,
                    self._device,
                )
                models.append(model)
                return model

            audit = initialize_steerer_model(
                profile,
                model_factory,
                backbone_path,
                torch_module=torch,
            )
            model = models[-1]
            models[:] = [model]
            self.initialization_audit = audit.to_json_dict()
            self._model = torch.nn.DataParallel(model.to(self._device))
            self._dataset = QNRF(
                root=config_object.dataset.root,
                list_path=config_object.dataset.train_set,
                num_samples=None,
                num_classes=config_object.dataset.num_classes,
                multi_scale=config_object.train.multi_scale,
                flip=config_object.train.flip,
                ignore_label=None,
                base_size=config_object.train.base_size,
                crop_size=config_object.train.image_size,
                min_unit=config_object.train.route_size,
                scale_factor=config_object.train.scale_factor,
            )
            if len(self._dataset) != profile.train_samples:
                raise ValueError("A0 Train loader must contain exactly 1,201 samples")
            self._dataset.AI_resize = False
            self._collate = default_collate
            generator = torch.Generator()
            generator.manual_seed((_SEED * 1_000_003) % (2**63 - 1))
            self._loader = torch.utils.data.DataLoader(
                self._dataset,
                batch_size=_PHYSICAL_BATCH,
                shuffle=True,
                num_workers=int(config_object.workers),
                pin_memory=True,
                drop_last=True,
                persistent_workers=False,
                collate_fn=default_collate,
                generator=generator,
                worker_init_fn=self._seed_worker,
            )
            self._optimizer = build_optimizer_cls(config_object.optimizer, model)
            updates_per_epoch = len(self._dataset) // _PHYSICAL_BATCH
            self._scheduler = build_scheduler(
                config_object.lr_config,
                self._optimizer,
                updates_per_epoch,
                _SCHEDULE_HORIZON,
            )

    @staticmethod
    def _seed_worker(_worker_id: int) -> None:
        import torch

        worker_seed = int(torch.initial_seed() % (2**32))
        random.seed(worker_seed)
        np.random.seed(worker_seed)

    def run_update(self) -> A0UpdateObservation:
        torch = self.torch_module
        started = time.perf_counter()
        initialize_cuda_memory_stats(torch, self._device)
        batch = next(iter(self._loader))
        images, labels, _sizes, name_metadata = batch
        images = images.to(self._device, non_blocking=True)
        labels = [label.to(self._device, non_blocking=True) for label in labels]
        self._model.train()
        self._optimizer.zero_grad(set_to_none=True)
        with self._scope(self._upstream_dir):
            result = self._model(images, labels, "train")
        loss = result["losses"].mean()
        tensors = [loss]
        density_values: list[float] = []
        for group in (result.get("pre_den", {}), result.get("gt_den", {})):
            if isinstance(group, Mapping):
                tensors.extend(group.values())
        for tensor in tensors:
            if not bool(torch.isfinite(tensor).all().item()):
                raise FloatingPointError("A0 produced non-finite loss or density")
            if tensor is not loss:
                density_values.append(float(tensor.detach().float().sum().item()))
        loss.backward()
        gradient = torch.nn.utils.clip_grad_norm_(self._model.parameters(), float("inf"))
        gradient_norm = float(gradient.detach().float().item())
        if not math.isfinite(gradient_norm):
            raise FloatingPointError("A0 produced a non-finite gradient")
        self._optimizer.step()
        self._scheduler.step_update(self.global_step)
        learning_rate = float(self._optimizer.param_groups[0]["lr"])
        self.global_step += 1
        torch.cuda.synchronize(self._device)
        return A0UpdateObservation(
            optimizer_steps=1,
            loss=float(loss.detach().float().item()),
            density_values=tuple(density_values),
            gradient_norm=gradient_norm,
            learning_rate=learning_rate,
            elapsed_seconds=max(time.perf_counter() - started, 0.0),
            peak_vram_mb=float(torch.cuda.max_memory_allocated(self._device) / (1024**2)),
            train_sample_tokens=tuple(str(name) for name in name_metadata[0]),
        )

    def checkpoint_state(self) -> dict[str, object]:
        model = getattr(self._model, "module", self._model)
        return {
            "model": model.state_dict(),
            "optimizer": self._optimizer.state_dict(),
            "scheduler": self._scheduler.state_dict(),
            "scaler": {},
        }

    def restore_checkpoint_state(self, state: dict[str, object]) -> None:
        if set(state) != {"model", "optimizer", "scheduler", "scaler"} or state["scaler"]:
            raise ValueError("A0 FP32 checkpoint components are invalid")
        model = getattr(self._model, "module", self._model)
        model.load_state_dict(state["model"], strict=True)
        self._optimizer.load_state_dict(state["optimizer"])
        self._scheduler.load_state_dict(state["scheduler"])
        scheduler_step = getattr(self._scheduler, "t", None)
        self.global_step = 1 if scheduler_step is None else int(scheduler_step) + 1

    def state_sha256s(self) -> dict[str, str]:
        state = self.checkpoint_state()
        return {name: state_sha256(value) for name, value in state.items()}


__all__ = [
    "A0GateResult",
    "A0Lineage",
    "A0UpdateObservation",
    "TorchOfficialA0Runtime",
    "initialize_cuda_memory_stats",
    "run_a0_gate",
    "state_sha256",
    "synthesize_a0_config",
]
