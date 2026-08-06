"""Stage-bounded control flow for the pinned STEERER training lane."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Literal, Mapping, Protocol, Sequence

import numpy as np

from droneai.steerer_training_profile import SteererTrainingProfile
from droneai.integrity import sha256_file
from droneai.integrity import is_sha256
from droneai.steerer_training_checkpoint import (
    CHECKPOINT_SCHEMA_VERSION,
    ResumeExpectations,
    capture_rng_state,
    restore_rng_state,
    save_checkpoint_with_policy,
    verify_resume,
)
from droneai.steerer_training_upstream import synthesize_official_config


Stage = Literal["T0", "T1", "T5", "T50"]
_STAGE_STOP_EPOCH = {"T0": 0, "T1": 1, "T5": 5, "T50": 50}
_SCHEDULE_HORIZON = 800
_SEED = 3035


@dataclass(frozen=True)
class TrainingLineage:
    config_sha256: str
    split_sha256s: Mapping[str, str]
    dataset_inventory_sha256: str
    backbone_sha256: str
    upstream_commit: str


@dataclass(frozen=True)
class AmpComparison:
    fp32_loss: float
    amp_loss: float
    fp32_count: float
    amp_count: float

    @property
    def relative_count_difference(self) -> float:
        denominator = max(abs(self.fp32_count), 1.0e-12)
        return abs(self.amp_count - self.fp32_count) / denominator

    @property
    def accepted(self) -> bool:
        relative_difference = self.relative_count_difference
        return (
            all(
                math.isfinite(value)
                for value in (
                    self.fp32_loss,
                    self.amp_loss,
                    self.fp32_count,
                    self.amp_count,
                )
            )
            and relative_difference < 1.0e-3
            and not math.isclose(relative_difference, 1.0e-3, rel_tol=1.0e-12)
        )


@dataclass(frozen=True)
class UpdateObservation:
    loss: float
    density_values: tuple[float, ...]
    gradient_norm: float
    learning_rate: float


@dataclass(frozen=True)
class EpochObservation:
    epoch: int
    optimizer_steps: int
    losses: tuple[float, ...]
    density_values: tuple[float, ...]
    gradient_norms: tuple[float, ...]
    learning_rates: tuple[float, ...]
    sample_tokens: tuple[str, ...]


@dataclass(frozen=True)
class ValidationObservation:
    sample_count: int
    loss: float
    counts: tuple[float, ...]
    density_values: tuple[float, ...]
    mae: float
    rmse: float


@dataclass(frozen=True)
class EpochRunResult:
    completed_epoch: int
    optimizer_steps: int
    learning_rates: tuple[float, ...]
    sample_tokens: tuple[str, ...]


@dataclass(frozen=True)
class StageResult:
    stage: str
    completed_epoch: int
    optimizer_steps: int
    validation_samples: int
    finite_loss: bool
    checkpoint_round_trip: bool
    physical_batch: int
    accumulation_steps: int
    elapsed_seconds: float


class TrainingEngine(Protocol):
    completed_epoch: int
    global_step: int
    lineage: TrainingLineage
    checkpoint_dir: Path
    environment_path: Path

    def set_seed(self, seed: int) -> None: ...

    def configure_batch(self, physical_batch: int, accumulation_steps: int) -> None: ...

    def probe_batch(self) -> None: ...

    def compare_amp_to_fp32(self) -> AmpComparison: ...

    def run_update(self, *, amp_enabled: bool) -> UpdateObservation: ...

    def run_epoch(self, epoch: int, *, amp_enabled: bool) -> EpochObservation: ...

    def validate(self) -> ValidationObservation: ...

    def checkpoint_state(self) -> dict[str, object]: ...

    def restore_checkpoint_state(self, state: dict[str, object]) -> None: ...

    def snapshot(self) -> object: ...

    def restore_snapshot(self, snapshot: object) -> None: ...

    def empty_cuda_cache(self) -> None: ...


def _canonical_config_sha256(config: Mapping[str, object]) -> str:
    stable = copy.deepcopy(dict(config))
    train = stable.get("train")
    if isinstance(train, dict):
        train["resume_path"] = None
    droneai = stable.get("droneai")
    if isinstance(droneai, dict):
        droneai.pop("stage", None)
        droneai.pop("stage_stop_epoch", None)
    encoded = json.dumps(
        stable, sort_keys=True, separators=(",", ":"), default=str
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _epoch_data_seed(seed: int, epoch: int) -> int:
    if seed != _SEED or not isinstance(epoch, int) or isinstance(epoch, bool) or epoch < 0:
        raise ValueError("epoch data seed requires seed 3035 and a non-negative epoch")
    return (seed * 1_000_003 + epoch) % (2**63 - 1)


def _seed_data_worker(_worker_id: int) -> None:
    import torch

    worker_seed = int(torch.initial_seed() % (2**32))
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def _record_resize_memory(
    dataset: object,
    *,
    sizes: object,
    names: Sequence[str],
    route_size: tuple[int, int],
) -> None:
    """Preserve the official train loop's per-image resize bookkeeping."""

    value = sizes
    for method_name in ("detach", "cpu"):
        method = getattr(value, method_name, None)
        if callable(method):
            value = method()
    numpy_method = getattr(value, "numpy", None)
    if callable(numpy_method):
        value = numpy_method()
    size_array = np.asarray(value)
    if size_array.ndim != 2 or size_array.shape[0] != len(names) or size_array.shape[1] < 2:
        raise ValueError("official STEERER batch sizes are malformed")
    pool = getattr(dataset, "resize_memory_pool", None)
    if not isinstance(pool, dict):
        raise ValueError("official STEERER dataset resize memory is unavailable")
    route_height, route_width = route_size
    for index, name in enumerate(names):
        if name in pool:
            continue
        patch_height = int(math.ceil(float(size_array[index, 0]) / route_height))
        patch_width = int(math.ceil(float(size_array[index, 1]) / route_width))
        pool[name] = {
            "avg_size": np.ones((patch_height, patch_width)),
            "load_num": np.zeros((patch_height, patch_width)),
            "size": size_array.copy(),
        }


def _validate_prepared_split(
    profile: SteererTrainingProfile,
    *,
    train_path: Path,
    validation_path: Path,
    inventory_path: Path,
) -> None:
    train_ids = tuple(train_path.read_text(encoding="utf-8").splitlines())
    validation_ids = tuple(validation_path.read_text(encoding="utf-8").splitlines())
    if (len(train_ids), len(validation_ids)) != (
        profile.train_count,
        profile.validation_count,
    ):
        raise ValueError("prepared split must contain the approved 961/240 samples")
    if (
        any(not sample_id for sample_id in train_ids + validation_ids)
        or len(set(train_ids)) != len(train_ids)
        or len(set(validation_ids)) != len(validation_ids)
        or set(train_ids).intersection(validation_ids)
    ):
        raise ValueError("prepared training and validation splits must be disjoint")
    inventory_ids: list[str] = []
    try:
        for line in inventory_path.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            if not isinstance(row, dict) or not isinstance(row.get("sample_id"), str):
                raise ValueError
            inventory_ids.append(row["sample_id"])
    except (json.JSONDecodeError, ValueError) as exc:
        raise ValueError("prepared dataset inventory is invalid") from exc
    if (
        len(inventory_ids) != profile.dataset_population
        or len(set(inventory_ids)) != len(inventory_ids)
        or set(inventory_ids) != set(train_ids).union(validation_ids)
    ):
        raise ValueError("prepared split must be complete for the approved inventory")


class PinnedUpstreamTrainingEngine:
    """Lazy boundary around the audited, pinned official STEERER primitives."""

    def __init__(
        self,
        *,
        profile: SteererTrainingProfile,
        stage: Stage,
        run_id: str,
        processed_root: str | Path,
        upstream_dir: str | Path,
        backbone_path: str | Path,
        device: str = "cuda:0",
        resume: Path | None = None,
        runtime_factory: Callable[..., object] | None = None,
    ) -> None:
        if not isinstance(profile, SteererTrainingProfile):
            raise TypeError("validated STEERER training profile is required")
        if stage not in _STAGE_STOP_EPOCH:
            raise ValueError("stage must be one of T0, T1, T5, or T50")
        self.profile = profile
        self.stage = stage
        self.run_id = run_id
        self.processed_root = Path(processed_root).resolve(strict=True)
        if self.processed_root != profile.processed_root.resolve(strict=False):
            raise ValueError("processed root must match the authoritative training profile")
        self.upstream_dir = Path(upstream_dir).resolve(strict=False)
        self.backbone_path = Path(backbone_path).resolve(strict=False)
        self.device = device
        self.resume = Path(resume) if resume is not None else None
        self.checkpoint_dir = profile.checkpoint_root / run_id
        self.environment_path = profile.result_root / run_id / "environment.json"
        self.completed_epoch = 0
        self.global_step = 0
        self.best_mae = 1.0e20
        self.best_rmse = 1.0e20
        self.learning_rates: list[float] = []
        self._seed = profile.seed
        self._runtime_factory = runtime_factory or _build_torch_runtime
        self._runtime: object | None = None
        self._lineage: TrainingLineage | None = None

    @property
    def lineage(self) -> TrainingLineage:
        if self._lineage is None:
            raise RuntimeError("training engine must be configured before lineage is available")
        return self._lineage

    @property
    def torch_module(self) -> object:
        runtime = self._require_runtime()
        return getattr(runtime, "torch_module")

    def _require_runtime(self) -> object:
        if self._runtime is None:
            raise RuntimeError("training engine batch plan is not configured")
        return self._runtime

    def set_seed(self, seed: int) -> None:
        if seed != self.profile.seed or seed != _SEED:
            raise ValueError("STEERER training seed must be 3035")
        self._seed = seed
        if self._runtime is not None:
            getattr(self._runtime, "set_seed")(seed)

    def configure_batch(self, physical_batch: int, accumulation_steps: int) -> None:
        config = synthesize_official_config(
            profile=self.profile,
            upstream_dir=self.upstream_dir,
            processed_root=self.processed_root,
            backbone_path=self.backbone_path,
            stage=self.stage,
            physical_batch=physical_batch,
            accumulation_steps=accumulation_steps,
            run_id=self.run_id,
            resume_path=self.resume,
        )
        train_path = self.processed_root / "train.txt"
        validation_path = self.processed_root / "val.txt"
        inventory_path = self.processed_root / "manifests" / "output-inventory.jsonl"
        for path in (train_path, validation_path, inventory_path):
            if not path.is_file() or path.is_symlink():
                raise ValueError(f"prepared training artifact is required: {path.name}")
        _validate_prepared_split(
            self.profile,
            train_path=train_path,
            validation_path=validation_path,
            inventory_path=inventory_path,
        )
        droneai = config.get("droneai")
        if not isinstance(droneai, Mapping) or droneai.get("schedule_horizon_epochs") != 800:
            raise ValueError("synthesized config must preserve the 800-epoch horizon")
        self._lineage = TrainingLineage(
            config_sha256=_canonical_config_sha256(config),
            split_sha256s={
                "train": sha256_file(train_path),
                "validation": sha256_file(validation_path),
            },
            dataset_inventory_sha256=sha256_file(inventory_path),
            backbone_sha256=self.profile.imagenet_backbone.sha256,
            upstream_commit=str(droneai["upstream_commit"]),
        )
        self._runtime = self._runtime_factory(
            profile=self.profile,
            config=config,
            upstream_dir=self.upstream_dir,
            backbone_path=self.backbone_path,
            device=self.device,
            physical_batch=physical_batch,
            accumulation_steps=accumulation_steps,
            seed=self._seed,
        )

    def probe_batch(self) -> None:
        getattr(self._require_runtime(), "probe_batch")()

    def empty_cuda_cache(self) -> None:
        if self._runtime is not None:
            getattr(self._runtime, "empty_cuda_cache")()
        self._runtime = None

    def compare_amp_to_fp32(self) -> AmpComparison:
        return getattr(self._require_runtime(), "compare_amp_to_fp32")()

    def run_update(self, *, amp_enabled: bool) -> UpdateObservation:
        observation = getattr(self._require_runtime(), "run_update")(
            amp_enabled=amp_enabled
        )
        self.global_step = int(getattr(self._runtime, "global_step"))
        self.learning_rates.append(observation.learning_rate)
        return observation

    def run_epoch(self, epoch: int, *, amp_enabled: bool) -> EpochObservation:
        observation = getattr(self._require_runtime(), "run_epoch")(
            epoch, amp_enabled=amp_enabled
        )
        self.completed_epoch = observation.epoch
        self.global_step = int(getattr(self._runtime, "global_step"))
        self.learning_rates.extend(observation.learning_rates)
        return observation

    def validate(self) -> ValidationObservation:
        return getattr(self._require_runtime(), "validate")()

    def checkpoint_state(self) -> dict[str, object]:
        return getattr(self._require_runtime(), "checkpoint_state")()

    def restore_checkpoint_state(self, state: dict[str, object]) -> None:
        getattr(self._require_runtime(), "restore_checkpoint_state")(state)
        self.global_step = int(getattr(self._runtime, "global_step"))

    def probe_snapshot(self) -> dict[str, object]:
        return getattr(self._require_runtime(), "probe_snapshot")()

    def set_progress(self, *, completed_epoch: int, global_step: int) -> None:
        self.completed_epoch = completed_epoch
        self.global_step = global_step
        setattr(self._require_runtime(), "global_step", global_step)

    def snapshot(self) -> object:
        return {
            "completed_epoch": self.completed_epoch,
            "global_step": self.global_step,
            "learning_rates": tuple(self.learning_rates),
            "checkpoint": copy.deepcopy(self.checkpoint_state()),
        }

    def restore_snapshot(self, snapshot: object) -> None:
        if not isinstance(snapshot, Mapping):
            raise TypeError("training snapshot is invalid")
        self.restore_checkpoint_state(dict(snapshot["checkpoint"]))  # type: ignore[arg-type]
        self.completed_epoch = int(snapshot["completed_epoch"])
        self.global_step = int(snapshot["global_step"])
        self.learning_rates = list(snapshot["learning_rates"])  # type: ignore[arg-type]


class _TorchPinnedRuntime:
    """Single-CUDA implementation assembled from the pinned upstream modules."""

    def __init__(
        self,
        *,
        profile: SteererTrainingProfile,
        config: Mapping[str, object],
        upstream_dir: Path,
        backbone_path: Path,
        device: str,
        physical_batch: int,
        accumulation_steps: int,
        seed: int,
    ) -> None:
        try:
            import torch
        except ImportError as exc:  # pragma: no cover - home5090 dependency
            raise RuntimeError("PyTorch is required for pinned STEERER training") from exc
        from droneai.steerer_adapter import _official_namespace_scope
        from droneai.steerer_training_upstream import initialize_steerer_model

        self.torch_module = torch
        self._scope = _official_namespace_scope
        self._upstream_dir = upstream_dir.resolve()
        self._device = torch.device(device)
        if self._device.type != "cuda" or not torch.cuda.is_available():
            raise RuntimeError("pinned STEERER training requires an available CUDA device")
        self._physical_batch = physical_batch
        self._accumulation_steps = accumulation_steps
        self._config_mapping = copy.deepcopy(dict(config))
        self.global_step = 0
        self._scaler: object | None = None
        self._scaler_enabled: bool | None = None
        self.set_seed(seed)
        self._seed = seed

        with self._scope(self._upstream_dir):
            try:
                from mmcv import Config
                from lib.datasets.qnrf import QNRF
                from lib.datasets.utils.collate import default_collate
                from lib.models.build_counter import Baseline_Counter
                from lib.solver.build import build_optimizer_cls
                from lib.solver.lr_scheduler_cls import build_scheduler
                from lib.core.cc_function import patch_forward
            except ImportError as exc:  # pragma: no cover - home5090 dependency
                raise RuntimeError(
                    "pinned STEERER/MMCV training dependencies are required"
                ) from exc

            config_object = Config(copy.deepcopy(self._config_mapping))
            models: list[object] = []

            def model_factory(*, pretrained_backbone: Path | None) -> object:
                candidate = Config(copy.deepcopy(self._config_mapping))
                candidate.network.pretrained_backbone = (
                    str(pretrained_backbone) if pretrained_backbone is not None else None
                )
                model = Baseline_Counter(
                    candidate.network,
                    candidate.dataset.den_factor,
                    candidate.train.route_size,
                    self._device,
                )
                models.append(model)
                return model

            initialize_steerer_model(
                profile,
                model_factory,
                backbone_path,
                torch_module=torch,
            )
            model = models[-1]
            self._model = torch.nn.DataParallel(model.to(self._device))
            self._optimizer = build_optimizer_cls(config_object.optimizer, model)
            self._train_dataset = QNRF(
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
            self._validation_dataset = QNRF(
                root=config_object.dataset.root,
                list_path=config_object.dataset.test_set,
                num_samples=None,
                num_classes=config_object.dataset.num_classes,
                multi_scale=False,
                flip=False,
                base_size=config_object.test.base_size,
                crop_size=(None, None),
                min_unit=config_object.train.route_size,
                downsample_rate=1,
            )
            workers = int(config_object.workers)
            self._workers = workers
            self._collate_fn = default_collate
            self._probe_loader = torch.utils.data.DataLoader(
                self._train_dataset,
                batch_size=physical_batch,
                shuffle=False,
                num_workers=0,
                pin_memory=True,
                drop_last=True,
                collate_fn=default_collate,
            )
            self._validation_loader = torch.utils.data.DataLoader(
                self._validation_dataset,
                batch_size=1,
                shuffle=False,
                num_workers=workers,
                pin_memory=True,
                collate_fn=default_collate,
            )
            physical_iterations = len(self._train_dataset) // physical_batch
            if physical_iterations == 0 or physical_iterations % accumulation_steps:
                raise ValueError(
                    "training batches must divide exactly into approved accumulation steps"
                )
            self._updates_per_epoch = physical_iterations // accumulation_steps
            self._scheduler = build_scheduler(
                config_object.lr_config,
                self._optimizer,
                self._updates_per_epoch,
                800,
            )
            self._patch_forward = patch_forward
            self._patch_batch_size = int(config_object.test.patch_batch_size)
            self._route_size = tuple(int(value) for value in config_object.train.route_size)
        self._train_iterator: object | None = None
        self._probe_batch: object | None = None
        self._current_sample_tokens: list[str] = []

    def _official_scope(self):
        return self._scope(self._upstream_dir)

    def set_seed(self, seed: int) -> None:
        random.seed(seed)
        np.random.seed(seed)
        torch = self.torch_module
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    def empty_cuda_cache(self) -> None:
        self._optimizer = None
        self._model = None
        self._train_iterator = None
        self._probe_batch = None
        self.torch_module.cuda.empty_cache()

    def _next_train_batch(self) -> object:
        if self._train_iterator is None:
            self._train_iterator = iter(self._make_train_loader(0))
        try:
            return next(self._train_iterator)  # type: ignore[arg-type]
        except StopIteration:
            raise RuntimeError("deterministic epoch loader ended before its update ceiling")

    def _make_train_loader(self, epoch: int) -> object:
        torch = self.torch_module
        generator = torch.Generator()
        generator.manual_seed(_epoch_data_seed(self._seed, epoch))
        return torch.utils.data.DataLoader(
            self._train_dataset,
            batch_size=self._physical_batch,
            shuffle=True,
            num_workers=self._workers,
            pin_memory=True,
            drop_last=True,
            persistent_workers=False,
            collate_fn=self._collate_fn,
            generator=generator,
            worker_init_fn=_seed_data_worker,
        )

    def _prepared_batch(self, batch: object) -> tuple[object, list[object]]:
        images, labels, _size, _names = batch  # type: ignore[misc]
        images = images.to(self._device, non_blocking=True)
        labels = [label.to(self._device, non_blocking=True) for label in labels]
        return images, labels

    def _autocast(self, enabled: bool):
        torch = self.torch_module
        return torch.autocast(device_type="cuda", dtype=torch.float16, enabled=enabled)

    def _forward_training(
        self, batch: object, *, amp_enabled: bool, record_resize: bool = False
    ) -> tuple[object, tuple[float, ...]]:
        torch = self.torch_module
        _images, _labels, sizes, name_metadata = batch  # type: ignore[misc]
        if record_resize:
            _record_resize_memory(
                self._train_dataset,
                sizes=sizes,
                names=tuple(name_metadata[0]),
                route_size=self._route_size,
            )
            self._current_sample_tokens.extend(str(name) for name in name_metadata[0])
            self._current_sample_tokens.append(
                hashlib.sha256(repr(name_metadata[1:]).encode("utf-8")).hexdigest()
            )
        images, labels = self._prepared_batch(batch)
        with self._official_scope(), self._autocast(amp_enabled):
            result = self._model(images, labels, "train")
        loss = result["losses"].mean()
        tensors = [loss]
        for group in (result.get("pre_den", {}), result.get("gt_den", {})):
            if isinstance(group, Mapping):
                tensors.extend(group.values())
        density_values: list[float] = []
        for tensor in tensors:
            if not bool(torch.isfinite(tensor).all().item()):
                raise FloatingPointError("non-finite loss or density tensor detected")
            if tensor is not loss:
                density_values.append(float(tensor.detach().float().sum().item()))
        return loss, tuple(density_values)

    def probe_batch(self) -> None:
        if self._probe_batch is None:
            self._probe_batch = next(iter(self._probe_loader))
        self._model.eval()
        self._optimizer.zero_grad(set_to_none=True)
        try:
            loss, _densities = self._forward_training(
                self._probe_batch, amp_enabled=False
            )
            scaler = self._ensure_scaler(False)
            scaler.scale(loss).backward()
            scaler.unscale_(self._optimizer)
            gradient_norm = self.torch_module.nn.utils.clip_grad_norm_(
                self._model.parameters(), float("inf")
            )
            if not math.isfinite(float(gradient_norm.item())):
                raise FloatingPointError("non-finite gradient norm detected")
            scaler.step(self._optimizer)
            scaler.update()
            self._scheduler.step_update(self.global_step)
            self.global_step += 1
        finally:
            self._optimizer.zero_grad(set_to_none=True)
            self._model.train()

    def probe_snapshot(self) -> dict[str, object]:
        def cpu_clone(value: object) -> object:
            if isinstance(value, dict):
                return {key: cpu_clone(item) for key, item in value.items()}
            if isinstance(value, list):
                return [cpu_clone(item) for item in value]
            if isinstance(value, tuple):
                return tuple(cpu_clone(item) for item in value)
            detach = getattr(value, "detach", None)
            if callable(detach):
                cloned = detach().cpu().clone()
                return cloned
            return copy.deepcopy(value)

        return cpu_clone(self.checkpoint_state())  # type: ignore[return-value]

    def compare_amp_to_fp32(self) -> AmpComparison:
        if self._probe_batch is None:
            self._probe_batch = next(iter(self._probe_loader))
        torch = self.torch_module
        self._model.eval()
        with torch.no_grad():
            fp32_loss, fp32_densities = self._forward_training(
                self._probe_batch, amp_enabled=False
            )
            try:
                amp_loss, amp_densities = self._forward_training(
                    self._probe_batch, amp_enabled=True
                )
                amp_loss_value = float(amp_loss.detach().float().item())
                amp_count = float(amp_densities[0])
            except FloatingPointError:
                amp_loss_value = float("nan")
                amp_count = float("nan")
        self._model.train()
        return AmpComparison(
            fp32_loss=float(fp32_loss.detach().float().item()),
            amp_loss=amp_loss_value,
            fp32_count=float(fp32_densities[0]),
            amp_count=amp_count,
        )

    def _ensure_scaler(self, enabled: bool) -> object:
        if self._scaler is not None and self._scaler_enabled != enabled:
            raise RuntimeError("AMP mode cannot change inside one run")
        if self._scaler is None:
            torch = self.torch_module
            try:
                self._scaler = torch.amp.GradScaler("cuda", enabled=enabled)
            except (AttributeError, TypeError):  # pragma: no cover - old Torch
                self._scaler = torch.cuda.amp.GradScaler(enabled=enabled)
            self._scaler_enabled = enabled
        return self._scaler

    def run_update(self, *, amp_enabled: bool) -> UpdateObservation:
        torch = self.torch_module
        scaler = self._ensure_scaler(amp_enabled)
        self._model.train()
        self._optimizer.zero_grad(set_to_none=True)
        losses: list[float] = []
        density_values: list[float] = []
        for _ in range(self._accumulation_steps):
            loss, densities = self._forward_training(
                self._next_train_batch(),
                amp_enabled=amp_enabled,
                record_resize=True,
            )
            losses.append(float(loss.detach().float().item()))
            density_values.extend(densities)
            scaler.scale(loss / self._accumulation_steps).backward()
        scaler.unscale_(self._optimizer)
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            self._model.parameters(), float("inf")
        )
        gradient_value = float(gradient_norm.detach().float().item())
        if not math.isfinite(gradient_value):
            self._optimizer.zero_grad(set_to_none=True)
            raise FloatingPointError("non-finite gradient norm detected")
        scaler.step(self._optimizer)
        scaler.update()
        self._scheduler.step_update(self.global_step)
        learning_rate = float(self._optimizer.param_groups[0]["lr"])
        self.global_step += 1
        return UpdateObservation(
            loss=float(sum(losses) / len(losses)),
            density_values=tuple(density_values),
            gradient_norm=gradient_value,
            learning_rate=learning_rate,
        )

    def run_epoch(self, epoch: int, *, amp_enabled: bool) -> EpochObservation:
        losses: list[float] = []
        densities: list[float] = []
        gradients: list[float] = []
        rates: list[float] = []
        self._current_sample_tokens = []
        self._train_iterator = iter(self._make_train_loader(epoch))
        try:
            for _ in range(self._updates_per_epoch):
                observation = self.run_update(amp_enabled=amp_enabled)
                losses.append(observation.loss)
                densities.extend(observation.density_values)
                gradients.append(observation.gradient_norm)
                rates.append(observation.learning_rate)
        finally:
            self._train_iterator = None
        if epoch > 5:
            self._train_dataset.AI_resize = False
        return EpochObservation(
            epoch=epoch,
            optimizer_steps=self._updates_per_epoch,
            losses=tuple(losses),
            density_values=tuple(densities),
            gradient_norms=tuple(gradients),
            learning_rates=tuple(rates),
            sample_tokens=tuple(self._current_sample_tokens),
        )

    def validate(self) -> ValidationObservation:
        torch = self.torch_module
        self._model.eval()
        losses: list[float] = []
        errors: list[float] = []
        counts: list[float] = []
        density_values: list[float] = []
        with torch.no_grad(), self._official_scope():
            for batch in self._validation_loader:
                images, labels = self._prepared_batch(batch)
                result = self._patch_forward(
                    self._model,
                    images,
                    labels,
                    self._patch_batch_size,
                    "val",
                )
                loss = result["losses"].mean()
                predicted = result["pre_den"]["1"].sum()
                target = labels[0].sum()
                for group_name in ("pre_den", "gt_den"):
                    group = result.get(group_name)
                    if not isinstance(group, Mapping) or not group:
                        raise ValueError(
                            f"validation result {group_name} must be a density mapping"
                        )
                    for tensor in group.values():
                        if not bool(torch.isfinite(tensor).all().item()):
                            raise FloatingPointError(
                                "non-finite validation density tensor detected"
                            )
                        density_values.append(
                            float(tensor.detach().float().sum().item())
                        )
                for name, tensor in (
                    ("validation loss", loss),
                    ("validation count", predicted),
                    ("validation count", target),
                ):
                    if not bool(torch.isfinite(tensor).all().item()):
                        raise FloatingPointError(f"non-finite {name} detected")
                predicted_value = float(predicted.detach().float().item())
                target_value = float(target.detach().float().item())
                losses.append(float(loss.detach().float().item()))
                counts.append(predicted_value)
                errors.append(predicted_value - target_value)
        if not counts:
            raise RuntimeError("full validation loader produced no samples")
        return ValidationObservation(
            sample_count=len(counts),
            loss=float(sum(losses) / len(losses)),
            counts=tuple(counts),
            density_values=tuple(density_values),
            mae=float(sum(abs(error) for error in errors) / len(errors)),
            rmse=float(math.sqrt(sum(error * error for error in errors) / len(errors))),
        )

    def checkpoint_state(self) -> dict[str, object]:
        model = getattr(self._model, "module", self._model)
        scaler = self._ensure_scaler(False) if self._scaler is None else self._scaler
        return {
            "model": model.state_dict(),
            "optimizer": self._optimizer.state_dict(),
            "scheduler": self._scheduler.state_dict(),
            "scaler": scaler.state_dict(),
        }

    def restore_checkpoint_state(self, state: dict[str, object]) -> None:
        if set(state) != {"model", "optimizer", "scheduler", "scaler"}:
            raise ValueError("project checkpoint components are incomplete")
        model = getattr(self._model, "module", self._model)
        model.load_state_dict(state["model"], strict=True)
        self._optimizer.load_state_dict(state["optimizer"])
        self._scheduler.load_state_dict(state["scheduler"])
        scaler_state = state["scaler"]
        enabled = bool(scaler_state)
        scaler = self._ensure_scaler(enabled)
        scaler.load_state_dict(scaler_state)
        scheduler_updates = getattr(self._scheduler, "t", None)
        if scheduler_updates is not None:
            self.global_step = int(scheduler_updates) + 1


def _build_torch_runtime(**kwargs: object) -> _TorchPinnedRuntime:
    return _TorchPinnedRuntime(**kwargs)  # type: ignore[arg-type]


def _set_process_seeds(seed: int, engine: TrainingEngine) -> None:
    random.seed(seed)
    np.random.seed(seed)
    engine.set_seed(seed)


def _finite_values(values: Sequence[float], *, name: str) -> None:
    if not values or not all(math.isfinite(float(value)) for value in values):
        raise FloatingPointError(f"non-finite {name} detected")


def _validate_update(observation: UpdateObservation) -> None:
    _finite_values((observation.loss,), name="loss")
    _finite_values(observation.density_values, name="density tensor")
    _finite_values((observation.gradient_norm,), name="gradient norm")
    _finite_values((observation.learning_rate,), name="learning rate")


def _validate_epoch(observation: EpochObservation) -> None:
    _finite_values(observation.losses, name="loss")
    _finite_values(observation.density_values, name="density tensor")
    _finite_values(observation.gradient_norms, name="gradient norm")
    _finite_values(observation.learning_rates, name="learning rate")


def _validate_validation(observation: ValidationObservation) -> None:
    if observation.sample_count < 0:
        raise ValueError("validation sample count cannot be negative")
    _finite_values((observation.loss,), name="validation loss")
    _finite_values(observation.counts, name="validation count")
    _finite_values(observation.density_values, name="validation density tensor")
    _finite_values((observation.mae, observation.rmse), name="validation metric")


def _is_cuda_oom(error: BaseException) -> bool:
    return isinstance(error, RuntimeError) and "CUDA out of memory" in str(error)


def _probe_batch_without_update(engine: TrainingEngine) -> None:
    snapshot_factory = getattr(engine, "probe_snapshot", None)
    state = (
        snapshot_factory()
        if callable(snapshot_factory)
        else copy.deepcopy(engine.checkpoint_state())
    )
    if not isinstance(state, dict):
        raise TypeError("training probe snapshot must be a checkpoint state mapping")
    torch_module = getattr(engine, "torch_module", None)
    rng = capture_rng_state(_SEED, torch_module=torch_module)
    completed_epoch = int(engine.completed_epoch)
    global_step = int(engine.global_step)
    try:
        engine.probe_batch()
    finally:
        engine.restore_checkpoint_state(state)
        restore_rng_state(rng, torch_module=torch_module)
        set_progress = getattr(engine, "set_progress", None)
        if callable(set_progress):
            set_progress(
                completed_epoch=completed_epoch,
                global_step=global_step,
            )
        else:
            engine.completed_epoch = completed_epoch
            engine.global_step = global_step


def _atomic_json(
    path: Path, payload: Mapping[str, object], *, replace_existing: bool = False
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink():
        raise ValueError("environment manifest cannot be a symlink")
    encoded = (
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")
    if path.exists():
        if path.read_bytes() != encoded:
            if not replace_existing:
                raise ValueError("existing environment manifest differs from current run")
        else:
            return
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        with temporary.open("xb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _write_environment(
    engine: TrainingEngine,
    *,
    run_id: str,
    device: str,
    physical_batch: int,
    accumulation_steps: int,
    oom_evidence: str | None,
    amp_comparison: AmpComparison,
    replace_existing: bool = False,
) -> str:
    def finite_or_none(value: float) -> float | None:
        return float(value) if math.isfinite(value) else None

    relative_difference = amp_comparison.relative_count_difference
    payload: dict[str, object] = {
        "schema_version": 1,
        "run_id": run_id,
        "seed": _SEED,
        "device": device,
        "schedule_horizon_epochs": _SCHEDULE_HORIZON,
        "batch": {
            "physical_batch": physical_batch,
            "accumulation_steps": accumulation_steps,
            "effective_batch": physical_batch * accumulation_steps,
        },
        "cuda_oom_evidence": oom_evidence,
        "amp": {
            "enabled": amp_comparison.accepted,
            "fp32_loss": finite_or_none(amp_comparison.fp32_loss),
            "amp_loss": finite_or_none(amp_comparison.amp_loss),
            "fp32_count": finite_or_none(amp_comparison.fp32_count),
            "amp_count": finite_or_none(amp_comparison.amp_count),
            "relative_count_difference": finite_or_none(relative_difference),
            "required_relative_count_difference_below": 1.0e-3,
        },
    }
    _atomic_json(
        engine.environment_path,
        payload,
        replace_existing=replace_existing,
    )
    return sha256_file(engine.environment_path)


def _checkpoint_components(engine: TrainingEngine) -> dict[str, Mapping[str, object]]:
    state = engine.checkpoint_state()
    required = {"model", "optimizer", "scheduler", "scaler"}
    if set(state) != required or not all(isinstance(state[key], Mapping) for key in required):
        raise ValueError("training engine checkpoint state must contain exact project components")
    return {key: state[key] for key in required}  # type: ignore[return-value]


def _persist_and_reload_checkpoint(
    engine: TrainingEngine,
    *,
    run_id: str,
    stage: Stage,
    epoch: int,
    current_mae: float,
    current_rmse: float,
    environment_manifest_sha256: str,
    parent_checkpoint_sha256: str | None = None,
    rng_state: Mapping[str, object] | None = None,
) -> str:
    components = _checkpoint_components(engine)
    previous_best_mae = float(getattr(engine, "best_mae", 1.0e20))
    previous_best_rmse = float(getattr(engine, "best_rmse", 1.0e20))
    best_mae = min(previous_best_mae, float(current_mae))
    best_rmse = min(previous_best_rmse, float(current_rmse))
    torch_module = getattr(engine, "torch_module", None)
    payload: dict[str, object] = {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "run_id": run_id,
        "stage": stage,
        "epoch": epoch,
        "global_step": int(engine.global_step),
        **components,
        "best_mae": best_mae,
        "best_rmse": best_rmse,
        "config_sha256": engine.lineage.config_sha256,
        "split_sha256s": dict(engine.lineage.split_sha256s),
        "dataset_inventory_sha256": engine.lineage.dataset_inventory_sha256,
        "backbone_sha256": engine.lineage.backbone_sha256,
        "upstream_commit": engine.lineage.upstream_commit,
        "rng": (
            dict(rng_state)
            if rng_state is not None
            else capture_rng_state(_SEED, torch_module=torch_module)
        ),
        "environment_manifest_sha256": environment_manifest_sha256,
    }
    saved = save_checkpoint_with_policy(
        engine.checkpoint_dir,
        payload,
        current_mae=current_mae,
        current_rmse=current_rmse,
        previous_best_mae=previous_best_mae,
        previous_best_rmse=previous_best_rmse,
        parent_checkpoint_sha256=parent_checkpoint_sha256,
        torch_module=torch_module,
    )
    last = saved.artifacts["last"]
    expectations = ResumeExpectations(
        run_id=run_id,
        config_sha256=engine.lineage.config_sha256,
        split_sha256s=engine.lineage.split_sha256s,
        dataset_inventory_sha256=engine.lineage.dataset_inventory_sha256,
        backbone_sha256=engine.lineage.backbone_sha256,
        upstream_commit=engine.lineage.upstream_commit,
        environment_manifest_sha256=environment_manifest_sha256,
    )
    restored = verify_resume(
        last,
        expected_sha256=last.sha256,
        expectations=expectations,
        torch_module=torch_module,
    )
    engine.restore_checkpoint_state(
        {key: restored[key] for key in ("model", "optimizer", "scheduler", "scaler")}
    )
    if hasattr(engine, "best_mae"):
        engine.best_mae = best_mae  # type: ignore[attr-defined]
    if hasattr(engine, "best_rmse"):
        engine.best_rmse = best_rmse  # type: ignore[attr-defined]
    return last.sha256


def _resume_manifest_hash(
    engine: TrainingEngine, *, run_id: str, resume: Path
) -> tuple[Path, str, str]:
    checkpoint_dir = engine.checkpoint_dir.resolve(strict=False)
    supplied = Path(resume)
    if supplied.is_symlink():
        raise ValueError("resume checkpoint must be a regular non-symlink file")
    resolved = supplied.resolve(strict=True)
    try:
        resolved.relative_to(checkpoint_dir)
    except ValueError as exc:
        raise ValueError("resume checkpoint must be inside the current run directory") from exc
    if resolved.parent != checkpoint_dir:
        raise ValueError("resume checkpoint must be in the current run directory root")
    if not resolved.is_file():
        raise ValueError("resume checkpoint must be a regular non-symlink file")
    manifest_path = checkpoint_dir / "checkpoint-manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, json.JSONDecodeError) as exc:
        raise ValueError("verified checkpoint manifest is required for resume") from exc
    if (
        not isinstance(manifest, dict)
        or manifest.get("schema_version") != CHECKPOINT_SCHEMA_VERSION
        or manifest.get("run_id") != run_id
        or not isinstance(manifest.get("checkpoints"), list)
    ):
        raise ValueError("checkpoint manifest lineage is invalid")
    matches = [
        entry
        for entry in manifest["checkpoints"]
        if isinstance(entry, dict) and entry.get("filename") == resolved.name
    ]
    if (
        len(matches) != 1
        or not is_sha256(matches[0].get("sha256"))
        or not is_sha256(matches[0].get("environment_manifest_sha256"))
    ):
        raise ValueError("resume checkpoint is not uniquely verified by the manifest")
    return (
        resolved,
        str(matches[0]["sha256"]).lower(),
        str(matches[0]["environment_manifest_sha256"]).lower(),
    )


def _restore_resume(
    engine: TrainingEngine,
    *,
    stage: Stage,
    run_id: str,
    resume: Path,
) -> str:
    path, expected_sha256, environment_manifest_sha256 = _resume_manifest_hash(
        engine, run_id=run_id, resume=resume
    )
    expectations = ResumeExpectations(
        run_id=run_id,
        config_sha256=engine.lineage.config_sha256,
        split_sha256s=engine.lineage.split_sha256s,
        dataset_inventory_sha256=engine.lineage.dataset_inventory_sha256,
        backbone_sha256=engine.lineage.backbone_sha256,
        upstream_commit=engine.lineage.upstream_commit,
        environment_manifest_sha256=environment_manifest_sha256,
    )
    torch_module = getattr(engine, "torch_module", None)
    state = verify_resume(
        path,
        expected_sha256=expected_sha256,
        expectations=expectations,
        torch_module=torch_module,
    )
    epoch = int(state["epoch"])
    resume_stage = str(state["stage"])
    accepted = (
        stage == "T5"
        and ((resume_stage == "T1" and epoch == 1) or (resume_stage == "T5" and 1 <= epoch < 5))
    ) or (
        stage == "T50"
        and ((resume_stage == "T5" and epoch == 5) or (resume_stage == "T50" and 5 <= epoch < 50))
    )
    if not accepted:
        predecessor = "T1" if stage == "T5" else "T5"
        raise ValueError(f"{stage} requires its verified {predecessor} predecessor checkpoint")
    engine.restore_checkpoint_state(
        {key: state[key] for key in ("model", "optimizer", "scheduler", "scaler")}
    )
    global_step = int(state["global_step"])
    set_progress = getattr(engine, "set_progress", None)
    if callable(set_progress):
        set_progress(completed_epoch=epoch, global_step=global_step)
    else:
        engine.completed_epoch = epoch
        engine.global_step = global_step
    if hasattr(engine, "best_mae"):
        engine.best_mae = float(state["best_mae"])  # type: ignore[attr-defined]
    if hasattr(engine, "best_rmse"):
        engine.best_rmse = float(state["best_rmse"])  # type: ignore[attr-defined]
    restore_rng_state(state["rng"], torch_module=torch_module)  # type: ignore[arg-type]
    return expected_sha256


def run_epochs(
    engine: TrainingEngine, *, stops: tuple[int, ...], schedule_horizon: int
) -> EpochRunResult:
    """Run staged epoch ceilings while preserving one scheduler timeline."""

    if schedule_horizon != _SCHEDULE_HORIZON:
        raise ValueError("STEERER schedule horizon must remain 800 epochs")
    if not stops or tuple(sorted(set(stops))) != stops or stops[-1] > 50:
        raise ValueError("epoch stops must be unique increasing ceilings through epoch 50")
    snapshot = engine.snapshot()
    rates: list[float] = []
    sample_tokens: list[str] = []
    optimizer_steps = 0
    completed_epoch = 0
    try:
        engine.completed_epoch = 0
        engine.global_step = 0
        if hasattr(engine, "learning_rates"):
            engine.learning_rates = []  # type: ignore[attr-defined]
        if hasattr(engine, "sample_tokens"):
            engine.sample_tokens = []  # type: ignore[attr-defined]
        for stop in stops:
            for epoch in range(completed_epoch + 1, stop + 1):
                observation = engine.run_epoch(epoch, amp_enabled=False)
                _validate_epoch(observation)
                if observation.epoch != epoch:
                    raise RuntimeError("training engine crossed the requested epoch ceiling")
                completed_epoch = epoch
                optimizer_steps += observation.optimizer_steps
                rates.extend(observation.learning_rates)
                sample_tokens.extend(observation.sample_tokens)
            if stop != stops[-1]:
                state = copy.deepcopy(engine.checkpoint_state())
                engine.restore_checkpoint_state(state)
    finally:
        engine.restore_snapshot(snapshot)
    return EpochRunResult(
        completed_epoch=completed_epoch,
        optimizer_steps=optimizer_steps,
        learning_rates=tuple(rates),
        sample_tokens=tuple(sample_tokens),
    )


def run_training_stage(
    profile: SteererTrainingProfile,
    *,
    stage: Stage,
    run_id: str,
    resume: Path | None = None,
    device: str = "cuda:0",
    engine: TrainingEngine,
) -> StageResult:
    """Execute only the optimizer and validation work authorized by ``stage``."""

    if not isinstance(profile, SteererTrainingProfile):
        raise TypeError("validated STEERER training profile is required")
    if profile.seed != _SEED:
        raise ValueError("STEERER training profile seed must be 3035")
    if stage not in _STAGE_STOP_EPOCH:
        raise ValueError("stage must be one of T0, T1, T5, or T50")
    if not run_id:
        raise ValueError("run_id is required")
    engine_stage = getattr(engine, "stage", stage)
    engine_run_id = getattr(engine, "run_id", run_id)
    if engine_stage != stage:
        raise ValueError("training engine stage differs from requested stage")
    if engine_run_id != run_id:
        raise ValueError("training engine run lineage differs from requested run")
    if stage in {"T0", "T1"} and resume is not None:
        raise ValueError("T0 and T1 cannot use a resume checkpoint")
    if stage in {"T5", "T50"} and resume is None:
        raise ValueError(f"{stage} requires a verified resume checkpoint")

    started = time.perf_counter()
    _set_process_seeds(_SEED, engine)
    engine.configure_batch(8, 1)
    physical_batch, accumulation_steps = 8, 1
    oom_evidence: str | None = None
    try:
        _probe_batch_without_update(engine)
    except BaseException as error:
        if not _is_cuda_oom(error):
            raise
        oom_evidence = f"{type(error).__name__}: CUDA out of memory"
        engine.empty_cuda_cache()
        physical_batch, accumulation_steps = 4, 2
        engine.configure_batch(physical_batch, accumulation_steps)
        _probe_batch_without_update(engine)
    parent_checkpoint_sha256: str | None = None
    if resume is not None:
        parent_checkpoint_sha256 = _restore_resume(
            engine,
            stage=stage,
            run_id=run_id,
            resume=resume,
        )
    amp_comparison = engine.compare_amp_to_fp32()
    _finite_values((amp_comparison.fp32_loss,), name="FP32 comparison loss")
    _finite_values((amp_comparison.fp32_count,), name="FP32 comparison count")
    amp_enabled = amp_comparison.accepted
    environment_manifest_sha256 = _write_environment(
        engine,
        run_id=run_id,
        device=device,
        physical_batch=physical_batch,
        accumulation_steps=accumulation_steps,
        oom_evidence=oom_evidence,
        amp_comparison=amp_comparison,
        replace_existing=resume is not None,
    )

    optimizer_steps = 0
    validation_samples = 0
    checkpoint_round_trip = False
    if stage == "T0":
        update = engine.run_update(amp_enabled=amp_enabled)
        _validate_update(update)
        optimizer_steps = 1
        checkpoint_round_trip = bool(_persist_and_reload_checkpoint(
            engine,
            run_id=run_id,
            stage=stage,
            epoch=0,
            current_mae=float(getattr(engine, "best_mae", 1.0e20)),
            current_rmse=float(getattr(engine, "best_rmse", 1.0e20)),
            environment_manifest_sha256=environment_manifest_sha256,
            parent_checkpoint_sha256=parent_checkpoint_sha256,
        ))
        completed_epoch = 0
    else:
        target_epoch = _STAGE_STOP_EPOCH[stage]
        for epoch in range(engine.completed_epoch + 1, target_epoch + 1):
            observation = engine.run_epoch(epoch, amp_enabled=amp_enabled)
            _validate_epoch(observation)
            if observation.epoch != epoch:
                raise RuntimeError("training engine crossed the requested epoch ceiling")
            optimizer_steps += observation.optimizer_steps
            if epoch < target_epoch:
                epoch_rng = capture_rng_state(
                    _SEED, torch_module=getattr(engine, "torch_module", None)
                )
                parent_checkpoint_sha256 = _persist_and_reload_checkpoint(
                    engine,
                    run_id=run_id,
                    stage=stage,
                    epoch=epoch,
                    current_mae=float(getattr(engine, "best_mae", 1.0e20)),
                    current_rmse=float(getattr(engine, "best_rmse", 1.0e20)),
                    environment_manifest_sha256=environment_manifest_sha256,
                    parent_checkpoint_sha256=parent_checkpoint_sha256,
                    rng_state=epoch_rng,
                )
                checkpoint_round_trip = True
        completed_epoch = engine.completed_epoch
        if completed_epoch != target_epoch:
            raise RuntimeError("training engine did not stop at the requested epoch ceiling")
        training_boundary_rng = capture_rng_state(
            _SEED, torch_module=getattr(engine, "torch_module", None)
        )
        validation = engine.validate()
        _validate_validation(validation)
        if validation.sample_count != profile.validation_count:
            raise RuntimeError(
                "validation must cover the complete approved validation split: "
                f"expected={profile.validation_count} observed={validation.sample_count}"
            )
        validation_samples = validation.sample_count
        checkpoint_round_trip = bool(_persist_and_reload_checkpoint(
            engine,
            run_id=run_id,
            stage=stage,
            epoch=completed_epoch,
            current_mae=validation.mae,
            current_rmse=validation.rmse,
            environment_manifest_sha256=environment_manifest_sha256,
            parent_checkpoint_sha256=parent_checkpoint_sha256,
            rng_state=training_boundary_rng,
        ))

    return StageResult(
        stage=stage,
        completed_epoch=completed_epoch,
        optimizer_steps=optimizer_steps,
        validation_samples=validation_samples,
        finite_loss=True,
        checkpoint_round_trip=checkpoint_round_trip,
        physical_batch=physical_batch,
        accumulation_steps=accumulation_steps,
        elapsed_seconds=max(time.perf_counter() - started, 0.0),
    )


__all__ = [
    "AmpComparison",
    "EpochObservation",
    "EpochRunResult",
    "PinnedUpstreamTrainingEngine",
    "StageResult",
    "TrainingEngine",
    "TrainingLineage",
    "UpdateObservation",
    "ValidationObservation",
    "run_epochs",
    "run_training_stage",
]
