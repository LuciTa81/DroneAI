from __future__ import annotations

from dataclasses import replace
import hashlib
import json
import pickle
import random
from pathlib import Path
from types import MethodType, SimpleNamespace

import numpy as np
import pytest

import scripts.run_steerer_ucf_training as training_cli
import droneai.steerer_training_runner as training_runner

from droneai.steerer_training_evidence import ValidationSampleObservation
from droneai.steerer_training_profile import load_training_profile
from droneai.steerer_training_runner import (
    AmpComparison,
    EpochObservation,
    TrainingLineage,
    PinnedUpstreamTrainingEngine,
    _TorchPinnedRuntime,
    _load_validation_annotation,
    _record_resize_memory,
    _validation_spatial_observation,
    _epoch_data_seed,
    UpdateObservation,
    ValidationObservation,
    run_epochs,
    run_training_stage,
)


PROFILE_PATH = Path("configs/training/steerer_ucf_qnrf_imagenet.home5090.json")
CONTAINER_DIGEST = "sha256:" + "c" * 64


def test_random_reference_uses_an_upstream_safe_empty_backbone_path() -> None:
    assert training_runner._upstream_backbone_argument(None) == ""
    assert training_runner._upstream_backbone_argument(Path("backbone.pth")) == "backbone.pth"


class FakeTrainingEngine:
    def __init__(self, tmp_path: Path) -> None:
        self.loss = 1.25
        self.density_values = (10.0, 5.0)
        self.gradient_norm = 0.5
        self.validation_count = 240
        self.validation_samples = tuple(
            ValidationSampleObservation(
                sample_id=f"validation-{index:04d}",
                target_count=12.0,
                predicted_count=10.0,
                game_l1=2.0,
                quadrant_zone_mae=0.5,
                localization_tp=8,
                localization_fp=2,
                localization_fn=4,
                latency_ms=10.0,
                peak_vram_mb=100.0,
                density_sum_count_difference=0.0,
            )
            for index in range(self.validation_count)
        )
        self.validation_loss = 0.75
        self.validation_counts = (10.0, 12.0)
        self.validation_pre_den = {"1": 10.0, "2": 5.0, "4": 2.5, "8": 1.25}
        self.validation_gt_den = {"1": 12.0, "2": 6.0, "4": 3.0, "8": 1.5}
        self.validation_mae = 2.0
        self.validation_rmse = 2.0
        self.completed_epoch = 0
        self.global_step = 0
        self.model_weight = 0
        self.learning_rates: list[float] = []
        self.sample_tokens: list[str] = []
        self.batch_plans: list[tuple[int, int]] = []
        self.checkpoint_dir = tmp_path / "checkpoints" / "run-3035"
        self.environment_path = tmp_path / "results" / "run-3035" / "environment.json"
        self.lineage = TrainingLineage(
            config_sha256="a" * 64,
            split_sha256s={"train": "b" * 64, "validation": "c" * 64},
            dataset_inventory_sha256="d" * 64,
            backbone_sha256="e" * 64,
            upstream_commit="f" * 40,
        )
        self.seed = None
        self.oom_on_batch8 = False
        self.oom_on_optimizer_step_batch8 = False
        self.probe_optimizer_steps = 0
        self.probe_error: Exception | None = None
        self.empty_cache_calls = 0
        self.amp_flags: list[bool] = []
        self.amp_comparison = AmpComparison(
            fp32_loss=1.0,
            amp_loss=1.0,
            fp32_count=100.0,
            amp_count=100.05,
        )
        self.resumed_amp_comparison: AmpComparison | None = None
        self.amp_compare_model_weights: list[int] = []
        self.scaler_mode: bool | None = None
        self.restored_scaler_modes: list[bool] = []
        self.torch_module = _FakeTorch()
        self.best_mae = 1.0e20
        self.best_rmse = 1.0e20
        self.validation_consumes_rng = False
        self.validation_epochs: list[int] = []
        self.fail_epoch: int | None = None
        self.container_image_digest = "sha256:" + "c" * 64

    def authoritative_environment_payload(
        self,
        *,
        stage: str,
        run_id: str,
        container_image_digest: str,
        physical_batch: int,
        accumulation_steps: int,
        metrics_path: Path,
        metrics_sha256: str,
    ) -> dict[str, object]:
        return {
            "schema_version": 2,
            "run_id": run_id,
            "stage": stage,
            "project": {"git_commit": "1" * 40, "profile_sha256": "2" * 64},
            "upstream": {
                "origin_url": "https://github.com/taohan10200/STEERER.git",
                "commit": self.lineage.upstream_commit,
                "clean": True,
                "license_sha256": "3" * 64,
                "config_sha256": "4" * 64,
            },
            "dataset": {
                "processed_root": "/workspace/data/datasets/fixture",
                "source_partition": "official_train_only",
                "official_test_accessed": False,
                "train_sha256": self.lineage.split_sha256s["train"],
                "validation_sha256": self.lineage.split_sha256s["validation"],
                "dataset_inventory_sha256": self.lineage.dataset_inventory_sha256,
                "population_sha256": "5" * 64,
                "test_sealed_sha256": "6" * 64,
                "split_manifest_sha256": "7" * 64,
            },
            "initialization": {
                "mode": "imagenet_backbone_only",
                "model_checkpoint_loaded": False,
                "backbone_path": "/workspace/data/checkpoints/backbone.pth",
                "backbone_sha256": self.lineage.backbone_sha256,
                "backbone_byte_size": 1,
            },
            "training": {"config_sha256": self.lineage.config_sha256},
            "runtime": {
                "container_image_digest": container_image_digest,
                "python": "3.12.3",
                "torch": "2.9.0a0",
                "cuda": "12.8",
                "gpu": "NVIDIA GeForce RTX 5090",
            },
            "batch": {
                "physical_batch": physical_batch,
                "accumulation_steps": accumulation_steps,
                "effective_batch": physical_batch * accumulation_steps,
            },
            "metrics": {"path": str(metrics_path), "sha256": metrics_sha256},
        }

    def set_seed(self, seed: int) -> None:
        self.seed = seed

    def configure_batch(self, physical_batch: int, accumulation_steps: int) -> None:
        self.batch_plans.append((physical_batch, accumulation_steps))

    def probe_batch(self) -> None:
        self._select_scaler(False)
        if self.probe_error is not None:
            raise self.probe_error
        if self.oom_on_batch8 and self.batch_plans[-1] == (8, 1):
            raise RuntimeError("CUDA out of memory while probing batch 8")
        self.probe_optimizer_steps += 1
        self.global_step += 1
        self.model_weight += 1
        if (
            self.oom_on_optimizer_step_batch8
            and self.batch_plans[-1] == (8, 1)
        ):
            raise RuntimeError("CUDA out of memory allocating AdamW state")
        return None

    def empty_cuda_cache(self) -> None:
        self.empty_cache_calls += 1

    def compare_amp_to_fp32(self) -> AmpComparison:
        self.amp_compare_model_weights.append(self.model_weight)
        if self.model_weight > 0 and self.resumed_amp_comparison is not None:
            return self.resumed_amp_comparison
        return self.amp_comparison

    def run_update(self, *, amp_enabled: bool) -> UpdateObservation:
        self._select_scaler(amp_enabled)
        self.amp_flags.append(amp_enabled)
        self.global_step += 1
        self.model_weight += 1
        lr = self._lr(self.global_step)
        self.learning_rates.append(lr)
        return UpdateObservation(
            loss=self.loss,
            density_values=self.density_values,
            gradient_norm=self.gradient_norm,
            learning_rate=lr,
        )

    def run_epoch(self, epoch: int, *, amp_enabled: bool) -> EpochObservation:
        if self.fail_epoch == epoch:
            raise RuntimeError(f"injected epoch failure: {epoch}")
        rates = []
        for _ in range(2):
            observation = self.run_update(amp_enabled=amp_enabled)
            rates.append(observation.learning_rate)
        self.completed_epoch = epoch
        epoch_rng = random.Random(_epoch_data_seed(3035, epoch))
        samples = [f"sample-{index}" for index in range(6)]
        epoch_rng.shuffle(samples)
        tokens = tuple(
            f"{epoch}:{sample}:aug-{epoch_rng.randrange(1000)}" for sample in samples
        )
        self.sample_tokens.extend(tokens)
        return EpochObservation(
            epoch=epoch,
            optimizer_steps=2,
            losses=(self.loss, self.loss),
            density_values=self.density_values,
            gradient_norms=(self.gradient_norm, self.gradient_norm),
            learning_rates=tuple(rates),
            sample_tokens=tokens,
        )

    def validate(self) -> ValidationObservation:
        self.validation_epochs.append(self.completed_epoch)
        if self.validation_consumes_rng:
            self.torch_module.cpu_state = b"validation-consumed"
        return ValidationObservation(
            sample_count=self.validation_count,
            loss=self.validation_loss,
            counts=self.validation_counts,
            mae=self.validation_mae,
            rmse=self.validation_rmse,
            density_values=tuple(self.validation_pre_den.values())
            + tuple(self.validation_gt_den.values()),
            samples=self.validation_samples,
        )

    def checkpoint_state(self) -> dict[str, object]:
        if self.scaler_mode is None:
            self._select_scaler(False)
        return {
            "model": {"weight": self.model_weight},
            "optimizer": {"step": self.global_step},
            "scheduler": {"global_step": self.global_step, "horizon": 800},
            "scaler": {"scale": 65536.0} if self.scaler_mode else {},
        }

    def probe_snapshot(self) -> dict[str, object]:
        return {
            "model": {"weight": self.model_weight},
            "optimizer": {"step": self.global_step},
            "scheduler": {"global_step": self.global_step, "horizon": 800},
            "scaler": (
                None
                if self.scaler_mode is None
                else ({"scale": 65536.0} if self.scaler_mode else {})
            ),
            "scaler_enabled": self.scaler_mode,
        }

    def restore_probe_snapshot(self, state: dict[str, object]) -> None:
        self.restore_checkpoint_core_state(state)
        enabled = state["scaler_enabled"]
        assert enabled is None or isinstance(enabled, bool)
        self.scaler_mode = enabled

    def restore_checkpoint_core_state(self, state: dict[str, object]) -> None:
        optimizer = state["optimizer"]
        scheduler = state["scheduler"]
        assert isinstance(optimizer, dict) and isinstance(scheduler, dict)
        self.global_step = int(optimizer["step"])
        model = state["model"]
        assert isinstance(model, dict)
        self.model_weight = int(model["weight"])
        assert scheduler["horizon"] == 800

    def restore_scaler_state(
        self, state: object, *, enabled: bool
    ) -> None:
        if not isinstance(state, dict) or bool(state) != enabled:
            raise ValueError("checkpoint scaler state is incompatible with AMP decision")
        self._select_scaler(enabled)
        self.restored_scaler_modes.append(enabled)

    def restore_checkpoint_state(self, state: dict[str, object]) -> None:
        self.restore_checkpoint_core_state(state)
        scaler = state["scaler"]
        enabled = bool(scaler)
        self._select_scaler(enabled)

    def _select_scaler(self, enabled: bool) -> None:
        if self.scaler_mode is not None and self.scaler_mode != enabled:
            raise RuntimeError("scaler mode was permanently selected before AMP decision")
        self.scaler_mode = enabled

    def snapshot(self) -> tuple[int, int, int, tuple[float, ...], tuple[str, ...]]:
        return (
            self.completed_epoch,
            self.global_step,
            self.model_weight,
            tuple(self.learning_rates),
            tuple(self.sample_tokens),
        )

    def restore_snapshot(
        self,
        snapshot: tuple[int, int, int, tuple[float, ...], tuple[str, ...]],
    ) -> None:
        (
            self.completed_epoch,
            self.global_step,
            self.model_weight,
            rates,
            sample_tokens,
        ) = snapshot
        self.learning_rates = list(rates)
        self.sample_tokens = list(sample_tokens)

    @staticmethod
    def _lr(global_step: int) -> float:
        return 0.0001 * (1.0 - (global_step - 1) / (800 * 2))


class _FakeCuda:
    def __init__(self) -> None:
        self.state = [b"cuda-state"]

    def manual_seed_all(self, _seed: int) -> None:
        return None

    def get_rng_state_all(self) -> list[bytes]:
        return list(self.state)

    def set_rng_state_all(self, state: list[bytes]) -> None:
        self.state = list(state)

    def empty_cache(self) -> None:
        return None


class _FakeTorch:
    def __init__(self) -> None:
        self.cpu_state = b"cpu-state"
        self.cuda = _FakeCuda()
        self.saved_epochs: list[int] = []

    def manual_seed(self, _seed: int) -> None:
        return None

    def get_rng_state(self) -> bytes:
        return self.cpu_state

    def set_rng_state(self, state: bytes) -> None:
        self.cpu_state = state

    def save(self, payload: object, path: str | Path) -> None:
        if isinstance(payload, dict) and isinstance(payload.get("epoch"), int):
            self.saved_epochs.append(payload["epoch"])
        with Path(path).open("wb") as stream:
            pickle.dump(payload, stream)

    def load(self, path: str | Path, **_kwargs: object) -> object:
        with Path(path).open("rb") as stream:
            return pickle.load(stream)


@pytest.fixture
def profile(tmp_path: Path):
    loaded = load_training_profile(PROFILE_PATH)
    return replace(
        loaded,
        checkpoint_root=tmp_path / "checkpoints",
        result_root=tmp_path / "results",
    )


@pytest.fixture
def fake_engine(tmp_path: Path) -> FakeTrainingEngine:
    return FakeTrainingEngine(tmp_path)


def test_t0_runs_one_update_and_checkpoint_reload(profile, fake_engine) -> None:
    """Skipping the optimizer update or checkpoint reload makes T0 meaningless."""

    result = run_training_stage(
        profile, stage="T0", run_id="run-3035", engine=fake_engine
    )

    assert result.optimizer_steps == 1
    assert result.finite_loss is True
    assert result.checkpoint_round_trip is True
    assert result.validation_samples == 0
    assert fake_engine.seed == 3035


def test_t0_binds_measured_metrics_environment_and_checkpoint(
    profile, fake_engine
) -> None:
    """Reordering or self-attesting any of the three authoritative artifacts is a bug."""

    result = run_training_stage(
        profile,
        stage="T0",
        run_id="run-3035",
        container_image_digest=fake_engine.container_image_digest,
        engine=fake_engine,
    )

    metrics_path = fake_engine.environment_path.parent / "metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert metrics == {
        "schema_version": 1,
        "run_id": "run-3035",
        "stage": "T0",
        "metrics": {"train_loss": 1.25},
    }
    metrics_sha = hashlib.sha256(metrics_path.read_bytes()).hexdigest()
    environment = json.loads(result.environment_path.read_text(encoding="utf-8"))
    assert environment["schema_version"] == 2
    assert environment["metrics"] == {
        "path": str(metrics_path),
        "sha256": metrics_sha,
    }
    environment_sha = hashlib.sha256(result.environment_path.read_bytes()).hexdigest()
    with result.checkpoint_path.open("rb") as stream:
        checkpoint = pickle.load(stream)
    assert checkpoint["environment_manifest_sha256"] == environment_sha
    assert result.metrics_path == metrics_path
    assert result.metrics_sha256 == metrics_sha
    assert result.environment_sha256 == environment_sha


def test_t1_runs_exactly_one_epoch_and_full_validation(profile, fake_engine) -> None:
    """Stopping early or sampling validation would violate the T1 ceiling."""

    result = run_training_stage(
        profile, stage="T1", run_id="run-3035", engine=fake_engine
    )

    assert result.completed_epoch == 1
    assert result.optimizer_steps == 2
    assert result.validation_samples == 240


def test_t1_rejects_duplicate_or_incomplete_validation_observations(
    profile, tmp_path: Path
) -> None:
    """A nominal sample_count cannot conceal missing or duplicate per-image evidence."""

    fake_engine = FakeTrainingEngine(tmp_path / "incomplete")
    fake_engine.validation_samples = fake_engine.validation_samples[:-1]
    with pytest.raises(ValueError, match="complete"):
        run_training_stage(
            profile, stage="T1", run_id="run-3035", engine=fake_engine
        )

    fake_engine = FakeTrainingEngine(tmp_path / "duplicate")
    fake_engine.validation_samples = (fake_engine.validation_samples[0],) * fake_engine.validation_count
    with pytest.raises(ValueError, match="unique"):
        run_training_stage(
            profile, stage="T1", run_id="run-3035", engine=fake_engine
        )


def test_validation_annotation_requires_exact_count_and_original_bounds(
    tmp_path: Path,
) -> None:
    """Prepared annotation corruption must fail before it influences localization."""

    annotation_dir = tmp_path / "jsons"
    annotation_dir.mkdir()
    annotation = annotation_dir / "sample-a.json"
    annotation.write_text(
        json.dumps(
            {
                "human_num": 2,
                "points": [[1.0, 2.0], [9.5, 7.5]],
                "source_width": 10,
                "source_height": 8,
            }
        ),
        encoding="utf-8",
    )

    loaded = _load_validation_annotation(tmp_path, "sample-a")

    assert loaded.points == ((1.0, 2.0), (9.5, 7.5))
    assert (loaded.width, loaded.height, loaded.count) == (10, 8, 2)

    payload = json.loads(annotation.read_text(encoding="utf-8"))
    payload["human_num"] = 3
    annotation.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="count"):
        _load_validation_annotation(tmp_path, "sample-a")


def test_torch_runtime_uses_validation_dataset_root_for_prepared_annotation(
    tmp_path: Path,
) -> None:
    """Validation must use the root owned by the constructed QNRF dataset."""

    annotation_dir = tmp_path / "jsons"
    annotation_dir.mkdir()
    (annotation_dir / "sample-a.json").write_text(
        json.dumps(
            {
                "human_num": 1,
                "points": [[1.0, 2.0]],
                "source_width": 10,
                "source_height": 8,
            }
        ),
        encoding="utf-8",
    )
    runtime = object.__new__(_TorchPinnedRuntime)
    runtime._validation_dataset = SimpleNamespace(root=str(tmp_path))

    loaded = runtime._prepared_validation_annotation("sample-a")

    assert loaded.sample_id == "sample-a"
    assert loaded.count == 1


def test_validation_spatial_observation_matches_density_and_point_contract() -> None:
    """Changing quadrant or 16-pixel matching formulas must change this result."""

    target = np.zeros((4, 4), dtype=np.float32)
    predicted = np.zeros((4, 4), dtype=np.float32)
    target[0, 0] = target[3, 3] = 1.0
    predicted[0, 0] = predicted[3, 2] = 1.0

    observation = _validation_spatial_observation(
        sample_id="sample-a",
        target_density=target,
        predicted_density=predicted,
        ground_truth_points=((0.0, 0.0), (30.0, 30.0)),
        predicted_points=((1.0, 1.0), (50.0, 50.0)),
        latency_ms=12.0,
        peak_vram_mb=256.0,
        localization_radius=16.0,
    )

    assert observation.target_count == 2.0
    assert observation.predicted_count == 2.0
    assert observation.game_l1 == 0.0
    assert observation.quadrant_zone_mae == 0.0
    assert (
        observation.localization_tp,
        observation.localization_fp,
        observation.localization_fn,
    ) == (1, 1, 1)
    assert observation.density_sum_count_difference == 0.0


def test_t1_to_t5_resume_matches_uninterrupted_lr_sequence(fake_engine) -> None:
    """Restarting the 800-epoch scheduler at T5 changes the experiment."""

    staged = run_epochs(fake_engine, stops=(1, 5), schedule_horizon=800)
    uninterrupted = run_epochs(fake_engine, stops=(5,), schedule_horizon=800)

    assert staged.learning_rates == uninterrupted.learning_rates
    assert staged.learning_rates[:2] == pytest.approx((0.0001, 0.0000999375))


def test_staged_epochs_match_uninterrupted_sample_and_augmentation_order(
    fake_engine,
) -> None:
    """Epoch restart must reproduce both shuffle and augmentation decisions."""

    staged = run_epochs(fake_engine, stops=(1, 5), schedule_horizon=800)
    uninterrupted = run_epochs(fake_engine, stops=(5,), schedule_horizon=800)

    assert staged.sample_tokens
    assert staged.sample_tokens == uninterrupted.sample_tokens


def test_adaptive_resize_stays_pinned_false_for_all_stages_and_resume() -> None:
    """The QNRF lane must never enter unapproved AI_resize code paths."""

    def runtime() -> _TorchPinnedRuntime:
        candidate = object.__new__(_TorchPinnedRuntime)
        candidate._train_dataset = type("Dataset", (), {"AI_resize": False})()
        candidate._updates_per_epoch = 1
        candidate._train_iterator = None
        candidate._current_sample_tokens = []
        candidate.global_step = 0
        candidate.loader_states = []

        def make_loader(self, epoch: int):
            state = bool(self._train_dataset.AI_resize)
            self.loader_states.append((epoch, state))
            return iter((f"epoch-{epoch}:resize-{state}",))

        def run_update(self, *, amp_enabled: bool) -> UpdateObservation:
            del amp_enabled
            token = next(self._train_iterator)
            self._current_sample_tokens.append(token)
            self.global_step += 1
            return UpdateObservation(1.0, (1.0,), 1.0, 0.0001)

        candidate._make_train_loader = MethodType(make_loader, candidate)
        candidate.run_update = MethodType(run_update, candidate)
        return candidate

    initial = runtime()
    assert initial._train_dataset.AI_resize is False
    assert initial._next_train_batch() == "epoch-0:resize-False"
    assert initial.loader_states == [(0, False)]

    for stage, epoch in (("T1", 1), ("T5", 5), ("T50", 50)):
        candidate = runtime()
        observation = candidate.run_epoch(epoch, amp_enabled=False)
        assert observation.sample_tokens == (f"epoch-{epoch}:resize-False",), stage
        assert candidate.loader_states == [(epoch, False)], stage

    uninterrupted = runtime()
    epoch6 = uninterrupted.run_epoch(6, amp_enabled=False)
    epoch7 = uninterrupted.run_epoch(7, amp_enabled=False)
    resumed = runtime()
    resumed_epoch7 = resumed.run_epoch(7, amp_enabled=False)

    assert epoch6.sample_tokens == ("epoch-6:resize-False",)
    assert epoch7.sample_tokens == ("epoch-7:resize-False",)
    assert resumed_epoch7.sample_tokens == epoch7.sample_tokens
    assert uninterrupted.loader_states == [(6, False), (7, False)]
    assert resumed.loader_states == [(7, False)]


def test_runner_stops_on_non_finite_loss(profile, fake_engine) -> None:
    """A NaN loss must stop before a checkpoint can legitimize the run."""

    fake_engine.loss = float("nan")

    with pytest.raises(FloatingPointError, match="non-finite.*loss"):
        run_training_stage(
            profile, stage="T0", run_id="run-3035", engine=fake_engine
        )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("density_values", (1.0, float("inf")), "density tensor"),
        ("gradient_norm", float("nan"), "gradient norm"),
    ],
)
def test_runner_stops_on_every_non_finite_training_value(
    profile, fake_engine, field: str, value: object, message: str
) -> None:
    """A finite aggregate loss must not hide an invalid density or gradient."""

    setattr(fake_engine, field, value)

    with pytest.raises(FloatingPointError, match=f"non-finite {message}"):
        run_training_stage(
            profile, stage="T0", run_id="run-3035", engine=fake_engine
        )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("validation_loss", float("nan"), "validation loss"),
        ("validation_counts", (10.0, float("inf")), "validation count"),
    ],
)
def test_runner_stops_on_every_non_finite_validation_value(
    profile, fake_engine, field: str, value: object, message: str
) -> None:
    """A full validation count cannot hide a non-finite loss or predicted count."""

    setattr(fake_engine, field, value)

    with pytest.raises(FloatingPointError, match=f"non-finite {message}"):
        run_training_stage(
            profile, stage="T1", run_id="run-3035", engine=fake_engine
        )


@pytest.mark.parametrize(("group", "scale"), [("pre", "2"), ("pre", "4"), ("gt", "8")])
def test_runner_checks_every_multiscale_validation_density(
    profile, fake_engine, group: str, scale: str
) -> None:
    """Finite x1 loss/count must not hide NaN/Inf in x2/x4/x8 density maps."""

    mapping = (
        fake_engine.validation_pre_den
        if group == "pre"
        else fake_engine.validation_gt_den
    )
    mapping[scale] = float("nan") if scale != "4" else float("inf")

    with pytest.raises(FloatingPointError, match="validation density tensor"):
        run_training_stage(
            profile, stage="T1", run_id="run-3035", engine=fake_engine
        )


def test_cuda_oom_probe_uses_only_approved_effective_batch_fallback(
    profile, fake_engine
) -> None:
    """OOM fallback must preserve effective batch 8 and leave audit evidence."""

    fake_engine.oom_on_batch8 = True

    result = run_training_stage(
        profile, stage="T0", run_id="run-3035", engine=fake_engine
    )

    assert (result.physical_batch, result.accumulation_steps) == (4, 2)
    assert fake_engine.batch_plans == [(8, 1), (4, 2)]
    assert fake_engine.empty_cache_calls == 1
    environment = json.loads(fake_engine.environment_path.read_text(encoding="utf-8"))
    assert environment["batch"] == {
        "physical_batch": 4,
        "accumulation_steps": 2,
        "effective_batch": 8,
    }
    assert "CUDA out of memory" in result.cuda_oom_evidence


def test_probe_full_optimizer_step_leaves_no_update_in_t0(profile, fake_engine) -> None:
    """The AdamW allocation probe must not become an unreported training update."""

    result = run_training_stage(
        profile, stage="T0", run_id="run-3035", engine=fake_engine
    )

    assert fake_engine.probe_optimizer_steps == 1
    assert result.optimizer_steps == 1
    assert fake_engine.global_step == 1
    assert fake_engine.model_weight == 1
    assert fake_engine.scaler_mode is True


def test_optimizer_step_only_oom_triggers_clean_4x2_rebuild(
    profile, fake_engine
) -> None:
    """OOM first seen during AdamW state allocation must use the one approved fallback."""

    fake_engine.oom_on_optimizer_step_batch8 = True

    result = run_training_stage(
        profile, stage="T0", run_id="run-3035", engine=fake_engine
    )

    assert fake_engine.batch_plans == [(8, 1), (4, 2)]
    assert fake_engine.probe_optimizer_steps == 2
    assert (result.physical_batch, result.accumulation_steps) == (4, 2)
    assert fake_engine.global_step == 1
    assert fake_engine.model_weight == 1


def test_non_oom_cuda_error_is_not_reclassified_as_batch_fallback(
    profile, fake_engine
) -> None:
    """Retrying an illegal CUDA access could conceal a corrupt run."""

    fake_engine.probe_error = RuntimeError("CUDA illegal memory access")

    with pytest.raises(RuntimeError, match="illegal memory access"):
        run_training_stage(
            profile, stage="T0", run_id="run-3035", engine=fake_engine
        )

    assert fake_engine.batch_plans == [(8, 1)]
    assert fake_engine.empty_cache_calls == 0


@pytest.mark.parametrize(
    ("comparison", "accepted"),
    [
        (AmpComparison(1.0, 1.0, 100.0, 100.099), True),
        (AmpComparison(1.0, 1.0, 100.0, 100.1), False),
        (AmpComparison(1.0, float("nan"), 100.0, 100.0), False),
    ],
)
def test_amp_is_used_only_after_finite_strict_numerical_gate(
    profile, fake_engine, comparison: AmpComparison, accepted: bool
) -> None:
    """AMP at or beyond the 1e-3 count boundary must fall back to FP32."""

    fake_engine.amp_comparison = comparison

    result = run_training_stage(
        profile, stage="T0", run_id="run-3035", engine=fake_engine
    )

    assert fake_engine.amp_flags == [accepted]
    assert result.amp_enabled is accepted


def test_amp_approved_fresh_stage_selects_enabled_scaler_only_after_probe(
    profile, fake_engine
) -> None:
    """A disabled probe scaler must not prevent an AMP-approved fresh update."""

    run_training_stage(profile, stage="T0", run_id="run-3035", engine=fake_engine)

    assert fake_engine.scaler_mode is True
    assert fake_engine.global_step == 1


def test_amp_checkpoint_resume_restores_compatible_scaler_after_comparison(
    profile, tmp_path: Path
) -> None:
    """A resumed AMP scaler must load only after resumed-weight numerical comparison."""

    t1_engine = FakeTrainingEngine(tmp_path)
    run_training_stage(profile, stage="T1", run_id="run-3035", engine=t1_engine)
    t5_engine = FakeTrainingEngine(tmp_path)

    run_training_stage(
        profile,
        stage="T5",
        run_id="run-3035",
        resume=t1_engine.checkpoint_dir / "last.pth",
        engine=t5_engine,
    )

    assert t5_engine.amp_compare_model_weights == [2]
    assert t5_engine.restored_scaler_modes[0] is True
    assert t5_engine.scaler_mode is True


def test_resume_rejects_scaler_state_incompatible_with_amp_decision(
    profile, tmp_path: Path
) -> None:
    """Silently loading an enabled scaler into an FP32 resume corrupts semantics."""

    t1_engine = FakeTrainingEngine(tmp_path)
    run_training_stage(profile, stage="T1", run_id="run-3035", engine=t1_engine)
    t5_engine = FakeTrainingEngine(tmp_path)
    t5_engine.resumed_amp_comparison = AmpComparison(1.0, 1.0, 100.0, 101.0)

    with pytest.raises(ValueError, match="scaler state is incompatible"):
        run_training_stage(
            profile,
            stage="T5",
            run_id="run-3035",
            resume=t1_engine.checkpoint_dir / "last.pth",
            engine=t5_engine,
        )


def test_stage_checkpoint_is_task5_payload_and_round_trips_from_disk(
    profile, fake_engine
) -> None:
    """An in-memory copy is not evidence that the atomic project checkpoint reloads."""

    result = run_training_stage(
        profile, stage="T1", run_id="run-3035", engine=fake_engine
    )

    checkpoint = fake_engine.checkpoint_dir / "last.pth"
    manifest = json.loads(
        (fake_engine.checkpoint_dir / "checkpoint-manifest.json").read_text(
            encoding="utf-8"
        )
    )
    with checkpoint.open("rb") as stream:
        payload = pickle.load(stream)
    assert result.checkpoint_round_trip is True
    assert set(payload) == {
        "schema_version", "run_id", "stage", "epoch", "global_step",
        "model", "optimizer", "scheduler", "scaler", "best_mae", "best_rmse",
        "config_sha256", "split_sha256s", "dataset_inventory_sha256",
        "backbone_sha256", "upstream_commit", "rng",
        "environment_manifest_sha256",
    }
    assert payload["scheduler"]["horizon"] == 800
    assert payload["best_mae"] == pytest.approx(2.0)
    assert payload["best_rmse"] == pytest.approx(2.0)
    assert fake_engine.best_mae == pytest.approx(2.0)
    assert fake_engine.best_rmse == pytest.approx(2.0)
    assert any(row["filename"] == "last.pth" for row in manifest["checkpoints"])


def test_worse_t5_metrics_preserve_t1_best_artifacts_and_payload_best(
    profile, tmp_path: Path
) -> None:
    """A worse later stage must not overwrite the true T1 best checkpoint."""

    t1_engine = FakeTrainingEngine(tmp_path)
    run_training_stage(profile, stage="T1", run_id="run-3035", engine=t1_engine)
    best_path = t1_engine.checkpoint_dir / "best-mae.pth"
    best_bytes = best_path.read_bytes()
    t1_manifest = json.loads(
        (t1_engine.checkpoint_dir / "checkpoint-manifest.json").read_text(
            encoding="utf-8"
        )
    )
    best_entry = next(
        row for row in t1_manifest["checkpoints"] if row["filename"] == "best-mae.pth"
    )

    t5_engine = FakeTrainingEngine(tmp_path)
    t5_engine.validation_mae = 5.0
    t5_engine.validation_rmse = 6.0
    run_training_stage(
        profile,
        stage="T5",
        run_id="run-3035",
        resume=t1_engine.checkpoint_dir / "last.pth",
        engine=t5_engine,
    )

    assert best_path.read_bytes() == best_bytes
    manifest = json.loads(
        (t5_engine.checkpoint_dir / "checkpoint-manifest.json").read_text(
            encoding="utf-8"
        )
    )
    assert next(
        row for row in manifest["checkpoints"] if row["filename"] == "best-mae.pth"
    ) == best_entry
    with (t5_engine.checkpoint_dir / "last.pth").open("rb") as stream:
        last_payload = pickle.load(stream)
    assert last_payload["best_mae"] == pytest.approx(2.0)
    assert last_payload["best_rmse"] == pytest.approx(2.0)


def test_t5_resumes_verified_t1_state_without_restarting_scheduler(
    profile, tmp_path: Path
) -> None:
    """Loading weights without optimizer/scheduler/RNG would diverge after T1."""

    t1_engine = FakeTrainingEngine(tmp_path)
    run_training_stage(profile, stage="T1", run_id="run-3035", engine=t1_engine)
    t1_rates = tuple(t1_engine.learning_rates)
    resume = t1_engine.checkpoint_dir / "last.pth"

    t5_engine = FakeTrainingEngine(tmp_path)
    result = run_training_stage(
        profile,
        stage="T5",
        run_id="run-3035",
        resume=resume,
        engine=t5_engine,
    )

    expected = tuple(FakeTrainingEngine._lr(step) for step in range(1, 11))
    assert result.completed_epoch == 5
    assert result.optimizer_steps == 8
    assert t1_rates + tuple(t5_engine.learning_rates) == pytest.approx(expected)
    manifest = json.loads(
        (t5_engine.checkpoint_dir / "checkpoint-manifest.json").read_text(
            encoding="utf-8"
        )
    )
    last = next(row for row in manifest["checkpoints"] if row["filename"] == "last.pth")
    assert last["parent_checkpoint_sha256"] is not None
    assert set(t5_engine.torch_module.saved_epochs) == {5}


def test_t800_runs_every_25_epoch_boundary_and_preserves_100_epoch_milestones(
    profile, tmp_path: Path
) -> None:
    """One end-only validation would erase restart safety and best-model evidence."""

    t1_engine = FakeTrainingEngine(tmp_path)
    run_training_stage(profile, stage="T1", run_id="run-3035", engine=t1_engine)
    t5_engine = FakeTrainingEngine(tmp_path)
    run_training_stage(
        profile,
        stage="T5",
        run_id="run-3035",
        resume=t1_engine.checkpoint_dir / "last.pth",
        engine=t5_engine,
    )
    t800_engine = FakeTrainingEngine(tmp_path)

    result = run_training_stage(
        profile,
        stage="T800",
        run_id="run-3035",
        resume=t5_engine.checkpoint_dir / "last.pth",
        engine=t800_engine,
    )

    expected_boundaries = list(range(25, 801, 25))
    assert result.completed_epoch == 800
    assert t800_engine.validation_epochs == expected_boundaries
    assert sorted(
        path.name for path in t800_engine.environment_path.parent.glob("metrics.epoch-*.json")
    ) == [f"metrics.epoch-{epoch:03d}.json" for epoch in expected_boundaries]
    assert (t800_engine.environment_path.parent / "metrics.t800.json").read_bytes() == (
        t800_engine.environment_path.parent / "metrics.epoch-800.json"
    ).read_bytes()
    assert sorted(
        path.name for path in t800_engine.checkpoint_dir.glob("milestone-*.pth")
    ) == ["milestone-001.pth", "milestone-005.pth"] + [
        f"milestone-{epoch:03d}.pth" for epoch in range(100, 801, 100)
    ]
    status = json.loads(
        (t800_engine.environment_path.parent / "status.json").read_text(encoding="utf-8")
    )
    assert status["state"] == "completed"
    assert status["epoch"] == 800
    assert status["checkpoint_sha256"] == result.checkpoint_sha256


def test_t800_interruption_resumes_from_last_verified_boundary(
    profile, tmp_path: Path
) -> None:
    """An interrupted partial epoch segment must restart from the last validated boundary."""

    t1_engine = FakeTrainingEngine(tmp_path)
    run_training_stage(profile, stage="T1", run_id="run-3035", engine=t1_engine)
    t5_engine = FakeTrainingEngine(tmp_path)
    run_training_stage(
        profile,
        stage="T5",
        run_id="run-3035",
        resume=t1_engine.checkpoint_dir / "last.pth",
        engine=t5_engine,
    )
    interrupted = FakeTrainingEngine(tmp_path)
    interrupted.fail_epoch = 26

    with pytest.raises(RuntimeError, match="injected epoch failure"):
        run_training_stage(
            profile,
            stage="T800",
            run_id="run-3035",
            resume=t5_engine.checkpoint_dir / "last.pth",
            engine=interrupted,
        )

    with (interrupted.checkpoint_dir / "last.pth").open("rb") as stream:
        boundary = pickle.load(stream)
    assert boundary["stage"] == "T800"
    assert boundary["epoch"] == 25
    retry = FakeTrainingEngine(tmp_path)
    result = run_training_stage(
        profile,
        stage="T800",
        run_id="run-3035",
        resume=interrupted.checkpoint_dir / "last.pth",
        engine=retry,
    )

    assert result.completed_epoch == 800
    assert retry.validation_epochs == list(range(50, 801, 25))


def test_t800_rejects_t1_as_an_unapproved_predecessor(profile, tmp_path: Path) -> None:
    t1_engine = FakeTrainingEngine(tmp_path)
    run_training_stage(profile, stage="T1", run_id="run-3035", engine=t1_engine)

    with pytest.raises(ValueError, match="T800.*T5.*T50.*T800"):
        run_training_stage(
            profile,
            stage="T800",
            run_id="run-3035",
            resume=t1_engine.checkpoint_dir / "last.pth",
            engine=FakeTrainingEngine(tmp_path),
        )


def test_t800_cuda_oom_stops_instead_of_changing_the_approved_batch(
    profile, tmp_path: Path
) -> None:
    t1_engine = FakeTrainingEngine(tmp_path)
    run_training_stage(profile, stage="T1", run_id="run-3035", engine=t1_engine)
    t5_engine = FakeTrainingEngine(tmp_path)
    run_training_stage(
        profile,
        stage="T5",
        run_id="run-3035",
        resume=t1_engine.checkpoint_dir / "last.pth",
        engine=t5_engine,
    )
    t800_engine = FakeTrainingEngine(tmp_path)
    t800_engine.oom_on_batch8 = True

    with pytest.raises(RuntimeError, match="CUDA out of memory"):
        run_training_stage(
            profile,
            stage="T800",
            run_id="run-3035",
            resume=t5_engine.checkpoint_dir / "last.pth",
            engine=t800_engine,
        )

    assert t800_engine.batch_plans == [(8, 1)]


def test_resume_is_restored_before_amp_comparison(profile, tmp_path: Path) -> None:
    """AMP eligibility must be measured on resumed weights, not fresh ImageNet weights."""

    t1_engine = FakeTrainingEngine(tmp_path)
    t1_engine.amp_comparison = AmpComparison(1.0, 1.0, 100.0, 101.0)
    run_training_stage(profile, stage="T1", run_id="run-3035", engine=t1_engine)
    t5_engine = FakeTrainingEngine(tmp_path)
    t5_engine.resumed_amp_comparison = AmpComparison(
        fp32_loss=1.0,
        amp_loss=1.0,
        fp32_count=100.0,
        amp_count=101.0,
    )

    result = run_training_stage(
        profile,
        stage="T5",
        run_id="run-3035",
        resume=t1_engine.checkpoint_dir / "last.pth",
        engine=t5_engine,
    )

    assert t5_engine.amp_compare_model_weights == [2]
    assert result.amp_enabled is False


def test_resume_hash_is_taken_from_manifest_before_deserialization(
    profile, tmp_path: Path
) -> None:
    """Hashing attacker-replaced bytes and trusting that same hash is not verification."""

    t1_engine = FakeTrainingEngine(tmp_path)
    run_training_stage(profile, stage="T1", run_id="run-3035", engine=t1_engine)
    resume = t1_engine.checkpoint_dir / "last.pth"
    resume.write_bytes(b"replaced checkpoint")
    t5_engine = FakeTrainingEngine(tmp_path)

    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        run_training_stage(
            profile,
            stage="T5",
            run_id="run-3035",
            resume=resume,
            engine=t5_engine,
        )

    assert t5_engine.global_step == 0


def test_resume_requires_existing_environment_evidence_before_restore(
    profile, tmp_path: Path
) -> None:
    """A missing environment file must not be silently regenerated during resume."""

    t1_engine = FakeTrainingEngine(tmp_path)
    run_training_stage(profile, stage="T1", run_id="run-3035", engine=t1_engine)
    resume = t1_engine.checkpoint_dir / "last.pth"
    with resume.open("rb") as stream:
        payload = pickle.load(stream)
    digest = payload["environment_manifest_sha256"]
    evidence = t1_engine.environment_path.parent / f"environment.{digest}.json"
    evidence.unlink()
    t5_engine = FakeTrainingEngine(tmp_path)

    with pytest.raises(ValueError, match="environment evidence is required"):
        run_training_stage(
            profile,
            stage="T5",
            run_id="run-3035",
            resume=resume,
            engine=t5_engine,
        )

    assert not evidence.exists()
    assert t5_engine.environment_path.is_file()
    assert t5_engine.global_step == 0


def test_checkpoint_environment_evidence_is_content_addressed_and_immutable(
    profile, tmp_path: Path
) -> None:
    """A checkpoint must authorize immutable bytes, not a mutable summary pathname."""

    engine = FakeTrainingEngine(tmp_path)
    run_training_stage(profile, stage="T1", run_id="run-3035", engine=engine)
    with (engine.checkpoint_dir / "last.pth").open("rb") as stream:
        payload = pickle.load(stream)
    digest = payload["environment_manifest_sha256"]
    evidence = engine.environment_path.parent / f"environment.{digest}.json"

    assert evidence.is_file()
    assert not evidence.is_symlink()
    assert hashlib.sha256(evidence.read_bytes()).hexdigest() == digest
    assert evidence.read_bytes() == engine.environment_path.read_bytes()


def test_resume_rejects_tampered_environment_without_overwriting_it(
    profile, tmp_path: Path
) -> None:
    """Manifest metadata cannot authorize environment bytes that no longer match."""

    t1_engine = FakeTrainingEngine(tmp_path)
    run_training_stage(profile, stage="T1", run_id="run-3035", engine=t1_engine)
    resume = t1_engine.checkpoint_dir / "last.pth"
    with resume.open("rb") as stream:
        payload = pickle.load(stream)
    digest = payload["environment_manifest_sha256"]
    evidence = t1_engine.environment_path.parent / f"environment.{digest}.json"
    tampered = b'{"tampered":true}\n'
    evidence.write_bytes(tampered)
    t5_engine = FakeTrainingEngine(tmp_path)

    with pytest.raises(ValueError, match="environment evidence SHA-256 mismatch"):
        run_training_stage(
            profile,
            stage="T5",
            run_id="run-3035",
            resume=resume,
            engine=t5_engine,
        )

    assert evidence.read_bytes() == tampered
    assert t5_engine.global_step == 0


def test_resume_rejects_symlinked_immutable_environment_evidence(
    profile, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A content-addressed filename must not authorize a symlink target."""

    t1_engine = FakeTrainingEngine(tmp_path)
    run_training_stage(profile, stage="T1", run_id="run-3035", engine=t1_engine)
    resume = t1_engine.checkpoint_dir / "last.pth"
    with resume.open("rb") as stream:
        payload = pickle.load(stream)
    digest = payload["environment_manifest_sha256"]
    evidence = t1_engine.environment_path.parent / f"environment.{digest}.json"
    original_is_symlink = Path.is_symlink
    monkeypatch.setattr(
        Path,
        "is_symlink",
        lambda self: self == evidence or original_is_symlink(self),
    )

    with pytest.raises(ValueError, match="immutable environment evidence is required"):
        run_training_stage(
            profile,
            stage="T5",
            run_id="run-3035",
            resume=resume,
            engine=FakeTrainingEngine(tmp_path),
        )


def test_crash_after_new_environment_write_keeps_old_checkpoint_resumable(
    profile, tmp_path: Path
) -> None:
    """A failed stage must not revoke the immutable evidence of its predecessor."""

    rejected_amp = AmpComparison(1.0, 1.0, 100.0, 101.0)
    t1_engine = FakeTrainingEngine(tmp_path)
    t1_engine.amp_comparison = rejected_amp
    run_training_stage(profile, stage="T1", run_id="run-3035", engine=t1_engine)
    old_resume = t1_engine.checkpoint_dir / "last.pth"
    with old_resume.open("rb") as stream:
        old_payload = pickle.load(stream)
    old_digest = old_payload["environment_manifest_sha256"]
    old_evidence = (
        t1_engine.environment_path.parent / f"environment.{old_digest}.json"
    )
    old_bytes = old_evidence.read_bytes()

    crashed = FakeTrainingEngine(tmp_path)
    crashed.resumed_amp_comparison = rejected_amp
    crashed.oom_on_batch8 = True
    crashed.loss = float("nan")
    with pytest.raises(FloatingPointError, match="non-finite.*loss"):
        run_training_stage(
            profile,
            stage="T5",
            run_id="run-3035",
            resume=old_resume,
            engine=crashed,
        )
    assert crashed.environment_path.read_bytes() == old_bytes
    assert old_evidence.read_bytes() == old_bytes

    retry = FakeTrainingEngine(tmp_path)
    retry.resumed_amp_comparison = rejected_amp
    retry.oom_on_batch8 = True
    retry_result = run_training_stage(
        profile,
        stage="T5",
        run_id="run-3035",
        resume=old_resume,
        engine=retry,
    )
    new_resume = retry.checkpoint_dir / "last.pth"
    with new_resume.open("rb") as stream:
        new_payload = pickle.load(stream)
    new_digest = retry_result.environment_sha256
    new_evidence = retry_result.environment_path
    assert new_digest != old_digest
    assert new_evidence.read_bytes() == retry.environment_path.read_bytes()
    assert new_payload["environment_manifest_sha256"] == new_digest

    old_evidence.unlink()
    t50_engine = FakeTrainingEngine(tmp_path)
    t50_engine.resumed_amp_comparison = rejected_amp
    t50_engine.oom_on_batch8 = True
    result = run_training_stage(
        profile,
        stage="T50",
        run_id="run-3035",
        resume=new_resume,
        engine=t50_engine,
    )
    assert result.completed_epoch == 50


def test_resume_manifest_filename_cannot_authorize_nested_checkpoint(
    profile, tmp_path: Path
) -> None:
    """A same-named nested file is not the manifest's root checkpoint artifact."""

    t1_engine = FakeTrainingEngine(tmp_path)
    run_training_stage(profile, stage="T1", run_id="run-3035", engine=t1_engine)
    nested = t1_engine.checkpoint_dir / "nested" / "last.pth"
    nested.parent.mkdir()
    nested.write_bytes((t1_engine.checkpoint_dir / "last.pth").read_bytes())

    with pytest.raises(ValueError, match="current run directory|manifest"):
        run_training_stage(
            profile,
            stage="T5",
            run_id="run-3035",
            resume=nested,
            engine=FakeTrainingEngine(tmp_path),
        )


def test_stage_ceiling_rejects_wrong_resume_predecessor(profile, tmp_path: Path) -> None:
    """T50 approval must not make a T1 checkpoint skip the approved T5 boundary."""

    t1_engine = FakeTrainingEngine(tmp_path)
    run_training_stage(profile, stage="T1", run_id="run-3035", engine=t1_engine)

    with pytest.raises(ValueError, match="T50.*T5|predecessor"):
        run_training_stage(
            profile,
            stage="T50",
            run_id="run-3035",
            resume=t1_engine.checkpoint_dir / "last.pth",
            engine=FakeTrainingEngine(tmp_path),
        )


def test_pinned_engine_defers_runtime_and_consumes_authoritative_synthesis(
    profile, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Importing or constructing the runner on Windows must not require CUDA/MMCV."""

    processed = tmp_path / "processed"
    processed.mkdir()
    train_ids = [f"img_{index:04d}" for index in range(961)]
    validation_ids = [f"img_{index:04d}" for index in range(961, 1201)]
    (processed / "train.txt").write_text("\n".join(train_ids) + "\n", encoding="utf-8")
    (processed / "val.txt").write_text(
        "\n".join(validation_ids) + "\n", encoding="utf-8"
    )
    manifests = processed / "manifests"
    manifests.mkdir()
    (manifests / "output-inventory.jsonl").write_text(
        "".join(json.dumps({"sample_id": sample_id}) + "\n" for sample_id in train_ids + validation_ids),
        encoding="utf-8",
    )
    backbone = tmp_path / "backbone.pth"
    backbone.write_bytes(b"backbone")
    local_profile = replace(
        profile,
        processed_root=processed.resolve(),
        imagenet_backbone=replace(
            profile.imagenet_backbone,
            path=backbone.resolve(),
            filename=backbone.name,
            byte_size=backbone.stat().st_size,
        ),
    )
    synthesis_calls: list[dict[str, object]] = []
    runtime_calls: list[tuple[dict[str, object], int, int]] = []

    def synthesize(**kwargs: object) -> dict[str, object]:
        synthesis_calls.append(dict(kwargs))
        return {
            "train": {"end_epoch": 800, "resume_path": None},
            "droneai": {
                "stage": "T1",
                "stage_stop_epoch": 1,
                "schedule_horizon_epochs": 800,
                "upstream_commit": local_profile.model_upstream.commit,
            },
        }

    def runtime_factory(**kwargs: object) -> object:
        runtime_calls.append(
            (
                kwargs["config"],  # type: ignore[arg-type]
                kwargs["physical_batch"],  # type: ignore[arg-type]
                kwargs["accumulation_steps"],  # type: ignore[arg-type]
            )
        )
        return object()

    monkeypatch.setattr(
        "droneai.steerer_training_runner.synthesize_official_config", synthesize
    )
    engine = PinnedUpstreamTrainingEngine(
        profile=local_profile,
        stage="T1",
        run_id="run-3035",
        processed_root=processed,
        upstream_dir=tmp_path / "STEERER",
        backbone_path=backbone,
        runtime_factory=runtime_factory,
    )

    assert synthesis_calls == []
    assert runtime_calls == []

    engine.configure_batch(8, 1)

    assert synthesis_calls[0]["profile"] is local_profile
    assert synthesis_calls[0]["processed_root"] == processed.resolve()
    assert synthesis_calls[0]["backbone_path"] == backbone.resolve()
    assert synthesis_calls[0]["physical_batch"] == 8
    assert synthesis_calls[0]["accumulation_steps"] == 1
    assert runtime_calls[0][1:] == (8, 1)
    assert engine.lineage.upstream_commit == local_profile.model_upstream.commit


def test_pinned_engine_rejects_incomplete_or_overlapping_prepared_split(
    profile, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A canonical path does not make mutated split contents approved."""

    processed = tmp_path / "processed"
    (processed / "manifests").mkdir(parents=True)
    (processed / "train.txt").write_text("same\n", encoding="utf-8")
    (processed / "val.txt").write_text("same\n", encoding="utf-8")
    (processed / "manifests" / "output-inventory.jsonl").write_text(
        json.dumps({"sample_id": "same"}) + "\n", encoding="utf-8"
    )
    local_profile = replace(profile, processed_root=processed.resolve())
    monkeypatch.setattr(
        "droneai.steerer_training_runner.synthesize_official_config",
        lambda **_kwargs: {
            "train": {"end_epoch": 800, "resume_path": None},
            "droneai": {
                "stage": "T1", "stage_stop_epoch": 1,
                "schedule_horizon_epochs": 800,
                "upstream_commit": local_profile.model_upstream.commit,
            },
        },
    )
    engine = PinnedUpstreamTrainingEngine(
        profile=local_profile,
        stage="T1",
        run_id="run-3035",
        processed_root=processed,
        upstream_dir=tmp_path / "STEERER",
        backbone_path=local_profile.imagenet_backbone.path,
        runtime_factory=lambda **_kwargs: object(),
    )

    with pytest.raises(ValueError, match="961/240|disjoint|complete"):
        engine.configure_batch(8, 1)


def test_cli_has_no_test_switch_and_requires_explicit_later_stage_approval() -> None:
    """A generic split or approval switch could expose sealed Test or skip a gate."""

    parser = training_cli._parser()
    option_strings = {
        option
        for action in parser._actions
        for option in action.option_strings
    }
    assert "--test" not in option_strings
    assert "--split" not in option_strings

    exit_code = training_cli.main(
        [
            "--config", str(PROFILE_PATH),
            "--stage", "T5",
            "--run-id", "run-3035",
            "--processed-root", "/workspace/data/datasets/ucf-qnrf-kaggle-apache/processed/steerer-training-v1",
            "--upstream-dir", "/workspace/upstreams/STEERER",
            "--backbone", "/workspace/data/checkpoints/backbones/hrnetv2_w48_imagenet_pretrained.pth",
            "--backbone-sha256", "0efec102d97f2ef58f0e258b2c3076b3704b93ffc2b73f64c8da5462c0037ef8",
            "--container-image-digest", CONTAINER_DIGEST,
            "--resume", "/workspace/data/checkpoints/steerer-ucf-training/run-3035/last.pth",
        ]
    )

    assert exit_code == 2


def test_cli_requires_matching_t800_approval_and_resume() -> None:
    parser = training_cli._parser()
    required = [
        "--config", str(PROFILE_PATH),
        "--stage", "T800",
        "--run-id", "run-3035",
        "--processed-root", "/workspace/data/datasets/ucf-qnrf-kaggle-apache/processed/steerer-training-v1",
        "--upstream-dir", "/workspace/upstreams/STEERER",
        "--backbone", "/workspace/data/checkpoints/backbones/hrnetv2_w48_imagenet_pretrained.pth",
        "--backbone-sha256", "0efec102d97f2ef58f0e258b2c3076b3704b93ffc2b73f64c8da5462c0037ef8",
        "--container-image-digest", CONTAINER_DIGEST,
        "--resume", "/workspace/data/checkpoints/steerer-ucf-training/run-3035/last.pth",
    ]

    approved = parser.parse_args(required + ["--approved-stage", "T800"])
    training_cli._validate_stage_ceiling(approved)

    mismatched = parser.parse_args(required + ["--approved-stage", "T50"])
    with pytest.raises(PermissionError, match="T800.*approved-stage"):
        training_cli._validate_stage_ceiling(mismatched)

    without_resume = parser.parse_args(
        [item for index, item in enumerate(required) if index not in {16, 17}]
        + ["--approved-stage", "T800"]
    )
    with pytest.raises(ValueError, match="T800.*resume"):
        training_cli._validate_stage_ceiling(without_resume)


def test_cli_backbone_hash_is_only_confirmation_of_profile_authority() -> None:
    """A caller-provided hash must never replace the checked-in approved hash."""

    exit_code = training_cli.main(
        [
            "--config", str(PROFILE_PATH),
            "--stage", "T0",
            "--run-id", "run-3035",
            "--processed-root", "/workspace/data/datasets/ucf-qnrf-kaggle-apache/processed/steerer-training-v1",
            "--upstream-dir", "/workspace/upstreams/STEERER",
            "--backbone", "/workspace/data/checkpoints/backbones/hrnetv2_w48_imagenet_pretrained.pth",
            "--backbone-sha256", "f" * 64,
            "--container-image-digest", CONTAINER_DIGEST,
        ]
    )

    assert exit_code == 2


def test_cli_rejects_test_backed_processed_root_even_when_profile_is_mutated(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A caller-supplied config must not redefine the canonical processed root to Test."""

    payload = json.loads(PROFILE_PATH.read_text(encoding="utf-8"))
    alternative = (
        "/workspace/data/datasets/ucf-qnrf-kaggle-apache/"
        "Test/processed/steerer-training-v1"
    )
    payload["storage"]["processed_root"] = alternative
    config = tmp_path / "mutated-profile.json"
    config.write_text(json.dumps(payload), encoding="utf-8")
    constructed = False

    class ForbiddenEngine:
        def __init__(self, **_kwargs: object) -> None:
            nonlocal constructed
            constructed = True

    monkeypatch.setattr(training_cli, "PinnedUpstreamTrainingEngine", ForbiddenEngine)

    exit_code = training_cli.main(
        [
            "--config", str(config),
            "--stage", "T0",
            "--run-id", "run-3035",
            "--processed-root", alternative,
            "--upstream-dir", "/workspace/upstreams/STEERER",
            "--backbone", "/workspace/data/checkpoints/backbones/hrnetv2_w48_imagenet_pretrained.pth",
            "--backbone-sha256", "0efec102d97f2ef58f0e258b2c3076b3704b93ffc2b73f64c8da5462c0037ef8",
            "--container-image-digest", CONTAINER_DIGEST,
        ]
    )

    assert exit_code == 2
    assert constructed is False


def test_cli_t0_constructs_pinned_engine_and_runs_hard_ceiling(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The CLI must route through the audited engine rather than upstream train_cc.py."""

    calls: dict[str, object] = {}

    class Engine:
        def __init__(self, **kwargs: object) -> None:
            calls["engine"] = kwargs

    expected = type(
        "Result",
        (),
        {
            "stage": "T0", "completed_epoch": 0, "optimizer_steps": 1,
            "validation_samples": 0, "finite_loss": True,
            "checkpoint_round_trip": True, "physical_batch": 8,
            "accumulation_steps": 1, "amp_enabled": True,
            "cuda_oom_evidence": None, "elapsed_seconds": 0.1,
            "metrics_path": Path("/workspace/data/results/run/metrics.json"),
            "metrics_sha256": "1" * 64,
            "environment_path": Path("/workspace/data/results/run/environment.json"),
            "environment_sha256": "2" * 64,
            "checkpoint_path": Path("/workspace/data/checkpoints/run/last.pth"),
            "checkpoint_sha256": "3" * 64,
        },
    )()

    def run(profile, **kwargs: object):
        calls["run"] = (profile, kwargs)
        return expected

    monkeypatch.setattr(training_cli, "PinnedUpstreamTrainingEngine", Engine)
    monkeypatch.setattr(training_cli, "run_training_stage", run)

    exit_code = training_cli.main(
        [
            "--config", str(PROFILE_PATH),
            "--stage", "T0",
            "--run-id", "run-3035",
            "--processed-root", "/workspace/data/datasets/ucf-qnrf-kaggle-apache/processed/steerer-training-v1",
            "--upstream-dir", "/workspace/upstreams/STEERER",
            "--backbone", "/workspace/data/checkpoints/backbones/hrnetv2_w48_imagenet_pretrained.pth",
            "--backbone-sha256", "0efec102d97f2ef58f0e258b2c3076b3704b93ffc2b73f64c8da5462c0037ef8",
            "--container-image-digest", CONTAINER_DIGEST,
        ]
    )

    assert exit_code == 0
    assert calls["engine"]["stage"] == "T0"  # type: ignore[index]
    assert calls["run"][1]["stage"] == "T0"  # type: ignore[index]
    assert calls["run"][1]["container_image_digest"] == CONTAINER_DIGEST  # type: ignore[index]


def test_cli_requires_an_actual_container_image_digest() -> None:
    """A Docker tag or missing digest cannot identify the runtime that produced weights."""

    action = next(
        action
        for action in training_cli._parser()._actions
        if action.dest == "container_image_digest"
    )
    assert action.required is True

    with pytest.raises(SystemExit):
        training_cli._parser().parse_args(
            ["--config", str(PROFILE_PATH), "--stage", "T0", "--run-id", "run"]
        )


def test_environment_json_records_rejected_nonfinite_amp_without_nan_token(
    profile, fake_engine
) -> None:
    """Evidence JSON must stay standards-compliant when AMP is rejected."""

    fake_engine.amp_comparison = AmpComparison(
        fp32_loss=1.0,
        amp_loss=float("nan"),
        fp32_count=100.0,
        amp_count=float("nan"),
    )

    run_training_stage(profile, stage="T0", run_id="run-3035", engine=fake_engine)

    text = fake_engine.environment_path.read_text(encoding="utf-8")
    assert "NaN" not in text
    assert json.loads(text)["schema_version"] == 2
    assert "amp" not in json.loads(text)


def test_runner_rejects_engine_constructed_for_another_stage_or_run(
    profile, fake_engine
) -> None:
    """A preconfigured engine must not bypass the requested stage/run lineage."""

    fake_engine.stage = "T1"
    fake_engine.run_id = "another-run"

    with pytest.raises(ValueError, match="engine.*stage|engine.*run"):
        run_training_stage(
            profile, stage="T0", run_id="run-3035", engine=fake_engine
        )


def test_checkpoint_rng_is_training_boundary_not_validation_side_effect(
    profile, fake_engine
) -> None:
    """T1 resume must begin where training stopped, independent of validation RNG use."""

    fake_engine.validation_consumes_rng = True

    run_training_stage(profile, stage="T1", run_id="run-3035", engine=fake_engine)

    with (fake_engine.checkpoint_dir / "last.pth").open("rb") as stream:
        payload = pickle.load(stream)
    assert payload["rng"]["torch_cpu"] == b"cpu-state"


def test_official_resize_memory_bookkeeping_is_preserved() -> None:
    """Omitting upstream resize memory changes later adaptive sampling behavior."""

    dataset = type("Dataset", (), {"resize_memory_pool": {}})()
    sizes = np.asarray([[768, 1024, 3], [500, 300, 3]])

    _record_resize_memory(
        dataset,
        sizes=sizes,
        names=("a", "b"),
        route_size=(256, 256),
    )

    assert dataset.resize_memory_pool["a"]["avg_size"].shape == (3, 4)
    assert dataset.resize_memory_pool["b"]["load_num"].shape == (2, 2)
