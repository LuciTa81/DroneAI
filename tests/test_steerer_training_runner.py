from __future__ import annotations

from dataclasses import replace
import json
import pickle
from pathlib import Path

import numpy as np
import pytest

import scripts.run_steerer_ucf_training as training_cli

from droneai.steerer_training_profile import load_training_profile
from droneai.steerer_training_runner import (
    AmpComparison,
    EpochObservation,
    TrainingLineage,
    PinnedUpstreamTrainingEngine,
    _record_resize_memory,
    UpdateObservation,
    ValidationObservation,
    run_epochs,
    run_training_stage,
)


PROFILE_PATH = Path("configs/training/steerer_ucf_qnrf_imagenet.home5090.json")


class FakeTrainingEngine:
    def __init__(self, tmp_path: Path) -> None:
        self.loss = 1.25
        self.density_values = (10.0, 5.0)
        self.gradient_norm = 0.5
        self.validation_count = 240
        self.validation_loss = 0.75
        self.validation_counts = (10.0, 12.0)
        self.completed_epoch = 0
        self.global_step = 0
        self.learning_rates: list[float] = []
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
        self.probe_error: Exception | None = None
        self.empty_cache_calls = 0
        self.amp_flags: list[bool] = []
        self.amp_comparison = AmpComparison(
            fp32_loss=1.0,
            amp_loss=1.0,
            fp32_count=100.0,
            amp_count=100.05,
        )
        self.torch_module = _FakeTorch()
        self.best_mae = 1.0e20
        self.best_rmse = 1.0e20
        self.validation_consumes_rng = False

    def set_seed(self, seed: int) -> None:
        self.seed = seed

    def configure_batch(self, physical_batch: int, accumulation_steps: int) -> None:
        self.batch_plans.append((physical_batch, accumulation_steps))

    def probe_batch(self) -> None:
        if self.probe_error is not None:
            raise self.probe_error
        if self.oom_on_batch8 and self.batch_plans[-1] == (8, 1):
            raise RuntimeError("CUDA out of memory while probing batch 8")
        return None

    def empty_cuda_cache(self) -> None:
        self.empty_cache_calls += 1

    def compare_amp_to_fp32(self) -> AmpComparison:
        return self.amp_comparison

    def run_update(self, *, amp_enabled: bool) -> UpdateObservation:
        self.amp_flags.append(amp_enabled)
        self.global_step += 1
        lr = self._lr(self.global_step)
        self.learning_rates.append(lr)
        return UpdateObservation(
            loss=self.loss,
            density_values=self.density_values,
            gradient_norm=self.gradient_norm,
            learning_rate=lr,
        )

    def run_epoch(self, epoch: int, *, amp_enabled: bool) -> EpochObservation:
        del amp_enabled
        rates = []
        for _ in range(2):
            observation = self.run_update(amp_enabled=False)
            rates.append(observation.learning_rate)
        self.completed_epoch = epoch
        return EpochObservation(
            epoch=epoch,
            optimizer_steps=2,
            losses=(self.loss, self.loss),
            density_values=self.density_values,
            gradient_norms=(self.gradient_norm, self.gradient_norm),
            learning_rates=tuple(rates),
        )

    def validate(self) -> ValidationObservation:
        if self.validation_consumes_rng:
            self.torch_module.cpu_state = b"validation-consumed"
        return ValidationObservation(
            sample_count=self.validation_count,
            loss=self.validation_loss,
            counts=self.validation_counts,
            mae=2.0,
            rmse=2.0,
        )

    def checkpoint_state(self) -> dict[str, object]:
        return {
            "model": {"weight": self.global_step},
            "optimizer": {"step": self.global_step},
            "scheduler": {"global_step": self.global_step, "horizon": 800},
            "scaler": {"enabled": True},
        }

    def restore_checkpoint_state(self, state: dict[str, object]) -> None:
        optimizer = state["optimizer"]
        scheduler = state["scheduler"]
        assert isinstance(optimizer, dict) and isinstance(scheduler, dict)
        self.global_step = int(optimizer["step"])
        assert scheduler["horizon"] == 800

    def snapshot(self) -> tuple[int, int, tuple[float, ...]]:
        return self.completed_epoch, self.global_step, tuple(self.learning_rates)

    def restore_snapshot(self, snapshot: tuple[int, int, tuple[float, ...]]) -> None:
        self.completed_epoch, self.global_step, rates = snapshot
        self.learning_rates = list(rates)

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


def test_t1_runs_exactly_one_epoch_and_full_validation(profile, fake_engine) -> None:
    """Stopping early or sampling validation would violate the T1 ceiling."""

    result = run_training_stage(
        profile, stage="T1", run_id="run-3035", engine=fake_engine
    )

    assert result.completed_epoch == 1
    assert result.optimizer_steps == 2
    assert result.validation_samples == 240


def test_t1_to_t5_resume_matches_uninterrupted_lr_sequence(fake_engine) -> None:
    """Restarting the 800-epoch scheduler at T5 changes the experiment."""

    staged = run_epochs(fake_engine, stops=(1, 5), schedule_horizon=800)
    uninterrupted = run_epochs(fake_engine, stops=(5,), schedule_horizon=800)

    assert staged.learning_rates == uninterrupted.learning_rates
    assert staged.learning_rates[:2] == pytest.approx((0.0001, 0.0000999375))


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
    assert "CUDA out of memory" in environment["cuda_oom_evidence"]


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

    run_training_stage(profile, stage="T0", run_id="run-3035", engine=fake_engine)

    assert fake_engine.amp_flags == [accepted]
    environment = json.loads(fake_engine.environment_path.read_text(encoding="utf-8"))
    assert environment["amp"]["enabled"] is accepted


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
    assert any(row["filename"] == "last.pth" for row in manifest["checkpoints"])


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
    assert {2, 3, 4, 5}.issubset(t5_engine.torch_module.saved_epochs)


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
            "--resume", "/workspace/data/checkpoints/steerer-ucf-training/run-3035/last.pth",
        ]
    )

    assert exit_code == 2


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
        ]
    )

    assert exit_code == 2


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
            "accumulation_steps": 1, "elapsed_seconds": 0.1,
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
        ]
    )

    assert exit_code == 0
    assert calls["engine"]["stage"] == "T0"  # type: ignore[index]
    assert calls["run"][1]["stage"] == "T0"  # type: ignore[index]


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
    assert json.loads(text)["amp"]["amp_loss"] is None


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
