from __future__ import annotations

import copy
import json
import pickle
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

import scripts.run_steerer_a_training as a_training_cli
import droneai.steerer_official_training as official_training

from droneai.steerer_official_data import load_official_training_data_profile
from droneai.steerer_official_training import (
    A0Lineage,
    A0UpdateObservation,
    A1EpochObservation,
    A1TestObservation,
    A1TestSampleObservation,
    initialize_cuda_memory_stats,
    run_a0_gate,
    run_a1_gate,
    state_sha256,
)
from droneai.steerer_training_upstream import UpstreamAudit


REPO_ROOT = Path(__file__).resolve().parents[1]
A_TRAINING_PROFILE = (
    REPO_ROOT
    / "configs/training/steerer_ucf_qnrf_official_code_reproduction.home5090.json"
)


class _FakeCuda:
    def __init__(self) -> None:
        self.state = [b"cuda-state"]

    def get_rng_state_all(self) -> list[bytes]:
        return list(self.state)

    def set_rng_state_all(self, state: list[bytes]) -> None:
        self.state = list(state)


class _FakeTorch:
    def __init__(self) -> None:
        self.cpu_state = b"cpu-state"
        self.cuda = _FakeCuda()

    def get_rng_state(self) -> bytes:
        return self.cpu_state

    def set_rng_state(self, state: bytes) -> None:
        self.cpu_state = state

    def save(self, payload: object, path: str | Path) -> None:
        with Path(path).open("wb") as stream:
            pickle.dump(payload, stream)

    def load(self, path: str | Path, **_kwargs: object) -> object:
        with Path(path).open("rb") as stream:
            return pickle.load(stream)


class _FakeA0Runtime:
    def __init__(self, *, bad_step_count: bool = False) -> None:
        self.torch_module = _FakeTorch()
        self.global_step = 0
        self.restore_calls = 0
        self.test_access_count = 0
        self.bad_step_count = bad_step_count
        self.initialization_audit = {
            "loaded_scope": "backbone_only",
            "model_checkpoint_loaded": False,
            "observed_weight_load_count": 1,
        }
        self.cuda_evidence = {
            "available": True,
            "device": "cuda:0",
            "matmul_finite": True,
        }
        self._state = {
            "model": {"weight": np.asarray([1.0, 2.0], dtype=np.float32)},
            "optimizer": {"state": {0: {"step": 1}}, "param_groups": [{"lr": 1e-4}]},
            "scheduler": {"t": 0, "horizon": 800},
            "scaler": {},
        }

    def run_update(self) -> A0UpdateObservation:
        self.global_step += 2 if self.bad_step_count else 1
        self._state["model"]["weight"][0] = 1.5
        return A0UpdateObservation(
            optimizer_steps=2 if self.bad_step_count else 1,
            loss=2.5,
            density_values=(10.0, 20.0),
            gradient_norm=3.0,
            learning_rate=5e-7,
            elapsed_seconds=0.25,
            peak_vram_mb=512.0,
            train_sample_tokens=("0001", "0002"),
        )

    def checkpoint_state(self) -> dict[str, object]:
        return copy.deepcopy(self._state)

    def restore_checkpoint_state(
        self, state: dict[str, object], *, global_step: int
    ) -> None:
        self.restore_calls += 1
        self._state = copy.deepcopy(state)
        self.global_step = global_step

    def state_sha256s(self) -> dict[str, str]:
        return {
            name: state_sha256(self._state[name])
            for name in ("model", "optimizer", "scheduler", "scaler")
        }


class _FakeA1Runtime(_FakeA0Runtime):
    def __init__(
        self, *, optimizer_steps: int = 150, test_samples: int = 334
    ) -> None:
        super().__init__()
        self.optimizer_steps = optimizer_steps
        self.test_samples = test_samples

    def run_epoch1(self) -> A1EpochObservation:
        self.global_step = self.optimizer_steps
        self._state["model"]["weight"][0] = 1.75
        self._state["optimizer"]["state"][0]["step"] = self.global_step
        self._state["scheduler"]["t"] = max(self.global_step - 1, 0)
        return A1EpochObservation(
            epoch=1,
            optimizer_steps=self.optimizer_steps,
            losses=(2.5,) * self.optimizer_steps,
            density_values=(10.0, 20.0),
            gradient_norms=(3.0,) * self.optimizer_steps,
            learning_rates=(5e-7,) * self.optimizer_steps,
            elapsed_seconds=12.5,
            peak_vram_mb=1024.0,
            train_sample_tokens=tuple(
                f"{index:04d}" for index in range(self.optimizer_steps * 8)
            ),
        )

    def evaluate_test(self) -> A1TestObservation:
        self.test_access_count = self.test_samples
        samples = tuple(
            A1TestSampleObservation(
                sample_id=f"{index + 1202:04d}",
                predicted_count=100.0,
                ground_truth_count=110.0,
                signed_error=-10.0,
                absolute_error=10.0,
                squared_error=100.0,
                latency_ms=25.0,
                peak_vram_mb=768.0,
            )
            for index in range(self.test_samples)
        )
        return A1TestObservation(
            sample_count=self.test_samples,
            loss=1.25,
            mae=10.0,
            rmse=10.0,
            mape_percent=100.0 / 11.0,
            signed_bias=-10.0,
            median_latency_ms=25.0,
            fps=40.0,
            peak_vram_mb=768.0,
            samples=samples,
        )


def _lineage() -> A0Lineage:
    return A0Lineage(
        profile_sha256="1" * 64,
        split_sha256s={"train": "2" * 64, "test": "3" * 64},
        dataset_manifest_sha256="4" * 64,
        dataset_content_sha256="5" * 64,
        backbone_sha256=(
            "0efec102d97f2ef58f0e258b2c3076b3704b93ffc2b73f64c8da5462c0037ef8"
        ),
        upstream_commit="5b1854dbc2d280f2326d67c65515d8baf9083810",
        upstream_config_sha256="8" * 64,
        project_commit="9" * 40,
        container_image_digest="sha256:" + "a" * 64,
    )


def test_a_profile_exposes_shared_upstream_and_backbone_contract() -> None:
    profile = load_official_training_data_profile(A_TRAINING_PROFILE)

    assert profile.model_upstream.commit == profile.upstream_commit
    assert profile.model_upstream.config_path == Path("configs/QNRF_final.py")
    assert profile.imagenet_backbone.sha256 == (
        "0efec102d97f2ef58f0e258b2c3076b3704b93ffc2b73f64c8da5462c0037ef8"
    )
    assert profile.initialization == "imagenet_backbone_only"


def test_state_sha256_is_order_stable_and_tensor_sensitive() -> None:
    left = {"b": [2, 3], "a": np.asarray([1.0], dtype=np.float32)}
    right = {"a": np.asarray([1.0], dtype=np.float32), "b": [2, 3]}
    changed = {"a": np.asarray([1.1], dtype=np.float32), "b": [2, 3]}

    assert state_sha256(left) == state_sha256(right)
    assert state_sha256(left) != state_sha256(changed)


def test_cuda_memory_stats_initializes_context_before_reset() -> None:
    calls: list[str] = []

    class FakeCuda:
        def current_device(self) -> int:
            calls.append("current_device")
            return 0

        def reset_peak_memory_stats(self, device: object) -> None:
            calls.append(f"reset:{device}")
            if calls[0] != "current_device":
                raise RuntimeError("Invalid device argument")

    class FakeTorch:
        cuda = FakeCuda()

    observed = initialize_cuda_memory_stats(FakeTorch(), "cuda:0")

    assert observed == 0
    assert calls == ["current_device", "reset:cuda:0"]


def test_a0_config_selects_train_only_and_keeps_effective_batch_eight(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    profile = load_official_training_data_profile(A_TRAINING_PROFILE)
    processed = tmp_path / "processed"
    processed.mkdir()
    backbone = tmp_path / "backbone.pth"
    backbone.write_bytes(b"pinned")
    upstream = tmp_path / "STEERER"
    upstream.mkdir()
    config_path = upstream / "configs" / "QNRF_final.py"
    config_path.parent.mkdir()
    config_path.write_text("# audited", encoding="utf-8")
    audit = UpstreamAudit(
        origin_url="https://github.com/taohan10200/STEERER",
        commit=profile.upstream_commit,
        clean=True,
        license_sha256="a" * 64,
        config_path=config_path,
        config_sha256="b" * 64,
        _config_bytes=b"# audited",
    )
    monkeypatch.setattr(
        official_training,
        "verify_imagenet_backbone",
        lambda _profile, _path: backbone,
    )
    monkeypatch.setattr(
        official_training,
        "load_audited_official_config",
        lambda _profile, _upstream: (
            {
                "dataset": {},
                "train": {"end_epoch": 800},
                "network": {},
            },
            audit,
        ),
    )
    profile = replace(profile, processed_root=processed)

    config, observed_audit = official_training.synthesize_a0_config(
        profile,
        upstream_dir=upstream,
        processed_root=processed,
        backbone_path=backbone,
        run_id="steerer-a0-unit",
    )

    assert observed_audit is audit
    assert config["dataset"]["train_set"] == "train.txt"
    assert config["dataset"]["test_set"] == "__A0_TEST_NOT_ACCESSED__"
    assert config["train"]["batch_size_per_gpu"] == 8
    assert config["train"]["resume_path"] is None
    assert config["droneai"]["effective_batch"] == 8
    assert config["droneai"]["test_access"] is False


def test_a1_config_is_fresh_and_exposes_only_official_test334(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    profile = load_official_training_data_profile(A_TRAINING_PROFILE)
    processed = tmp_path / "processed"
    processed.mkdir()
    backbone = tmp_path / "backbone.pth"
    backbone.write_bytes(b"pinned")
    upstream = tmp_path / "STEERER"
    upstream.mkdir()
    config_path = upstream / "configs" / "QNRF_final.py"
    config_path.parent.mkdir()
    config_path.write_text("# audited", encoding="utf-8")
    audit = UpstreamAudit(
        origin_url="https://github.com/taohan10200/STEERER",
        commit=profile.upstream_commit,
        clean=True,
        license_sha256="a" * 64,
        config_path=config_path,
        config_sha256="b" * 64,
        _config_bytes=b"# audited",
    )
    monkeypatch.setattr(
        official_training, "verify_imagenet_backbone", lambda _profile, _path: backbone
    )
    monkeypatch.setattr(
        official_training,
        "load_audited_official_config",
        lambda _profile, _upstream: (
            {"dataset": {}, "train": {"end_epoch": 800}, "network": {}},
            audit,
        ),
    )
    profile = replace(profile, processed_root=processed)

    config, _ = official_training.synthesize_a1_config(
        profile,
        upstream_dir=upstream,
        processed_root=processed,
        backbone_path=backbone,
        run_id="steerer-a1-unit",
    )

    assert config["dataset"]["train_set"] == "train.txt"
    assert config["dataset"]["test_set"] == "test.txt"
    assert config["train"]["resume_path"] is None
    assert config["droneai"]["fresh_start"] is True
    assert config["droneai"]["test_selection"] == "official Test, test-selected"
    assert config["droneai"]["validation_access"] is False


def test_a0_gate_runs_exactly_one_update_and_restores_all_state(tmp_path: Path) -> None:
    profile = load_official_training_data_profile(A_TRAINING_PROFILE)
    runtime = _FakeA0Runtime()

    result = run_a0_gate(
        profile,
        run_id="steerer-a0-unit",
        runtime=runtime,
        lineage=_lineage(),
        output_dir=tmp_path / "results",
        checkpoint_dir=tmp_path / "checkpoints",
        environment={"python": "3.12.3", "torch": "fake", "cuda": "fake"},
    )

    assert result.report.score == 100
    assert result.report.status == "PASS_COMMERCIAL_CANDIDATE"
    assert result.checkpoint_path.name == "last.pth"
    assert result.checkpoint_sha256
    assert runtime.global_step == 1
    assert runtime.restore_calls == 1
    assert runtime.test_access_count == 0
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert manifest["stage"] == "A0"
    assert manifest["optimizer_steps"] == 1
    assert manifest["checkpoint_round_trip"] is True
    assert manifest["rng_round_trip"] is True
    assert manifest["test_access_count"] == 0
    assert manifest["training_settings"]["effective_batch"] == 8
    assert manifest["initialization"]["model_checkpoint_loaded"] is False
    assert not list(tmp_path.rglob("*.tmp"))


def test_a0_gate_rejects_more_than_one_optimizer_update(tmp_path: Path) -> None:
    profile = load_official_training_data_profile(A_TRAINING_PROFILE)

    with pytest.raises(RuntimeError, match="exactly one optimizer update"):
        run_a0_gate(
            profile,
            run_id="steerer-a0-too-many",
            runtime=_FakeA0Runtime(bad_step_count=True),
            lineage=_lineage(),
            output_dir=tmp_path / "results",
            checkpoint_dir=tmp_path / "checkpoints",
            environment={"python": "3.12.3"},
        )


def test_a1_gate_runs_fresh_full_epoch_and_complete_official_test(
    tmp_path: Path,
) -> None:
    profile = load_official_training_data_profile(A_TRAINING_PROFILE)
    runtime = _FakeA1Runtime()

    result = run_a1_gate(
        profile,
        run_id="steerer-a1-unit",
        runtime=runtime,
        lineage=_lineage(),
        output_dir=tmp_path / "results",
        checkpoint_dir=tmp_path / "checkpoints",
        environment={"python": "3.12.3", "torch": "fake", "cuda": "fake"},
    )

    assert result.report.score == 100
    assert result.report.status == "PASS_COMMERCIAL_CANDIDATE"
    assert runtime.global_step == 150
    assert runtime.test_access_count == 334
    assert runtime.restore_calls == 1
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert manifest["stage"] == "A1"
    assert manifest["epoch"] == 1
    assert manifest["optimizer_steps"] == 150
    assert manifest["train_sample_count"] == 1200
    assert manifest["test_sample_count"] == 334
    assert manifest["selection_protocol"] == "official Test, test-selected"
    assert manifest["fresh_start"] is True
    assert manifest["initialization"]["loaded_scope"] == "backbone_only"
    assert manifest["checkpoint"]["sha256"] == result.checkpoint_sha256
    assert not list(tmp_path.rglob("*.tmp"))


@pytest.mark.parametrize(
    ("optimizer_steps", "test_samples", "message"),
    [(149, 334, "150 optimizer updates"), (150, 333, "334 Test samples")],
)
def test_a1_gate_rejects_incomplete_epoch_or_test(
    tmp_path: Path, optimizer_steps: int, test_samples: int, message: str
) -> None:
    profile = load_official_training_data_profile(A_TRAINING_PROFILE)

    with pytest.raises(RuntimeError, match=message):
        run_a1_gate(
            profile,
            run_id=f"steerer-a1-{optimizer_steps}-{test_samples}",
            runtime=_FakeA1Runtime(
                optimizer_steps=optimizer_steps, test_samples=test_samples
            ),
            lineage=_lineage(),
            output_dir=tmp_path / "results",
            checkpoint_dir=tmp_path / "checkpoints",
            environment={"python": "3.12.3"},
        )


def test_a0_cli_has_no_resume_validation_test_or_model_checkpoint_override() -> None:
    parser = a_training_cli._parser()
    required = [
        "A0",
        "--config",
        str(A_TRAINING_PROFILE),
        "--run-id",
        "steerer-a0-unit",
        "--processed-root",
        "/workspace/data/datasets/qnrf/prepared",
        "--g2-manifest",
        "/workspace/data/datasets/qnrf/prepared/manifests/dataset-preparation.json",
        "--upstream-dir",
        "/workspace/upstreams/STEERER",
        "--backbone",
        "/workspace/data/checkpoints/backbones/hrnetv2_w48_imagenet_pretrained.pth",
        "--container-image-digest",
        "sha256:" + "a" * 64,
    ]

    parsed = parser.parse_args(required)
    assert parsed.stage == "A0"
    for forbidden in ("--resume", "--validation", "--test", "--model-checkpoint"):
        with pytest.raises(SystemExit):
            parser.parse_args(required + [forbidden, "/tmp/forbidden"])


def test_a1_cli_is_bounded_and_has_no_resume_or_checkpoint_override() -> None:
    parser = a_training_cli._parser()
    required = [
        "A1",
        "--config",
        str(A_TRAINING_PROFILE),
        "--run-id",
        "steerer-a1-unit",
        "--processed-root",
        "/workspace/data/datasets/qnrf/prepared",
        "--g2-manifest",
        "/workspace/data/datasets/qnrf/prepared/manifests/dataset-preparation.json",
        "--upstream-dir",
        "/workspace/upstreams/STEERER",
        "--backbone",
        "/workspace/data/checkpoints/backbones/hrnetv2_w48_imagenet_pretrained.pth",
        "--container-image-digest",
        "sha256:" + "a" * 64,
    ]

    parsed = parser.parse_args(required)
    assert parsed.stage == "A1"
    for forbidden in ("--resume", "--validation", "--model-checkpoint", "--epochs"):
        with pytest.raises(SystemExit):
            parser.parse_args(required + [forbidden, "/tmp/forbidden"])
