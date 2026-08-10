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
    run_a0_gate,
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

    def restore_checkpoint_state(self, state: dict[str, object]) -> None:
        self.restore_calls += 1
        self._state = copy.deepcopy(state)

    def state_sha256s(self) -> dict[str, str]:
        return {
            name: state_sha256(self._state[name])
            for name in ("model", "optimizer", "scheduler", "scaler")
        }


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
