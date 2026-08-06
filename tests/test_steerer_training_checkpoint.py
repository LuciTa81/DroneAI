from __future__ import annotations

import json
import pickle
import random
from pathlib import Path

import numpy as np
import pytest

from droneai.steerer_training_checkpoint import (
    CHECKPOINT_PAYLOAD_KEYS,
    capture_rng_state,
    load_training_checkpoint,
    restore_rng_state,
    save_checkpoint_with_policy,
    save_training_checkpoint,
    verify_resume,
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


class _FakeCpuTorch:
    def __init__(self) -> None:
        self.cpu_state = b"cpu-state"

    def get_rng_state(self) -> bytes:
        return self.cpu_state

    def set_rng_state(self, state: bytes) -> None:
        self.cpu_state = state


def _state(*, epoch: int = 1, stage: str = "T1") -> dict[str, object]:
    return {
        "schema_version": 1,
        "run_id": "run-3035",
        "stage": stage,
        "epoch": epoch,
        "global_step": 16,
        "model": {"weight": [1, 2, 3]},
        "optimizer": {"lr": 0.0001, "step": 16},
        "scheduler": {"last_epoch": epoch, "horizon": 800},
        "scaler": {"scale": 65536.0},
        "best_mae": 12.5,
        "best_rmse": 21.0,
        "config_sha256": "a" * 64,
        "split_sha256s": {"train": "b" * 64, "validation": "c" * 64},
        "dataset_inventory_sha256": "d" * 64,
        "backbone_sha256": "e" * 64,
        "upstream_commit": "f" * 40,
        "rng": {"seed": 3035, "python": (3, (), None), "numpy": ("MT19937", np.array([1], dtype=np.uint32), 0, 0, 0.0), "torch_cpu": b"cpu-state", "torch_cuda": [b"cuda-state"]},
        "environment_manifest_sha256": "0" * 64,
    }


def test_checkpoint_round_trip_restores_optimizer_scheduler_and_rng(tmp_path: Path) -> None:
    state = _state()
    saved = save_training_checkpoint(tmp_path / "last.pth", state, torch_module=_FakeTorch())

    restored = load_training_checkpoint(
        saved.path, expected_sha256=saved.sha256, torch_module=_FakeTorch()
    )

    assert set(restored) == CHECKPOINT_PAYLOAD_KEYS
    assert restored["epoch"] == 1
    assert restored["optimizer"] == state["optimizer"]
    assert restored["scheduler"] == state["scheduler"]
    assert restored["scaler"] == state["scaler"]
    assert restored["rng"]["seed"] == 3035  # type: ignore[index]


def test_resume_rejects_hash_mismatch_and_wrong_lineage(tmp_path: Path) -> None:
    checkpoint = save_training_checkpoint(
        tmp_path / "last.pth", _state(), torch_module=_FakeTorch()
    )

    with pytest.raises(ValueError, match="SHA-256"):
        verify_resume(
            checkpoint,
            expected_sha256="0" * 64,
            expected_run_id="another-run",
            torch_module=_FakeTorch(),
        )
    with pytest.raises(ValueError, match="lineage"):
        verify_resume(
            checkpoint,
            expected_sha256=checkpoint.sha256,
            expected_run_id="another-run",
            torch_module=_FakeTorch(),
        )


def test_checkpoint_rejects_missing_or_extra_payload_keys(tmp_path: Path) -> None:
    missing = _state()
    del missing["scaler"]
    with pytest.raises(ValueError, match="payload keys"):
        save_training_checkpoint(tmp_path / "missing.pth", missing, torch_module=_FakeTorch())

    extra = _state()
    extra["untracked"] = True
    with pytest.raises(ValueError, match="payload keys"):
        save_training_checkpoint(tmp_path / "extra.pth", extra, torch_module=_FakeTorch())


def test_checkpoint_rejects_non_hex_upstream_commit(tmp_path: Path) -> None:
    state = _state()
    state["upstream_commit"] = "z" * 40

    with pytest.raises(ValueError, match="upstream commit"):
        save_training_checkpoint(tmp_path / "invalid-commit.pth", state, torch_module=_FakeTorch())


def test_captured_rng_restores_python_numpy_and_torch_state() -> None:
    fake_torch = _FakeTorch()
    random.seed(3035)
    np.random.seed(3035)
    captured = capture_rng_state(3035, torch_module=fake_torch)
    expected_python = random.random()
    expected_numpy = float(np.random.random())

    random.random()
    np.random.random()
    fake_torch.cpu_state = b"changed"
    fake_torch.cuda.state = [b"changed-cuda"]
    restore_rng_state(captured, torch_module=fake_torch)

    assert random.random() == expected_python
    assert float(np.random.random()) == expected_numpy
    assert fake_torch.cpu_state == b"cpu-state"
    assert fake_torch.cuda.state == [b"cuda-state"]


def test_rng_capture_and_restore_allow_cpu_only_torch() -> None:
    fake_torch = _FakeCpuTorch()

    captured = capture_rng_state(3035, torch_module=fake_torch)
    fake_torch.cpu_state = b"changed"
    restore_rng_state(captured, torch_module=fake_torch)

    assert captured["torch_cuda"] == []
    assert fake_torch.cpu_state == b"cpu-state"


def test_checkpoint_policy_writes_last_best_milestone_and_atomic_manifest(
    tmp_path: Path,
) -> None:
    state = _state(epoch=5, stage="T5")

    result = save_checkpoint_with_policy(
        tmp_path,
        state,
        current_mae=12.5,
        current_rmse=21.0,
        parent_checkpoint_sha256="1" * 64,
        torch_module=_FakeTorch(),
    )

    assert set(result.artifacts) == {"last", "best-mae", "best-rmse", "milestone-005"}
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert not list(tmp_path.glob("*.tmp"))
    assert {entry["filename"] for entry in manifest["checkpoints"]} == {
        "last.pth", "best-mae.pth", "best-rmse.pth", "milestone-005.pth"
    }
    assert all(entry["parent_checkpoint_sha256"] == "1" * 64 for entry in manifest["checkpoints"])
    assert all(entry["byte_count"] > 0 for entry in manifest["checkpoints"])
