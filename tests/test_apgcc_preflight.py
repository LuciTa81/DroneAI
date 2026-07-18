from __future__ import annotations

import importlib
import json
import subprocess
from pathlib import Path
from typing import Sequence

import pytest


UPSTREAM_COMMIT = "e3e997bf592a70fd34233a432a974200d8c0c847"
FROZEN = (
    "gdown==5.2.0",
    "opencv-python-headless==4.12.0.88",
    "PySocks==1.7.1",
    "easydict==1.13",
    "tensorboardX==2.6.2.2",
)


def _setup():
    return importlib.import_module("scripts.setup_apgcc_home5090")


def _preflight():
    return importlib.import_module("scripts.run_apgcc_preflight")


def _artifact(path: Path, payload: bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return path


def _overlay(path: Path, lines: Sequence[str] = FROZEN) -> Path:
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _venv(path: Path) -> Path:
    (path / "bin").mkdir(parents=True)
    (path / "bin/python").write_text("fixture", encoding="utf-8")
    (path / "pyvenv.cfg").write_text(
        "include-system-site-packages = true\n", encoding="utf-8"
    )
    return path


class SetupRunner:
    def __init__(self, *, cuda: bool = True) -> None:
        self.cuda = cuda
        self.calls: list[tuple[str, ...]] = []

    def __call__(self, command: Sequence[str]) -> subprocess.CompletedProcess[str]:
        call = tuple(str(part) for part in command)
        self.calls.append(call)
        if call[-3:] == ("-m", "pip", "check"):
            return subprocess.CompletedProcess(
                call, 0, "No broken requirements found.\n", ""
            )
        if len(call) >= 2 and call[-2] == "-c" and "importlib.metadata" in call[-1]:
            payload = {
                "gdown": "5.2.0",
                "opencv-python-headless": "4.12.0.88",
                "PySocks": "1.7.1",
                "easydict": "1.13",
                "tensorboardX": "2.6.2.2",
                "opencv-python": None,
            }
            return subprocess.CompletedProcess(call, 0, json.dumps(payload), "")
        if len(call) >= 2 and call[-2] == "-c" and "torch.cuda" in call[-1]:
            payload = {
                "cuda_available": self.cuda,
                "torch": "2.9-fixture",
                "torchvision": "0.24-fixture",
                "gpu": "RTX 5090 fixture" if self.cuda else None,
                "matmul": 32.0 if self.cuda else None,
                "apgcc_imports": self.cuda,
            }
            return subprocess.CompletedProcess(call, 0, json.dumps(payload), "")
        return subprocess.CompletedProcess(call, 0, "", "")


class GitRunner:
    def __init__(
        self,
        *,
        upstream_commit: str = UPSTREAM_COMMIT,
        upstream_dirty: str = "",
        repo_commit: str = "d" * 40,
        repo_dirty: str = "",
    ) -> None:
        self.upstream_commit = upstream_commit
        self.upstream_dirty = upstream_dirty
        self.repo_commit = repo_commit
        self.repo_dirty = repo_dirty

    def __call__(self, command: Sequence[str]) -> subprocess.CompletedProcess[str]:
        call = tuple(str(part) for part in command)
        is_repo = Path(call[2]).name == "DroneAI"
        if call[-2:] == ("rev-parse", "HEAD"):
            commit = self.repo_commit if is_repo else self.upstream_commit
            return subprocess.CompletedProcess(call, 0, commit + "\n", "")
        if call[-2:] == ("status", "--porcelain"):
            dirty = self.repo_dirty if is_repo else self.upstream_dirty
            return subprocess.CompletedProcess(call, 0, dirty, "")
        raise AssertionError(f"unexpected command: {call}")


def test_apgcc_overlay_is_exact_and_does_not_replace_ngc_torch(tmp_path: Path) -> None:
    setup = _setup()
    assert setup.parse_requirements(_overlay(tmp_path / "ok.txt")) == FROZEN

    with pytest.raises(ValueError, match="frozen APGCC allowlist"):
        setup.parse_requirements(_overlay(tmp_path / "bad.txt", (*FROZEN, "torch==1.12")))


def test_apgcc_setup_verifies_imports_and_cuda_without_inference(tmp_path: Path) -> None:
    setup = _setup()
    runner = SetupRunner()
    result = setup.setup_environment(
        requirements_path=_overlay(tmp_path / "requirements.txt"),
        venv_path=_venv(tmp_path / "venv"),
        upstream_path=tmp_path / "APGCC",
        runner=runner,
    )

    install = next(call for call in runner.calls if "install" in call)
    assert install[-(len(FROZEN) + 1) :] == ("--no-deps", *FROZEN)
    assert not any(part.startswith(("torch==", "torchvision==")) for part in install)
    assert result["status"] == "PASS"
    assert result["cuda"]["cuda_available"] is True
    assert result["cuda"]["apgcc_imports"] is True
    assert result["inference_executed"] is False


def test_capture_apgcc_preflight_hashes_frozen_identities(tmp_path: Path) -> None:
    preflight = _preflight()
    result = preflight.capture_preflight(
        environment={"status": "PASS", "cuda": {"cuda_available": True}},
        repo_path=tmp_path / "DroneAI",
        upstream_path=tmp_path / "APGCC",
        expected_upstream_commit=UPSTREAM_COMMIT,
        checkpoint_path=_artifact(tmp_path / "SHHA_best.pth", b"checkpoint"),
        dataset_config_path=_artifact(tmp_path / "dataset.json", b"dataset"),
        split_manifest_path=_artifact(tmp_path / "split.json", b"split"),
        sample_manifest_path=_artifact(tmp_path / "samples.json", b"samples"),
        model_profile_path=_artifact(tmp_path / "model.json", b"model"),
        runner=GitRunner(),
    )

    assert result["status"] == "PASS"
    assert result["model_id"] == "apgcc"
    assert result["inference_executed"] is False
    assert result["upstream"]["commit"] == UPSTREAM_COMMIT
    assert result["checkpoint"]["sha256"] == preflight.sha256_file(
        tmp_path / "SHHA_best.pth"
    )
    assert result["model_profile"]["sha256"] == preflight.sha256_file(
        tmp_path / "model.json"
    )


def test_capture_apgcc_preflight_rejects_dirty_upstream(tmp_path: Path) -> None:
    preflight = _preflight()
    with pytest.raises(ValueError, match="upstream repository is dirty"):
        preflight.capture_preflight(
            environment={"status": "PASS", "cuda": {"cuda_available": True}},
            repo_path=tmp_path / "DroneAI",
            upstream_path=tmp_path / "APGCC",
            expected_upstream_commit=UPSTREAM_COMMIT,
            checkpoint_path=_artifact(tmp_path / "checkpoint.pth", b"checkpoint"),
            dataset_config_path=_artifact(tmp_path / "dataset.json", b"dataset"),
            split_manifest_path=_artifact(tmp_path / "split.json", b"split"),
            sample_manifest_path=_artifact(tmp_path / "samples.json", b"samples"),
            model_profile_path=_artifact(tmp_path / "model.json", b"model"),
            runner=GitRunner(upstream_dirty=" M apgcc/models/APGCC.py\n"),
        )


def test_apgcc_model_profile_pins_native_point_contract() -> None:
    profile = json.loads(
        Path("configs/models/apgcc.official.json").read_text(encoding="utf-8")
    )
    assert profile["repository"]["commit"] == UPSTREAM_COMMIT
    assert profile["architecture"]["backbone"] == "VGG16-BN"
    assert profile["architecture"]["decoder"] == "IFI"
    assert profile["outputs"]["native"] == ["pred_logits", "pred_points", "offset"]
    assert profile["outputs"]["count_derivation"] == (
        "number of in-bounds pred_points whose person-class probability exceeds 0.5"
    )
    assert profile["checkpoint"]["training_dataset"] == "ShanghaiTech Part A"
