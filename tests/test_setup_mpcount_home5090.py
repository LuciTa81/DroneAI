from __future__ import annotations

import importlib
import json
import subprocess
from pathlib import Path
from typing import Sequence

import pytest


FROZEN = (
    "gdown==5.2.0",
    "opencv-python-headless==4.12.0.88",
    "PySocks==1.7.1",
    "pandas==2.2.2",
)


def _setup():
    return importlib.import_module("scripts.setup_mpcount_home5090")


def _overlay(path: Path, lines: Sequence[str] = FROZEN) -> Path:
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _venv(path: Path, *, system_packages: bool = True) -> Path:
    (path / "bin").mkdir(parents=True)
    (path / "bin" / "python").write_text("fixture", encoding="utf-8")
    (path / "pyvenv.cfg").write_text(
        "include-system-site-packages = "
        + ("true" if system_packages else "false")
        + "\n",
        encoding="utf-8",
    )
    return path


class FakeRunner:
    def __init__(self, *, cuda: bool = True) -> None:
        self.cuda = cuda
        self.calls: list[tuple[str, ...]] = []

    def __call__(self, command: Sequence[str]) -> subprocess.CompletedProcess[str]:
        call = tuple(str(part) for part in command)
        self.calls.append(call)
        if call[-3:] == ("-m", "pip", "check"):
            return subprocess.CompletedProcess(call, 0, "No broken requirements found.\n", "")
        if len(call) >= 2 and call[-2] == "-c" and "importlib.metadata" in call[-1]:
            payload = {
                "gdown": "5.2.0",
                "opencv-python-headless": "4.12.0.88",
                "PySocks": "1.7.1",
                "pandas": "2.2.2",
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
                "mpcount_imports": self.cuda,
            }
            return subprocess.CompletedProcess(call, 0, json.dumps(payload), "")
        return subprocess.CompletedProcess(call, 0, "", "")


def test_requirements_are_exact_and_forbid_torch(tmp_path: Path) -> None:
    setup = _setup()
    assert setup.parse_requirements(_overlay(tmp_path / "ok.txt")) == FROZEN

    with pytest.raises(ValueError, match="frozen MPCount allowlist"):
        setup.parse_requirements(
            _overlay(tmp_path / "bad.txt", (*FROZEN, "torch==2.0.1"))
        )


def test_setup_installs_overlay_without_dependencies_or_torch(tmp_path: Path) -> None:
    setup = _setup()
    runner = FakeRunner()
    result = setup.setup_environment(
        requirements_path=_overlay(tmp_path / "requirements.txt"),
        venv_path=_venv(tmp_path / "venv"),
        upstream_path=tmp_path / "MPCount",
        runner=runner,
    )

    install = next(call for call in runner.calls if "install" in call)
    assert install[-(len(FROZEN) + 1) :] == ("--no-deps", *FROZEN)
    assert not any(part.startswith(("torch==", "torchvision==")) for part in install)
    assert result["status"] == "PASS"
    assert result["cuda"]["cuda_available"] is True


def test_existing_venv_must_inherit_ngc_runtime(tmp_path: Path) -> None:
    setup = _setup()
    with pytest.raises(ValueError, match="include-system-site-packages=true"):
        setup.setup_environment(
            requirements_path=_overlay(tmp_path / "requirements.txt"),
            venv_path=_venv(tmp_path / "venv", system_packages=False),
            upstream_path=tmp_path / "MPCount",
            verify_only=True,
            runner=FakeRunner(),
        )


def test_verification_fails_closed_on_missing_cuda(tmp_path: Path) -> None:
    setup = _setup()
    with pytest.raises(RuntimeError, match="CUDA"):
        setup.setup_environment(
            requirements_path=_overlay(tmp_path / "requirements.txt"),
            venv_path=_venv(tmp_path / "venv"),
            upstream_path=tmp_path / "MPCount",
            verify_only=True,
            runner=FakeRunner(cuda=False),
        )
