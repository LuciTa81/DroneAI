from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Sequence

import pytest

from scripts import setup_pet_home5090 as setup


FROZEN = (
    "gdown==5.2.0",
    "opencv-python-headless==4.12.0.88",
    "PySocks==1.7.1",
)


def _overlay(path: Path, lines: Sequence[str] = FROZEN) -> Path:
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _venv(path: Path, *, system_packages: bool = True) -> Path:
    (path / "bin").mkdir(parents=True)
    (path / "bin" / "python").write_text("fixture", encoding="utf-8")
    (path / "pyvenv.cfg").write_text(
        "include-system-site-packages = " + ("true" if system_packages else "false") + "\n",
        encoding="utf-8",
    )
    return path


class FakeRunner:
    def __init__(self, *, cuda: bool = True, pip_check: str = "") -> None:
        self.cuda = cuda
        self.pip_check = pip_check
        self.calls: list[tuple[str, ...]] = []

    def __call__(self, command: Sequence[str]) -> subprocess.CompletedProcess[str]:
        call = tuple(str(part) for part in command)
        self.calls.append(call)
        if call[-3:] == ("-m", "pip", "check"):
            return subprocess.CompletedProcess(call, int(bool(self.pip_check)), self.pip_check, "")
        if len(call) >= 2 and call[-2] == "-c" and "importlib.metadata" in call[-1]:
            payload = {
                "gdown": "5.2.0",
                "opencv-python-headless": "4.12.0.88",
                "PySocks": "1.7.1",
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
            }
            return subprocess.CompletedProcess(call, 0, json.dumps(payload), "")
        return subprocess.CompletedProcess(call, 0, "", "")


def test_requirements_must_match_small_exact_allowlist(tmp_path: Path) -> None:
    assert setup.parse_requirements(_overlay(tmp_path / "ok.txt")) == FROZEN

    with pytest.raises(ValueError, match="frozen PET allowlist"):
        setup.parse_requirements(
            _overlay(tmp_path / "bad.txt", (*FROZEN, "torch==1.12.1"))
        )


def test_existing_venv_must_inherit_ngc_runtime(tmp_path: Path) -> None:
    path = _venv(tmp_path / "venv", system_packages=False)
    with pytest.raises(ValueError, match="include-system-site-packages=true"):
        setup.setup_environment(
            requirements_path=_overlay(tmp_path / "requirements.txt"),
            venv_path=path,
            verify_only=True,
            runner=FakeRunner(),
        )


def test_setup_installs_overlay_without_dependencies_or_torch(tmp_path: Path) -> None:
    runner = FakeRunner()
    result = setup.setup_environment(
        requirements_path=_overlay(tmp_path / "requirements.txt"),
        venv_path=_venv(tmp_path / "venv"),
        runner=runner,
    )

    install = next(call for call in runner.calls if "install" in call)
    assert install[-4:] == ("--no-deps", *FROZEN)
    assert not any(part.startswith(("torch==", "torchvision==")) for part in install)
    assert result["status"] == "PASS"
    assert result["cuda"]["cuda_available"] is True


def test_verification_fails_closed_on_missing_cuda(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="CUDA"):
        setup.setup_environment(
            requirements_path=_overlay(tmp_path / "requirements.txt"),
            venv_path=_venv(tmp_path / "venv"),
            verify_only=True,
            runner=FakeRunner(cuda=False),
        )


def test_cli_emits_machine_readable_status(tmp_path: Path, monkeypatch, capsys) -> None:
    runner = FakeRunner()
    monkeypatch.setattr(setup, "_run", runner)
    code = setup.main(
        [
            "--requirements",
            str(_overlay(tmp_path / "requirements.txt")),
            "--venv",
            str(_venv(tmp_path / "venv")),
            "--verify-only",
        ]
    )
    payload = json.loads(capsys.readouterr().out)
    assert code == 0
    assert payload["status"] == "PASS"
    assert payload["verify_only"] is True
