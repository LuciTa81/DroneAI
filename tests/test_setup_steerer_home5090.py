from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Sequence

import pytest

from scripts import setup_steerer_home5090 as setup


MMCV = "mmcv==1.7.2"
HEADLESS = "opencv-python-headless==4.12.0.88"
FROZEN_REQUIREMENTS = (
    MMCV,
    "addict==2.4.0",
    "packaging==25.0",
    "Pillow==11.3.0",
    "PyYAML==6.0.2",
    "yapf==0.43.0",
    HEADLESS,
    "timm==1.0.19",
    "einops==0.8.1",
    "dict_recursive_update==1.0.1",
    "yacs==0.1.8",
    "tensorboardX==2.6.2.2",
    "fvcore==0.1.5.post20221221",
)
FROZEN_VERSIONS = {
    name.replace("_", "-").lower(): version
    for requirement in FROZEN_REQUIREMENTS
    for name, _, version in (requirement.partition("=="),)
}
KNOWN_PIP_CHECK_LINES = (
    "mmcv 1.7.2 requires opencv-python, which is not installed.",
    "nvidia-resiliency-ext 0.4.1+cuda13 requires pynvml, which is not installed.",
)


def _write_overlay(path: Path, *extra_lines: str) -> Path:
    lines = [*FROZEN_REQUIREMENTS, *extra_lines]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _make_existing_venv(
    path: Path, *, include_system_site_packages: str | None = "true"
) -> Path:
    (path / "bin").mkdir(parents=True)
    config = "home = /usr/local/bin\n"
    if include_system_site_packages is not None:
        config += (
            "include-system-site-packages = "
            f"{include_system_site_packages}\n"
        )
    (path / "pyvenv.cfg").write_text(config, encoding="utf-8")
    (path / "bin" / "python").write_text("fake python\n", encoding="utf-8")
    return path


class FakeRunner:
    def __init__(
        self,
        *,
        gui_version: str | None = None,
        headless_version: str | None = "4.12.0.88",
        installed_overrides: dict[str, str | None] | None = None,
        import_returncode: int = 0,
        pip_check_lines: Sequence[str] = KNOWN_PIP_CHECK_LINES,
    ) -> None:
        self.gui_version = gui_version
        self.headless_version = headless_version
        self.installed_versions: dict[str, str | None] = dict(FROZEN_VERSIONS)
        self.installed_versions["opencv-python-headless"] = headless_version
        self.installed_versions.update(installed_overrides or {})
        self.import_returncode = import_returncode
        self.pip_check_lines = tuple(pip_check_lines)
        self.calls: list[tuple[str, ...]] = []

    def __call__(self, command: Sequence[str]) -> subprocess.CompletedProcess[str]:
        call = tuple(str(part) for part in command)
        self.calls.append(call)
        if call[-3:] == ("-m", "pip", "check"):
            output = "\n".join(self.pip_check_lines)
            if output:
                output += "\n"
            return subprocess.CompletedProcess(call, int(bool(output)), output, "")
        if len(call) >= 2 and call[-2] == "-c" and "importlib.metadata" in call[-1]:
            output = json.dumps(
                {**self.installed_versions, "opencv-python": self.gui_version}
            )
            return subprocess.CompletedProcess(call, 0, output + "\n", "")
        if len(call) >= 2 and call[-2] == "-c" and "import cv2" in call[-1]:
            return subprocess.CompletedProcess(
                call,
                self.import_returncode,
                "",
                "import failed" if self.import_returncode else "",
            )
        return subprocess.CompletedProcess(call, 0, "", "")


@pytest.mark.parametrize(
    "bad_line",
    [
        "mmcv>=1.7.2",
        "opencv-python-headless>=4.12.0.88",
        "opencv-python==5.0.0.93",
    ],
)
def test_requirement_parser_rejects_unpinned_mmcv_or_opencv(
    tmp_path: Path, bad_line: str
) -> None:
    overlay = _write_overlay(tmp_path / "requirements.txt")
    text = overlay.read_text(encoding="utf-8")
    if bad_line.startswith("mmcv"):
        text = text.replace(MMCV, bad_line)
    elif bad_line.startswith("opencv-python-headless"):
        text = text.replace(HEADLESS, bad_line)
    else:
        text += bad_line + "\n"
    overlay.write_text(text, encoding="utf-8")

    with pytest.raises(ValueError, match="MMCV|OpenCV|opencv"):
        setup.parse_requirements(overlay)


@pytest.mark.parametrize("duplicate", [MMCV, HEADLESS])
def test_requirement_parser_rejects_duplicate_mmcv_or_opencv(
    tmp_path: Path, duplicate: str
) -> None:
    overlay = _write_overlay(tmp_path / "requirements.txt", duplicate)

    with pytest.raises(ValueError, match="duplicate"):
        setup.parse_requirements(overlay)


@pytest.mark.parametrize(
    "forbidden_requirement",
    [
        "torch==0.0.0",
        "opencv-contrib-python==4.12.0.88",
    ],
)
def test_requirement_parser_rejects_torch_or_opencv_contrib(
    tmp_path: Path, forbidden_requirement: str
) -> None:
    overlay = _write_overlay(tmp_path / "requirements.txt", forbidden_requirement)

    with pytest.raises(ValueError, match="frozen 13-package allowlist"):
        setup.parse_requirements(overlay)


@pytest.mark.parametrize("mutation", ["extra", "missing", "changed"])
def test_requirement_parser_rejects_any_frozen_allowlist_mutation(
    tmp_path: Path, mutation: str
) -> None:
    lines = list(FROZEN_REQUIREMENTS)
    if mutation == "extra":
        lines.append("surprise==1.0.0")
    elif mutation == "missing":
        lines.remove("timm==1.0.19")
    else:
        lines[lines.index("einops==0.8.1")] = "einops==0.8.2"
    overlay = tmp_path / "requirements.txt"
    overlay.write_text("\n".join(lines) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="frozen 13-package allowlist"):
        setup.parse_requirements(overlay)


def test_setup_reuses_existing_venv_and_installs_mmcv_last_without_deps(
    tmp_path: Path,
) -> None:
    overlay = _write_overlay(tmp_path / "requirements.txt")
    venv = _make_existing_venv(tmp_path / "venv")
    marker = venv / "keep.txt"
    marker.write_text("preserve", encoding="utf-8")
    runner = FakeRunner()

    summary = setup.setup_environment(
        requirements_path=overlay,
        venv_path=venv,
        runner=runner,
    )

    pip_mutation_calls = [
        call
        for call in runner.calls
        if call[1:3] == ("-m", "pip") and call[3] in {"install", "uninstall"}
    ]
    assert pip_mutation_calls[0][3:] == ("uninstall", "-y", "opencv-python")
    assert pip_mutation_calls[1][3] == "install"
    assert pip_mutation_calls[1][4:] == tuple(
        line
        for line in overlay.read_text(encoding="utf-8").splitlines()
        if line != MMCV
    )
    assert pip_mutation_calls[2][3:] == ("install", "--no-deps", MMCV)
    assert pip_mutation_calls[3][3:] == (
        "install",
        "--no-deps",
        "--force-reinstall",
        HEADLESS,
    )
    assert len(pip_mutation_calls) == 4
    assert not any(
        token.startswith("opencv-python==")
        for call in pip_mutation_calls
        for token in call
    )
    assert all(call[1:3] != ("-m", "venv") for call in runner.calls)
    assert marker.read_text(encoding="utf-8") == "preserve"
    assert summary["venv_created"] is False
    assert summary["status"] == "PASS"


def test_verify_only_performs_no_create_or_install(tmp_path: Path) -> None:
    overlay = _write_overlay(tmp_path / "requirements.txt")
    venv = _make_existing_venv(tmp_path / "venv")
    runner = FakeRunner()

    summary = setup.setup_environment(
        requirements_path=overlay,
        venv_path=venv,
        verify_only=True,
        runner=runner,
    )

    assert summary["verify_only"] is True
    assert not any(call[1:3] == ("-m", "venv") for call in runner.calls)
    assert not any(
        call[1:3] == ("-m", "pip") and call[3] in {"install", "uninstall"}
        for call in runner.calls
    )


@pytest.mark.parametrize("setting", ["false", None])
def test_setup_rejects_venv_without_system_site_packages_before_commands(
    tmp_path: Path, setting: str | None
) -> None:
    overlay = _write_overlay(tmp_path / "requirements.txt")
    venv = _make_existing_venv(
        tmp_path / "venv", include_system_site_packages=setting
    )
    runner = FakeRunner()

    with pytest.raises(ValueError, match="include-system-site-packages=true"):
        setup.setup_environment(
            requirements_path=overlay,
            venv_path=venv,
            runner=runner,
        )

    assert runner.calls == []


def test_pip_check_allows_only_the_two_exact_known_lines() -> None:
    assert setup.validate_pip_check("\n".join(KNOWN_PIP_CHECK_LINES), returncode=1) == list(
        KNOWN_PIP_CHECK_LINES
    )
    assert setup.validate_pip_check("No broken requirements found.\n", returncode=0) == []

    with pytest.raises(RuntimeError, match="unexpected pip check failure"):
        setup.validate_pip_check(
            "\n".join((*KNOWN_PIP_CHECK_LINES, "surprise 1.0 requires missing")),
            returncode=1,
        )


@pytest.mark.parametrize(
    ("runner", "message"),
    [
        (FakeRunner(gui_version="5.0.0.93"), "must be absent"),
        (FakeRunner(headless_version=None), "must be exactly 4.12.0.88"),
        (FakeRunner(headless_version="4.11.0.86"), "must be exactly 4.12.0.88"),
        (FakeRunner(import_returncode=1), "required STEERER imports failed"),
    ],
)
def test_verification_rejects_wrong_opencv_or_failed_imports(
    tmp_path: Path, runner: FakeRunner, message: str
) -> None:
    venv = _make_existing_venv(tmp_path / "venv")

    with pytest.raises((RuntimeError, ValueError), match=message):
        setup.verify_environment(venv_path=venv, runner=runner)


@pytest.mark.parametrize(
    ("distribution", "observed"),
    [
        ("mmcv", "1.7.1"),
        ("timm", "1.0.18"),
        ("fvcore", None),
    ],
)
def test_verification_rejects_stale_or_missing_frozen_distribution(
    tmp_path: Path, distribution: str, observed: str | None
) -> None:
    venv = _make_existing_venv(tmp_path / "venv")
    runner = FakeRunner(installed_overrides={distribution: observed})

    with pytest.raises(ValueError, match="installed frozen package mismatch"):
        setup.verify_environment(venv_path=venv, runner=runner)


def test_metadata_probe_queries_all_frozen_distributions(tmp_path: Path) -> None:
    venv = _make_existing_venv(tmp_path / "venv")
    runner = FakeRunner()

    verification = setup.verify_environment(venv_path=venv, runner=runner)

    metadata_code = next(
        call[-1] for call in runner.calls if "importlib.metadata" in call[-1]
    )
    assert all(repr(name) in metadata_code for name in FROZEN_VERSIONS)
    assert verification["installed_versions"] == FROZEN_VERSIONS


def test_import_probe_explicitly_adds_resolved_repo_src(tmp_path: Path) -> None:
    venv = _make_existing_venv(tmp_path / "venv")
    runner = FakeRunner()

    setup.verify_environment(venv_path=venv, runner=runner)

    import_code = next(call[-1] for call in runner.calls if "import cv2" in call[-1])
    resolved_src = str((setup.REPO_ROOT / "src").resolve())
    assert "sys.path.insert(0," in import_code
    assert repr(resolved_src) in import_code
    assert import_code.index("sys.path.insert") < import_code.index(
        "from droneai.steerer_adapter import STEERERAdapter"
    )


def test_main_emits_machine_readable_json_status(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    overlay = _write_overlay(tmp_path / "requirements.txt")
    venv = _make_existing_venv(tmp_path / "venv")
    runner = FakeRunner()
    monkeypatch.setattr(setup, "_run_command", runner)

    status = setup.main(
        [
            "--requirements",
            str(overlay),
            "--venv",
            str(venv),
            "--verify-only",
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    assert status == 0
    assert payload["status"] == "PASS"
    assert payload["verify_only"] is True
    assert payload["opencv_python"] is None
    assert payload["opencv_python_headless"] == "4.12.0.88"
    assert payload["installed_versions"] == FROZEN_VERSIONS
    assert "mmcv" not in payload
    import_call = next(call for call in runner.calls if "import cv2" in call[-1])
    assert all(
        module in import_call[-1]
        for module in (
            "cv2",
            "mmcv",
            "torch",
            "timm",
            "einops",
            "yacs",
            "fvcore",
            "dict_recursive_update",
            "tensorboardX",
        )
    )
