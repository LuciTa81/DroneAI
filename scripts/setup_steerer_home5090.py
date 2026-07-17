"""Create or verify the isolated, headless STEERER home5090 environment."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REQUIREMENTS = REPO_ROOT / "requirements/models/steerer-home5090.txt"
DEFAULT_VENV = Path("/workspace/.venvs/steerer")
EXPECTED_MMCV = "mmcv==1.7.2"
EXPECTED_HEADLESS = "opencv-python-headless==4.12.0.88"
EXPECTED_HEADLESS_VERSION = "4.12.0.88"
FROZEN_REQUIREMENTS = (
    EXPECTED_MMCV,
    "addict==2.4.0",
    "packaging==25.0",
    "Pillow==11.3.0",
    "PyYAML==6.0.2",
    "yapf==0.43.0",
    EXPECTED_HEADLESS,
    "timm==1.0.19",
    "einops==0.8.1",
    "dict_recursive_update==1.0.1",
    "yacs==0.1.8",
    "tensorboardX==2.6.2.2",
    "fvcore==0.1.5.post20221221",
)
KNOWN_PIP_CHECK_LINES = frozenset(
    {
        "mmcv 1.7.2 requires opencv-python, which is not installed.",
        (
            "nvidia-resiliency-ext 0.4.1+cuda13 requires pynvml, "
            "which is not installed."
        ),
    }
)
REQUIRED_IMPORTS = (
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
_EXACT_PIN = re.compile(r"^(?P<name>[A-Za-z0-9][A-Za-z0-9._-]*)==(?P<version>[^\s;#]+)$")


@dataclass(frozen=True)
class ParsedRequirements:
    mmcv: str
    non_mmcv: tuple[str, ...]


CommandRunner = Callable[[Sequence[str]], subprocess.CompletedProcess[str]]


def _normalized_name(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


FROZEN_BY_NAME = {
    _normalized_name(requirement.partition("==")[0]): requirement
    for requirement in FROZEN_REQUIREMENTS
}
FROZEN_VERSIONS = {
    name: requirement.partition("==")[2]
    for name, requirement in FROZEN_BY_NAME.items()
}


def parse_requirements(path: str | Path) -> ParsedRequirements:
    """Parse an exact-pin overlay and isolate the one permitted MMCV pin."""

    source = Path(path)
    entries: list[tuple[str, str]] = []
    seen: set[str] = set()
    for line_number, raw_line in enumerate(
        source.read_text(encoding="utf-8").splitlines(), start=1
    ):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        match = _EXACT_PIN.fullmatch(line)
        if match is None:
            lowered = line.lower()
            if "mmcv" in lowered:
                raise ValueError(f"MMCV must use an exact == pin at line {line_number}")
            if "opencv" in lowered:
                raise ValueError(f"OpenCV must use an exact == pin at line {line_number}")
            raise ValueError(f"requirement must use an exact == pin at line {line_number}")
        normalized = _normalized_name(match.group("name"))
        if normalized in seen:
            raise ValueError(f"duplicate requirement: {normalized}")
        seen.add(normalized)
        entries.append((normalized, line))

    by_name = {name: requirement for name, requirement in entries}
    if by_name.get("mmcv") != EXPECTED_MMCV:
        raise ValueError(f"MMCV must be exactly {EXPECTED_MMCV}")
    if "opencv-python" in by_name:
        raise ValueError("GUI opencv-python is forbidden; use only headless OpenCV")
    if by_name.get("opencv-python-headless") != EXPECTED_HEADLESS:
        raise ValueError(f"OpenCV must be exactly {EXPECTED_HEADLESS}")
    if by_name != FROZEN_BY_NAME:
        missing = sorted(FROZEN_BY_NAME.keys() - by_name.keys())
        unexpected = sorted(by_name.keys() - FROZEN_BY_NAME.keys())
        changed = sorted(
            name
            for name in FROZEN_BY_NAME.keys() & by_name.keys()
            if by_name[name] != FROZEN_BY_NAME[name]
        )
        raise ValueError(
            "requirements must match the frozen 13-package allowlist; "
            f"missing={missing}; unexpected={unexpected}; changed={changed}"
        )

    return ParsedRequirements(
        mmcv=by_name["mmcv"],
        non_mmcv=tuple(
            requirement for name, requirement in entries if name != "mmcv"
        ),
    )


def _run_command(command: Sequence[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [str(part) for part in command],
        check=False,
        capture_output=True,
        text=True,
    )


def _require_success(
    result: subprocess.CompletedProcess[str], *, operation: str
) -> None:
    if result.returncode == 0:
        return
    detail = (result.stderr or result.stdout or "no command output").strip()
    raise RuntimeError(f"{operation} failed (exit {result.returncode}): {detail}")


def _venv_python(venv_path: Path) -> Path:
    config_path = venv_path / "pyvenv.cfg"
    if not venv_path.is_dir() or not config_path.is_file():
        raise ValueError(f"existing path is not a Python venv: {venv_path}")
    settings: dict[str, str] = {}
    for raw_line in config_path.read_text(encoding="utf-8").splitlines():
        key, separator, value = raw_line.partition("=")
        if separator:
            settings[key.strip().lower()] = value.strip().lower()
    if settings.get("include-system-site-packages") != "true":
        raise ValueError(
            "venv pyvenv.cfg must set include-system-site-packages=true: "
            f"{config_path}"
        )
    python = venv_path / "bin/python"
    if not python.is_file():
        raise ValueError(f"venv Python is missing: {python}")
    return python


def _ensure_venv(
    *,
    venv_path: Path,
    verify_only: bool,
    runner: CommandRunner,
) -> tuple[Path, bool]:
    if venv_path.exists():
        return _venv_python(venv_path), False
    if verify_only:
        raise FileNotFoundError(f"cannot verify missing venv: {venv_path}")

    venv_path.parent.mkdir(parents=True, exist_ok=True)
    result = runner(
        (
            sys.executable,
            "-m",
            "venv",
            "--system-site-packages",
            str(venv_path),
        )
    )
    _require_success(result, operation="venv creation")
    return _venv_python(venv_path), True


def validate_pip_check(output: str, *, returncode: int) -> list[str]:
    """Allow only the two evidence-backed metadata failures."""

    if returncode not in {0, 1}:
        raise RuntimeError(f"pip check failed unexpectedly with exit {returncode}")
    lines = [
        line.strip()
        for line in output.splitlines()
        if line.strip() and line.strip() != "No broken requirements found."
    ]
    unexpected = [line for line in lines if line not in KNOWN_PIP_CHECK_LINES]
    if unexpected:
        raise RuntimeError(
            "unexpected pip check failure(s): " + " | ".join(unexpected)
        )
    if returncode == 1 and not lines:
        raise RuntimeError("pip check exited 1 without a recognized failure")
    return lines


def verify_environment(
    *,
    venv_path: str | Path,
    runner: CommandRunner | None = None,
) -> dict[str, object]:
    """Verify package identity, imports, and narrowly allowed metadata errors."""

    command_runner = runner or _run_command
    resolved_venv = Path(venv_path)
    python = _venv_python(resolved_venv)
    metadata_names = (*FROZEN_VERSIONS, "opencv-python")
    metadata_code = (
        "import importlib.metadata, json\n"
        f"NAMES = {metadata_names!r}\n"
        "def version(name):\n"
        "    try:\n"
        "        return importlib.metadata.version(name)\n"
        "    except importlib.metadata.PackageNotFoundError:\n"
        "        return None\n"
        "print(json.dumps({name: version(name) for name in NAMES}))"
    )
    metadata_result = command_runner((str(python), "-c", metadata_code))
    _require_success(metadata_result, operation="OpenCV package metadata probe")
    try:
        metadata = json.loads(metadata_result.stdout)
    except (json.JSONDecodeError, TypeError) as exc:
        raise RuntimeError("OpenCV package metadata probe returned invalid JSON") from exc
    if not isinstance(metadata, dict):
        raise RuntimeError("OpenCV package metadata probe must return an object")

    gui_version = metadata.get("opencv-python")
    headless_version = metadata.get("opencv-python-headless")
    if gui_version is not None:
        raise ValueError(
            f"GUI opencv-python must be absent, found version {gui_version}"
        )
    if headless_version != EXPECTED_HEADLESS_VERSION:
        raise ValueError(
            "opencv-python-headless must be exactly "
            f"{EXPECTED_HEADLESS_VERSION}, found {headless_version}"
        )
    installed_versions = {name: metadata.get(name) for name in FROZEN_VERSIONS}
    mismatches = [
        f"{name}: expected {expected}, found {installed_versions[name]}"
        for name, expected in FROZEN_VERSIONS.items()
        if installed_versions[name] != expected
    ]
    if mismatches:
        raise ValueError(
            "installed frozen package mismatch: " + " | ".join(mismatches)
        )

    resolved_src = str((REPO_ROOT / "src").resolve())
    import_code = (
        "import sys\n"
        f"sys.path.insert(0, {resolved_src!r})\n"
        "import " + ", ".join(REQUIRED_IMPORTS) + "\n"
        "from droneai.steerer_adapter import STEERERAdapter"
    )
    import_result = command_runner((str(python), "-c", import_code))
    _require_success(import_result, operation="required STEERER imports")

    pip_check_result = command_runner((str(python), "-m", "pip", "check"))
    pip_check_output = "\n".join(
        part.strip()
        for part in (pip_check_result.stdout, pip_check_result.stderr)
        if part and part.strip()
    )
    allowed_pip_check = validate_pip_check(
        pip_check_output,
        returncode=pip_check_result.returncode,
    )
    return {
        "opencv_python": gui_version,
        "opencv_python_headless": headless_version,
        "installed_versions": installed_versions,
        "required_imports": list(REQUIRED_IMPORTS),
        "pip_check_allowed_issues": allowed_pip_check,
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def setup_environment(
    *,
    requirements_path: str | Path = DEFAULT_REQUIREMENTS,
    venv_path: str | Path = DEFAULT_VENV,
    verify_only: bool = False,
    runner: CommandRunner | None = None,
) -> dict[str, object]:
    """Create/reuse, optionally install, and always verify the STEERER venv."""

    command_runner = runner or _run_command
    requirements = Path(requirements_path).resolve()
    venv = Path(venv_path)
    parsed = parse_requirements(requirements)
    python, venv_created = _ensure_venv(
        venv_path=venv,
        verify_only=verify_only,
        runner=command_runner,
    )

    if not verify_only:
        uninstall_gui_result = command_runner(
            (str(python), "-m", "pip", "uninstall", "-y", "opencv-python")
        )
        _require_success(
            uninstall_gui_result,
            operation="GUI OpenCV removal",
        )
        non_mmcv_result = command_runner(
            (str(python), "-m", "pip", "install", *parsed.non_mmcv)
        )
        _require_success(non_mmcv_result, operation="non-MMCV dependency install")
        mmcv_result = command_runner(
            (str(python), "-m", "pip", "install", "--no-deps", parsed.mmcv)
        )
        _require_success(mmcv_result, operation="MMCV no-deps install")
        restore_headless_result = command_runner(
            (
                str(python),
                "-m",
                "pip",
                "install",
                "--no-deps",
                "--force-reinstall",
                EXPECTED_HEADLESS,
            )
        )
        _require_success(
            restore_headless_result,
            operation="headless OpenCV restoration",
        )

    verification = verify_environment(venv_path=venv, runner=command_runner)
    return {
        "schema_version": 1,
        "status": "PASS",
        "venv": str(venv),
        "venv_created": venv_created,
        "verify_only": verify_only,
        "requirements": str(requirements),
        "requirements_sha256": _sha256(requirements),
        **verification,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create or verify the isolated headless STEERER home5090 venv."
    )
    parser.add_argument("--venv", type=Path, default=DEFAULT_VENV)
    parser.add_argument("--requirements", type=Path, default=DEFAULT_REQUIREMENTS)
    parser.add_argument("--verify-only", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        summary = setup_environment(
            requirements_path=args.requirements,
            venv_path=args.venv,
            verify_only=args.verify_only,
        )
        status = 0
    except Exception as exc:
        summary = {
            "schema_version": 1,
            "status": "FAIL",
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
        status = 1
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return status


if __name__ == "__main__":
    sys.exit(main())
