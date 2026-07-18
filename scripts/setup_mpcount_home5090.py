"""Create or verify MPCount's isolated overlay on the home5090 NGC runtime."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Callable, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REQUIREMENTS = REPO_ROOT / "requirements/models/mpcount-home5090.txt"
DEFAULT_VENV = Path("/workspace/.venvs/mpcount")
DEFAULT_UPSTREAM = Path("/workspace/upstreams/MPCount")
FROZEN_REQUIREMENTS = (
    "gdown==5.2.0",
    "opencv-python-headless==4.12.0.88",
    "PySocks==1.7.1",
    "scikit-image==0.26.0",
    "ImageIO==2.37.3",
    "tifffile==2026.7.14",
    "lazy_loader==0.4",
)
FROZEN_VERSIONS = {
    requirement.partition("==")[0]: requirement.partition("==")[2]
    for requirement in FROZEN_REQUIREMENTS
}
KNOWN_PIP_CHECK = frozenset(
    {
        "nvidia-resiliency-ext 0.4.1+cuda13 requires pynvml, which is not installed."
    }
)
_EXACT_PIN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*==[^\s;#]+$")

Runner = Callable[[Sequence[str]], subprocess.CompletedProcess[str]]


def _normalize(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def parse_requirements(path: str | Path) -> tuple[str, ...]:
    values: list[str] = []
    for number, raw in enumerate(
        Path(path).read_text(encoding="utf-8").splitlines(), start=1
    ):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if _EXACT_PIN.fullmatch(line) is None:
            raise ValueError(
                f"MPCount requirement must use an exact pin at line {number}"
            )
        values.append(line)

    expected = {_normalize(item.partition("==")[0]): item for item in FROZEN_REQUIREMENTS}
    observed = {_normalize(item.partition("==")[0]): item for item in values}
    if len(observed) != len(values) or observed != expected:
        raise ValueError(
            "requirements must match the frozen MPCount allowlist; "
            f"expected={list(FROZEN_REQUIREMENTS)!r}; observed={values!r}"
        )
    return tuple(values)


def _run(command: Sequence[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [str(part) for part in command],
        check=False,
        capture_output=True,
        text=True,
    )


def _success(result: subprocess.CompletedProcess[str], operation: str) -> None:
    if result.returncode == 0:
        return
    detail = (result.stderr or result.stdout or "no output").strip()
    raise RuntimeError(f"{operation} failed (exit {result.returncode}): {detail}")


def _venv_python(path: Path) -> Path:
    config = path / "pyvenv.cfg"
    python = path / "bin/python"
    if not path.is_dir() or not config.is_file() or not python.is_file():
        raise ValueError(f"existing path is not a Python venv: {path}")
    settings: dict[str, str] = {}
    for raw in config.read_text(encoding="utf-8").splitlines():
        key, separator, value = raw.partition("=")
        if separator:
            settings[key.strip().lower()] = value.strip().lower()
    if settings.get("include-system-site-packages") != "true":
        raise ValueError(
            f"venv must set include-system-site-packages=true: {config}"
        )
    return python


def _ensure_venv(
    path: Path, *, verify_only: bool, runner: Runner
) -> tuple[Path, bool]:
    if path.exists():
        return _venv_python(path), False
    if verify_only:
        raise FileNotFoundError(f"cannot verify missing venv: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    result = runner(
        (sys.executable, "-m", "venv", "--system-site-packages", str(path))
    )
    _success(result, "MPCount venv creation")
    return _venv_python(path), True


def _parse_json_output(
    result: subprocess.CompletedProcess[str], *, operation: str
) -> dict[str, object]:
    _success(result, operation)
    try:
        value = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"{operation} returned invalid JSON") from exc
    if not isinstance(value, dict):
        raise RuntimeError(f"{operation} must return a JSON object")
    return value


def _verify(python: Path, *, upstream: Path, runner: Runner) -> dict[str, object]:
    names = (*FROZEN_VERSIONS, "opencv-python")
    metadata_code = (
        "import importlib.metadata as m, json\n"
        f"names={names!r}\n"
        "def version(name):\n"
        " try: return m.version(name)\n"
        " except m.PackageNotFoundError: return None\n"
        "print(json.dumps({name: version(name) for name in names}))\n"
    )
    metadata = _parse_json_output(
        runner((str(python), "-c", metadata_code)),
        operation="MPCount package metadata",
    )
    if metadata.get("opencv-python") is not None:
        raise ValueError("GUI opencv-python is forbidden in the MPCount venv")
    mismatches = [
        f"{name}: expected {version}, found {metadata.get(name)}"
        for name, version in FROZEN_VERSIONS.items()
        if metadata.get(name) != version
    ]
    if mismatches:
        raise ValueError("MPCount overlay mismatch: " + " | ".join(mismatches))

    runtime_code = (
        "import json, sys\n"
        f"sys.path.insert(0, {str(upstream.resolve())!r})\n"
        "import cv2, gdown, socks, torch, torchvision, numpy, scipy, matplotlib, einops\n"
        "import skimage, imageio, tifffile, lazy_loader\n"
        "from PIL import Image\n"
        "from models.models import DGModel_final\n"
        "from utils.misc import divide_img_into_patches, get_padding\n"
        "available=torch.cuda.is_available()\n"
        "gpu=torch.cuda.get_device_name(0) if available else None\n"
        "value=float((torch.ones((32,32),device='cuda') @ "
        "torch.ones((32,32),device='cuda'))[0,0].item()) if available else None\n"
        "print(json.dumps({'cuda_available':available,'torch':torch.__version__,"
        "'torchvision':torchvision.__version__,'gpu':gpu,'matmul':value,"
        "'mpcount_imports':True,'metrics_imports':True}))\n"
    )
    cuda = _parse_json_output(
        runner((str(python), "-c", runtime_code)),
        operation="MPCount CUDA/import smoke",
    )
    if (
        cuda.get("cuda_available") is not True
        or cuda.get("matmul") != 32.0
        or cuda.get("mpcount_imports") is not True
        or cuda.get("metrics_imports") is not True
    ):
        raise RuntimeError(f"MPCount CUDA/import smoke failed: {cuda}")

    pip_check = runner((str(python), "-m", "pip", "check"))
    output = "\n".join(
        part.strip() for part in (pip_check.stdout, pip_check.stderr) if part.strip()
    )
    lines = [
        line.strip()
        for line in output.splitlines()
        if line.strip() and line.strip() != "No broken requirements found."
    ]
    unexpected = [line for line in lines if line not in KNOWN_PIP_CHECK]
    if pip_check.returncode not in {0, 1} or unexpected:
        raise RuntimeError(
            "unexpected MPCount pip check failure: " + " | ".join(lines)
        )

    return {
        "installed_versions": {
            name: metadata[name] for name in FROZEN_VERSIONS
        },
        "opencv_python": metadata.get("opencv-python"),
        "cuda": cuda,
        "pip_check_allowed_issues": lines,
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
    upstream_path: str | Path = DEFAULT_UPSTREAM,
    verify_only: bool = False,
    runner: Runner | None = None,
) -> dict[str, object]:
    command_runner = runner or _run
    requirements = Path(requirements_path).resolve()
    frozen = parse_requirements(requirements)
    venv = Path(venv_path)
    python, created = _ensure_venv(
        venv, verify_only=verify_only, runner=command_runner
    )
    if not verify_only:
        install = command_runner(
            (str(python), "-m", "pip", "install", "--no-deps", *frozen)
        )
        _success(install, "MPCount overlay install")
    verification = _verify(
        python, upstream=Path(upstream_path), runner=command_runner
    )
    return {
        "schema_version": 1,
        "status": "PASS",
        "venv": str(venv),
        "venv_created": created,
        "verify_only": verify_only,
        "requirements": str(requirements),
        "requirements_sha256": _sha256(requirements),
        "upstream": str(Path(upstream_path)),
        **verification,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create or verify the isolated MPCount home5090 overlay."
    )
    parser.add_argument("--venv", type=Path, default=DEFAULT_VENV)
    parser.add_argument("--requirements", type=Path, default=DEFAULT_REQUIREMENTS)
    parser.add_argument("--upstream", type=Path, default=DEFAULT_UPSTREAM)
    parser.add_argument("--verify-only", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        payload = setup_environment(
            requirements_path=args.requirements,
            venv_path=args.venv,
            upstream_path=args.upstream,
            verify_only=args.verify_only,
        )
        code = 0
    except Exception as exc:
        payload = {
            "schema_version": 1,
            "status": "FAIL",
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
        code = 1
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
    return code


if __name__ == "__main__":
    sys.exit(main())
