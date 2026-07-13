from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _run(command: list[str]) -> str | None:
    try:
        return subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None


def collect_environment() -> dict[str, Any]:
    report: dict[str, Any] = {
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
        "executable": sys.executable,
        "colab_release_tag": os.environ.get("COLAB_RELEASE_TAG"),
        "git_commit": _run(["git", "rev-parse", "HEAD"]),
        "nvidia_smi": _run(
            [
                "nvidia-smi",
                "--query-gpu=name,driver_version,memory.total",
                "--format=csv,noheader",
            ]
        ),
    }

    try:
        import torch

        report["torch"] = {
            "version": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
            "device_count": torch.cuda.device_count(),
            "device_names": [
                torch.cuda.get_device_name(index)
                for index in range(torch.cuda.device_count())
            ],
        }
    except Exception as exc:  # pragma: no cover - environment-dependent
        report["torch"] = {"error": f"{type(exc).__name__}: {exc}"}

    try:
        import tensorflow as tf

        report["tensorflow"] = {
            "version": tf.__version__,
            "gpu_devices": [device.name for device in tf.config.list_physical_devices("GPU")],
        }
    except Exception as exc:  # pragma: no cover - environment-dependent
        report["tensorflow"] = {"error": f"{type(exc).__name__}: {exc}"}

    return report


def write_environment_report(output: str | Path) -> Path:
    target = Path(output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(collect_environment(), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return target
