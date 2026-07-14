"""Stage 0: verify that one configured backend can produce reproducible runs."""

from __future__ import annotations

import json
import hashlib
import subprocess
import sys
from pathlib import Path
from typing import Any, Sequence

from droneai.runtime_probe import collect_environment
from droneai.runtime_profile import RuntimeProfile
from droneai.scoring import CheckResult, StageReport, score_stage

DEFAULT_SEEDS = (17, 42, 2026)
REQUIRED_DRIVE_DIRS = ("datasets", "checkpoints", "runs")


def _version_tuple(version_text: str) -> tuple[int, int]:
    parts = version_text.split(".")
    return int(parts[0]), int(parts[1])


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _artifact_roundtrip(runs_dir: Path) -> tuple[bool, str]:
    probe = runs_dir / "stage-0" / ".write-probe.json"
    expected = {"stage": 0, "ok": True}
    try:
        _write_json(probe, expected)
        actual = json.loads(probe.read_text(encoding="utf-8"))
        probe.unlink(missing_ok=True)
        return actual == expected, str(probe.parent)
    except OSError as exc:
        return False, f"{type(exc).__name__}: {exc}"


def _cuda_smoke(environment: dict[str, Any]) -> tuple[bool, str]:
    torch_info = environment.get("torch", {})
    if not torch_info.get("cuda_available"):
        return False, "CUDA unavailable"
    try:
        import torch

        size = 32
        left = torch.ones((size, size), device="cuda")
        right = torch.ones((size, size), device="cuda")
        product = left @ right
        maximum_error = float((product - size).abs().max().item())
        return maximum_error == 0.0, f"CUDA matmul max_error={maximum_error}"
    except Exception as exc:  # pragma: no cover - requires a GPU runtime
        return False, f"{type(exc).__name__}: {exc}"


def run_unit_tests(repo_root: Path) -> tuple[bool, str]:
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "-q"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )
    output = "\n".join(part.strip() for part in (result.stdout, result.stderr) if part.strip())
    return result.returncode == 0, output[-2000:] or f"exit code {result.returncode}"


def build_stage0_checks(
    *,
    environment: dict[str, Any],
    drive_root: Path | None = None,
    storage_root: Path | None = None,
    required_storage_dirs: Sequence[Path] | None = None,
    results_dir: Path | None = None,
    seeds: Sequence[int],
    unit_tests_passed: bool,
    unit_test_evidence: str,
    environment_report_written: bool,
    config_snapshot_written: bool,
) -> list[CheckResult]:
    storage_root = storage_root or drive_root
    if storage_root is None:
        raise ValueError("storage_root or drive_root is required")
    results_dir = results_dir or storage_root / "runs"
    torch_info = environment.get("torch", {})
    gpu_names = torch_info.get("device_names") or []
    required_dirs = list(required_storage_dirs or (storage_root / name for name in REQUIRED_DRIVE_DIRS))
    artifact_ok, artifact_evidence = _artifact_roundtrip(results_dir)
    cuda_ok, cuda_evidence = _cuda_smoke(environment)

    python_version = f"{sys.version_info.major}.{sys.version_info.minor}"
    versions_recorded = bool(environment.get("python") and torch_info.get("version"))
    git_commit = environment.get("git_commit")
    git_dirty = environment.get("git_dirty")

    return [
        CheckResult(
            "runtime.python", "runtime", "Python 3.10 or newer", 5, _version_tuple(python_version) >= (3, 10),
            expected=">=3.10", observed=python_version,
        ),
        CheckResult(
            "runtime.torch", "runtime", "PyTorch imports successfully", 5, bool(torch_info.get("version")),
            expected="version recorded", observed=str(torch_info.get("version") or torch_info.get("error")),
        ),
        CheckResult(
            "runtime.cuda", "runtime", "CUDA is available to PyTorch", 15, bool(torch_info.get("cuda_available")), True,
            expected="true", observed=str(torch_info.get("cuda_available")),
        ),
        CheckResult(
            "runtime.gpu", "runtime", "GPU identity is recorded", 5, bool(gpu_names),
            expected="at least one device", observed=", ".join(gpu_names) or "none",
        ),
        CheckResult(
            "runtime.cuda_smoke", "runtime", "A tensor operation succeeds on CUDA", 5, cuda_ok, True,
            expected="matmul max_error=0.0", observed=cuda_evidence,
        ),
        CheckResult(
            "storage.root", "storage", "Configured storage root exists", 5, storage_root.is_dir(),
            expected=str(storage_root), observed=str(storage_root.resolve()) if storage_root.exists() else "missing",
        ),
        CheckResult(
            "storage.layout", "storage", "Dataset, checkpoint and result folders exist", 5,
            all(path.is_dir() for path in required_dirs),
            expected=", ".join(str(path) for path in required_dirs),
            observed=", ".join(str(path) for path in required_dirs if path.is_dir()) or "none",
        ),
        CheckResult(
            "storage.writable", "storage", "Run artifacts survive a write/read roundtrip", 10, artifact_ok, True,
            expected="JSON roundtrip succeeds", observed=artifact_evidence,
        ),
        CheckResult(
            "storage.environment_report", "storage", "Environment report is persisted", 5,
            environment_report_written, True,
            expected="environment.json written", observed=str(environment_report_written),
        ),
        CheckResult(
            "repro.git_commit", "reproducibility", "Exact Git commit is captured", 8, bool(git_commit), True,
            expected="40-character commit SHA", observed=str(git_commit),
        ),
        CheckResult(
            "repro.git_clean", "reproducibility", "Working tree state is recorded and clean", 4,
            git_dirty is False,
            expected="git_dirty=false", observed=str(git_dirty),
        ),
        CheckResult(
            "repro.versions", "reproducibility", "Python and PyTorch versions are recorded", 4,
            versions_recorded,
            expected="both versions", observed=f"python={environment.get('python')}; torch={torch_info.get('version')}",
        ),
        CheckResult(
            "repro.seeds", "reproducibility", "At least three unique seeds are fixed", 5,
            len(set(seeds)) >= 3,
            expected=">=3 unique seeds", observed=", ".join(map(str, seeds)),
        ),
        CheckResult(
            "repro.config", "reproducibility", "Stage configuration snapshot is persisted", 4,
            config_snapshot_written,
            expected="stage0_config.json written", observed=str(config_snapshot_written),
        ),
        CheckResult(
            "automation.tests", "automation", "Repository unit tests pass", 10,
            unit_tests_passed, True,
            expected="pytest exit code 0", observed=unit_test_evidence,
        ),
        CheckResult(
            "automation.schema", "automation", "Runtime report contains required top-level fields", 5,
            all(key in environment for key in ("captured_at", "python", "platform", "git_commit", "torch")),
            expected="captured_at, python, platform, git_commit, torch",
            observed=", ".join(sorted(environment.keys())),
        ),
    ]


def run_stage0(
    *,
    drive_root: str | Path | None = None,
    profile: RuntimeProfile | None = None,
    repo_root: str | Path,
    output_dir: str | Path | None = None,
    seeds: Sequence[int] = DEFAULT_SEEDS,
) -> StageReport:
    repo_root = Path(repo_root)
    if profile is not None and drive_root is not None:
        raise ValueError("profile and drive_root are mutually exclusive")

    if profile is not None:
        storage_root = profile.storage.root
        required_storage_dirs = (
            profile.storage.datasets_dir,
            profile.storage.checkpoints_dir,
            profile.storage.results_dir,
        )
        results_dir = profile.storage.results_dir
        backend = {
            "id": profile.backend_id,
            "role": profile.role,
            "container_name": profile.container_name,
            "container_image": profile.container_image,
            "container_image_digest": profile.container_image_digest,
        }
        profile_record = {
            "path": str(profile.source_path),
            "sha256": profile.source_sha256,
        }
        license_scope = profile.license_scope
    else:
        if drive_root is None:
            raise ValueError("profile or drive_root is required")
        storage_root = Path(drive_root)
        required_storage_dirs = tuple(storage_root / name for name in REQUIRED_DRIVE_DIRS)
        results_dir = storage_root / "runs"
        backend = {
            "id": "colab_drive",
            "role": "legacy_argument",
            "container_name": None,
            "container_image": None,
            "container_image_digest": None,
        }
        profile_record = {"path": None, "sha256": None}
        license_scope = "research_only"

    output_dir = Path(output_dir) if output_dir else results_dir / "stage-0"
    for directory in required_storage_dirs:
        directory.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    environment = collect_environment(cwd=repo_root)
    environment_path = output_dir / "environment.json"
    config_path = output_dir / "stage0_config.json"
    try:
        _write_json(environment_path, environment)
        environment_written = environment_path.is_file()
    except OSError:
        environment_written = False

    config = {
        "stage_id": "stage-0",
        "threshold": 85,
        "runtime_backend": backend,
        "storage": {
            "root": str(storage_root),
            "datasets_dir": str(required_storage_dirs[0]),
            "checkpoints_dir": str(required_storage_dirs[1]),
            "results_dir": str(results_dir),
        },
        "repo_root": str(repo_root),
        "seeds": list(seeds),
        "profile": profile_record,
        "license_scope": license_scope,
        "production_approved": False,
    }
    try:
        _write_json(config_path, config)
        config_written = config_path.is_file()
    except OSError:
        config_written = False

    tests_passed, test_evidence = run_unit_tests(repo_root)
    checks = build_stage0_checks(
        environment=environment,
        storage_root=storage_root,
        required_storage_dirs=required_storage_dirs,
        results_dir=results_dir,
        seeds=seeds,
        unit_tests_passed=tests_passed,
        unit_test_evidence=test_evidence,
        environment_report_written=environment_written,
        config_snapshot_written=config_written,
    )
    report = score_stage(
        stage_id="stage-0",
        stage_name="Experiment foundation readiness",
        threshold=85,
        checks=checks,
    )
    _write_json(output_dir / "score.json", report.to_dict())
    (output_dir / "score.md").write_text(report.to_markdown(), encoding="utf-8")

    artifact_names = ("environment.json", "stage0_config.json", "score.json", "score.md")
    artifacts = {}
    for name in artifact_names:
        path = output_dir / name
        artifacts[name] = {
            "path": name,
            "bytes": path.stat().st_size,
            "sha256": _sha256_file(path),
        }
    manifest = {
        "schema_version": 1,
        "run_type": "foundation_cuda_smoke",
        "runtime_backend": backend,
        "source": {
            "git_commit": environment.get("git_commit"),
            "git_dirty": environment.get("git_dirty"),
            "upstream_commit": None,
        },
        "runtime": {
            "python": environment.get("python"),
            "torch": environment.get("torch"),
            "nvidia_smi": environment.get("nvidia_smi"),
        },
        "dataset": {"id": None, "artifact_sha256": None},
        "split": {"id": None, "sha256": None},
        "seeds": list(seeds),
        "license_scope": license_scope,
        "production_approved": False,
        "profile": profile_record,
        "artifacts": artifacts,
    }
    _write_json(output_dir / "run_manifest.json", manifest)
    return report
