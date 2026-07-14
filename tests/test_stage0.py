import hashlib
import json
from pathlib import Path

from droneai.scoring import score_stage
from droneai.runtime_profile import RuntimeProfile, StoragePaths
from droneai.stage0 import build_stage0_checks, run_stage0


def test_stage0_weights_total_one_hundred(tmp_path: Path) -> None:
    drive_root = tmp_path / "drive"
    for name in ("datasets", "checkpoints", "runs"):
        (drive_root / name).mkdir(parents=True)

    environment = {
        "captured_at": "2026-01-01T00:00:00+00:00",
        "python": "3.12",
        "platform": "test",
        "git_commit": "a" * 40,
        "git_dirty": False,
        "torch": {
            "version": "2.6.0",
            "cuda_available": False,
            "device_names": [],
        },
    }
    checks = build_stage0_checks(
        environment=environment,
        drive_root=drive_root,
        seeds=(17, 42, 2026),
        unit_tests_passed=True,
        unit_test_evidence="tests passed",
        environment_report_written=True,
        config_snapshot_written=True,
    )
    report = score_stage(
        stage_id="stage-0",
        stage_name="Experiment foundation readiness",
        threshold=85,
        checks=checks,
    )
    assert sum(check.weight for check in checks) == 100
    assert report.status == "BLOCKED"
    assert "runtime.cuda" in report.to_dict()["failed_blockers"]
    cuda_smoke = next(check for check in checks if check.check_id == "runtime.cuda_smoke")
    assert cuda_smoke.expected == "matmul max_error=0.0"


def test_stage0_profile_writes_hash_bearing_research_manifest(
    tmp_path: Path, monkeypatch
) -> None:
    storage_root = tmp_path / "storage"
    profile_source = tmp_path / "home5090.json"
    profile_source.write_text('{"backend_id":"home5090_docker"}', encoding="utf-8")
    profile = RuntimeProfile(
        schema_version=1,
        backend_id="home5090_docker",
        role="default",
        repo_root=tmp_path / "repo",
        container_name="crowd-jupyter",
        container_image="nvcr.io/nvidia/pytorch:25.10-py3",
        container_image_digest="nvcr.io/nvidia/pytorch@sha256:" + "b" * 64,
        storage=StoragePaths(
            root=storage_root,
            datasets_dir=storage_root / "datasets",
            checkpoints_dir=storage_root / "checkpoints",
            results_dir=storage_root / "results",
            host_bind_source=Path("/mnt/crowd-data/CrowdCounting"),
        ),
        license_scope="research_only",
        large_artifacts_in_git=False,
        source_path=profile_source,
        source_sha256=hashlib.sha256(profile_source.read_bytes()).hexdigest(),
    )
    environment = {
        "captured_at": "2026-07-14T00:00:00+00:00",
        "python": "3.12.3",
        "platform": "Linux",
        "git_commit": "a" * 40,
        "git_dirty": False,
        "nvidia_smi": "NVIDIA GeForce RTX 5090, 580.82.07, 32607 MiB",
        "torch": {
            "version": "2.9.0a0+nv25.10",
            "cuda_runtime": "13.0",
            "cuda_available": True,
            "device_names": ["NVIDIA GeForce RTX 5090"],
        },
    }
    monkeypatch.setattr("droneai.stage0.collect_environment", lambda cwd: environment)
    monkeypatch.setattr(
        "droneai.stage0._cuda_smoke", lambda _: (True, "CUDA matmul max_error=0.0")
    )
    monkeypatch.setattr("droneai.stage0.run_unit_tests", lambda _: (True, "34 passed"))

    report = run_stage0(profile=profile, repo_root=profile.repo_root)

    output_dir = storage_root / "results" / "stage-0"
    manifest = json.loads((output_dir / "run_manifest.json").read_text(encoding="utf-8"))
    assert report.status == "PASS"
    assert manifest["runtime_backend"] == {
        "id": "home5090_docker",
        "role": "default",
        "container_name": "crowd-jupyter",
        "container_image": "nvcr.io/nvidia/pytorch:25.10-py3",
        "container_image_digest": "nvcr.io/nvidia/pytorch@sha256:" + "b" * 64,
    }
    assert manifest["source"]["git_commit"] == "a" * 40
    assert manifest["source"]["upstream_commit"] is None
    assert manifest["dataset"] == {"id": None, "artifact_sha256": None}
    assert manifest["split"] == {"id": None, "sha256": None}
    assert manifest["seeds"] == [17, 42, 2026]
    assert manifest["license_scope"] == "research_only"
    assert manifest["production_approved"] is False
    assert manifest["profile"]["sha256"] == profile.source_sha256

    for name, artifact in manifest["artifacts"].items():
        path = output_dir / artifact["path"]
        assert path.name == name
        assert artifact["bytes"] == path.stat().st_size
        assert artifact["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
        assert len(artifact["sha256"]) == 64
