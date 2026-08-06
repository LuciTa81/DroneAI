from __future__ import annotations

import hashlib
import json
import pickle
import random
import subprocess
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pytest

from droneai.integrity import sha256_file
from droneai.steerer_training_gate import (
    AuthoritativeTrainingInputs,
    REQUIRED_T1_METRICS,
    score_training_stage,
    verify_authoritative_training_evidence,
    weights_by_category,
)
from droneai.steerer_training_checkpoint import save_checkpoint_with_policy
from droneai.steerer_training_profile import load_training_profile
from droneai.steerer_training_runner import _canonical_config_sha256
from droneai.steerer_training_upstream import synthesize_official_config
from scripts import score_steerer_ucf_training as scoring_cli
from tests.test_steerer_training_upstream import OFFICIAL_CONFIG


PROFILE_PATH = Path("configs/training/steerer_ucf_qnrf_imagenet.home5090.json")


class _FakeTorch:
    def save(self, payload: object, path: str | Path) -> None:
        with Path(path).open("wb") as stream:
            pickle.dump(payload, stream)

    def load(self, path: str | Path, **_kwargs: object) -> object:
        with Path(path).open("rb") as stream:
            return pickle.load(stream)


def _git(root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=root, check=True, capture_output=True, text=True
    ).stdout.strip()


def _init_repo(root: Path) -> str:
    _git(root, "init", "-q")
    _git(root, "add", ".")
    _git(
        root,
        "-c", "user.name=DroneAI tests",
        "-c", "user.email=tests@example.invalid",
        "commit", "-qm", "fixture",
    )
    return _git(root, "rev-parse", "HEAD")


def _json_write(path: Path, payload: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return path


def _jsonl_write(path: Path, rows: list[dict[str, object]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(
        b"".join(
            json.dumps(row, sort_keys=True, separators=(",", ":")).encode("utf-8") + b"\n"
            for row in rows
        )
    )
    return path


@dataclass
class _AuthorityFixture:
    inputs: AuthoritativeTrainingInputs
    profile: object
    project_root: Path
    upstream_root: Path
    processed_root: Path
    backbone: Path
    checkpoint: Path
    checkpoint_manifest: Path
    environment: Path
    metrics: Path


def _authoritative_fixture(
    tmp_path: Path,
    *,
    stage: str = "T1",
    metric_overrides: dict[str, object] | None = None,
) -> _AuthorityFixture:
    project_root = tmp_path / "project"
    profile_path = (
        project_root
        / "configs"
        / "training"
        / "steerer_ucf_qnrf_imagenet.home5090.json"
    )
    profile_path.parent.mkdir(parents=True)
    profile_path.write_bytes(PROFILE_PATH.read_bytes())
    project_commit = _init_repo(project_root)

    upstream_root = tmp_path / "STEERER"
    (upstream_root / "configs").mkdir(parents=True)
    (upstream_root / "LICENSE").write_text("official license fixture\n", encoding="utf-8")
    (upstream_root / "configs" / "QNRF_final.py").write_text(
        OFFICIAL_CONFIG, encoding="utf-8"
    )
    upstream_commit = _init_repo(upstream_root)
    _git(
        upstream_root,
        "remote", "add", "origin", "https://github.com/taohan10200/STEERER.git",
    )

    backbone = tmp_path / "backbones" / "hrnetv2_w48_imagenet_pretrained.pth"
    backbone.parent.mkdir()
    backbone.write_bytes(b"verified ImageNet backbone fixture")
    processed_root = tmp_path / "processed"
    checkpoint_root = tmp_path / "checkpoints"
    result_root = tmp_path / "results"
    loaded = load_training_profile(profile_path)
    profile = replace(
        loaded,
        model_upstream=replace(
            loaded.model_upstream,
            commit=upstream_commit,
            license_sha256=sha256_file(upstream_root / "LICENSE"),
        ),
        imagenet_backbone=replace(
            loaded.imagenet_backbone,
            path=backbone.resolve(),
            sha256=sha256_file(backbone),
            byte_size=backbone.stat().st_size,
        ),
        processed_root=processed_root.resolve(),
        checkpoint_root=checkpoint_root.resolve(),
        result_root=result_root.resolve(),
    )

    sample_ids = [f"sample-{index:04d}" for index in range(1201)]
    train_ids, validation_ids = sample_ids[:961], sample_ids[961:]
    processed_root.mkdir()
    (processed_root / "images").mkdir()
    (processed_root / "jsons").mkdir()
    (processed_root / "manifests").mkdir()
    (processed_root / "train.txt").write_bytes(("\n".join(train_ids) + "\n").encode())
    (processed_root / "val.txt").write_bytes(("\n".join(validation_ids) + "\n").encode())
    train_sha = sha256_file(processed_root / "train.txt")
    validation_sha = sha256_file(processed_root / "val.txt")
    image_bytes = b"fixture image bytes"
    image_sha = hashlib.sha256(image_bytes).hexdigest()
    annotation_sha = hashlib.sha256(b"fixture source annotation").hexdigest()
    input_rows: list[dict[str, object]] = []
    output_rows: list[dict[str, object]] = []
    for sample_id in sample_ids:
        image = processed_root / "images" / f"{sample_id}.jpg"
        image.write_bytes(image_bytes)
        annotation = _json_write(
            processed_root / "jsons" / f"{sample_id}.json",
            {
                "annotation_sha256": annotation_sha,
                "boxes": [[0.0, 0.0, 0.0, 0.0]],
                "human_num": 1,
                "points": [[0.0, 0.0]],
                "sample_id": sample_id,
                "scale": [1.0, 1.0],
                "source_height": 1,
                "source_partition": "official_train_only",
                "source_sha256": image_sha,
                "source_width": 1,
            },
        )
        input_rows.append(
            {
                "annotation_sha256": annotation_sha,
                "count": 1,
                "height": 1,
                "image_sha256": image_sha,
                "sample_id": sample_id,
                "width": 1,
            }
        )
        output_rows.append(
            {
                "count": 1,
                "height": 1,
                "image_sha256": image_sha,
                "json_sha256": sha256_file(annotation),
                "sample_id": sample_id,
                "width": 1,
            }
        )
    input_inventory = _jsonl_write(
        processed_root / "manifests" / "input-inventory.jsonl", input_rows
    )
    output_inventory = _jsonl_write(
        processed_root / "manifests" / "output-inventory.jsonl", output_rows
    )
    population_rows = (
        f"{row['sample_id']},{row['image_sha256']},{row['annotation_sha256']},1,low"
        for row in input_rows
    )
    population_sha = hashlib.sha256("\n".join(population_rows).encode()).hexdigest()
    split_manifest = _json_write(
        processed_root / "split-manifest.json",
        {
            "band_counts": {"low": {"population": 1201, "train": 961, "validation": 240}},
            "population_sha256": population_sha,
            "seed": 3035,
            "train_ids": train_ids,
            "train_sha256": train_sha,
            "validation_ids": validation_ids,
            "validation_sha256": validation_sha,
        },
    )
    test_sealed = _json_write(
        processed_root / "test-sealed.json",
        {"accessed": False, "role": "sealed", "sample_count": 334},
    )
    _json_write(
        processed_root / "manifests" / "training-manifest.json",
        {
            "converted_sample_count": 1201,
            "count_mismatches": 0,
            "input_inventory_sha256": sha256_file(input_inventory),
            "out_of_bounds_boxes": 0,
            "out_of_bounds_points": 0,
            "output_inventory_sha256": sha256_file(output_inventory),
            "sample_count": 1201,
            "source_partition": "official_train_only",
            "train_count": 961,
            "validation_count": 240,
        },
    )

    run_id = "run-3035"
    config = synthesize_official_config(
        profile=profile,
        upstream_dir=upstream_root,
        processed_root=processed_root,
        backbone_path=backbone,
        stage=stage,
        physical_batch=8,
        accumulation_steps=1,
        run_id=run_id,
    )
    config_sha = _canonical_config_sha256(config)
    metric_values: dict[str, object] = (
        {"train_loss": 1.0}
        if stage == "T0"
        else {name: 1.0 for name in REQUIRED_T1_METRICS}
    )
    metric_values.update(metric_overrides or {})
    metrics = _json_write(
        result_root / run_id / "metrics.json",
        {
            "schema_version": 1,
            "run_id": run_id,
            "stage": stage,
            "metrics": metric_values,
        },
    )
    environment_payload = {
        "schema_version": 2,
        "run_id": run_id,
        "stage": stage,
        "project": {
            "git_commit": project_commit,
            "profile_sha256": sha256_file(profile_path),
        },
        "upstream": {
            "origin_url": "https://github.com/taohan10200/STEERER.git",
            "commit": upstream_commit,
            "clean": True,
            "license_sha256": sha256_file(upstream_root / "LICENSE"),
            "config_sha256": sha256_file(upstream_root / "configs" / "QNRF_final.py"),
        },
        "dataset": {
            "processed_root": str(processed_root.resolve()),
            "source_partition": "official_train_only",
            "official_test_accessed": False,
            "train_sha256": train_sha,
            "validation_sha256": validation_sha,
            "dataset_inventory_sha256": sha256_file(output_inventory),
            "population_sha256": population_sha,
            "test_sealed_sha256": sha256_file(test_sealed),
            "split_manifest_sha256": sha256_file(split_manifest),
        },
        "initialization": {
            "mode": "imagenet_backbone_only",
            "model_checkpoint_loaded": False,
            "backbone_path": str(backbone.resolve()),
            "backbone_sha256": sha256_file(backbone),
            "backbone_byte_size": backbone.stat().st_size,
        },
        "training": {"config_sha256": config_sha},
        "runtime": {
            "container_image_digest": "sha256:" + "c" * 64,
            "python": "3.13.0",
            "torch": "2.8.0",
            "cuda": "12.8",
            "gpu": "NVIDIA GeForce RTX 5090",
        },
        "batch": {"physical_batch": 8, "accumulation_steps": 1, "effective_batch": 8},
        "metrics": {"path": str(metrics.resolve()), "sha256": sha256_file(metrics)},
    }
    environment_bytes = (
        json.dumps(environment_payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode()
    environment_sha = hashlib.sha256(environment_bytes).hexdigest()
    environment = result_root / run_id / f"environment.{environment_sha}.json"
    environment.write_bytes(environment_bytes)

    state = {
        "schema_version": 1,
        "run_id": run_id,
        "stage": stage,
        "epoch": {"T0": 0, "T1": 1, "T5": 5, "T50": 50, "T800": 800}[stage],
        "global_step": 16,
        "model": {"weight": [1]},
        "optimizer": {"step": 16},
        "scheduler": {
            "last_epoch": {
                "T0": 0,
                "T1": 1,
                "T5": 5,
                "T50": 50,
                "T800": 800,
            }[stage],
            "horizon": 800,
        },
        "scaler": {},
        "best_mae": 1.0,
        "best_rmse": 1.0,
        "config_sha256": config_sha,
        "split_sha256s": {"train": train_sha, "validation": validation_sha},
        "dataset_inventory_sha256": sha256_file(output_inventory),
        "backbone_sha256": sha256_file(backbone),
        "upstream_commit": upstream_commit,
        "rng": {
            "seed": 3035,
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch_cpu": b"cpu",
            "torch_cuda": [],
        },
        "environment_manifest_sha256": environment_sha,
    }
    checkpoint_result = save_checkpoint_with_policy(
        checkpoint_root / run_id,
        state,
        current_mae=1.0,
        current_rmse=1.0,
        torch_module=_FakeTorch(),
    )
    checkpoint = checkpoint_result.artifacts["last"].path
    checkpoint_manifest = checkpoint_result.manifest_path
    inputs = AuthoritativeTrainingInputs(
        stage=stage,
        run_id=run_id,
        profile_path=profile_path,
        project_repo_root=project_root,
        upstream_dir=upstream_root,
        processed_root=processed_root,
        backbone_path=backbone,
        checkpoint_path=checkpoint,
        checkpoint_manifest_path=checkpoint_manifest,
        environment_path=environment,
        metrics_path=metrics,
    )
    return _AuthorityFixture(
        inputs=inputs,
        profile=profile,
        project_root=project_root,
        upstream_root=upstream_root,
        processed_root=processed_root,
        backbone=backbone,
        checkpoint=checkpoint,
        checkpoint_manifest=checkpoint_manifest,
        environment=environment,
        metrics=metrics,
    )


def _artifact(tmp_path: Path, name: str, content: bytes = b"evidence") -> dict[str, str]:
    path = tmp_path / name
    path.write_bytes(content)
    return {"path": str(path), "sha256": sha256_file(path)}


def _evidence(tmp_path: Path) -> dict[str, object]:
    artifacts = {
        name: _artifact(tmp_path, f"{name}.bin", name.encode("utf-8"))
        for name in (
            "environment",
            "dataset_inventory",
            "train_split",
            "validation_split",
            "backbone",
            "input_checkpoint",
            "output_checkpoint",
            "config",
            "rights_decision",
        )
    }
    environment_bytes = b"immutable environment evidence"
    environment_sha256 = hashlib.sha256(environment_bytes).hexdigest()
    environment_path = tmp_path / f"environment.{environment_sha256}.json"
    environment_path.write_bytes(environment_bytes)
    artifacts["environment"] = {"path": str(environment_path), "sha256": environment_sha256}
    return {
        "schema_version": 1,
        "run_id": "steerer-qnrf-imagenet-20260806-a",
        "stage": "T1",
        "provenance": {
            "git_commit": "a" * 40,
            "upstream_commit": "5b1854dbc2d280f2326d67c65515d8baf9083810",
            "container_image_digest": "sha256:" + "c" * 64,
            "python": "3.13.0",
            "torch": "2.8.0",
            "cuda": "12.8",
            "gpu": "NVIDIA GeForce RTX 5090",
            "upstream_clean": True,
        },
        "split": {
            "dataset_population": 1201,
            "train_count": 961,
            "validation_count": 240,
            "overlap": 0,
            "complete": True,
            "official_test_accessed": False,
        },
        "initialization": {"model_checkpoint_loaded": False},
        "stability": {"finite": True, "checkpoint_round_trip": True},
        "metrics": {name: 1.0 for name in REQUIRED_T1_METRICS},
        "artifacts": artifacts,
    }


def test_t0_explicitly_does_not_require_full_validation(tmp_path: Path) -> None:
    fixture = _authoritative_fixture(tmp_path, stage="T0")
    evidence = verify_authoritative_training_evidence(
        fixture.inputs, _profile_override=fixture.profile, torch_module=_FakeTorch()
    )

    report = score_training_stage(evidence, stage="T0")

    assert report.threshold == 70
    assert report.status == "PASS_COMMERCIAL_CANDIDATE"


def test_authoritative_bundle_is_complete_immutable_and_atomic(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _authoritative_fixture(tmp_path)
    evidence = verify_authoritative_training_evidence(
        fixture.inputs, _profile_override=fixture.profile, torch_module=_FakeTorch()
    )
    output = tmp_path / "bundle"
    scoring_cli.write_stage_result(evidence, stage="T1", output_dir=output)
    assert {path.name for path in output.iterdir()} == {
        "score.json", "score.md", "metrics.json", "environment.json", "lineage.json", "README.md"
    }
    assert json.loads((output / "score.json").read_text(encoding="utf-8"))["status"] == "PASS_COMMERCIAL_CANDIDATE"
    with pytest.raises(FileExistsError, match="new and empty"):
        scoring_cli.write_stage_result(evidence, stage="T1", output_dir=output)

    failed_output = tmp_path / "nested" / "result"
    original_write = scoring_cli._write_new
    writes = 0

    def fail_during_bundle(path: Path, content: str | bytes) -> None:
        nonlocal writes
        writes += 1
        if writes == 3:
            raise OSError("injected bundle write failure")
        original_write(path, content)

    monkeypatch.setattr(scoring_cli, "_write_new", fail_during_bundle)

    with pytest.raises(OSError, match="injected bundle write failure"):
        scoring_cli.write_stage_result(evidence, stage="T1", output_dir=failed_output)

    assert not failed_output.exists()
    assert not list(failed_output.parent.glob(".result.*"))


def test_score_rejects_free_form_self_attested_evidence(tmp_path: Path) -> None:
    evidence = _evidence(tmp_path)
    evidence["provenance"]["git_commit"] = "a" * 40  # type: ignore[index]
    evidence["split"]["official_test_accessed"] = False  # type: ignore[index]
    evidence["initialization"]["model_checkpoint_loaded"] = False  # type: ignore[index]

    with pytest.raises(TypeError, match="AuthoritativeTrainingEvidence"):
        score_training_stage(evidence, stage="T1")


def test_authoritative_fixture_passes_only_after_recomputed_verification(
    tmp_path: Path,
) -> None:
    fixture = _authoritative_fixture(tmp_path)

    evidence = verify_authoritative_training_evidence(
        fixture.inputs,
        _profile_override=fixture.profile,
        torch_module=_FakeTorch(),
    )
    report = score_training_stage(evidence, stage="T1")

    assert report.status == "PASS_COMMERCIAL_CANDIDATE"
    assert report.score == 100
    assert report.threshold == 75
    assert sum(check.weight for check in report.checks) == 100
    assert weights_by_category(report) == {
        "provenance": 25,
        "split": 20,
        "stability": 20,
        "validation": 20,
        "artifacts": 15,
    }
    mutable_copy = evidence.evidence
    mutable_copy["initialization"]["model_checkpoint_loaded"] = True  # type: ignore[index]
    assert score_training_stage(evidence, stage="T1").status == "PASS_COMMERCIAL_CANDIDATE"


def test_stage_evidence_description_names_the_scored_stage(tmp_path: Path) -> None:
    fixture = _authoritative_fixture(tmp_path, stage="T5")
    evidence = verify_authoritative_training_evidence(
        fixture.inputs,
        _profile_override=fixture.profile,
        torch_module=_FakeTorch(),
    )

    report = score_training_stage(evidence, stage="T5")
    validation_check = next(
        check for check in report.checks if check.check_id == "validation.stage_evidence"
    )

    assert validation_check.observed == "all 17 required T5 validation metrics recorded"


def test_t800_checkpoint_payload_and_score_are_authoritative(tmp_path: Path) -> None:
    fixture = _authoritative_fixture(tmp_path, stage="T800")
    evidence = verify_authoritative_training_evidence(
        fixture.inputs,
        _profile_override=fixture.profile,
        torch_module=_FakeTorch(),
    )

    report = score_training_stage(evidence, stage="T800")

    assert report.threshold == 90
    assert report.score == 100
    assert report.status == "PASS_COMMERCIAL_CANDIDATE"
    assert report.failed_blockers == ()


def test_score_cli_accepts_t800_evidence_stage(tmp_path: Path) -> None:
    args = scoring_cli._parser().parse_args(
        [
            "--stage", "T800",
            "--run-id", "run-3035",
            "--profile", str(tmp_path / "profile.json"),
            "--project-repo", str(tmp_path / "project"),
            "--upstream-dir", str(tmp_path / "upstream"),
            "--processed-root", str(tmp_path / "processed"),
            "--backbone", str(tmp_path / "backbone.pth"),
            "--checkpoint", str(tmp_path / "last.pth"),
            "--checkpoint-manifest", str(tmp_path / "checkpoint-manifest.json"),
            "--environment", str(tmp_path / "environment.json"),
            "--metrics", str(tmp_path / "metrics.json"),
            "--output-dir", str(tmp_path / "score-bundle"),
        ]
    )

    assert args.stage == "T800"


def test_matching_self_declared_hash_cannot_replace_profile_backbone(
    tmp_path: Path,
) -> None:
    fixture = _authoritative_fixture(tmp_path)
    arbitrary = tmp_path / fixture.backbone.name
    arbitrary.write_bytes(b"arbitrary attacker bytes")

    with pytest.raises(ValueError, match="backbone.*canonical|backbone.*profile"):
        verify_authoritative_training_evidence(
            replace(fixture.inputs, backbone_path=arbitrary),
            _profile_override=fixture.profile,
            torch_module=_FakeTorch(),
        )


def test_authority_recomputes_project_head_and_clean_status(tmp_path: Path) -> None:
    fixture = _authoritative_fixture(tmp_path)
    (fixture.project_root / "untracked.txt").write_text("dirty\n", encoding="utf-8")

    with pytest.raises(ValueError, match="project.*clean"):
        verify_authoritative_training_evidence(
            fixture.inputs,
            _profile_override=fixture.profile,
            torch_module=_FakeTorch(),
        )


@pytest.mark.parametrize("defect", ("dirty", "wrong-origin"))
def test_authority_rejects_dirty_or_wrong_upstream(
    tmp_path: Path, defect: str
) -> None:
    fixture = _authoritative_fixture(tmp_path)
    if defect == "dirty":
        (fixture.upstream_root / "untracked.txt").write_text("dirty\n", encoding="utf-8")
    else:
        _git(fixture.upstream_root, "remote", "set-url", "origin", "https://example.test/foreign.git")

    with pytest.raises(ValueError, match="upstream|pinned|clean|origin"):
        verify_authoritative_training_evidence(
            fixture.inputs,
            _profile_override=fixture.profile,
            torch_module=_FakeTorch(),
        )


@pytest.mark.parametrize("defect", ("count", "overlap"))
def test_authority_recomputes_split_count_and_overlap(
    tmp_path: Path, defect: str
) -> None:
    fixture = _authoritative_fixture(tmp_path)
    train = fixture.processed_root / "train.txt"
    validation = fixture.processed_root / "val.txt"
    train_ids = train.read_text(encoding="utf-8").splitlines()
    if defect == "count":
        train_ids.pop()
    else:
        train_ids[-1] = validation.read_text(encoding="utf-8").splitlines()[0]
    train.write_bytes(("\n".join(train_ids) + "\n").encode())

    with pytest.raises(ValueError, match="split|961|disjoint|hash"):
        verify_authoritative_training_evidence(
            fixture.inputs,
            _profile_override=fixture.profile,
            torch_module=_FakeTorch(),
        )


def test_authority_rejects_wrong_backbone_bytes(tmp_path: Path) -> None:
    fixture = _authoritative_fixture(tmp_path)
    fixture.backbone.write_bytes(b"tampered backbone")

    with pytest.raises(ValueError, match="backbone.*size|backbone.*SHA"):
        verify_authoritative_training_evidence(
            fixture.inputs,
            _profile_override=fixture.profile,
            torch_module=_FakeTorch(),
        )


def test_authority_rejects_foreign_checkpoint_lineage_even_with_updated_manifest_hash(
    tmp_path: Path,
) -> None:
    fixture = _authoritative_fixture(tmp_path)
    manifest = json.loads(fixture.checkpoint_manifest.read_text(encoding="utf-8"))
    manifest["checkpoints"].append({"filename": "foreign.pth"})
    _json_write(fixture.checkpoint_manifest, manifest)
    with pytest.raises(ValueError, match="checkpoint policy entry"):
        verify_authoritative_training_evidence(
            fixture.inputs,
            _profile_override=fixture.profile,
            torch_module=_FakeTorch(),
        )
    manifest["checkpoints"].pop()
    _json_write(fixture.checkpoint_manifest, manifest)

    with fixture.checkpoint.open("rb") as stream:
        state = pickle.load(stream)
    state["dataset_inventory_sha256"] = "f" * 64
    with fixture.checkpoint.open("wb") as stream:
        pickle.dump(state, stream)
    entry = next(row for row in manifest["checkpoints"] if row["filename"] == fixture.checkpoint.name)
    entry["sha256"] = sha256_file(fixture.checkpoint)
    entry["byte_count"] = fixture.checkpoint.stat().st_size
    _json_write(fixture.checkpoint_manifest, manifest)

    with pytest.raises(ValueError, match="dataset_inventory_sha256|provenance"):
        verify_authoritative_training_evidence(
            fixture.inputs,
            _profile_override=fixture.profile,
            torch_module=_FakeTorch(),
        )


@pytest.mark.parametrize("defect", ("missing", "tampered"))
def test_authority_requires_checkpoint_bound_immutable_environment(
    tmp_path: Path, defect: str
) -> None:
    fixture = _authoritative_fixture(tmp_path)
    if defect == "missing":
        fixture.environment.unlink()
    else:
        fixture.environment.write_bytes(b'{"tampered":true}\n')

    with pytest.raises(ValueError, match="environment"):
        verify_authoritative_training_evidence(
            fixture.inputs,
            _profile_override=fixture.profile,
            torch_module=_FakeTorch(),
        )


def test_authority_requires_checkpoint_bound_metrics(tmp_path: Path) -> None:
    fixture = _authoritative_fixture(tmp_path)
    _json_write(
        fixture.metrics,
        {"schema_version": 1, "run_id": "run-3035", "stage": "T1", "metrics": {"mae": 0}},
    )

    with pytest.raises(ValueError, match="metrics.*SHA|metrics.*hash"):
        verify_authoritative_training_evidence(
            fixture.inputs,
            _profile_override=fixture.profile,
            torch_module=_FakeTorch(),
        )


def test_production_cli_has_no_test_or_trust_override_option() -> None:
    destinations = {action.dest for action in scoring_cli._parser()._actions}

    assert {
        "profile", "project_repo", "upstream_dir", "processed_root", "backbone",
        "checkpoint", "checkpoint_manifest", "environment", "metrics",
    }.issubset(destinations)
    assert not any("test" in destination or "override" in destination for destination in destinations)
