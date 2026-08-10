from __future__ import annotations

import importlib
import json
import os
import pickle
from pathlib import Path
import sys

import pytest


def _subject():
    return importlib.import_module("droneai.steerer_official_longrun")


def _upstream_fixture(tmp_path: Path) -> Path:
    upstream = tmp_path / "STEERER"
    (upstream / "tools").mkdir(parents=True)
    (upstream / "configs").mkdir()
    (upstream / "tools" / "train_cc.py").write_text("# pinned trainer\n", encoding="utf-8")
    (upstream / "configs" / "QNRF_final.py").write_text("# pinned config\n", encoding="utf-8")
    return upstream


def test_official_validation_schedule_matches_all_399_upstream_evaluations() -> None:
    epochs = _subject().official_validation_epochs()

    assert len(epochs) == 399
    assert epochs[:14] == tuple(range(7, 99, 7))
    assert epochs[14:34] == tuple(range(105, 201, 5))
    assert epochs[34:100] == tuple(range(204, 400, 3))
    assert epochs[100:200] == tuple(range(402, 601, 2))
    assert epochs[200:] == tuple(range(602, 801))
    assert epochs[-1] == 800


def test_official_command_has_only_path_overrides_and_pinned_raw_entrypoint(
    tmp_path: Path,
) -> None:
    module = _subject()
    upstream = _upstream_fixture(tmp_path)
    processed = tmp_path / "processed"
    processed.mkdir()
    backbone = tmp_path / "hrnet.pth"
    backbone.write_bytes(b"backbone")
    log_root = tmp_path / "checkpoints" / "official-output"

    command = module.build_official_a800_command(
        python_executable=tmp_path / "venv" / "python",
        upstream_dir=upstream,
        processed_root=processed,
        backbone_path=backbone,
        log_root=log_root,
    )

    assert command[:7] == (
        str((tmp_path / "venv" / "python").resolve()),
        str((upstream / "tools" / "train_cc.py").resolve()),
        "--cfg",
        str((upstream / "configs" / "QNRF_final.py").resolve()),
        "--launcher",
        "none",
        "--cfg-options",
    )
    assert command[7:] == (
        f"dataset.root={processed.resolve().as_posix()}/",
        f"network.pretrained_backbone={backbone.resolve().as_posix()}",
        f"log_dir={log_root.resolve().as_posix()}",
    )
    forbidden = ("optimizer", "end_epoch", "val_span", "batch_size", "scale_factor")
    assert not any(name in argument for name in forbidden for argument in command[7:])


def test_official_resume_adds_only_verified_upstream_resume_path(tmp_path: Path) -> None:
    module = _subject()
    upstream = _upstream_fixture(tmp_path)
    processed = tmp_path / "processed"
    processed.mkdir()
    backbone = tmp_path / "hrnet.pth"
    backbone.write_bytes(b"backbone")
    log_root = tmp_path / "checkpoints" / "official-output"
    resume = log_root / "QNRF" / "MocHRBackbone_hrnet48" / "run"
    resume.mkdir(parents=True)
    (resume / "checkpoint.pth.tar").write_bytes(b"checkpoint")

    command = module.build_official_a800_command(
        python_executable=tmp_path / "venv" / "python",
        upstream_dir=upstream,
        processed_root=processed,
        backbone_path=backbone,
        log_root=log_root,
        resume_run_dir=resume,
    )

    assert command[-1] == f"train.resume_path={resume.resolve().as_posix()}"
    assert sum(argument.startswith("train.") for argument in command) == 1


def test_a800_cli_exposes_no_hyperparameter_or_model_checkpoint_override() -> None:
    cli = importlib.import_module("scripts.run_steerer_a800_official")
    parser = cli._parser()
    required = [
        "start",
        "--config",
        "configs/training/steerer.json",
        "--run-id",
        "steerer-qnrf-official-a800-v1",
        "--processed-root",
        "/workspace/data/datasets/qnrf/prepared",
        "--g2-manifest",
        "/workspace/data/datasets/qnrf/prepared/manifests/dataset-preparation.json",
        "--upstream-dir",
        "/workspace/upstreams/STEERER",
        "--backbone",
        "/workspace/data/checkpoints/backbones/hrnet.pth",
        "--container-image-digest",
        "sha256:" + "a" * 64,
    ]

    parsed = parser.parse_args(required)
    assert parsed.action == "start"
    for forbidden in (
        "--epochs",
        "--test-interval",
        "--optimizer",
        "--model-checkpoint",
        "--batch-size",
    ):
        with pytest.raises(SystemExit):
            parser.parse_args(required + [forbidden, "1"])


def _lineage(module):
    return module.OfficialA800Lineage(
        project_commit="1" * 40,
        upstream_commit="2" * 40,
        upstream_config_sha256="3" * 64,
        profile_sha256="4" * 64,
        dataset_manifest_sha256="5" * 64,
        dataset_content_sha256="6" * 64,
        train_split_sha256="7" * 64,
        test_split_sha256="8" * 64,
        backbone_sha256="9" * 64,
        container_image_digest="sha256:" + "a" * 64,
    )


def test_raw_launcher_records_exact_schedule_and_completed_attempt(
    tmp_path: Path,
) -> None:
    module = _subject()
    child = tmp_path / "child.py"
    child.write_text("print('raw-upstream-child', flush=True)\n", encoding="utf-8")

    result = module.run_raw_official_training(
        run_id="steerer-qnrf-official-a800-unit",
        command=(sys.executable, str(child)),
        cwd=tmp_path,
        result_dir=tmp_path / "results",
        checkpoint_dir=tmp_path / "checkpoints",
        lineage=_lineage(module),
        environment={"runtime": "unit"},
        resume=False,
    )

    assert result.return_code == 0
    assert result.attempt == 1
    status = json.loads(result.status_path.read_text(encoding="utf-8"))
    manifest = json.loads(result.attempt_manifest_path.read_text(encoding="utf-8"))
    assert status["state"] == "completed"
    assert status["return_code"] == 0
    assert manifest["execution"] == "pinned raw upstream tools/train_cc.py"
    assert manifest["official_validation_count"] == 399
    assert manifest["official_validation_epochs"][-199:] == list(range(602, 801))
    assert manifest["resume"] is False
    assert manifest["lineage"]["upstream_commit"] == "2" * 40
    assert not list(tmp_path.rglob("*.tmp"))


def test_raw_launcher_refuses_to_overwrite_an_existing_fresh_run(tmp_path: Path) -> None:
    module = _subject()
    result_dir = tmp_path / "results"
    result_dir.mkdir()
    sentinel = result_dir / "sentinel.txt"
    sentinel.write_text("preserve", encoding="utf-8")

    with pytest.raises(FileExistsError, match="fresh A800"):
        module.run_raw_official_training(
            run_id="steerer-qnrf-official-a800-unit",
            command=(sys.executable, "-c", "raise SystemExit(0)"),
            cwd=tmp_path,
            result_dir=result_dir,
            checkpoint_dir=tmp_path / "checkpoints",
            lineage=_lineage(module),
            environment={},
            resume=False,
        )

    assert sentinel.read_text(encoding="utf-8") == "preserve"
    assert not (tmp_path / "checkpoints").exists()


@pytest.mark.skipif(os.name != "posix", reason="production lock uses Linux flock")
def test_live_run_lock_rejection_preserves_current_status_and_attempts(
    tmp_path: Path,
) -> None:
    module = _subject()
    from droneai.steerer_training_longrun import RunLock

    result_dir = tmp_path / "results"
    checkpoint_dir = tmp_path / "checkpoints"
    result_dir.mkdir()
    checkpoint_dir.mkdir()
    status = result_dir / "launcher-status.json"
    status.write_text('{"state":"running"}\n', encoding="utf-8")
    before = status.read_bytes()

    with RunLock(result_dir / "run.lock", run_id="steerer-qnrf-official-a800-unit"):
        with pytest.raises(FileExistsError, match="already active"):
            module.run_raw_official_training(
                run_id="steerer-qnrf-official-a800-unit",
                command=(sys.executable, "-c", "raise SystemExit(0)"),
                cwd=tmp_path,
                result_dir=result_dir,
                checkpoint_dir=checkpoint_dir,
                lineage=_lineage(module),
                environment={},
                resume=True,
            )

    assert status.read_bytes() == before
    assert not list(result_dir.glob("attempt-*-manifest.json"))


class _FakeTorch:
    def load(self, path: str | Path, **_kwargs: object) -> object:
        with Path(path).open("rb") as stream:
            return pickle.load(stream)


def test_resume_checkpoint_requires_hash_and_exact_official_payload(tmp_path: Path) -> None:
    module = _subject()
    checkpoint = tmp_path / "checkpoint.pth.tar"
    payload = {
        "epoch": 7,
        "best_MAE": 100.0,
        "best_MSE": 200.0,
        "state_dict": {"weight": [1, 2]},
        "optimizer": {"state": {}, "param_groups": []},
    }
    checkpoint.write_bytes(pickle.dumps(payload))
    from droneai.integrity import sha256_file

    observed = module.verify_official_resume_checkpoint(
        checkpoint,
        expected_sha256=sha256_file(checkpoint),
        torch_module=_FakeTorch(),
    )

    assert observed["epoch"] == 7
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        module.verify_official_resume_checkpoint(
            checkpoint,
            expected_sha256="0" * 64,
            torch_module=_FakeTorch(),
        )

    payload["rng"] = {}
    checkpoint.write_bytes(pickle.dumps(payload))
    with pytest.raises(ValueError, match="exact raw-upstream schema"):
        module.verify_official_resume_checkpoint(
            checkpoint,
            expected_sha256=sha256_file(checkpoint),
            torch_module=_FakeTorch(),
        )
