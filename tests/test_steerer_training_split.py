from __future__ import annotations

import hashlib
import importlib.util
import inspect
import json
import subprocess
import sys
from pathlib import Path

import pytest
from PIL import Image

import droneai.steerer_training_split as training_split
from droneai.integrity import sha256_file
from droneai.steerer_training_split import build_training_split, write_training_split
from droneai.ucf_qnrf import UCFQNRFRecord


REPO_ROOT = Path(__file__).resolve().parents[1]
PROFILE_PATH = REPO_ROOT / "configs/training/steerer_ucf_qnrf_imagenet.home5090.json"


def _record(sample_id: str, band: str) -> UCFQNRFRecord:
    return UCFQNRFRecord(
        sample_id=sample_id,
        image_path=Path("/training-only") / f"{sample_id}.jpg",
        annotation_path=Path("/training-only") / f"{sample_id}_ann.mat",
        image_sha256=hashlib.sha256(f"image:{sample_id}".encode()).hexdigest(),
        annotation_sha256=hashlib.sha256(f"annotation:{sample_id}".encode()).hexdigest(),
        width=1,
        height=1,
        points=(),
        density_band=band,
    )


@pytest.fixture
def records() -> tuple[UCFQNRFRecord, ...]:
    return tuple(
        _record(f"{band}-{index:04d}", band)
        for band, size in (("low", 601), ("medium", 400), ("high", 200))
        for index in range(size)
    )


def test_split_is_deterministic_disjoint_complete_and_stratified(
    records: tuple[UCFQNRFRecord, ...],
) -> None:
    first = build_training_split(records, seed=3035, validation_count=240)
    second = build_training_split(tuple(reversed(records)), seed=3035, validation_count=240)

    assert first.train_ids == second.train_ids
    assert first.validation_ids == second.validation_ids
    assert len(first.train_ids) == 961
    assert len(first.validation_ids) == 240
    assert set(first.train_ids).isdisjoint(first.validation_ids)
    assert set(first.train_ids) | set(first.validation_ids) == {r.sample_id for r in records}
    assert first.band_counts == {
        "high": {"population": 200, "train": 160, "validation": 40},
        "low": {"population": 601, "train": 481, "validation": 120},
        "medium": {"population": 400, "train": 320, "validation": 80},
    }


def test_split_api_cannot_accept_a_test_root() -> None:
    assert "test_root" not in inspect.signature(build_training_split).parameters


def test_split_artifacts_are_atomic_and_keep_test_population_sealed(
    tmp_path: Path, records: tuple[UCFQNRFRecord, ...]
) -> None:
    split = build_training_split(records, seed=3035, validation_count=240)
    output = tmp_path / "split"

    write_training_split(output, split)

    assert (output / "train.txt").read_text(encoding="utf-8").splitlines() == list(split.train_ids)
    assert (output / "val.txt").read_text(encoding="utf-8").splitlines() == list(split.validation_ids)
    assert hashlib.sha256((output / "train.txt").read_bytes()).hexdigest() == split.train_sha256
    assert hashlib.sha256((output / "val.txt").read_bytes()).hexdigest() == split.validation_sha256
    assert json.loads((output / "test-sealed.json").read_text(encoding="utf-8")) == {
        "role": "sealed",
        "sample_count": 334,
        "accessed": False,
    }
    manifest = json.loads((output / "split-manifest.json").read_text(encoding="utf-8"))
    assert manifest["train_sha256"] == split.train_sha256
    assert manifest["validation_sha256"] == split.validation_sha256
    assert "training-only" not in (output / "split-manifest.json").read_text(encoding="utf-8")


def test_split_writer_requires_a_new_empty_output_directory(
    tmp_path: Path, records: tuple[UCFQNRFRecord, ...]
) -> None:
    output = tmp_path / "split"
    output.mkdir()

    with pytest.raises(FileExistsError):
        write_training_split(output, build_training_split(records, seed=3035, validation_count=240))


@pytest.mark.parametrize(
    "failed_artifact",
    ("train.txt", "val.txt", "test-sealed.json", "split-manifest.json"),
)
def test_split_writer_cleans_temp_files_when_a_split_artifact_replace_fails(
    tmp_path: Path,
    records: tuple[UCFQNRFRecord, ...],
    monkeypatch: pytest.MonkeyPatch,
    failed_artifact: str,
) -> None:
    """A failed split-file replace must not expose a partial destination or leak a temp file."""

    output = tmp_path / "split"
    real_replace = training_split.os.replace

    def fail_artifact_replace(source: Path, destination: Path) -> None:
        if Path(destination).name == failed_artifact:
            raise OSError(f"injected {failed_artifact} replacement failure")
        real_replace(source, destination)

    monkeypatch.setattr(training_split.os, "replace", fail_artifact_replace)

    with pytest.raises(OSError, match=f"injected {failed_artifact} replacement failure"):
        write_training_split(
            output, build_training_split(records, seed=3035, validation_count=240)
        )

    assert not output.exists()
    assert not list(tmp_path.rglob("*.tmp"))


def test_prepare_cli_exposes_train_root_but_no_test_or_dataset_root() -> None:
    help_text = subprocess.run(
        [sys.executable, "scripts/prepare_steerer_ucf_training.py", "--help"],
        text=True,
        capture_output=True,
        check=True,
    ).stdout

    assert "--train-root" in help_text
    assert "--config" in help_text
    assert "--output-root" in help_text
    assert "--test-root" not in help_text
    assert "--dataset-root" not in help_text
    assert "--output-dir" not in help_text
    assert "--seed" not in help_text
    assert "--validation-count" not in help_text


def _prepare_module():
    spec = importlib.util.spec_from_file_location(
        "prepare_steerer_ucf_training_test",
        REPO_ROOT / "scripts/prepare_steerer_ucf_training.py",
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_prepare_cli_rejects_incomplete_population_before_creating_output(
    tmp_path: Path, records: tuple[UCFQNRFRecord, ...], monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _prepare_module()
    monkeypatch.setattr(module, "index_ucf_qnrf_train", lambda _: records[:-1])
    output = tmp_path / "prepared"

    assert module.main(
        [
            "--config", str(PROFILE_PATH),
            "--train-root", str(tmp_path / "Train"),
            "--output-root", str(output),
        ]
    ) == 2

    assert not output.exists()


def test_prepare_cli_writes_exact_approved_split_counts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    source_image = tmp_path / "source.jpg"
    source_annotation = tmp_path / "source_ann.mat"
    Image.new("RGB", (2, 2), color=(12, 34, 56)).save(source_image, quality=91)
    source_annotation.write_bytes(b"pre-indexed annotation")
    source_sha256 = sha256_file(source_image)
    annotation_sha256 = sha256_file(source_annotation)
    records = tuple(
        UCFQNRFRecord(
            sample_id=f"sample-{index:04d}",
            image_path=source_image,
            annotation_path=source_annotation,
            image_sha256=source_sha256,
            annotation_sha256=annotation_sha256,
            width=2,
            height=2,
            points=((0.0, 0.0),),
            density_band="low",
        )
        for index in range(1201)
    )
    module = _prepare_module()
    monkeypatch.setattr(module, "index_ucf_qnrf_train", lambda _: records)
    output = tmp_path / "prepared"

    assert module.main(
        [
            "--config", str(PROFILE_PATH),
            "--train-root", str(tmp_path / "Train"),
            "--output-root", str(output),
        ]
    ) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "prepared"
    assert (result["sample_count"], result["train_count"], result["validation_count"]) == (
        1201,
        961,
        240,
    )
    assert result["manifest_path"] == str(output / "manifests" / "training-manifest.json")
    assert len(result["manifest_sha256"]) == 64
    manifest = json.loads((output / "manifests" / "training-manifest.json").read_text())
    assert manifest["source_partition"] == "official_train_only"
    assert manifest["converted_sample_count"] == 1201
    assert (manifest["train_count"], manifest["validation_count"]) == (961, 240)
    assert (
        manifest["count_mismatches"],
        manifest["out_of_bounds_points"],
        manifest["out_of_bounds_boxes"],
    ) == (0, 0, 0)
    assert len((output / "train.txt").read_text().splitlines()) == 961
    assert len((output / "val.txt").read_text().splitlines()) == 240
    assert len(list((output / "images").glob("*.jpg"))) == 1201
    assert len(list((output / "jsons").glob("*.json"))) == 1201
    assert len((output / "manifests" / "input-inventory.jsonl").read_text().splitlines()) == 1201
    assert len((output / "manifests" / "output-inventory.jsonl").read_text().splitlines()) == 1201
    source_bytes = source_image.read_bytes()
    for image in (output / "images").glob("*.jpg"):
        assert image.read_bytes() == source_bytes
        assert sha256_file(image) == source_sha256
    assert not list(output.rglob("*.tmp"))
