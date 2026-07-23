from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from PIL import Image
from scipy.io import savemat

from droneai.integrity import sha256_file
from droneai.round2_config import DatasetLane, ModelLane, Round2Config
from droneai.round2_manifest import (
    build_round2_manifest,
    verify_round2_manifest,
    write_round2_manifest,
)


def _write_json(path: Path, payload: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _write_ucf(root: Path) -> None:
    root.mkdir(parents=True)
    for index, count in ((1, 2), (2, 3)):
        image = root / f"img_{index:04d}.jpg"
        Image.new("RGB", (60, 40), color=(index, 2, 3)).save(image)
        points = np.asarray(
            [[float(5 + offset), float(6 + offset)] for offset in range(count)],
            dtype=np.float32,
        )
        savemat(root / f"img_{index:04d}_ann.mat", {"annPoints": points})


def _write_jhu(root: Path) -> None:
    val = root / "val"
    (val / "images").mkdir(parents=True)
    (val / "gt").mkdir(parents=True)
    labels: list[str] = []
    for index, count in ((1, 2), (2, 3)):
        sample_id = f"{index:04d}"
        Image.new("RGB", (60, 40), color=(index, 2, 3)).save(
            val / "images" / f"{sample_id}.jpg"
        )
        (val / "gt" / f"{sample_id}.txt").write_text(
            "".join(
                f"{10 + offset} {11 + offset} 8 10 1 0\n"
                for offset in range(count)
            ),
            encoding="utf-8",
        )
        labels.append(f"{sample_id},{count},stadium,0,0")
    (val / "image_labels.txt").write_text(
        "\n".join(labels) + "\n", encoding="utf-8"
    )


def _write_up_count(root: Path) -> None:
    rows: list[dict[str, object]] = []
    for index, split in ((1, "val"), (2, "test")):
        sequence = f"{index:04d}"
        stem = f"{sequence}__0000__60.0"
        image = root / "images" / sequence / f"{stem}.jpg"
        annotation = (
            root / "normalized_annotations" / sequence / f"{stem}.json"
        )
        image.parent.mkdir(parents=True)
        annotation.parent.mkdir(parents=True)
        Image.new("RGB", (60, 40), color=(index, 2, 3)).save(image)
        annotation.write_text(
            json.dumps({"points": [[10, 11], [12, 13]]}), encoding="utf-8"
        )
        rows.append(
            {
                "sample_id": f"{sequence}/{stem}",
                "split": split,
                "group_id": sequence,
                "frame_id": 0,
                "image_path": image.relative_to(root).as_posix(),
                "annotation_path": annotation.relative_to(root).as_posix(),
                "image_sha256": sha256_file(image),
                "annotation_sha256": sha256_file(annotation),
                "width": 60,
                "height": 40,
                "point_count": 2,
                "condition_tags": {
                    "altitude_band": "mid",
                    "density_band": "low",
                },
            }
        )
    (root / "inventory.jsonl").write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8"
    )


@pytest.fixture
def round2_fixture(tmp_path: Path) -> tuple[Round2Config, dict[str, Path]]:
    roots = {
        "ucf-qnrf-kaggle-apache": tmp_path / "ucf",
        "jhu-crowd-plus-v2": tmp_path / "jhu",
        "up-count-v1": tmp_path / "up-count",
    }
    _write_ucf(roots["ucf-qnrf-kaggle-apache"])
    _write_jhu(roots["jhu-crowd-plus-v2"])
    _write_up_count(roots["up-count-v1"])
    manifests = {
        dataset_id: _write_json(
            tmp_path / "manifests" / f"{dataset_id}.json",
            {"schema_version": 1, "dataset_id": dataset_id},
        )
        for dataset_id in roots
    }
    config = Round2Config(
        round_id="round-2-fixture",
        action="frozen_internal_reference_evaluation",
        models=(
            ModelLane("steerer", "density_and_points", "PASS_COMMERCIAL_CANDIDATE"),
            ModelLane("dm-count", "density", "PASS_COMMERCIAL_CANDIDATE"),
            ModelLane("mpcount", "density", "PASS_COMMERCIAL_CANDIDATE"),
        ),
        datasets=(
            DatasetLane("ucf-qnrf-kaggle-apache", "test", 2, "PASS_COMMERCIAL_CANDIDATE", manifests["ucf-qnrf-kaggle-apache"]),
            DatasetLane("jhu-crowd-plus-v2", "val", 2, "PASS_RESEARCH_ONLY", manifests["jhu-crowd-plus-v2"]),
            DatasetLane("up-count-v1", "val_test", 2, "PASS_RESEARCH_ONLY", manifests["up-count-v1"]),
        ),
        total_samples_per_model=6,
        combined_rights_scope="PASS_RESEARCH_ONLY",
        fine_tuning=False,
    )
    return config, roots


def test_manifest_contains_exact_counts_and_aggregate_hash(
    round2_fixture: tuple[Round2Config, dict[str, Path]],
) -> None:
    config, roots = round2_fixture

    payload = build_round2_manifest(config, roots)

    assert payload["counts"] == {
        "ucf-qnrf-kaggle-apache": 2,
        "jhu-crowd-plus-v2": 2,
        "up-count-v1": 2,
    }
    assert len(payload["sample_manifest_sha256"]) == 64
    assert len(payload["samples"]) == 6
    assert len({row["global_sample_id"] for row in payload["samples"]}) == 6
    assert len({row["split_id"] for row in payload["samples"]}) == 3


def test_point_shortlist_changes_config_identity_not_sample_identity(
    round2_fixture: tuple[Round2Config, dict[str, Path]],
) -> None:
    legacy_config, roots = round2_fixture
    point_config = replace(
        legacy_config,
        models=(
            ModelLane(
                "steerer",
                "density_and_points",
                "PASS_COMMERCIAL_CANDIDATE",
            ),
            ModelLane("pet", "points", "PASS_RESEARCH_ONLY"),
            ModelLane(
                "apgcc",
                "points",
                "PASS_COMMERCIAL_CANDIDATE",
            ),
        ),
    )

    legacy = build_round2_manifest(legacy_config, roots)
    point = build_round2_manifest(point_config, roots)

    assert point["config_identity_sha256"] != legacy["config_identity_sha256"]
    assert point["sample_manifest_sha256"] == legacy["sample_manifest_sha256"]


def test_verify_manifest_rejects_changed_annotation(
    round2_fixture: tuple[Round2Config, dict[str, Path]],
) -> None:
    config, roots = round2_fixture
    payload = build_round2_manifest(config, roots)
    annotation = roots["jhu-crowd-plus-v2"] / "val/gt/0001.txt"
    annotation.write_text(
        annotation.read_text(encoding="utf-8").replace("10 11", "14 15"),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="annotation SHA-256"):
        verify_round2_manifest(payload, config, roots)


def test_manifest_write_is_atomic_and_self_verifying(
    round2_fixture: tuple[Round2Config, dict[str, Path]], tmp_path: Path
) -> None:
    config, roots = round2_fixture
    output = tmp_path / "result/sample-manifest.json"

    digest = write_round2_manifest(output, config=config, dataset_roots=roots)

    assert output.is_file()
    assert not output.with_suffix(".json.tmp").exists()
    assert digest == sha256_file(output)
    verified = verify_round2_manifest(
        json.loads(output.read_text(encoding="utf-8")), config, roots
    )
    assert len(verified) == 6
