from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

import json

from droneai.jhu_crowd import index_jhu_validation, prepare_jhu_validation
from droneai.round2_samples import evaluation_samples_from_jhu


def _write_jhu_sample(
    root: Path,
    sample_id: str,
    *,
    points: tuple[tuple[int, int], ...],
    scene: str = "stadium",
    weather: int = 0,
    distractor: int = 0,
) -> None:
    image_root = root / "images"
    gt_root = root / "gt"
    image_root.mkdir(parents=True, exist_ok=True)
    gt_root.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (100, 80), color=(10, 20, 30)).save(
        image_root / f"{sample_id}.jpg"
    )
    (gt_root / f"{sample_id}.txt").write_text(
        "".join(f"{x} {y} 8 10 1 0\n" for x, y in points),
        encoding="utf-8",
    )
    label_path = root / "image_labels.txt"
    with label_path.open("a", encoding="utf-8") as stream:
        stream.write(
            f"{sample_id},{len(points)},{scene},{weather},{distractor}\n"
        )


def test_index_jhu_validation_reads_official_points_and_labels(
    tmp_path: Path,
) -> None:
    val = tmp_path / "val"
    _write_jhu_sample(val, "0001", points=((10, 11), (20, 21)))

    rows = index_jhu_validation(val, expected_samples=1)

    assert rows[0].count == 2
    assert rows[0].points == ((10.0, 11.0), (20.0, 21.0))
    assert rows[0].condition_tags == {
        "scene": "stadium",
        "weather": "none",
        "distractor": "false",
        "density_band": "low",
    }


def test_index_jhu_validation_rejects_declared_count_mismatch(
    tmp_path: Path,
) -> None:
    val = tmp_path / "val"
    _write_jhu_sample(val, "0001", points=((10, 11),))
    (val / "image_labels.txt").write_text(
        "0001,2,stadium,0,0\n", encoding="utf-8"
    )

    with pytest.raises(ValueError, match="declared count"):
        index_jhu_validation(val, expected_samples=1)


def test_index_jhu_validation_requires_exact_frozen_count(tmp_path: Path) -> None:
    val = tmp_path / "val"
    _write_jhu_sample(val, "0001", points=((10, 11),))

    with pytest.raises(ValueError, match="expected 500"):
        index_jhu_validation(val, expected_samples=500)


def test_jhu_out_of_bounds_points_are_count_preserved_but_not_localization_truth(
    tmp_path: Path,
) -> None:
    val = tmp_path / "val"
    _write_jhu_sample(val, "0001", points=((100, 79), (130, 90)))

    rows = index_jhu_validation(val, expected_samples=1)

    assert rows[0].count == 2
    assert rows[0].normalization_corrections == 2
    assert rows[0].point_localization_valid is False
    assert all(0 <= x < 100 and 0 <= y < 80 for x, y in rows[0].points)

    sample = evaluation_samples_from_jhu(rows, split_id="fixture-jhu")
    assert sample[0].ground_truth_count == 2.0
    assert sample[0].has_point_annotations is False
    assert sample[0].ground_truth_points == ()
    assert sample[0].ground_truth_density is None
    assert sample[0].condition_tags["point_localization_valid"] == "false"


def test_prepare_jhu_validation_writes_stage1_inventory_without_editing_source(
    tmp_path: Path,
) -> None:
    val = tmp_path / "val"
    _write_jhu_sample(val, "0001", points=((10, 11), (20, 21)))
    source_annotation = (val / "gt/0001.txt").read_bytes()

    count = prepare_jhu_validation(
        dataset_root=tmp_path,
        validation_root=val,
        expected_samples=1,
    )

    assert count == 1
    rows = [
        json.loads(line)
        for line in (tmp_path / "inventory.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert rows[0]["split"] == "val"
    assert rows[0]["group_id"] == "0001"
    assert rows[0]["condition_tags"]["scene"] == "stadium"
    normalized = json.loads(
        (tmp_path / rows[0]["annotation_path"]).read_text(encoding="utf-8")
    )
    assert normalized["points"] == [[10.0, 11.0], [20.0, 21.0]]
    assert (val / "gt/0001.txt").read_bytes() == source_annotation


def test_prepare_jhu_validation_records_official_label_corrections(
    tmp_path: Path,
) -> None:
    val = tmp_path / "val"
    _write_jhu_sample(val, "0001", points=((100, 80),))

    prepare_jhu_validation(
        dataset_root=tmp_path,
        validation_root=val,
        expected_samples=1,
    )

    row = json.loads((tmp_path / "inventory.jsonl").read_text(encoding="utf-8"))
    normalized = json.loads(
        (tmp_path / row["annotation_path"]).read_text(encoding="utf-8")
    )
    assert row["point_count"] == 1
    assert row["normalization_corrections"] == 1
    assert row["condition_tags"]["point_localization_valid"] == "false"
    assert normalized["normalization"]["corrected_count"] == 1
    assert normalized["normalization"]["point_localization_valid"] is False
    assert normalized["points"][0][0] < 100
    assert normalized["points"][0][1] < 80


def test_prepare_jhu_validation_refuses_to_replace_inventory(tmp_path: Path) -> None:
    val = tmp_path / "val"
    _write_jhu_sample(val, "0001", points=((10, 11),))
    (tmp_path / "inventory.jsonl").write_text("existing\n", encoding="utf-8")

    with pytest.raises(FileExistsError, match="inventory"):
        prepare_jhu_validation(
            dataset_root=tmp_path,
            validation_root=val,
            expected_samples=1,
        )
