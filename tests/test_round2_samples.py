from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

from droneai.round2_samples import (
    evaluation_samples_from_up_count,
    load_round2_samples,
)


def test_up_count_rows_convert_to_existing_evaluation_contract(
    tmp_path: Path,
) -> None:
    image = tmp_path / "images/0001/frame.jpg"
    annotation = tmp_path / "normalized_annotations/0001/frame.json"
    image.parent.mkdir(parents=True)
    annotation.parent.mkdir(parents=True)
    Image.new("RGB", (40, 30), color=(1, 2, 3)).save(image)
    annotation.write_text(
        json.dumps({"points": [[3, 4], [10, 12]]}), encoding="utf-8"
    )
    from droneai.integrity import sha256_file

    rows = (
        {
            "sample_id": "0001/frame",
            "split": "val",
            "group_id": "0001",
            "image_path": "images/0001/frame.jpg",
            "annotation_path": "normalized_annotations/0001/frame.json",
            "image_sha256": sha256_file(image),
            "annotation_sha256": sha256_file(annotation),
            "width": 40,
            "height": 30,
            "point_count": 2,
            "condition_tags": {
                "altitude_band": "mid",
                "density_band": "low",
            },
        },
    )

    samples = evaluation_samples_from_up_count(
        rows,
        dataset_root=tmp_path,
        split_id="up-count-val-test-166-v1",
    )

    assert samples[0].sample_id == "0001/frame"
    assert samples[0].dataset_id == "up-count-v1"
    assert samples[0].ground_truth_count == 2.0
    assert samples[0].condition_tags["official_partition"] == "val"
    assert samples[0].ground_truth_density is not None
    assert float(samples[0].ground_truth_density.sum()) == 2.0


def test_round2_loader_dispatches_up_count_inventory(tmp_path: Path) -> None:
    image = tmp_path / "images/0001/0001__0000__60.0.jpg"
    annotation = (
        tmp_path
        / "normalized_annotations/0001/0001__0000__60.0.json"
    )
    image.parent.mkdir(parents=True)
    annotation.parent.mkdir(parents=True)
    Image.new("RGB", (40, 30), color=(1, 2, 3)).save(image)
    annotation.write_text(json.dumps({"points": [[3, 4]]}), encoding="utf-8")
    from droneai.integrity import sha256_file

    row = {
        "sample_id": "0001/0001__0000__60.0",
        "split": "val",
        "group_id": "0001",
        "image_path": image.relative_to(tmp_path).as_posix(),
        "annotation_path": annotation.relative_to(tmp_path).as_posix(),
        "image_sha256": sha256_file(image),
        "annotation_sha256": sha256_file(annotation),
        "width": 40,
        "height": 30,
        "point_count": 1,
        "condition_tags": {
            "altitude_band": "mid",
            "density_band": "low",
        },
    }
    (tmp_path / "inventory.jsonl").write_text(
        json.dumps(row) + "\n", encoding="utf-8"
    )

    samples = load_round2_samples(
        dataset_id="up-count-v1",
        dataset_root=tmp_path,
        partition="val_test",
        expected_samples=1,
        split_id="fixture-up-count-v1",
    )

    assert len(samples) == 1
    assert samples[0].sample_id == row["sample_id"]
