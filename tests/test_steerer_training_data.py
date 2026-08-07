from __future__ import annotations

import json
import math
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from PIL import Image
from scipy.io import savemat

from droneai.integrity import sha256_file
from droneai.steerer_training_data import (
    convert_training_sample,
    prepare_training_dataset,
)
from droneai.steerer_training_split import build_training_split
from droneai.ucf_qnrf import index_ucf_qnrf_train


def _record(tmp_path: Path, *, sample_id: str = "img_0001"):
    train = tmp_path / "Train"
    train.mkdir(exist_ok=True)
    image_path = train / f"{sample_id}.jpg"
    Image.new("RGB", (40, 24), color=(12, 34, 56)).save(image_path, quality=91)
    savemat(
        train / f"{sample_id}_ann.mat",
        {"annPoints": np.asarray([[0.0, 0.0], [39.0, 23.0], [20.0, 12.0]])},
    )
    return next(
        record
        for record in index_ucf_qnrf_train(train)
        if record.sample_id == sample_id
    )


def test_convert_sample_preserves_count_and_scales_geometry(tmp_path: Path) -> None:
    """Dropping points or deriving boxes outside the source image is a conversion bug."""

    record = _record(tmp_path)

    result = convert_training_sample(record, output_root=tmp_path / "output")

    payload = json.loads(result.json_path.read_text(encoding="utf-8"))
    assert payload["human_num"] == record.count == len(payload["points"])
    assert payload["source_sha256"] == record.image_sha256
    assert payload["scale"] == [1.0, 1.0]
    assert (result.width, result.height) == (record.width, record.height)
    assert all(0 <= x < result.width and 0 <= y < result.height for x, y in payload["points"])
    assert all(
        0 <= x1 <= x2 < result.width and 0 <= y1 <= y2 < result.height
        for x1, y1, x2, y2 in payload["boxes"]
    )


def test_convert_sample_copies_the_exact_source_image_bytes(tmp_path: Path) -> None:
    """Re-encoding a JPEG changes its evidence hash and is therefore a bug."""

    record = _record(tmp_path)

    result = convert_training_sample(record, output_root=tmp_path / "output")

    assert result.image_path.read_bytes() == record.image_path.read_bytes()
    assert sha256_file(result.image_path) == record.image_sha256


def test_convert_sample_keeps_indexed_right_bottom_edges_strictly_in_bounds(
    tmp_path: Path,
) -> None:
    """Rounding a normalized edge back to width or height would break the loader contract."""

    indexed = _record(tmp_path)
    record = replace(
        indexed,
        points=(
            (
                math.nextafter(float(indexed.width), 0.0),
                math.nextafter(float(indexed.height), 0.0),
            ),
        ),
    )

    result = convert_training_sample(record, output_root=tmp_path / "output")

    point = json.loads(result.json_path.read_text(encoding="utf-8"))["points"][0]
    assert 0 <= point[0] < result.width
    assert 0 <= point[1] < result.height


def test_preparation_refuses_non_empty_output(tmp_path: Path) -> None:
    """Writing into an owned output can mix incompatible dataset versions."""

    record = _record(tmp_path)
    split = build_training_split((record,), seed=3035, validation_count=0)
    output = tmp_path / "output"
    output.mkdir()
    (output / "existing.txt").write_text("owned", encoding="utf-8")

    with pytest.raises(FileExistsError, match="new or empty"):
        prepare_training_dataset((record,), split, output_root=output)


def test_preparation_writes_exact_images_lists_and_integrity_manifest(tmp_path: Path) -> None:
    """A dataset missing a converted row or its provenance inventory is unusable."""

    first = _record(tmp_path, sample_id="img_0001")
    second = _record(tmp_path, sample_id="img_0002")
    records = (first, second)
    split = build_training_split(records, seed=3035, validation_count=1)

    manifest_path = prepare_training_dataset(records, split, output_root=tmp_path / "output")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    output = manifest_path.parents[1]
    assert manifest["source_partition"] == "official_train_only"
    assert manifest["converted_sample_count"] == 2
    assert (manifest["train_count"], manifest["validation_count"]) == (1, 1)
    assert manifest["count_mismatches"] == 0
    assert manifest["out_of_bounds_points"] == 0
    assert manifest["out_of_bounds_boxes"] == 0
    assert len(manifest["input_inventory_sha256"]) == 64
    assert len(manifest["output_inventory_sha256"]) == 64
    assert (output / "train.txt").read_text(encoding="utf-8").splitlines() == list(split.train_ids)
    assert (output / "val.txt").read_text(encoding="utf-8").splitlines() == list(split.validation_ids)
    for record in records:
        image = output / "images" / f"{record.sample_id}.jpg"
        annotation = output / "jsons" / f"{record.sample_id}.json"
        assert sha256_file(image) == record.image_sha256
        assert json.loads(annotation.read_text(encoding="utf-8"))["human_num"] == record.count
