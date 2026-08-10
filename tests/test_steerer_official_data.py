from __future__ import annotations

import json
from pathlib import Path

from PIL import Image
from scipy.io import savemat

from droneai.steerer_official_data import (
    convert_official_test_sample,
    load_prepared_official_test,
    official_qnrf_prepared_size,
    prepare_official_test_partition,
)


def _raw_sample(root: Path, *, points: list[list[float]]) -> None:
    root.mkdir(parents=True)
    Image.new("RGB", (640, 480), color=(20, 40, 60)).save(
        root / "img_0001.jpg", quality=90
    )
    savemat(root / "img_0001_ann.mat", {"annPoints": points})


def test_official_qnrf_prepared_size_matches_pinned_reference() -> None:
    assert official_qnrf_prepared_size(2500, 1875) == (2512, 1888)
    assert official_qnrf_prepared_size(1024, 768) == (1040, 784)
    assert official_qnrf_prepared_size(640, 480) == (1024, 784)


def test_conversion_preserves_count_and_clips_edge_points(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    _raw_sample(raw, points=[[10.0, 20.0], [641.0, 482.0]])
    output = tmp_path / "converted"

    converted = convert_official_test_sample(
        raw / "img_0001.jpg",
        raw / "img_0001_ann.mat",
        prepared_id="1202",
        source_sample_id="test_img_0001",
        output_root=output,
    )

    assert converted.count == 2
    assert converted.width == 1024
    assert converted.height == 784
    assert converted.boundary_corrections == 1
    annotation = json.loads(converted.annotation_path.read_text(encoding="utf-8"))
    assert len(annotation["points"]) == 2
    assert annotation["points"][1] == [1023.0, 783.0]
    with Image.open(converted.image_path) as image:
        assert image.size == (1024, 784)


def test_partition_manifest_is_hash_verified_and_loadable(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    _raw_sample(raw, points=[[10.0, 20.0], [30.0, 40.0]])
    output = tmp_path / "prepared"

    manifest = prepare_official_test_partition(
        raw,
        output_root=output,
        expected_samples=1,
    )
    records = load_prepared_official_test(
        output,
        manifest_path=manifest,
        expected_samples=1,
    )

    assert len(records) == 1
    assert records[0].sample_id == "test_img_0001"
    assert records[0].count == 2
    assert records[0].image_sha256
    assert records[0].annotation_sha256
