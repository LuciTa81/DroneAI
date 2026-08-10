from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image
from scipy.io import savemat

from droneai.ucf_qnrf import (
    apply_official_train_validation_split,
    index_ucf_qnrf_test,
    index_ucf_qnrf_train,
    prepare_evaluation_sample,
    read_ucf_qnrf_points,
    select_stratified_smoke,
)


def _write_sample(root: Path, index: int, count: int) -> None:
    width, height = 80, 64
    image_path = root / f"img_{index:04d}.jpg"
    Image.new("RGB", (width, height), color=(index % 255, 20, 30)).save(image_path)
    points = np.asarray(
        [
            [float((point_index * 7) % width), float((point_index * 11) % height)]
            for point_index in range(count)
        ],
        dtype=np.float32,
    ).reshape((-1, 2))
    savemat(root / f"img_{index:04d}_ann.mat", {"annPoints": points})


def test_read_ucf_qnrf_points_reads_annpoints(tmp_path: Path) -> None:
    annotation = tmp_path / "img_0001_ann.mat"
    savemat(annotation, {"annPoints": np.asarray([[1.5, 2.5], [3.0, 4.0]])})

    assert read_ucf_qnrf_points(annotation) == ((1.5, 2.5), (3.0, 4.0))


def test_index_uses_only_explicit_train_root_and_hashes_both_assets(tmp_path: Path) -> None:
    train = tmp_path / "Train"
    test = tmp_path / "Test"
    train.mkdir()
    test.mkdir()
    _write_sample(train, 1, 3)
    (test / "must-not-be-read.jpg").write_bytes(b"not an image")

    records = index_ucf_qnrf_train(train)

    assert len(records) == 1
    assert records[0].sample_id == "img_0001"
    assert records[0].count == 3
    assert len(records[0].image_sha256) == 64
    assert len(records[0].annotation_sha256) == 64
    assert all("Test" not in str(path) for path in (records[0].image_path, records[0].annotation_path))


def test_test_index_prefixes_partition_identity_and_requires_annotation_pairs(
    tmp_path: Path,
) -> None:
    test = tmp_path / "Test"
    test.mkdir()
    _write_sample(test, 1, 3)
    _write_sample(test, 2, 4)

    records = index_ucf_qnrf_test(test)

    assert [row.sample_id for row in records] == ["test_img_0001", "test_img_0002"]
    assert [row.count for row in records] == [3, 4]
    assert all(row.image_path.parent == test.resolve() for row in records)

    (test / "img_0002_ann.mat").unlink()
    with pytest.raises(FileNotFoundError, match="annotation missing"):
        index_ucf_qnrf_test(test)


def test_index_matches_official_out_of_bounds_filter_and_keeps_edge_points(tmp_path: Path) -> None:
    train = tmp_path / "Train"
    train.mkdir()
    image = train / "img_0001.jpg"
    Image.new("RGB", (10, 10), color=(0, 0, 0)).save(image)
    savemat(
        train / "img_0001_ann.mat",
        {
            "annPoints": np.asarray(
                [[-1.0, 5.0], [10.0, 10.0], [11.0, 5.0]], dtype=np.float32
            )
        },
    )

    record = index_ucf_qnrf_train(train)[0]

    assert record.count == 1
    assert record.normalization_corrections == 3
    assert 0 <= record.points[0][0] < 10
    assert 0 <= record.points[0][1] < 10


def test_official_train_validation_split_is_exact_disjoint_and_complete(tmp_path: Path) -> None:
    train = tmp_path / "Train"
    train.mkdir()
    for index in range(1, 7):
        _write_sample(train, index, index)
    records = index_ucf_qnrf_train(train)
    train_list = tmp_path / "qnrf_train.txt"
    validation_list = tmp_path / "qnrf_val.txt"
    train_list.write_text("img_0001.jpg\nimg_0003.jpg img_0005.jpg\n", encoding="utf-8")
    validation_list.write_text("img_0002.jpg\nimg_0004.jpg\nimg_0006.jpg\n", encoding="utf-8")

    official_train, official_validation = apply_official_train_validation_split(
        tuple(reversed(records)),
        train_list_path=train_list,
        validation_list_path=validation_list,
    )

    assert [row.sample_id for row in official_train] == ["img_0001", "img_0003", "img_0005"]
    assert [row.sample_id for row in official_validation] == [
        "img_0002",
        "img_0004",
        "img_0006",
    ]


def test_official_split_rejects_overlap_or_missing_samples(tmp_path: Path) -> None:
    train = tmp_path / "Train"
    train.mkdir()
    for index in range(1, 4):
        _write_sample(train, index, index)
    records = index_ucf_qnrf_train(train)
    train_list = tmp_path / "qnrf_train.txt"
    validation_list = tmp_path / "qnrf_val.txt"
    train_list.write_text("img_0001.jpg img_0002.jpg", encoding="utf-8")
    validation_list.write_text("img_0002.jpg", encoding="utf-8")

    try:
        apply_official_train_validation_split(
            records,
            train_list_path=train_list,
            validation_list_path=validation_list,
        )
    except ValueError as error:
        assert "overlap" in str(error) or "complete" in str(error)
    else:
        raise AssertionError("overlapping or incomplete official split must fail")


def test_smoke_selection_is_deterministic_and_density_stratified(tmp_path: Path) -> None:
    train = tmp_path / "Train"
    train.mkdir()
    counts = [100 + index for index in range(12)]
    counts += [700 + index for index in range(12)]
    counts += [1700 + index for index in range(12)]
    for index, count in enumerate(counts, start=1):
        _write_sample(train, index, count)
    records = index_ucf_qnrf_train(train)

    first = select_stratified_smoke(records, sample_count=36, seed=20260715)
    second = select_stratified_smoke(tuple(reversed(records)), sample_count=36, seed=20260715)

    assert [row.sample_id for row in first] == [row.sample_id for row in second]
    assert {row.density_band for row in first} == {"low", "medium", "high"}
    assert {band: sum(row.density_band == band for row in first) for band in ("low", "medium", "high")} == {
        "low": 12,
        "medium": 12,
        "high": 12,
    }


def test_prepare_evaluation_sample_preserves_count_and_quadrant_geometry(tmp_path: Path) -> None:
    train = tmp_path / "Train"
    train.mkdir()
    _write_sample(train, 1, 5)
    record = index_ucf_qnrf_train(train)[0]

    sample = prepare_evaluation_sample(record, split_id="ucf-qnrf-train-val-v1")

    assert sample.ground_truth_count == 5
    assert sample.annotation_sha256 == record.annotation_sha256
    assert sample.has_point_annotations is True
    assert len(sample.ground_truth_points) == 5
    assert sample.ground_truth_density is not None
    assert float(sample.ground_truth_density.sum()) == 5.0
    assert sample.ground_truth_density.shape == (8, 10)
    assert sample.condition_tags["density_band"] == "low"
    assert sample.condition_tags["source_partition"] == "official_validation"
    assert [zone.zone_id for zone in sample.zones] == ["q00", "q01", "q10", "q11"]
