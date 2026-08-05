from __future__ import annotations

import json
import zipfile
from pathlib import Path

from droneai.up_count_remote import (
    build_up_count_catalog,
    extract_selected_up_count,
    select_up_count_members,
)


def _write_fixture_archives(tmp_path: Path) -> tuple[Path, Path, Path]:
    labels_zip = tmp_path / "labels.zip"
    images_zip = tmp_path / "images.zip"
    splits = tmp_path / "splits"
    splits.mkdir()
    (splits / "train.txt").write_text("0000\n", encoding="utf-8")
    (splits / "val.txt").write_text("0001\n", encoding="utf-8")
    (splits / "test.txt").write_text("0002\n", encoding="utf-8")
    with zipfile.ZipFile(labels_zip, "w") as labels, zipfile.ZipFile(
        images_zip, "w"
    ) as images:
        for sequence in ("0001", "0002"):
            for frame in (0, 30, 60):
                stem = f"{sequence}__{frame:04d}__60.0"
                labels.writestr(
                    f"UP-COUNT/labels/{sequence}/{stem}.txt", "1 2\n3 4\n"
                )
                images.writestr(
                    f"UP-COUNT/images/{sequence}/{stem}.jpg",
                    f"image-{sequence}-{frame}".encode(),
                )
    return labels_zip, images_zip, splits


def test_catalog_and_selection_match_official_member_names(tmp_path: Path) -> None:
    labels_zip, images_zip, splits = _write_fixture_archives(tmp_path)

    catalog = build_up_count_catalog(labels_zip=labels_zip, split_dir=splits)
    with zipfile.ZipFile(images_zip) as images:
        selected = select_up_count_members(
            catalog,
            image_infos=images.infolist(),
            sample_count=4,
            namespace="round2-upcount-v1",
            minimum_frame_gap=30,
        )

    assert len(catalog) == 6
    assert len(selected) == 4
    assert {row["split"] for row in selected} == {"val", "test"}
    assert all(str(row["image_member"]).startswith("UP-COUNT/images/") for row in selected)


def test_selected_extraction_is_verified_and_resumable(tmp_path: Path) -> None:
    labels_zip, images_zip, splits = _write_fixture_archives(tmp_path)
    catalog = build_up_count_catalog(labels_zip=labels_zip, split_dir=splits)
    output = tmp_path / "selected.partial"
    with zipfile.ZipFile(images_zip) as images:
        selected = select_up_count_members(
            catalog,
            image_infos=images.infolist(),
            sample_count=4,
            namespace="round2-upcount-v1",
            minimum_frame_gap=30,
        )
        first = extract_selected_up_count(
            image_zip=images,
            labels_zip=labels_zip,
            split_dir=splits,
            selected=selected,
            dataset_root=output,
            archive_identity={
                "url": "https://example.invalid/images.zip",
                "size": 123,
                "md5": "0" * 32,
            },
            selection_identity={
                "namespace": "round2-upcount-v1",
                "minimum_frame_gap": 30,
            },
        )
        second = extract_selected_up_count(
            image_zip=images,
            labels_zip=labels_zip,
            split_dir=splits,
            selected=selected,
            dataset_root=output,
            archive_identity={
                "url": "https://example.invalid/images.zip",
                "size": 123,
                "md5": "0" * 32,
            },
            selection_identity={
                "namespace": "round2-upcount-v1",
                "minimum_frame_gap": 30,
            },
        )

    assert first == second
    manifest = json.loads((output / "acquisition/selected-members.json").read_text())
    assert manifest["selected_samples"] == 4
    assert len(manifest["members"]) == 4
    assert len(list((output / "images").rglob("*.jpg"))) == 4
    assert len(list((output / "labels").rglob("*.txt"))) == 4
    assert all(len(row["sha256"]) == 64 for row in manifest["members"])
