import json
from pathlib import Path

from PIL import Image

from droneai.up_count import prepare_up_count


def test_prepare_up_count_builds_normalized_inventory(tmp_path: Path) -> None:
    image_root = tmp_path / "images"
    label_root = tmp_path / "labels"
    split_dir = tmp_path / "splits"
    for sequence, split in (("0000", "train"), ("0001", "val"), ("0002", "test")):
        (image_root / sequence).mkdir(parents=True)
        (label_root / sequence).mkdir(parents=True)
        split_dir.mkdir(parents=True, exist_ok=True)
        stem = f"{sequence}__0000__60.0"
        Image.new("RGB", (12, 10)).save(image_root / sequence / f"{stem}.jpg")
        (label_root / sequence / f"{stem}.txt").write_text(
            "1 2\n3.5 4.5\n12 10\n3 -4\n", encoding="utf-8"
        )
        (split_dir / f"{split}.txt").write_text(f"{sequence}\n", encoding="utf-8")

    count = prepare_up_count(
        dataset_root=tmp_path,
        image_root=image_root,
        label_root=label_root,
        split_dir=split_dir,
    )
    rows = [json.loads(line) for line in (tmp_path / "inventory.jsonl").read_text().splitlines()]
    assert count == 3
    assert {row["split"] for row in rows} == {"train", "val", "test"}
    assert rows[0]["point_count"] == 4
    assert rows[0]["normalization_corrections"] == 3
    normalized = json.loads((tmp_path / rows[0]["annotation_path"]).read_text())
    assert normalized["points"] == [[1, 2], [3, 4], [11, 9], [3, 0]]
    assert rows[0]["condition_tags"] == {"altitude_band": "mid", "density_band": "low"}


def test_prepare_up_count_can_use_official_image_subset(tmp_path: Path) -> None:
    image_root = tmp_path / "images"
    label_root = tmp_path / "labels" / "0000"
    split_dir = tmp_path / "splits"
    image_root.mkdir()
    label_root.mkdir(parents=True)
    split_dir.mkdir()
    (split_dir / "train.txt").write_text("0000\n", encoding="utf-8")
    (split_dir / "val.txt").write_text("0001\n", encoding="utf-8")
    (split_dir / "test.txt").write_text("0002\n", encoding="utf-8")
    for frame in ("0000", "0001"):
        stem = f"0000__{frame}__30.0"
        (label_root / f"{stem}.txt").write_text("1 2\n", encoding="utf-8")
    Image.new("RGB", (12, 10)).save(image_root / "0000__0000__30.0.jpg")

    count = prepare_up_count(
        dataset_root=tmp_path,
        image_root=image_root,
        label_root=tmp_path / "labels",
        split_dir=split_dir,
        allow_image_subset=True,
    )
    assert count == 1
