"""Normalize the official UP-COUNT point labels and build inventory.jsonl."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image

from droneai.stage1 import sha256_file

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def _read_split_ids(split_dir: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for split in ("train", "val", "test"):
        for value in (split_dir / f"{split}.txt").read_text(encoding="utf-8").splitlines():
            sequence_id = value.strip()
            if sequence_id:
                if sequence_id in result:
                    raise ValueError(f"sequence {sequence_id} occurs in multiple split files")
                result[sequence_id] = split
    return result


def _parse_points(label_path: Path) -> list[list[float]]:
    points: list[list[float]] = []
    for line_number, line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        values = line.split()
        if len(values) != 2:
            raise ValueError(f"{label_path}:{line_number}: expected x y")
        points.append([float(values[0]), float(values[1])])
    return points


def _altitude_from_stem(stem: str) -> float:
    try:
        return float(stem.rsplit("__", 1)[1])
    except (IndexError, ValueError) as exc:
        raise ValueError(f"cannot parse altitude from {stem}") from exc


def _altitude_band(value: float) -> str:
    if value < 40:
        return "low"
    if value < 80:
        return "mid"
    return "high"


def _density_band(count: int) -> str:
    if count < 50:
        return "low"
    if count < 300:
        return "medium"
    return "high"


def prepare_up_count(
    *,
    dataset_root: Path,
    image_root: Path,
    label_root: Path,
    split_dir: Path,
    allow_image_subset: bool = False,
) -> int:
    split_ids = _read_split_ids(split_dir)
    images_by_stem: dict[str, Path] = {}
    for path in image_root.rglob("*"):
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
            if path.stem in images_by_stem:
                raise ValueError(f"duplicate image stem: {path.stem}")
            images_by_stem[path.stem] = path

    normalized_root = dataset_root / "normalized_annotations"
    rows: list[dict] = []
    matched_image_stems: set[str] = set()
    for label_path in sorted(label_root.rglob("*.txt")):
        sequence_id = label_path.parent.name
        split = split_ids.get(sequence_id)
        if not split:
            raise ValueError(f"sequence {sequence_id} has no official split")
        image_path = images_by_stem.get(label_path.stem)
        if image_path is None:
            if allow_image_subset:
                continue
            raise FileNotFoundError(f"image missing for {label_path.stem}")
        matched_image_stems.add(label_path.stem)

        points = _parse_points(label_path)
        altitude = _altitude_from_stem(label_path.stem)
        with Image.open(image_path) as image:
            width, height = image.size

        normalized_path = normalized_root / sequence_id / f"{label_path.stem}.json"
        normalized_path.parent.mkdir(parents=True, exist_ok=True)
        normalized_path.write_text(
            json.dumps({"points": points}, ensure_ascii=False, separators=(",", ":")),
            encoding="utf-8",
        )
        rows.append(
            {
                "sample_id": f"{sequence_id}/{label_path.stem}",
                "split": split,
                "group_id": sequence_id,
                "image_path": image_path.relative_to(dataset_root).as_posix(),
                "annotation_path": normalized_path.relative_to(dataset_root).as_posix(),
                "image_sha256": sha256_file(image_path),
                "annotation_sha256": sha256_file(normalized_path),
                "width": width,
                "height": height,
                "point_count": len(points),
                "condition_tags": {
                    "altitude_band": _altitude_band(altitude),
                    "density_band": _density_band(len(points)),
                },
            }
        )

    if not rows:
        raise ValueError("no UP-COUNT labels found")
    unmatched_images = set(images_by_stem) - matched_image_stems
    if unmatched_images:
        raise FileNotFoundError(f"labels missing for images: {sorted(unmatched_images)[:5]}")
    inventory_path = dataset_root / "inventory.jsonl"
    inventory_path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False, sort_keys=True) for row in rows) + "\n",
        encoding="utf-8",
    )
    return len(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare UP-COUNT for the Stage 1 gate")
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--label-root", type=Path, required=True)
    parser.add_argument("--split-dir", type=Path, required=True)
    parser.add_argument(
        "--allow-image-subset",
        action="store_true",
        help="Normalize only labels with a matching image (for the official 202-sequence sample pack).",
    )
    args = parser.parse_args()
    count = prepare_up_count(
        dataset_root=args.dataset_root,
        image_root=args.image_root,
        label_root=args.label_root,
        split_dir=args.split_dir,
        allow_image_subset=args.allow_image_subset,
    )
    print(f"Prepared {count} UP-COUNT samples")


if __name__ == "__main__":
    main()
