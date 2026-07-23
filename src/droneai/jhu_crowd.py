"""Read-only JHU-CROWD++ official-validation indexer."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from PIL import Image

from droneai.evaluation_contract import Point
from droneai.integrity import sha256_file


DATASET_ID = "jhu-crowd-plus-v2"
WEATHER_NAMES = {0: "none", 1: "fog_haze", 2: "rain", 3: "snow"}


@dataclass(frozen=True)
class JHUCrowdRecord:
    sample_id: str
    image_path: Path
    annotation_path: Path
    image_sha256: str
    annotation_sha256: str
    width: int
    height: int
    points: tuple[Point, ...]
    condition_tags: dict[str, str]

    @property
    def count(self) -> int:
        return len(self.points)


def _density_band(count: int) -> str:
    if count < 50:
        return "low"
    if count < 300:
        return "medium"
    return "high"


def _image_labels(path: Path) -> dict[str, tuple[int, str, str, str]]:
    labels: dict[str, tuple[int, str, str, str]] = {}
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        values = [value.strip() for value in line.split(",")]
        if len(values) != 5:
            raise ValueError(f"{path}:{line_number}: expected five CSV fields")
        sample_id, count_text, scene, weather_text, distractor_text = values
        if sample_id in labels:
            raise ValueError(f"duplicate JHU image label: {sample_id}")
        try:
            count = int(count_text)
            weather = WEATHER_NAMES[int(weather_text)]
            distractor = str(bool(int(distractor_text))).lower()
        except (ValueError, KeyError) as exc:
            raise ValueError(f"{path}:{line_number}: invalid image label") from exc
        labels[sample_id] = (count, scene, weather, distractor)
    if not labels:
        raise ValueError("JHU image_labels.txt is empty")
    return labels


def _head_points(path: Path, *, width: int, height: int) -> tuple[Point, ...]:
    points: list[Point] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        values = line.split()
        if len(values) != 6:
            raise ValueError(f"{path}:{line_number}: expected six integers")
        try:
            center_x, center_y, box_width, box_height, occlusion, blur = map(
                int, values
            )
        except ValueError as exc:
            raise ValueError(f"{path}:{line_number}: invalid integer row") from exc
        if box_width < 0 or box_height < 0 or occlusion not in {1, 2, 3} or blur not in {0, 1}:
            raise ValueError(f"{path}:{line_number}: invalid JHU head metadata")
        if not 0 <= center_x < width or not 0 <= center_y < height:
            raise ValueError(f"{path}:{line_number}: head center is out of bounds")
        points.append((float(center_x), float(center_y)))
    return tuple(points)


def index_jhu_validation(
    validation_root: str | Path,
    *,
    expected_samples: int = 500,
) -> tuple[JHUCrowdRecord, ...]:
    root = Path(validation_root)
    image_root = root / "images"
    gt_root = root / "gt"
    label_path = root / "image_labels.txt"
    if not image_root.is_dir() or not gt_root.is_dir() or not label_path.is_file():
        raise FileNotFoundError(
            "JHU validation requires images/, gt/, and image_labels.txt"
        )
    labels = _image_labels(label_path)
    images = sorted(image_root.glob("*.jpg"))
    if len(images) != expected_samples:
        raise ValueError(
            f"JHU validation expected {expected_samples} samples, observed {len(images)}"
        )
    records: list[JHUCrowdRecord] = []
    for image_path in images:
        sample_id = image_path.stem
        annotation_path = gt_root / f"{sample_id}.txt"
        if not annotation_path.is_file() or sample_id not in labels:
            raise FileNotFoundError(f"JHU annotation or image label missing: {sample_id}")
        with Image.open(image_path) as image:
            width, height = image.size
        points = _head_points(annotation_path, width=width, height=height)
        declared_count, scene, weather, distractor = labels[sample_id]
        if declared_count != len(points):
            raise ValueError(
                f"JHU declared count mismatch for {sample_id}: "
                f"declared count={declared_count}, rows={len(points)}"
            )
        records.append(
            JHUCrowdRecord(
                sample_id=sample_id,
                image_path=image_path.resolve(),
                annotation_path=annotation_path.resolve(),
                image_sha256=sha256_file(image_path),
                annotation_sha256=sha256_file(annotation_path),
                width=width,
                height=height,
                points=points,
                condition_tags={
                    "scene": scene,
                    "weather": weather,
                    "distractor": distractor,
                    "density_band": _density_band(len(points)),
                },
            )
        )
    if set(labels) != {record.sample_id for record in records}:
        raise ValueError("JHU image_labels.txt differs from validation images")
    return tuple(records)


def prepare_jhu_validation(
    *,
    dataset_root: str | Path,
    validation_root: str | Path,
    expected_samples: int = 500,
) -> int:
    """Write Stage 1 inventory/JSON labels without changing official JHU files."""

    root = Path(dataset_root).resolve()
    validation = Path(validation_root).resolve()
    try:
        validation.relative_to(root)
    except ValueError as exc:
        raise ValueError("JHU validation root must stay inside dataset root") from exc
    inventory_path = root / "inventory.jsonl"
    normalized_root = root / "normalized_annotations"
    staging_root = root / "normalized_annotations.tmp"
    inventory_staging = root / "inventory.jsonl.tmp"
    if inventory_path.exists():
        raise FileExistsError(f"JHU inventory already exists: {inventory_path}")
    if normalized_root.exists() or staging_root.exists() or inventory_staging.exists():
        raise FileExistsError("JHU normalized output or staging path already exists")

    records = index_jhu_validation(
        validation,
        expected_samples=expected_samples,
    )
    staging_root.mkdir(parents=True)
    rows: list[dict[str, object]] = []
    for record in records:
        normalized_path = staging_root / "val" / f"{record.sample_id}.json"
        normalized_path.parent.mkdir(parents=True, exist_ok=True)
        normalized_path.write_text(
            json.dumps(
                {
                    "points": [[x, y] for x, y in record.points],
                    "normalization": {
                        "policy": "official_head_centers_preserved",
                        "corrected_count": 0,
                        "source_annotation": record.annotation_path.relative_to(
                            root
                        ).as_posix(),
                        "source_annotation_sha256": record.annotation_sha256,
                    },
                },
                ensure_ascii=False,
                separators=(",", ":"),
            ),
            encoding="utf-8",
        )
        final_annotation = normalized_root / "val" / normalized_path.name
        rows.append(
            {
                "sample_id": record.sample_id,
                "split": "val",
                "group_id": record.sample_id,
                "image_path": record.image_path.relative_to(root).as_posix(),
                "annotation_path": final_annotation.relative_to(root).as_posix(),
                "image_sha256": record.image_sha256,
                "annotation_sha256": sha256_file(normalized_path),
                "width": record.width,
                "height": record.height,
                "point_count": record.count,
                "normalization_corrections": 0,
                "condition_tags": dict(sorted(record.condition_tags.items())),
            }
        )
    inventory_staging.write_text(
        "\n".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) for row in rows
        )
        + "\n",
        encoding="utf-8",
    )
    staging_root.replace(normalized_root)
    inventory_staging.replace(inventory_path)
    return len(rows)
