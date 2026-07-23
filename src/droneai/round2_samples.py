"""Convert normalized Round 2 dataset records to the common evaluation contract."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from droneai.evaluation_contract import EvaluationSample, Point, ZoneBox
from droneai.integrity import sha256_file
from droneai.jhu_crowd import DATASET_ID as JHU_DATASET_ID
from droneai.jhu_crowd import JHUCrowdRecord, index_jhu_validation
from droneai.ucf_qnrf import DATASET_ID as UCF_DATASET_ID
from droneai.ucf_qnrf import (
    UCFQNRFRecord,
    index_ucf_qnrf_partition,
    prepare_evaluation_sample,
)
from droneai.up_count import select_up_count_reference


UP_COUNT_DATASET_ID = "up-count-v1"


def _point_density(
    points: Sequence[Point], *, width: int, height: int, downsample: int = 8
) -> np.ndarray:
    density = np.zeros(
        (math.ceil(height / downsample), math.ceil(width / downsample)),
        dtype=np.float32,
    )
    for x, y in points:
        density[min(int(y / downsample), density.shape[0] - 1), min(int(x / downsample), density.shape[1] - 1)] += 1.0
    return density


def _quadrant_zones(width: int, height: int) -> tuple[ZoneBox, ...]:
    x_mid, y_mid = width / 2.0, height / 2.0
    return (
        ZoneBox("q00", 0.0, 0.0, x_mid, y_mid),
        ZoneBox("q01", x_mid, 0.0, float(width), y_mid),
        ZoneBox("q10", 0.0, y_mid, x_mid, float(height)),
        ZoneBox("q11", x_mid, y_mid, float(width), float(height)),
    )


def evaluation_samples_from_ucf(
    records: Iterable[UCFQNRFRecord], *, split_id: str
) -> tuple[EvaluationSample, ...]:
    return tuple(
        prepare_evaluation_sample(
            record,
            split_id=split_id,
            dataset_id=UCF_DATASET_ID,
            source_partition="official_test",
        )
        for record in records
    )


def evaluation_samples_from_jhu(
    records: Iterable[JHUCrowdRecord], *, split_id: str
) -> tuple[EvaluationSample, ...]:
    samples: list[EvaluationSample] = []
    for record in records:
        point_localization_valid = record.point_localization_valid
        samples.append(
            EvaluationSample(
                sample_id=record.sample_id,
                dataset_id=JHU_DATASET_ID,
                split_id=split_id,
                image_path=record.image_path,
                source_sha256=record.image_sha256,
                width=record.width,
                height=record.height,
                ground_truth_count=float(record.count),
                annotation_sha256=record.annotation_sha256,
                ground_truth_points=(record.points if point_localization_valid else ()),
                ground_truth_density=(
                    _point_density(
                        record.points, width=record.width, height=record.height
                    )
                    if point_localization_valid
                    else None
                ),
                condition_tags={
                    **record.condition_tags,
                    "source_partition": "official_validation",
                    "point_localization_valid": str(
                        point_localization_valid
                    ).lower(),
                },
                zones=_quadrant_zones(record.width, record.height),
                has_point_annotations=point_localization_valid,
            )
        )
    return tuple(samples)


def evaluation_samples_from_up_count(
    rows: Iterable[dict[str, object]],
    *,
    dataset_root: str | Path,
    split_id: str,
) -> tuple[EvaluationSample, ...]:
    root = Path(dataset_root)
    samples: list[EvaluationSample] = []
    for row in rows:
        image_path = root / str(row["image_path"])
        annotation_path = root / str(row["annotation_path"])
        if sha256_file(image_path) != row["image_sha256"]:
            raise ValueError(f"UP-COUNT image SHA-256 mismatch: {row['sample_id']}")
        if sha256_file(annotation_path) != row["annotation_sha256"]:
            raise ValueError(
                f"UP-COUNT annotation SHA-256 mismatch: {row['sample_id']}"
            )
        payload = json.loads(annotation_path.read_text(encoding="utf-8"))
        raw_points = payload.get("points")
        if not isinstance(raw_points, list):
            raise ValueError("UP-COUNT normalized annotation requires points")
        points = tuple((float(point[0]), float(point[1])) for point in raw_points)
        width, height = int(row["width"]), int(row["height"])
        if len(points) != int(row["point_count"]):
            raise ValueError("UP-COUNT normalized point count mismatch")
        tags = {
            str(key): str(value)
            for key, value in dict(row.get("condition_tags") or {}).items()
        }
        tags.update(
            {
                "official_partition": str(row["split"]),
                "source_partition": "official_validation_or_test",
            }
        )
        samples.append(
            EvaluationSample(
                sample_id=str(row["sample_id"]),
                dataset_id=UP_COUNT_DATASET_ID,
                split_id=split_id,
                image_path=image_path.resolve(),
                source_sha256=str(row["image_sha256"]),
                width=width,
                height=height,
                ground_truth_count=float(len(points)),
                annotation_sha256=str(row["annotation_sha256"]),
                ground_truth_points=points,
                ground_truth_density=_point_density(
                    points, width=width, height=height
                ),
                condition_tags=tags,
                zones=_quadrant_zones(width, height),
                has_point_annotations=True,
            )
        )
    return tuple(samples)


def _read_inventory(path: Path) -> tuple[dict[str, object], ...]:
    rows: list[dict[str, object]] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"inventory line {line_number} must be an object")
        rows.append(value)
    return tuple(rows)


def load_round2_samples(
    *,
    dataset_id: str,
    dataset_root: str | Path,
    partition: str,
    expected_samples: int,
    split_id: str,
) -> tuple[EvaluationSample, ...]:
    root = Path(dataset_root)
    if dataset_id == UCF_DATASET_ID:
        if partition != "test":
            raise ValueError("Round 2 UCF-QNRF lane requires partition=test")
        canonical = root / "raw/UCF-QNRF_ECCV18/Test"
        image_root = canonical if canonical.is_dir() else root
        records = index_ucf_qnrf_partition(
            image_root,
            partition="test",
            expected_samples=expected_samples,
        )
        return evaluation_samples_from_ucf(records, split_id=split_id)
    if dataset_id == JHU_DATASET_ID:
        if partition != "val":
            raise ValueError("Round 2 JHU lane requires partition=val")
        validation_root = root / "val" if (root / "val").is_dir() else root
        records = index_jhu_validation(
            validation_root, expected_samples=expected_samples
        )
        return evaluation_samples_from_jhu(records, split_id=split_id)
    if dataset_id == UP_COUNT_DATASET_ID:
        if partition != "val_test":
            raise ValueError("Round 2 UP-COUNT lane requires partition=val_test")
        inventory_path = root / "inventory.jsonl"
        if not inventory_path.is_file():
            raise FileNotFoundError(f"UP-COUNT inventory missing: {inventory_path}")
        selected = select_up_count_reference(
            _read_inventory(inventory_path),
            sample_count=expected_samples,
            namespace="round2-upcount-v1",
            minimum_frame_gap=30,
        )
        return evaluation_samples_from_up_count(
            selected,
            dataset_root=root,
            split_id=split_id,
        )
    raise ValueError(f"unsupported Round 2 dataset: {dataset_id}")
