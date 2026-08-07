"""Deterministic conversion of indexed UCF-QNRF training assets for STEERER."""

from __future__ import annotations

import json
import math
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from scipy.spatial.distance import cdist

from droneai.integrity import sha256_file
from droneai.steerer_training_split import SteererTrainingSplit, write_training_split
from droneai.ucf_qnrf import UCFQNRFRecord


@dataclass(frozen=True)
class ConvertedTrainingSample:
    """Paths and immutable source dimensions for one converted training sample."""

    sample_id: str
    image_path: Path
    json_path: Path
    width: int
    height: int
    image_sha256: str
    json_sha256: str
    count: int


def _atomic_write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        with temporary.open("xb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _atomic_copy_bytes(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    try:
        with source.open("rb") as input_stream, temporary.open("xb") as output_stream:
            shutil.copyfileobj(input_stream, output_stream, length=1024 * 1024)
            output_stream.flush()
            os.fsync(output_stream.fileno())
        os.replace(temporary, destination)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _normalized_points(record: UCFQNRFRecord) -> np.ndarray:
    """Use the indexed points, only resolving an exact right/bottom edge."""

    if record.width <= 0 or record.height <= 0:
        raise ValueError("indexed training sample must have positive image dimensions")
    points = np.asarray(record.points, dtype=np.float64).reshape((-1, 2))
    if not np.isfinite(points).all():
        raise ValueError(f"indexed training sample has non-finite points: {record.sample_id}")
    if (
        np.any(points[:, 0] < 0.0)
        or np.any(points[:, 0] > record.width)
        or np.any(points[:, 1] < 0.0)
        or np.any(points[:, 1] > record.height)
    ):
        raise ValueError(f"indexed training sample has out-of-bounds points: {record.sample_id}")
    if len(points):
        points[:, 0] = np.minimum(points[:, 0], math.nextafter(float(record.width), 0.0))
        points[:, 1] = np.minimum(points[:, 1], math.nextafter(float(record.height), 0.0))
    return points


def clipped_xyxy(
    points: np.ndarray, half_size: np.ndarray, *, width: int, height: int
) -> np.ndarray:
    """Create inclusive, image-bounded ``xyxy`` boxes around point annotations."""

    if width <= 0 or height <= 0:
        raise ValueError("image dimensions must be positive")
    if len(points) == 0:
        return np.empty((0, 4), dtype=np.float32)
    maximum_x, maximum_y = float(width - 1), float(height - 1)
    return np.column_stack(
        (
            np.clip(points[:, 0] - half_size, 0.0, maximum_x),
            np.clip(points[:, 1] - half_size, 0.0, maximum_y),
            np.clip(points[:, 0] + half_size, 0.0, maximum_x),
            np.clip(points[:, 1] + half_size, 0.0, maximum_y),
        )
    ).astype(np.float32)


def point_boxes(points: np.ndarray, *, width: int, height: int) -> np.ndarray:
    """Derive bounded audit boxes from each point's local neighbour distance."""

    if len(points) == 0:
        return np.empty((0, 4), dtype=np.float32)
    distances = cdist(points, points)
    np.fill_diagonal(distances, np.inf)
    neighbour_count = min(3, max(len(points) - 1, 1))
    local = np.partition(distances, kth=neighbour_count - 1, axis=1)[:, :neighbour_count].mean(axis=1)
    if len(points) == 1:
        local[:] = 32.0
    half = np.clip(local * 0.25, 2.0, 64.0)
    return clipped_xyxy(points, half, width=width, height=height)


def _json_bytes(payload: object) -> bytes:
    return (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")


def convert_training_sample(
    record: UCFQNRFRecord, *, output_root: str | Path
) -> ConvertedTrainingSample:
    """Copy one source image verbatim and write the STEERER point annotation JSON."""

    root = Path(output_root)
    points = _normalized_points(record)
    boxes = point_boxes(points, width=record.width, height=record.height)
    image_path = root / "images" / f"{record.sample_id}.jpg"
    json_path = root / "jsons" / f"{record.sample_id}.json"
    _atomic_copy_bytes(record.image_path, image_path)
    if sha256_file(image_path) != record.image_sha256:
        raise ValueError(f"source image hash changed during copy: {record.sample_id}")
    payload = {
        "annotation_sha256": record.annotation_sha256,
        "boxes": boxes.tolist(),
        "human_num": record.count,
        "points": points.tolist(),
        "sample_id": record.sample_id,
        "scale": [1.0, 1.0],
        "source_height": record.height,
        "source_partition": "official_train_only",
        "source_sha256": record.image_sha256,
        "source_width": record.width,
    }
    _atomic_write_bytes(json_path, _json_bytes(payload))
    return ConvertedTrainingSample(
        sample_id=record.sample_id,
        image_path=image_path,
        json_path=json_path,
        width=record.width,
        height=record.height,
        image_sha256=record.image_sha256,
        json_sha256=sha256_file(json_path),
        count=record.count,
    )


def _inventory_bytes(rows: Iterable[dict[str, object]]) -> bytes:
    return b"".join(
        json.dumps(row, sort_keys=True, separators=(",", ":")).encode("utf-8") + b"\n"
        for row in rows
    )


def _validate_records_and_split(
    records: tuple[UCFQNRFRecord, ...], split: SteererTrainingSplit
) -> None:
    if not records:
        raise ValueError("training preparation requires at least one indexed record")
    record_ids = [record.sample_id for record in records]
    if len(record_ids) != len(set(record_ids)):
        raise ValueError("training preparation requires unique sample IDs")
    split_ids = split.train_ids + split.validation_ids
    if len(split_ids) != len(set(split_ids)) or set(split_ids) != set(record_ids):
        raise ValueError("training split must be disjoint and complete for indexed records")
    for record in records:
        _normalized_points(record)


def _prepare_destination(destination: Path) -> None:
    if destination.exists():
        if not destination.is_dir() or any(destination.iterdir()):
            raise FileExistsError(
                f"training output must be a new or empty directory: {destination}"
            )
    destination.parent.mkdir(parents=True, exist_ok=True)


def prepare_training_dataset(
    records: Iterable[UCFQNRFRecord],
    split: SteererTrainingSplit,
    *,
    output_root: str | Path,
) -> Path:
    """Atomically write a complete, training-only STEERER dataset and manifest."""

    rows = tuple(records)
    destination = Path(output_root)
    _validate_records_and_split(rows, split)
    _prepare_destination(destination)
    parent_temporary = Path(tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent))
    staging = parent_temporary / "dataset"
    try:
        write_training_split(staging, split)
        input_inventory = [
            {
                "annotation_sha256": record.annotation_sha256,
                "count": record.count,
                "height": record.height,
                "image_sha256": record.image_sha256,
                "sample_id": record.sample_id,
                "width": record.width,
            }
            for record in sorted(rows, key=lambda row: row.sample_id)
        ]
        _atomic_write_bytes(
            staging / "manifests" / "input-inventory.jsonl", _inventory_bytes(input_inventory)
        )
        converted = [convert_training_sample(record, output_root=staging) for record in rows]
        result_by_id = {result.sample_id: result for result in converted}
        output_inventory = [
            {
                "count": result.count,
                "height": result.height,
                "image_sha256": result.image_sha256,
                "json_sha256": result.json_sha256,
                "sample_id": result.sample_id,
                "width": result.width,
            }
            for result in sorted(converted, key=lambda result: result.sample_id)
        ]
        _atomic_write_bytes(
            staging / "manifests" / "output-inventory.jsonl", _inventory_bytes(output_inventory)
        )

        count_mismatches = 0
        out_of_bounds_points = 0
        out_of_bounds_boxes = 0
        for record in rows:
            result = result_by_id[record.sample_id]
            payload = json.loads(result.json_path.read_text(encoding="utf-8"))
            points = payload["points"]
            boxes = payload["boxes"]
            count_mismatches += int(payload["human_num"] != record.count or len(points) != record.count)
            out_of_bounds_points += sum(
                not (0 <= x < record.width and 0 <= y < record.height) for x, y in points
            )
            out_of_bounds_boxes += sum(
                not (0 <= x1 <= x2 < record.width and 0 <= y1 <= y2 < record.height)
                for x1, y1, x2, y2 in boxes
            )
        if count_mismatches or out_of_bounds_points or out_of_bounds_boxes:
            raise ValueError("converted training data failed integrity validation")

        input_inventory_path = staging / "manifests" / "input-inventory.jsonl"
        output_inventory_path = staging / "manifests" / "output-inventory.jsonl"
        manifest = {
            "converted_sample_count": len(converted),
            "count_mismatches": count_mismatches,
            "input_inventory_sha256": sha256_file(input_inventory_path),
            "out_of_bounds_boxes": out_of_bounds_boxes,
            "out_of_bounds_points": out_of_bounds_points,
            "output_inventory_sha256": sha256_file(output_inventory_path),
            "sample_count": len(converted),
            "source_partition": "official_train_only",
            "train_count": len(split.train_ids),
            "validation_count": len(split.validation_ids),
        }
        manifest_path = staging / "manifests" / "training-manifest.json"
        _atomic_write_bytes(manifest_path, _json_bytes(manifest))
        if destination.exists():
            destination.rmdir()
        os.replace(staging, destination)
        return destination / "manifests" / "training-manifest.json"
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    finally:
        parent_temporary.rmdir()
