"""Exact, isolated QNRF image preparation for the pinned STEERER code path."""

from __future__ import annotations

import json
import math
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, __version__ as PILLOW_VERSION

from droneai.integrity import sha256_file
from droneai.steerer_official_reproduction import UPSTREAM_COMMIT
from droneai.ucf_qnrf import (
    UCFQNRFRecord,
    density_band,
    read_ucf_qnrf_points,
)


@dataclass(frozen=True)
class ConvertedOfficialSample:
    sample_id: str
    prepared_id: str
    image_path: Path
    annotation_path: Path
    source_image_sha256: str
    source_annotation_sha256: str
    image_sha256: str
    annotation_sha256: str
    width: int
    height: int
    points: tuple[tuple[float, float], ...]
    boundary_corrections: int

    @property
    def count(self) -> int:
        return len(self.points)


def official_qnrf_prepared_size(width: int, height: int) -> tuple[int, int]:
    """Match ``prepare_QNRF.py::resize_images`` including its next-16 behavior."""

    if width <= 0 or height <= 0:
        raise ValueError("official QNRF preparation requires positive dimensions")
    width_ratio = width / 1024.0
    height_ratio = height / 768.0
    if width_ratio < 1.0 or height_ratio < 1.0:
        if width_ratio > height_ratio:
            prepared_height = 768
            prepared_width = int(width / height_ratio)
            prepared_width = (prepared_width // 16 + 1) * 16
        else:
            prepared_height = int(height / width_ratio)
            prepared_height = (prepared_height // 16 + 1) * 16
            prepared_width = 1024
    else:
        prepared_width = (width // 16 + 1) * 16
        prepared_height = (height // 16 + 1) * 16
    return prepared_width, prepared_height


def _official_points(
    raw_points: tuple[tuple[float, float], ...],
    *,
    source_size: tuple[int, int],
    prepared_size: tuple[int, int],
) -> tuple[tuple[tuple[float, float], ...], int]:
    source_width, source_height = source_size
    prepared_width, prepared_height = prepared_size
    points = np.asarray(raw_points, dtype=np.float64).reshape((-1, 2))
    if not np.isfinite(points).all():
        raise ValueError("official QNRF points must be finite")
    if len(points):
        points *= np.asarray(
            (prepared_width / source_width, prepared_height / source_height),
            dtype=np.float64,
        )
        points = points.astype(np.int64).astype(np.float64)
        before = points.copy()
        points[:, 0] = np.clip(points[:, 0], 0, prepared_width - 1)
        points[:, 1] = np.clip(points[:, 1], 0, prepared_height - 1)
        corrections = int(np.any(points != before, axis=1).sum())
    else:
        corrections = 0
    return tuple((float(x), float(y)) for x, y in points), corrections


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def convert_official_test_sample(
    image_path: str | Path,
    annotation_path: str | Path,
    *,
    prepared_id: str,
    source_sample_id: str,
    output_root: str | Path,
) -> ConvertedOfficialSample:
    source_image = Path(image_path).resolve()
    source_annotation = Path(annotation_path).resolve()
    if not source_image.is_file() or not source_annotation.is_file():
        raise FileNotFoundError("official QNRF source image and annotation are required")
    if not prepared_id.isdigit() or len(prepared_id) != 4:
        raise ValueError("prepared QNRF identity must be a four-digit number")
    if not source_sample_id.startswith("test_img_"):
        raise ValueError("prepared Test sample must preserve its source Test identity")

    root = Path(output_root)
    destination_image = root / "images" / f"{prepared_id}.jpg"
    destination_annotation = root / "jsons" / f"{prepared_id}.json"
    if destination_image.exists() or destination_annotation.exists():
        raise FileExistsError("prepared QNRF sample destination already exists")
    destination_image.parent.mkdir(parents=True, exist_ok=True)
    destination_annotation.parent.mkdir(parents=True, exist_ok=True)

    with Image.open(source_image) as opened:
        image = opened.convert("RGB")
        source_width, source_height = image.size
        prepared_width, prepared_height = official_qnrf_prepared_size(
            source_width, source_height
        )
        prepared = image.resize(
            (prepared_width, prepared_height), Image.Resampling.BILINEAR
        )
        prepared.save(destination_image, quality=95)

    raw_points = read_ucf_qnrf_points(source_annotation)
    points, corrections = _official_points(
        raw_points,
        source_size=(source_width, source_height),
        prepared_size=(prepared_width, prepared_height),
    )
    _write_json(
        destination_annotation,
        {
            "human_num": len(raw_points),
            "img_id": f"{prepared_id}.jpg",
            "points": [list(point) for point in points],
            "prepared_size": [prepared_width, prepared_height],
            "preprocessing_reference": (
                "lib/datasets/prepare/prepare_QNRF.py::resize_images/writer_jsons"
            ),
            "source_annotation_sha256": sha256_file(source_annotation),
            "source_image_sha256": sha256_file(source_image),
            "source_sample_id": source_sample_id,
        },
    )
    return ConvertedOfficialSample(
        sample_id=source_sample_id,
        prepared_id=prepared_id,
        image_path=destination_image.resolve(),
        annotation_path=destination_annotation.resolve(),
        source_image_sha256=sha256_file(source_image),
        source_annotation_sha256=sha256_file(source_annotation),
        image_sha256=sha256_file(destination_image),
        annotation_sha256=sha256_file(destination_annotation),
        width=prepared_width,
        height=prepared_height,
        points=points,
        boundary_corrections=corrections,
    )


def _source_pairs(raw_test_root: Path) -> tuple[tuple[Path, Path], ...]:
    images = sorted(raw_test_root.glob("img_*.jpg"))
    pairs: list[tuple[Path, Path]] = []
    for image in images:
        annotation = raw_test_root / f"{image.stem}_ann.mat"
        if not annotation.is_file():
            raise FileNotFoundError(f"official Test annotation is missing: {annotation}")
        pairs.append((image, annotation))
    if not pairs:
        raise ValueError("official Test preparation found no source images")
    return tuple(pairs)


def prepare_official_test_partition(
    raw_test_root: str | Path,
    *,
    output_root: str | Path,
    expected_samples: int = 334,
) -> Path:
    raw_root = Path(raw_test_root).resolve()
    destination = Path(output_root).resolve()
    if not raw_root.is_dir():
        raise FileNotFoundError(f"official Test source root is missing: {raw_root}")
    if isinstance(expected_samples, bool) or expected_samples <= 0:
        raise ValueError("expected Test sample count must be positive")
    pairs = _source_pairs(raw_root)
    if len(pairs) != expected_samples:
        raise ValueError(
            f"official Test preparation requires {expected_samples} samples; "
            f"observed={len(pairs)}"
        )
    if destination.exists() and (not destination.is_dir() or any(destination.iterdir())):
        raise FileExistsError(f"prepared Test destination must be new or empty: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging_parent = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent)
    )
    staging = staging_parent / "dataset"
    try:
        converted: list[ConvertedOfficialSample] = []
        for index, (image, annotation) in enumerate(pairs, start=1):
            converted.append(
                convert_official_test_sample(
                    image,
                    annotation,
                    prepared_id=f"{1201 + index:04d}",
                    source_sample_id=f"test_{image.stem}",
                    output_root=staging,
                )
            )
        (staging / "test.txt").write_text(
            "".join(f"{row.prepared_id}\n" for row in converted),
            encoding="utf-8",
        )
        samples = [
            {
                "annotation_path": row.annotation_path.relative_to(staging).as_posix(),
                "annotation_sha256": row.annotation_sha256,
                "boundary_corrections": row.boundary_corrections,
                "count": row.count,
                "height": row.height,
                "image_path": row.image_path.relative_to(staging).as_posix(),
                "image_sha256": row.image_sha256,
                "prepared_id": row.prepared_id,
                "sample_id": row.sample_id,
                "source_annotation_sha256": row.source_annotation_sha256,
                "source_image_sha256": row.source_image_sha256,
                "width": row.width,
            }
            for row in converted
        ]
        manifest_path = staging / "manifests" / "test-preparation.json"
        _write_json(
            manifest_path,
            {
                "schema_version": 1,
                "dataset_id": "ucf-qnrf-kaggle-apache",
                "partition": "official_test",
                "sample_count": len(samples),
                "source_root": str(raw_root),
                "upstream_commit": UPSTREAM_COMMIT,
                "preprocessing_reference": (
                    "lib/datasets/prepare/prepare_QNRF.py::resize_images/writer_jsons"
                ),
                "image_encoding": "Pillow bilinear JPEG quality=95",
                "pillow_version": PILLOW_VERSION,
                "test_list_sha256": sha256_file(staging / "test.txt"),
                "samples": samples,
            },
        )
        if destination.exists():
            destination.rmdir()
        os.replace(staging, destination)
        return destination / "manifests" / "test-preparation.json"
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    finally:
        staging_parent.rmdir()


def _contained_file(root: Path, relative: Any, *, label: str) -> Path:
    if not isinstance(relative, str) or not relative.strip():
        raise ValueError(f"prepared {label} path is missing")
    target = (root / relative).resolve()
    try:
        target.relative_to(root)
    except ValueError as error:
        raise ValueError(f"prepared {label} path escapes the dataset root") from error
    if not target.is_file():
        raise FileNotFoundError(f"prepared {label} is missing: {target}")
    return target


def load_prepared_official_test(
    prepared_root: str | Path,
    *,
    manifest_path: str | Path,
    expected_samples: int = 334,
) -> tuple[UCFQNRFRecord, ...]:
    root = Path(prepared_root).resolve()
    manifest_file = Path(manifest_path).resolve()
    if not root.is_dir() or not manifest_file.is_file():
        raise FileNotFoundError("prepared Test root and manifest are required")
    try:
        manifest_file.relative_to(root)
    except ValueError as error:
        raise ValueError("prepared Test manifest must stay inside its dataset root") from error
    payload = json.loads(manifest_file.read_text(encoding="utf-8"))
    if (
        not isinstance(payload, dict)
        or payload.get("schema_version") != 1
        or payload.get("dataset_id") != "ucf-qnrf-kaggle-apache"
        or payload.get("partition") != "official_test"
        or payload.get("upstream_commit") != UPSTREAM_COMMIT
        or payload.get("sample_count") != expected_samples
    ):
        raise ValueError("prepared Test manifest identity or sample count is invalid")
    raw_samples = payload.get("samples")
    if not isinstance(raw_samples, list) or len(raw_samples) != expected_samples:
        raise ValueError("prepared Test manifest has incomplete samples")
    records: list[UCFQNRFRecord] = []
    for raw in raw_samples:
        if not isinstance(raw, dict):
            raise ValueError("prepared Test sample manifest rows must be objects")
        image = _contained_file(root, raw.get("image_path"), label="image")
        annotation = _contained_file(
            root, raw.get("annotation_path"), label="annotation"
        )
        if (
            sha256_file(image) != raw.get("image_sha256")
            or sha256_file(annotation) != raw.get("annotation_sha256")
        ):
            raise ValueError("prepared Test artifact hash verification failed")
        annotation_payload = json.loads(annotation.read_text(encoding="utf-8"))
        points = tuple(
            (float(point[0]), float(point[1]))
            for point in annotation_payload.get("points", [])
        )
        width = raw.get("width")
        height = raw.get("height")
        if (
            isinstance(width, bool)
            or not isinstance(width, int)
            or isinstance(height, bool)
            or not isinstance(height, int)
            or width <= 0
            or height <= 0
            or len(points) != raw.get("count")
            or any(not (0 <= x < width and 0 <= y < height) for x, y in points)
        ):
            raise ValueError("prepared Test dimensions, points, or count are invalid")
        records.append(
            UCFQNRFRecord(
                sample_id=str(raw.get("sample_id")),
                image_path=image,
                annotation_path=annotation,
                image_sha256=str(raw.get("image_sha256")),
                annotation_sha256=str(raw.get("annotation_sha256")),
                width=width,
                height=height,
                points=points,
                density_band=density_band(len(points)),
                normalization_corrections=int(raw.get("boundary_corrections", 0)),
            )
        )
    if len({record.sample_id for record in records}) != expected_samples:
        raise ValueError("prepared Test sample identities must be unique")
    return tuple(records)
