"""Normalize official UP-COUNT labels into the DroneAI data contract."""

from __future__ import annotations

import json
import hashlib
import math
from collections import defaultdict
from pathlib import Path
from typing import Iterable

from PIL import Image

from droneai.stage1 import sha256_file

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
NORMALIZATION_POLICY = "up-count-official-loader-a6d3664"


def _selection_key(namespace: str, sequence_id: str, frame_id: int) -> str:
    value = f"{namespace}:{sequence_id}:{frame_id}".encode("utf-8")
    return hashlib.sha256(value).hexdigest()


def _frame_id(row: dict[str, object]) -> int:
    value = row.get("frame_id")
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    stem = str(row.get("sample_id") or "").rsplit("/", 1)[-1]
    try:
        return int(stem.split("__")[1])
    except (IndexError, ValueError) as exc:
        raise ValueError(f"cannot parse UP-COUNT frame ID: {stem}") from exc


def _sequence_quotas(
    groups: dict[str, list[dict[str, object]]],
    *,
    sample_count: int,
    namespace: str,
) -> dict[str, int]:
    if sample_count <= 0 or sum(len(rows) for rows in groups.values()) < sample_count:
        raise ValueError("UP-COUNT selection requires enough eligible frames")
    sequences = sorted(groups)
    quotas = {sequence: 0 for sequence in sequences}
    if len(sequences) > sample_count:
        chosen = sorted(
            sequences,
            key=lambda sequence: _selection_key(namespace, sequence, -1),
        )[:sample_count]
        return {sequence: int(sequence in chosen) for sequence in sequences}
    for sequence in sequences:
        quotas[sequence] = 1
    remaining = sample_count - len(sequences)
    while remaining:
        capacities = {
            sequence: len(groups[sequence]) - quotas[sequence]
            for sequence in sequences
            if len(groups[sequence]) > quotas[sequence]
        }
        if not capacities:
            raise ValueError("UP-COUNT sequence capacity cannot satisfy selection")
        total_capacity = sum(capacities.values())
        ideals = {
            sequence: remaining * capacity / total_capacity
            for sequence, capacity in capacities.items()
        }
        awarded = 0
        for sequence in capacities:
            addition = min(math.floor(ideals[sequence]), capacities[sequence])
            quotas[sequence] += addition
            awarded += addition
        remaining -= awarded
        if not remaining:
            break
        order = sorted(
            capacities,
            key=lambda sequence: (
                -(ideals[sequence] - math.floor(ideals[sequence])),
                _selection_key(namespace, sequence, -2),
            ),
        )
        for sequence in order:
            if remaining == 0:
                break
            if quotas[sequence] < len(groups[sequence]):
                quotas[sequence] += 1
                remaining -= 1
    return quotas


def select_up_count_reference(
    inventory: Iterable[dict[str, object]],
    *,
    sample_count: int,
    namespace: str,
    minimum_frame_gap: int,
) -> tuple[dict[str, object], ...]:
    if not namespace or minimum_frame_gap < 0:
        raise ValueError("UP-COUNT selection requires a namespace and nonnegative gap")
    groups: dict[str, list[dict[str, object]]] = defaultdict(list)
    seen_ids: set[str] = set()
    for source_row in inventory:
        row = dict(source_row)
        if row.get("split") not in {"val", "test"}:
            continue
        sample_id = str(row.get("sample_id") or "")
        sequence = str(row.get("group_id") or "")
        if not sample_id or not sequence or sample_id in seen_ids:
            raise ValueError("UP-COUNT inventory requires unique IDs and group IDs")
        seen_ids.add(sample_id)
        row["frame_id"] = _frame_id(row)
        groups[sequence].append(row)
    if not groups:
        raise ValueError("UP-COUNT selection found no validation/test frames")
    quotas = _sequence_quotas(groups, sample_count=sample_count, namespace=namespace)
    selected: list[dict[str, object]] = []
    for sequence, rows in sorted(groups.items()):
        quota = quotas[sequence]
        ordered = sorted(
            rows,
            key=lambda row: _selection_key(
                namespace, sequence, int(row["frame_id"])
            ),
        )
        accepted: list[dict[str, object]] = []
        deferred: list[dict[str, object]] = []
        for row in ordered:
            frame = int(row["frame_id"])
            if all(
                abs(frame - int(existing["frame_id"])) >= minimum_frame_gap
                for existing in accepted
            ):
                accepted.append({**row, "gap_relaxed": False})
            else:
                deferred.append(row)
            if len(accepted) == quota:
                break
        if len(accepted) < quota:
            accepted_ids = {str(row["sample_id"]) for row in accepted}
            for row in deferred + ordered:
                if str(row["sample_id"]) in accepted_ids:
                    continue
                accepted.append({**row, "gap_relaxed": True})
                accepted_ids.add(str(row["sample_id"]))
                if len(accepted) == quota:
                    break
        if len(accepted) != quota:
            raise ValueError(f"UP-COUNT quota shortfall for sequence {sequence}")
        selected.extend(accepted)
    return tuple(
        sorted(
            selected,
            key=lambda row: _selection_key(
                namespace, str(row["group_id"]), int(row["frame_id"])
            ),
        )
    )


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


def _normalize_like_official_loader(
    points: list[list[float]], width: int, height: int
) -> tuple[list[list[int]], int]:
    """Match the official loader: integer cast, then clip to image bounds."""

    normalized: list[list[int]] = []
    corrected = 0
    for x, y in points:
        nx = min(max(int(x), 0), width - 1)
        ny = min(max(int(y), 0), height - 1)
        normalized.append([nx, ny])
        if nx != x or ny != y:
            corrected += 1
    return normalized, corrected


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

        source_points = _parse_points(label_path)
        altitude = _altitude_from_stem(label_path.stem)
        with Image.open(image_path) as image:
            width, height = image.size
        points, corrected_count = _normalize_like_official_loader(
            source_points, width, height
        )

        normalized_path = normalized_root / sequence_id / f"{label_path.stem}.json"
        normalized_path.parent.mkdir(parents=True, exist_ok=True)
        normalized_path.write_text(
            json.dumps(
                {
                    "points": points,
                    "normalization": {
                        "policy": NORMALIZATION_POLICY,
                        "corrected_count": corrected_count,
                        "source_label": label_path.as_posix(),
                    },
                },
                ensure_ascii=False,
                separators=(",", ":"),
            ),
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
                "normalization_corrections": corrected_count,
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
