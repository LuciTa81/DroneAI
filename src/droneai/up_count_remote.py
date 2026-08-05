"""Select and verify a small UP-COUNT reference set from the official remote ZIP."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import zlib
import zipfile
from pathlib import Path, PurePosixPath
from typing import Iterable, Mapping, Protocol

from droneai.up_count import select_up_count_reference


class ReadableZip(Protocol):
    def infolist(self) -> list[zipfile.ZipInfo]: ...

    def getinfo(self, name: str) -> zipfile.ZipInfo: ...

    def open(self, name: str | zipfile.ZipInfo, mode: str = "r"): ...


def _split_map(split_dir: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for split in ("train", "val", "test"):
        path = split_dir / f"{split}.txt"
        if not path.is_file():
            raise FileNotFoundError(f"UP-COUNT split file missing: {path}")
        for raw in path.read_text(encoding="utf-8").splitlines():
            sequence = raw.strip()
            if not sequence:
                continue
            if sequence in result:
                raise ValueError(f"UP-COUNT sequence occurs in two splits: {sequence}")
            result[sequence] = split
    return result


def _frame_id(stem: str) -> int:
    try:
        return int(stem.split("__")[1])
    except (IndexError, ValueError) as exc:
        raise ValueError(f"invalid UP-COUNT frame name: {stem}") from exc


def _safe_member(name: str, *, kind: str) -> tuple[str, str]:
    path = PurePosixPath(name)
    expected = ("UP-COUNT", kind)
    if path.is_absolute() or ".." in path.parts or path.parts[:2] != expected:
        raise ValueError(f"unsafe UP-COUNT {kind} member: {name}")
    if len(path.parts) != 4:
        raise ValueError(f"unexpected UP-COUNT {kind} member: {name}")
    return path.parts[2], path.stem


def build_up_count_catalog(
    *, labels_zip: str | Path, split_dir: str | Path
) -> tuple[dict[str, object], ...]:
    """Read label member names/counts without requiring the 28.4 GB image ZIP."""

    splits = _split_map(Path(split_dir))
    rows: list[dict[str, object]] = []
    seen: set[str] = set()
    with zipfile.ZipFile(labels_zip) as labels:
        for info in sorted(labels.infolist(), key=lambda value: value.filename):
            if info.is_dir() or not info.filename.endswith(".txt"):
                continue
            sequence, stem = _safe_member(info.filename, kind="labels")
            if sequence not in splits:
                raise ValueError(f"UP-COUNT label sequence has no split: {sequence}")
            sample_id = f"{sequence}/{stem}"
            if sample_id in seen:
                raise ValueError(f"duplicate UP-COUNT label member: {sample_id}")
            seen.add(sample_id)
            point_count = sum(
                bool(line.strip()) for line in labels.read(info).splitlines()
            )
            rows.append(
                {
                    "sample_id": sample_id,
                    "split": splits[sequence],
                    "group_id": sequence,
                    "frame_id": _frame_id(stem),
                    "point_count": point_count,
                    "label_member": info.filename,
                }
            )
    if not rows:
        raise ValueError("UP-COUNT label archive contains no samples")
    return tuple(rows)


def select_up_count_members(
    catalog: Iterable[dict[str, object]],
    *,
    image_infos: Iterable[zipfile.ZipInfo],
    sample_count: int,
    namespace: str,
    minimum_frame_gap: int,
) -> tuple[dict[str, object], ...]:
    """Apply the frozen selector and bind every row to one official image member."""

    images: dict[str, zipfile.ZipInfo] = {}
    for info in image_infos:
        if info.is_dir() or not info.filename.lower().endswith(".jpg"):
            continue
        sequence, stem = _safe_member(info.filename, kind="images")
        sample_id = f"{sequence}/{stem}"
        if sample_id in images:
            raise ValueError(f"duplicate UP-COUNT image member: {sample_id}")
        images[sample_id] = info
    selected = select_up_count_reference(
        catalog,
        sample_count=sample_count,
        namespace=namespace,
        minimum_frame_gap=minimum_frame_gap,
    )
    result: list[dict[str, object]] = []
    for row in selected:
        sample_id = str(row["sample_id"])
        if sample_id not in images:
            raise FileNotFoundError(f"UP-COUNT image member missing: {sample_id}")
        info = images[sample_id]
        result.append(
            {
                **row,
                "image_member": info.filename,
                "image_file_size": info.file_size,
                "image_compress_size": info.compress_size,
                "image_crc32": f"{info.CRC:08x}",
            }
        )
    return tuple(result)


def _hash_and_crc(path: Path) -> tuple[str, int, int]:
    digest = hashlib.sha256()
    checksum = 0
    size = 0
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
            checksum = zlib.crc32(block, checksum)
            size += len(block)
    return digest.hexdigest(), checksum & 0xFFFFFFFF, size


def _extract_verified_member(
    archive: ReadableZip, *, member: str, output: Path
) -> dict[str, object]:
    info = archive.getinfo(member)
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = output.with_suffix(output.suffix + ".part")
    if output.exists():
        sha256, crc32, size = _hash_and_crc(output)
        if size != info.file_size or crc32 != info.CRC:
            raise ValueError(f"existing selected UP-COUNT image is corrupt: {output}")
    else:
        if staging.exists():
            raise FileExistsError(f"incomplete UP-COUNT staging file exists: {staging}")
        with archive.open(info) as source, staging.open("xb") as destination:
            shutil.copyfileobj(source, destination, length=1024 * 1024)
        sha256, crc32, size = _hash_and_crc(staging)
        if size != info.file_size or crc32 != info.CRC:
            raise ValueError(f"remote UP-COUNT member failed ZIP verification: {member}")
        staging.replace(output)
    return {
        "file_size": size,
        "compressed_size": info.compress_size,
        "crc32": f"{crc32:08x}",
        "sha256": sha256,
    }


def _copy_exact(source: Path, output: Path) -> str:
    payload = source.read_bytes()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        if output.read_bytes() != payload:
            raise ValueError(f"existing provenance file differs: {output}")
    else:
        staging = output.with_suffix(output.suffix + ".tmp")
        if staging.exists():
            raise FileExistsError(f"provenance staging file exists: {staging}")
        staging.write_bytes(payload)
        staging.replace(output)
    return hashlib.sha256(payload).hexdigest()


def extract_selected_up_count(
    *,
    image_zip: ReadableZip,
    labels_zip: str | Path,
    split_dir: str | Path,
    selected: Iterable[dict[str, object]],
    dataset_root: str | Path,
    archive_identity: Mapping[str, object],
    selection_identity: Mapping[str, object],
) -> dict[str, object]:
    """Extract selected members atomically and verify completed files on resume."""

    root = Path(dataset_root)
    root.mkdir(parents=True, exist_ok=True)
    labels_path = Path(labels_zip)
    splits_path = Path(split_dir)
    source_hash = _copy_exact(labels_path, root / "source/labels.zip")
    split_hashes = {
        split: _copy_exact(
            splits_path / f"{split}.txt", root / f"splits/{split}.txt"
        )
        for split in ("train", "val", "test")
    }
    member_rows: list[dict[str, object]] = []
    with zipfile.ZipFile(labels_path) as labels:
        for row in selected:
            sequence = str(row["group_id"])
            stem = str(row["sample_id"]).split("/", 1)[1]
            label_member = str(row["label_member"])
            image_member = str(row["image_member"])
            label_payload = labels.read(label_member)
            label_output = root / "labels" / sequence / f"{stem}.txt"
            label_output.parent.mkdir(parents=True, exist_ok=True)
            if label_output.exists():
                if label_output.read_bytes() != label_payload:
                    raise ValueError(f"existing selected label differs: {label_output}")
            else:
                label_output.write_bytes(label_payload)
            image_output = root / "images" / sequence / f"{stem}.jpg"
            evidence = _extract_verified_member(
                image_zip, member=image_member, output=image_output
            )
            member_rows.append(
                {
                    "sample_id": row["sample_id"],
                    "split": row["split"],
                    "group_id": sequence,
                    "point_count": row["point_count"],
                    "gap_relaxed": row["gap_relaxed"],
                    "label_member": label_member,
                    "image_member": image_member,
                    **evidence,
                }
            )
    manifest = {
        "schema_version": 1,
        "archive_identity": dict(sorted(archive_identity.items())),
        "labels_zip_sha256": source_hash,
        "split_sha256": split_hashes,
        "selection_identity": dict(sorted(selection_identity.items())),
        "selected_samples": len(member_rows),
        "members": member_rows,
    }
    manifest_path = root / "acquisition/selected-members.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    if manifest_path.exists():
        if json.loads(manifest_path.read_text(encoding="utf-8")) != manifest:
            raise ValueError("existing UP-COUNT selection manifest differs")
    else:
        staging = manifest_path.with_suffix(".json.tmp")
        if staging.exists():
            raise FileExistsError(f"selection manifest staging file exists: {staging}")
        staging.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(staging, manifest_path)
    return manifest
