"""Deterministic training-only UCF-QNRF train/validation splits."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

from droneai.ucf_qnrf import UCFQNRFRecord, _stable_key


@dataclass(frozen=True)
class SteererTrainingSplit:
    seed: int
    train_ids: tuple[str, ...]
    validation_ids: tuple[str, ...]
    band_counts: dict[str, dict[str, int]]
    population_sha256: str
    train_sha256: str
    validation_sha256: str


def _sha256_lines(values: Iterable[str], *, final_newline: bool = False) -> str:
    text = "\n".join(values)
    if final_newline:
        text += "\n"
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _population_sha256(records: Iterable[UCFQNRFRecord]) -> str:
    rows = (
        f"{record.sample_id},{record.image_sha256},{record.annotation_sha256},"
        f"{record.count},{record.density_band}"
        for record in sorted(records, key=lambda record: record.sample_id)
    )
    return _sha256_lines(rows)


def _validation_quotas(
    band_sizes: dict[str, int], *, population_count: int, validation_count: int
) -> dict[str, int]:
    quotas = {
        band: validation_count * size // population_count
        for band, size in band_sizes.items()
    }
    remaining = validation_count - sum(quotas.values())
    remainders = sorted(
        band_sizes,
        key=lambda band: (
            -(validation_count * band_sizes[band] % population_count),
            band,
        ),
    )
    for band in remainders[:remaining]:
        quotas[band] += 1
    return quotas


def build_training_split(
    records: Iterable[UCFQNRFRecord], *, seed: int, validation_count: int
) -> SteererTrainingSplit:
    """Split only supplied training records into deterministic train and validation IDs."""

    rows = tuple(records)
    if not rows:
        raise ValueError("training split requires at least one record")
    if validation_count < 0 or validation_count > len(rows):
        raise ValueError("validation count must be between zero and the training population")
    if len({record.sample_id for record in rows}) != len(rows):
        raise ValueError("training split requires unique sample IDs")

    by_band: dict[str, list[UCFQNRFRecord]] = {}
    for record in rows:
        by_band.setdefault(record.density_band, []).append(record)
    band_sizes = {band: len(by_band[band]) for band in by_band}
    quotas = _validation_quotas(
        band_sizes, population_count=len(rows), validation_count=validation_count
    )
    validation_ids = tuple(
        record.sample_id
        for band in sorted(by_band)
        for record in sorted(
            by_band[band], key=lambda record: _stable_key(record.sample_id, seed, f"steerer-val-{band}")
        )[: quotas[band]]
    )
    validation_set = set(validation_ids)
    train_ids = tuple(
        record.sample_id
        for record in sorted(
            (record for record in rows if record.sample_id not in validation_set),
            key=lambda record: _stable_key(record.sample_id, seed, "steerer-train"),
        )
    )
    band_counts = {
        band: {
            "population": len(by_band[band]),
            "train": len(by_band[band]) - quotas[band],
            "validation": quotas[band],
        }
        for band in sorted(by_band)
    }
    return SteererTrainingSplit(
        seed=seed,
        train_ids=train_ids,
        validation_ids=validation_ids,
        band_counts=band_counts,
        population_sha256=_population_sha256(rows),
        train_sha256=_sha256_lines(train_ids, final_newline=True),
        validation_sha256=_sha256_lines(validation_ids, final_newline=True),
    )


def _atomic_write_bytes(path: Path, payload: bytes) -> None:
    """Replace one split artifact only after its bytes are durable."""

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


def _write_lines(path: Path, values: tuple[str, ...]) -> None:
    _atomic_write_bytes(path, ("\n".join(values) + "\n").encode("utf-8"))


def write_training_split(output_dir: str | Path, split: SteererTrainingSplit) -> Path:
    """Atomically create a sealed training split directory at ``output_dir``."""

    destination = Path(output_dir)
    if destination.exists():
        raise FileExistsError(f"split output must be a new directory: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent))
    try:
        _write_lines(temporary / "train.txt", split.train_ids)
        _write_lines(temporary / "val.txt", split.validation_ids)
        _atomic_write_bytes(
            temporary / "test-sealed.json",
            (
                json.dumps(
                    {"role": "sealed", "sample_count": 334, "accessed": False},
                    indent=2,
                )
                + "\n"
            ).encode("utf-8"),
        )
        _atomic_write_bytes(
            temporary / "split-manifest.json",
            (json.dumps(asdict(split), indent=2, sort_keys=True) + "\n").encode("utf-8"),
        )
        os.replace(temporary, destination)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return destination
