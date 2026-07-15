from __future__ import annotations

import csv
import io
import json
import math
from dataclasses import asdict
from pathlib import Path
from typing import Iterable

from droneai.evaluation_contract import ScalarEvaluation
from droneai.integrity import sha256_file

REVIEW_BUDGET_BYTES = 25 * 1024 * 1024


def _json_safe(payload: object) -> object:
    if isinstance(payload, float) and not math.isfinite(payload):
        return None
    if isinstance(payload, dict):
        normalized: dict[str, object] = {}
        for key, value in payload.items():
            normalized_key = str(key)
            if normalized_key in normalized:
                raise ValueError(f"JSON key collision after normalization: {key!r}")
            normalized[normalized_key] = _json_safe(value)
        return normalized
    if isinstance(payload, (list, tuple)):
        return [_json_safe(value) for value in payload]
    return payload


def _write_new_text(path: Path, content: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="") as handle:
        handle.write(content)
    return path


def write_json(path: str | Path, payload: object) -> Path:
    content = json.dumps(
        _json_safe(payload),
        indent=2,
        ensure_ascii=False,
        sort_keys=True,
        allow_nan=False,
    )
    return _write_new_text(Path(path), f"{content}\n")


def _flat_record(record: ScalarEvaluation) -> dict[str, object]:
    row = asdict(record)
    extra = row.pop("extra_metrics")
    conditions = row.pop("condition_values")
    for key, value in extra.items():
        if key in row:
            raise ValueError(f"extra metric collides with scalar field: {key}")
        row[key] = value
    for key, value in sorted(conditions.items()):
        field = f"condition_{key}"
        if field in row:
            raise ValueError(f"condition collides with scalar field: {field}")
        row[field] = value
    return {key: _json_safe(value) for key, value in row.items()}


def write_predictions_csv(
    path: str | Path,
    records: Iterable[ScalarEvaluation],
) -> Path:
    rows = [_flat_record(record) for record in records]
    fieldnames = sorted({key for row in rows for key in row})
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return _write_new_text(Path(path), buffer.getvalue())


def artifact_reference(
    path: str | Path,
    *,
    base_dir: str | Path,
) -> dict[str, str]:
    base = Path(base_dir).resolve()
    target = Path(path).resolve()
    if not target.is_file():
        raise FileNotFoundError(f"artifact missing: {target}")
    try:
        relative = target.relative_to(base)
    except ValueError as exc:
        raise ValueError("artifact must be inside base directory") from exc
    return {"path": relative.as_posix(), "sha256": sha256_file(target)}


def enforce_review_budget(
    root: str | Path,
    limit_bytes: int = REVIEW_BUDGET_BYTES,
) -> int:
    if limit_bytes < 0:
        raise ValueError("review budget limit must be non-negative")
    bundle_root = Path(root)
    if not bundle_root.is_dir():
        raise FileNotFoundError(f"review bundle directory missing: {bundle_root}")
    paths = [bundle_root, *sorted(bundle_root.rglob("*"))]
    for path in paths:
        is_junction = getattr(path, "is_junction", lambda: False)
        if path.is_symlink() or is_junction():
            raise ValueError(f"review bundle cannot contain a symlink or junction: {path}")
    total = sum(path.stat().st_size for path in paths[1:] if path.is_file())
    if total > limit_bytes:
        raise ValueError(
            f"review bundle exceeds budget: {total} bytes "
            f"(limit={limit_bytes} bytes; default=25 MiB)"
        )
    return total
