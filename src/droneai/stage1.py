"""Stage 1: verify dataset rights, integrity, annotations and split isolation."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

from droneai.scoring import CheckResult, StageReport, score_stage

ALLOWED_SPLITS = {"train", "val", "test"}
SAFE_SPLIT_UNITS = {"sequence", "camera", "date", "zone", "flight_session", "event_session"}


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"inventory line {line_number} must be a JSON object")
        rows.append(value)
    return rows


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _check_asset(
    row: dict[str, Any], dataset_root: Path, path_key: str, hash_key: str
) -> tuple[bool, str, str | None]:
    relative = row.get(path_key)
    expected_hash = row.get(hash_key)
    if not relative or not expected_hash:
        return False, f"{path_key}/{hash_key} missing", None
    path = dataset_root / str(relative)
    if not path.is_file():
        return False, f"missing: {relative}", None
    observed_hash = sha256_file(path)
    if observed_hash != expected_hash:
        return False, f"checksum mismatch: {relative}", observed_hash
    return True, str(relative), observed_hash


def _annotation_points(path: Path) -> list[list[float]]:
    payload = _read_json(path)
    points = payload.get("points")
    if not isinstance(points, list):
        raise ValueError("annotation must contain a points list")
    return points


def build_stage1_checks(
    *, manifest: dict[str, Any], inventory: Iterable[dict[str, Any]], dataset_root: Path
) -> list[CheckResult]:
    rows = list(inventory)
    source = manifest.get("source") or {}
    rights = manifest.get("rights") or {}
    annotation = manifest.get("annotation") or {}
    split_config = manifest.get("split") or {}
    coverage = manifest.get("coverage") or {}

    intended_use = str(manifest.get("intended_use") or "")
    allowed_uses = set(rights.get("allowed_uses") or [])
    rights_verified = rights.get("status") == "verified" and bool(rights.get("evidence_url"))

    asset_failures: list[str] = []
    checksum_failures: list[str] = []
    point_failures: list[str] = []
    image_hash_splits: dict[str, set[str]] = defaultdict(set)
    group_splits: dict[str, set[str]] = defaultdict(set)
    split_counts: dict[str, int] = defaultdict(int)
    tag_keys: set[str] = set()

    for row in rows:
        split = str(row.get("split") or "")
        group = str(row.get("group_id") or "")
        split_counts[split] += 1
        group_splits[group].add(split)
        tag_keys.update((row.get("condition_tags") or {}).keys())

        image_ok, image_evidence, image_hash = _check_asset(
            row, dataset_root, "image_path", "image_sha256"
        )
        annotation_ok, annotation_evidence, _ = _check_asset(
            row, dataset_root, "annotation_path", "annotation_sha256"
        )
        if not image_ok:
            asset_failures.append(image_evidence)
        if not annotation_ok:
            asset_failures.append(annotation_evidence)
        if not image_hash or not row.get("annotation_sha256"):
            checksum_failures.append(str(row.get("sample_id") or "unknown"))
        elif image_ok:
            image_hash_splits[image_hash].add(split)

        if annotation_ok:
            try:
                width = int(row["width"])
                height = int(row["height"])
                points = _annotation_points(dataset_root / str(row["annotation_path"]))
                declared_count = int(row["point_count"])
                if declared_count != len(points):
                    point_failures.append(f"{row.get('sample_id')}: count {declared_count}!={len(points)}")
                for point in points:
                    if (
                        not isinstance(point, list)
                        or len(point) != 2
                        or not 0 <= float(point[0]) < width
                        or not 0 <= float(point[1]) < height
                    ):
                        point_failures.append(f"{row.get('sample_id')}: out-of-bounds point {point}")
                        break
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                point_failures.append(f"{row.get('sample_id')}: {type(exc).__name__}: {exc}")

    invalid_groups = {group: splits for group, splits in group_splits.items() if not group or len(splits) != 1}
    duplicate_hashes = {digest: splits for digest, splits in image_hash_splits.items() if len(splits) > 1}
    required_splits = set(split_config.get("required_splits") or ALLOWED_SPLITS)
    missing_splits = required_splits - {name for name, count in split_counts.items() if count > 0}
    required_tags = set(coverage.get("required_condition_tags") or [])
    missing_tags = required_tags - tag_keys
    split_unit = str(split_config.get("unit") or "")

    source_ok = bool(source.get("official_url") and source.get("citation"))
    schema_ok = all(
        annotation.get(key)
        for key in ("format", "point_semantics", "coordinate_system")
    )
    pair_ok = bool(rows) and not asset_failures
    points_ok = bool(rows) and pair_ok and not point_failures
    checksum_ok = bool(rows) and not checksum_failures and not asset_failures
    required_split_ok = not missing_splits and not (set(split_counts) - ALLOWED_SPLITS)

    return [
        CheckResult(
            "access.source", "license/access", "Official source and citation are recorded", 5, source_ok,
            expected="official_url and citation", observed=str(source_ok), evidence=str(source.get("official_url") or ""),
        ),
        CheckResult(
            "access.rights", "license/access", "Usage-rights evidence is verified", 5, rights_verified, True,
            expected="status=verified and evidence_url", observed=str(rights.get("status") or "missing"),
            evidence=str(rights.get("evidence_url") or ""),
        ),
        CheckResult(
            "access.intended_use", "license/access", "Declared use is allowed by the dataset rights", 10,
            bool(intended_use and intended_use in allowed_uses), True,
            expected=f"{intended_use} in allowed_uses", observed=", ".join(sorted(allowed_uses)) or "none",
        ),
        CheckResult(
            "annotation.schema", "annotation integrity", "Point annotation semantics are explicit", 5, schema_ok,
            expected="format, point_semantics, coordinate_system", observed=str(schema_ok),
        ),
        CheckResult(
            "annotation.pairs", "annotation integrity", "Every inventory row has intact image and annotation files", 10,
            pair_ok, True, expected="all pairs exist and match hashes",
            observed="all pairs verified" if pair_ok else "; ".join(asset_failures[:5]) or "empty inventory",
        ),
        CheckResult(
            "annotation.points", "annotation integrity", "Point counts match and coordinates stay in bounds", 10,
            points_ok, True, expected="count matches; 0 <= x < width; 0 <= y < height",
            observed="all points valid" if points_ok else "; ".join(point_failures[:5]) or "files unavailable",
        ),
        CheckResult(
            "split.unit", "split integrity", "Split unit prevents adjacent-frame leakage", 5,
            split_unit in SAFE_SPLIT_UNITS, True, expected=", ".join(sorted(SAFE_SPLIT_UNITS)), observed=split_unit or "missing",
        ),
        CheckResult(
            "split.groups", "split integrity", "No sequence/camera/session group crosses splits", 10,
            bool(rows) and not invalid_groups, True, expected="one split per group",
            observed="no group overlap" if rows and not invalid_groups else str(invalid_groups)[:500] or "empty inventory",
        ),
        CheckResult(
            "split.duplicates", "split integrity", "No exact image duplicate crosses splits", 10,
            bool(rows) and checksum_ok and not duplicate_hashes, True, expected="no SHA-256 shared across splits",
            observed="no duplicate overlap" if rows and checksum_ok and not duplicate_hashes else str(duplicate_hashes)[:500] or "unverified",
        ),
        CheckResult(
            "coverage.splits", "condition coverage", "Train, validation and test inventories are non-empty", 10,
            bool(rows) and required_split_ok, True, expected=", ".join(sorted(required_splits)),
            observed=str(dict(sorted(split_counts.items()))) if rows else "empty inventory",
        ),
        CheckResult(
            "coverage.tags", "condition coverage", "Required operating-condition tags are represented", 10,
            bool(rows) and not missing_tags, expected=", ".join(sorted(required_tags)) or "none",
            observed=", ".join(sorted(tag_keys)) or "none",
        ),
        CheckResult(
            "manifest.checksums", "manifest", "All files are pinned by SHA-256", 5, checksum_ok, True,
            expected="image and annotation SHA-256 verified", observed="verified" if checksum_ok else ", ".join(checksum_failures[:5]) or "asset failure",
        ),
        CheckResult(
            "manifest.snapshot", "manifest", "Manifest and inventory are versioned inputs", 5,
            bool(manifest.get("schema_version") == 1 and manifest.get("inventory_path") and rows),
            expected="schema_version=1 and non-empty inventory", observed=f"schema={manifest.get('schema_version')}; rows={len(rows)}",
        ),
    ]


def run_stage1(
    *, manifest_path: str | Path, dataset_root: str | Path, output_dir: str | Path
) -> StageReport:
    manifest_path = Path(manifest_path)
    dataset_root = Path(dataset_root)
    output_dir = Path(output_dir)
    manifest = _read_json(manifest_path)
    inventory_path = dataset_root / str(manifest.get("inventory_path") or "inventory.jsonl")
    inventory = _read_jsonl(inventory_path) if inventory_path.is_file() else []
    checks = build_stage1_checks(manifest=manifest, inventory=inventory, dataset_root=dataset_root)
    report = score_stage(
        stage_id="stage-1", stage_name="Dataset readiness and split integrity", threshold=85, checks=checks
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "manifest.snapshot.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (output_dir / "score.json").write_text(
        json.dumps(report.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (output_dir / "score.md").write_text(report.to_markdown(), encoding="utf-8")
    return report
