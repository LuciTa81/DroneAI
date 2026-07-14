"""Stage 3B: deterministic, test-isolated DM-Count clean-protocol smoke gate."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any

from droneai.integrity import is_sha256, verify_artifact_reference
from droneai.scoring import CheckResult, StageReport, score_stage

PINNED_COMMIT = "cc5f2132e0d1328909f31b6d665b8e0b15c30467"
EXPECTED_TOTAL = 300
EXPECTED_TRAIN = 240
EXPECTED_VALIDATION = 60
SPLIT_SEED = 2026


def _canonical_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def build_clean_split(
    part_a_dir: str | Path,
    *,
    seed: int = SPLIT_SEED,
    train_count: int = EXPECTED_TRAIN,
) -> dict[str, Any]:
    """Build a stable 240/60 split using only ShanghaiTech train_data."""

    part_a_dir = Path(part_a_dir)
    image_dir = part_a_dir / "train_data" / "images"
    gt_dir = part_a_dir / "train_data" / "ground-truth"
    image_paths = sorted(image_dir.glob("*.jpg"), key=lambda path: path.name)
    if len(image_paths) != EXPECTED_TOTAL:
        raise ValueError(f"expected {EXPECTED_TOTAL} training images, found {len(image_paths)}")
    if not 0 < train_count < len(image_paths):
        raise ValueError("train_count must leave non-empty train and validation partitions")

    ranked: list[tuple[str, Path, Path]] = []
    for image_path in image_paths:
        ground_truth = gt_dir / f"GT_{image_path.stem}.mat"
        if not ground_truth.is_file():
            raise FileNotFoundError(ground_truth)
        relative_image = image_path.relative_to(part_a_dir).as_posix()
        rank = hashlib.sha256(f"stage3b:{seed}:{relative_image}".encode("utf-8")).hexdigest()
        ranked.append((rank, image_path, ground_truth))

    ranked.sort(key=lambda item: (item[0], item[1].name))
    train_names = {item[1].name for item in ranked[:train_count]}
    records: list[dict[str, Any]] = []
    for image_path in image_paths:
        ground_truth = gt_dir / f"GT_{image_path.stem}.mat"
        records.append(
            {
                "image": image_path.relative_to(part_a_dir).as_posix(),
                "ground_truth": ground_truth.relative_to(part_a_dir).as_posix(),
                "split": "train" if image_path.name in train_names else "validation",
                "source_partition": "train_data",
                "image_bytes": image_path.stat().st_size,
                "ground_truth_bytes": ground_truth.stat().st_size,
            }
        )

    assignments = [{"image": row["image"], "split": row["split"]} for row in records]
    split_hash = _canonical_hash({"seed": seed, "assignments": assignments})
    return {
        "schema_version": 1,
        "dataset": "ShanghaiTech Part A",
        "seed": seed,
        "method": "sha256-ranked-image-path",
        "counts": {
            "total": len(records),
            "train": sum(row["split"] == "train" for row in records),
            "validation": sum(row["split"] == "validation" for row in records),
        },
        "split_hash": split_hash,
        "records": records,
    }


def write_split_manifest(split: dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "image",
        "ground_truth",
        "split",
        "source_partition",
        "image_bytes",
        "ground_truth_bytes",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(split["records"])


def _finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def build_stage3b_checks(
    evidence: dict[str, Any], *, artifact_root: str | Path | None = None
) -> list[CheckResult]:
    split = evidence.get("split") or {}
    isolation = evidence.get("test_isolation") or {}
    smoke = evidence.get("smoke") or {}
    traceability = evidence.get("traceability") or {}
    artifacts = evidence.get("artifacts") or {}

    counts = split.get("counts") or {}
    count_ok = counts == {
        "total": EXPECTED_TOTAL,
        "train": EXPECTED_TRAIN,
        "validation": EXPECTED_VALIDATION,
    } and int(split.get("overlap", -1)) == 0
    reproducible_ok = (
        split.get("seed") == SPLIT_SEED
        and is_sha256(split.get("split_hash"))
        and split.get("repeat_hash") == split.get("split_hash")
    )
    isolation_ok = (
        isolation.get("validation_source") == "training_partition_only"
        and int(isolation.get("test_paths_seen", -1)) == 0
        and int(isolation.get("test_evaluations", -1)) == 0
        and bool(isolation.get("split_frozen_before_smoke"))
    )
    smoke_ok = (
        int(smoke.get("epochs_completed", -1)) >= 1
        and int(smoke.get("validation_samples", -1)) == EXPECTED_VALIDATION
        and bool(smoke.get("checkpoint_exists"))
        and bool(smoke.get("best_model_exists"))
        and _finite(smoke.get("validation_mae"))
        and _finite(smoke.get("validation_rmse"))
    )
    traceability_schema_ok = (
        traceability.get("upstream_commit") == PINNED_COMMIT
        and bool(traceability.get("gpu"))
        and is_sha256(traceability.get("checkpoint_sha256"))
        and is_sha256(traceability.get("best_model_sha256"))
        and is_sha256(traceability.get("config_snapshot_sha256"))
        and bool(traceability.get("checkpoint"))
        and bool(traceability.get("best_model"))
        and bool(traceability.get("config_snapshot"))
    )
    traceability_results = {
        name: verify_artifact_reference(
            {"path": traceability.get(path_key), "sha256": traceability.get(hash_key)},
            base_dir=artifact_root,
        )
        for name, path_key, hash_key in (
            ("checkpoint", "checkpoint", "checkpoint_sha256"),
            ("best_model", "best_model", "best_model_sha256"),
            ("config_snapshot", "config_snapshot", "config_snapshot_sha256"),
        )
    }
    traceability_ok = traceability_schema_ok and all(
        result[0] for result in traceability_results.values()
    )
    review_results = {
        name: verify_artifact_reference(
            {"path": artifacts.get(path_key), "sha256": artifacts.get(hash_key)},
            base_dir=artifact_root,
        )
        for name, path_key, hash_key in (
            ("manifest_csv", "manifest_csv", "manifest_sha256"),
            ("split_json", "split_json", "split_sha256"),
        )
    }
    output_references_ok = all(
        artifacts.get(key) for key in ("evidence_json", "score_json", "score_md")
    )
    artifacts_ok = output_references_ok and all(result[0] for result in review_results.values())
    traceability_observed = "; ".join(
        f"{key}={result[1]}" for key, result in traceability_results.items()
    )
    review_observed = "; ".join(
        f"{key}={result[1]}" for key, result in review_results.items()
    ) + f"; output_references={output_references_ok}"

    return [
        CheckResult("split.counts", "split integrity", "Frozen split contains 240 train and 60 validation images without overlap", 25, count_ok, True, expected="240/60 from 300; overlap=0", observed=str({**counts, "overlap": split.get("overlap")})),
        CheckResult("split.reproducibility", "reproducibility", "The split is deterministic for seed 2026", 20, reproducible_ok, True, expected="matching 64-character split hashes", observed=f"seed={split.get('seed')}; hash={split.get('split_hash')}; repeat={split.get('repeat_hash')}"),
        CheckResult("evaluation.test_isolation", "test isolation", "Validation uses only the training partition and never evaluates test_data", 20, isolation_ok, True, expected="training_partition_only; zero test paths/evaluations", observed=str(isolation)),
        CheckResult("smoke.training", "training smoke", "At least one clean epoch, validation pass, checkpoint and best model complete", 15, smoke_ok, True, expected="epoch>=1; validation=60; finite metrics and artifacts", observed=str(smoke)),
        CheckResult("traceability.bundle", "traceability", "Pinned code, GPU, config and model hashes are recalculated", 10, traceability_ok, True, expected=PINNED_COMMIT, observed=traceability_observed),
        CheckResult("review.bundle", "review", "Split artifacts are hash-verified and output references are persisted", 10, artifacts_ok, True, expected="verified manifest/split plus three output references", observed=review_observed),
    ]


def run_stage3b_gate(*, evidence: dict[str, Any], output_dir: str | Path) -> StageReport:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report = score_stage(
        stage_id="stage-3b",
        stage_name="DM-Count clean-protocol smoke",
        threshold=90,
        checks=build_stage3b_checks(evidence, artifact_root=output_dir),
        success_status="PASS_RESEARCH_ONLY",
    )
    (output_dir / "evidence.snapshot.json").write_text(
        json.dumps(evidence, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (output_dir / "score.json").write_text(
        json.dumps(report.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (output_dir / "score.md").write_text(report.to_markdown(), encoding="utf-8")
    return report
