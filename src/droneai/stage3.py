"""Stage 3: score faithful and leakage-free DM-Count reproductions separately."""

from __future__ import annotations

import json
import math
import statistics
from pathlib import Path
from typing import Any

from droneai.integrity import is_sha256, verify_artifact_reference
from droneai.scoring import CheckResult, StageReport, score_stage

PINNED_COMMIT = "cc5f2132e0d1328909f31b6d665b8e0b15c30467"
PAPER_MAE = 59.7
PAPER_RMSE = 95.7
EXPECTED_TRAIN = 300
EXPECTED_TEST = 182
EXPECTED_SEEDS = {2026, 2027, 2028}


def _finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _relative_gap(observed: float, target: float) -> float:
    return abs(observed - target) / target


def build_stage3_checks(
    evidence: dict[str, Any], *, artifact_root: str | Path | None = None
) -> list[CheckResult]:
    """Build the fixed 100-point gate from persisted Stage 3 evidence."""

    upstream = evidence.get("upstream") or {}
    dataset = evidence.get("dataset") or {}
    faithful_runs = evidence.get("faithful_runs") or []
    clean = evidence.get("clean_protocol") or {}
    artifacts = evidence.get("artifacts") or {}
    runtime = evidence.get("runtime") or {}

    faithful_seeds = {int(run["seed"]) for run in faithful_runs if "seed" in run}
    complete_runs = [
        run
        for run in faithful_runs
        if _finite(run.get("mae"))
        and _finite(run.get("rmse"))
        and int(run.get("prediction_count", -1)) == EXPECTED_TEST
    ]
    mae_values = [float(run["mae"]) for run in complete_runs]
    rmse_values = [float(run["rmse"]) for run in complete_runs]
    median_mae = statistics.median(mae_values) if mae_values else float("inf")
    median_rmse = statistics.median(rmse_values) if rmse_values else float("inf")
    mae_gap = _relative_gap(median_mae, PAPER_MAE) if _finite(median_mae) else float("inf")
    rmse_gap = _relative_gap(median_rmse, PAPER_RMSE) if _finite(median_rmse) else float("inf")
    paper_gap_ok = mae_gap <= 0.05 and rmse_gap <= 0.05
    stability = (
        statistics.pstdev(mae_values) / median_mae
        if len(mae_values) >= 2 and median_mae > 0
        else float("inf")
    )

    dataset_ok = (
        dataset.get("id") == "shanghaitech-part-a"
        and int(dataset.get("train_images", -1)) == EXPECTED_TRAIN
        and int(dataset.get("test_images", -1)) == EXPECTED_TEST
        and is_sha256(dataset.get("split_hash"))
    )
    rights_ok = (
        dataset.get("license_status") == "verified"
        and dataset.get("commercial_use") is False
    )
    evaluation_ok = len(complete_runs) == len(faithful_runs) and bool(complete_runs)
    seeds_ok = faithful_seeds == EXPECTED_SEEDS and len(complete_runs) == 3
    clean_ok = (
        clean.get("validation_source") == "training_partition_only"
        and clean.get("test_evaluations_per_seed") == 1
        and set(clean.get("seeds") or []) == EXPECTED_SEEDS
        and bool(clean.get("split_frozen_before_test"))
    )
    artifact_results = {
        key: verify_artifact_reference(artifacts.get(key), base_dir=artifact_root)
        for key in ("config_snapshot", "predictions", "checkpoint_hashes", "environment")
    }
    artifacts_ok = all(result[0] for result in artifact_results.values())
    artifact_observed = "; ".join(
        f"{key}={result[1]}" for key, result in artifact_results.items()
    )
    runtime_ok = (
        bool(runtime.get("gpu"))
        and _finite(runtime.get("median_seconds_per_image"))
        and float(runtime.get("median_seconds_per_image")) > 0
    )

    return [
        CheckResult("provenance.commit", "traceability", "Official repository commit is pinned", 10, upstream.get("commit") == PINNED_COMMIT, True, expected=PINNED_COMMIT, observed=str(upstream.get("commit"))),
        CheckResult("provenance.code_license", "traceability", "Official implementation license is recorded", 5, upstream.get("license") == "MIT", expected="MIT", observed=str(upstream.get("license"))),
        CheckResult("dataset.contract", "evaluation integrity", "ShanghaiTech Part A official split is intact", 10, dataset_ok, True, expected="300 train, 182 test and split hash", observed=str({key: dataset.get(key) for key in ("id", "train_images", "test_images", "split_hash")})),
        CheckResult("dataset.rights", "evaluation integrity", "Dataset usage rights have explicit evidence", 5, rights_ok, expected="verified rights; commercial_use=false", observed=f"status={dataset.get('license_status')}; commercial_use={dataset.get('commercial_use')}"),
        CheckResult("evaluation.coverage", "evaluation integrity", "Every faithful run has finite MAE/RMSE and 182 predictions", 10, evaluation_ok, True, expected="finite metrics and 182 predictions per run", observed=f"complete={len(complete_runs)}/{len(faithful_runs)}"),
        CheckResult("reproduction.paper_gap", "paper gap", "Three-seed median MAE and RMSE are within 5% of the paper", 20, paper_gap_ok, True, expected="MAE 59.7 and RMSE 95.7 within 5%", observed=f"median_mae={median_mae:.3f} gap={mae_gap:.3%}; median_rmse={median_rmse:.3f} gap={rmse_gap:.3%}"),
        CheckResult("reproduction.seeds", "seed stability", "All three frozen seeds completed", 10, seeds_ok, True, expected=str(sorted(EXPECTED_SEEDS)), observed=str(sorted(faithful_seeds))),
        CheckResult("reproduction.stability", "seed stability", "MAE coefficient of variation is at most 10%", 10, stability <= 0.10, expected="<=10%", observed=f"{stability:.3%}"),
        CheckResult("evaluation.test_isolation", "evaluation integrity", "Clean lane selects checkpoints without the test set", 10, clean_ok, True, expected="train-only validation; one final test per seed", observed=str(clean)),
        CheckResult("artifacts.bundle", "traceability", "Config, predictions, checkpoint hashes and environment are hash-verified", 5, artifacts_ok, True, expected="four path/SHA-256 records matching persisted files", observed=artifact_observed),
        CheckResult("runtime.measurement", "runtime", "GPU inference time is finite and recorded", 5, runtime_ok, expected="GPU and positive median seconds/image", observed=str(runtime)),
    ]


def run_stage3_gate(*, evidence_path: str | Path, output_dir: str | Path) -> StageReport:
    evidence_path = Path(evidence_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    report = score_stage(
        stage_id="stage-3",
        stage_name="DM-Count official reproduction",
        threshold=85,
        checks=build_stage3_checks(evidence, artifact_root=evidence_path.parent),
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
