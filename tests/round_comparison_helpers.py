from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

from droneai.integrity import sha256_file
from droneai.round_scorecard import ScorecardConfig, load_scorecard


SCORECARD_PATH = Path("configs/evaluation/round1_cctv_scorecard.json")


def write_json(path: Path, payload: object) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return sha256_file(path)


def write_predictions(path: Path, *, family: str, changed_sample: str | None = None) -> None:
    fields = [
        "sample_id",
        "ground_truth_count",
        "density_band",
        "condition_source_partition",
        "normalized_error",
        "zone_mae",
        "game_l1",
        "density_psnr",
        "density_ssim",
        "localization_precision",
        "localization_recall",
        "localization_f1",
        "localization_mean_distance",
        "latency_ms",
        "peak_vram_mb",
        "output_type",
        "failure_state",
        "signed_error",
        "absolute_error",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    output_type = {
        "density": "density",
        "points": "points",
        "density_and_points": "hybrid",
    }[family]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for index in range(36):
            sample_id = changed_sample if index == 0 and changed_sample else f"img_{index:04d}"
            band = ("low", "medium", "high")[index // 12]
            has_density = family in {"density", "density_and_points"}
            has_points = family in {"points", "density_and_points"}
            writer.writerow(
                {
                    "sample_id": sample_id,
                    "ground_truth_count": 100 + index,
                    "density_band": band,
                    "condition_source_partition": "official_validation",
                    "normalized_error": 0.1,
                    "zone_mae": 10.0,
                    "game_l1": 20.0 if has_density else "",
                    "density_psnr": 25.0 if has_density else "",
                    "density_ssim": 0.8 if has_density else "",
                    "localization_precision": 0.75 if has_points else "",
                    "localization_recall": 0.65 if has_points else "",
                    "localization_f1": 0.7 if has_points else "",
                    "localization_mean_distance": 6.0 if has_points else "",
                    "latency_ms": 70.0,
                    "peak_vram_mb": 4096.0,
                    "output_type": output_type,
                    "failure_state": "",
                    "signed_error": 1.0,
                    "absolute_error": 1.0,
                }
            )


def make_run(
    results_root: Path,
    *,
    model_id: str,
    family: str,
    rights_scope: str,
    checkpoint_status: str = "VERIFIED_DISJOINT",
) -> dict[str, object]:
    run = results_root / model_id / "run"
    write_predictions(run / "predictions.csv", family=family)
    panels = [f"figures/panel-{index:02d}.png" for index in range(12)]
    for panel_path in panels:
        panel = run / panel_path
        panel.parent.mkdir(parents=True, exist_ok=True)
        panel.write_bytes(b"png fixture")
    write_json(
        run / "selection_manifest.json",
        {"schema_version": 1, "selected_count": 12, "shortfall": 0, "panel_paths": panels},
    )
    primary_name = "localization_f1" if family == "points" else "game_l1"
    primary_value = 0.7 if family == "points" else 20.0
    metrics = {
        "expected_samples": 36,
        "recorded_samples": 36,
        "successful_samples": 36,
        "explicit_failures": 0,
        "mae": 10.0,
        "rmse": 12.0,
        "signed_bias": 1.0,
        "median_latency_ms": 70.0,
        "throughput_fps_batch1": 14.2857,
        "peak_vram_mb": 4096.0,
        "band_signed_bias": {"low": 1.0, "medium": 1.0, "high": 1.0},
        "spatial_metric_name": primary_name,
        "spatial_mean": primary_value,
    }
    metrics_hash = write_json(run / "metrics.json", metrics)
    rights_decision = {
        "status": rights_scope,
        "component_ids": {
            "code": f"{model_id}-code",
            "dataset": "ucf-qnrf-kaggle-apache",
            "pretrained_weights": f"{model_id}-checkpoint",
            "derived_weights": "not-created",
            "deployment": "pending",
        },
        "ambiguous_components": [
            "pretrained_weights",
            "derived_weights",
            "deployment",
        ],
        "restricted_components": [],
    }
    write_json(run / "rights-decision.json", rights_decision)
    artifact_paths = [
        run / "predictions.csv",
        run / "selection_manifest.json",
        run / "metrics.json",
        run / "rights-decision.json",
        *(run / panel for panel in panels),
    ]
    evidence_hash = write_json(
        run / "evidence-manifest.json",
        {
            "schema_version": 1,
            "artifacts": [
                {
                    "path": path.relative_to(run).as_posix(),
                    "sha256": sha256_file(path),
                }
                for path in artifact_paths
            ],
        },
    )
    score_hash = write_json(
        run / "score.json",
        {
            "schema_version": 2,
            "status": "PASS_RESEARCH_ONLY",
            "decision_scope": "research_only",
            "score": 90.0,
            "checks": [
                {
                    "check_id": "provenance.bundle",
                    "evidence": f"evidence-manifest.json@{evidence_hash}",
                }
            ],
        },
    )
    return {
        "model_id": model_id,
        "family": family,
        "queue_state": "completed",
        "rights_scope": rights_scope,
        "dataset_id": "ucf-qnrf-kaggle-apache",
        "split_id": "dm-count-upstream-qnrf-val-smoke-36-v1",
        "split_role": "validation",
        "expected_samples": 36,
        "checkpoint_training_split_status": checkpoint_status,
        "comparison_scope": "compatibility_smoke",
        "checkpoint_split_evidence": "fixture checkpoint split evidence",
        "gate_order": ["benchmark"],
        "accepted_evidence": [
            {
                "gate": "benchmark",
                "artifact_kind": "score",
                "path": f"{model_id}/run/score.json",
                "sha256": score_hash,
            },
            {
                "gate": "benchmark",
                "artifact_kind": "metrics",
                "path": f"{model_id}/run/metrics.json",
                "sha256": metrics_hash,
            },
        ],
    }


def make_round_fixture(tmp_path: Path) -> tuple[Path, Path, ScorecardConfig]:
    config = load_scorecard(SCORECARD_PATH)
    results = tmp_path / "results"
    families = (
        "density",
        "density_and_points",
        "points",
        "density",
        "points",
        "density",
    )
    models = [
        make_run(
            results,
            model_id=model_id,
            family=family,
            rights_scope="PASS_COMMERCIAL_CANDIDATE",
        )
        for model_id, family in zip(config.required_model_ids, families, strict=True)
    ]
    models[-1]["queue_state"] = "active"
    queue = tmp_path / "queue.json"
    queue.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "round_id": config.round_id,
                "active_model": "csrnet",
                "models": models,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return queue, results, config


def reanchor_run(queue: Path, results: Path, model_id: str) -> None:
    run = results / model_id / "run"
    evidence_path = run / "evidence-manifest.json"
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    for artifact in evidence["artifacts"]:
        target = run / artifact["path"]
        artifact["sha256"] = sha256_file(target)
    evidence_hash = write_json(evidence_path, evidence)
    score_path = run / "score.json"
    score = json.loads(score_path.read_text(encoding="utf-8"))
    score["checks"][0]["evidence"] = f"evidence-manifest.json@{evidence_hash}"
    score_hash = write_json(score_path, score)
    payload = json.loads(queue.read_text(encoding="utf-8"))
    model = next(item for item in payload["models"] if item["model_id"] == model_id)
    accepted = next(
        item
        for item in model["accepted_evidence"]
        if item["artifact_kind"] == "score"
    )
    accepted["sha256"] = score_hash
    queue.write_text(json.dumps(payload, indent=2), encoding="utf-8")
