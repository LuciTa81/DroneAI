from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from droneai.evaluation_artifacts import artifact_reference
from droneai.integrity import sha256_file
from droneai.round2_comparison import (
    Round2Run,
    build_round2_comparison,
    write_round2_comparison,
)


MODELS = ("steerer", "dm-count", "mpcount")
POINT_MODELS = ("steerer", "pet", "apgcc")
DATASETS = (
    ("ucf-qnrf-kaggle-apache", 2, "PASS_COMMERCIAL_CANDIDATE"),
    ("jhu-crowd-plus-v2", 2, "PASS_RESEARCH_ONLY"),
    ("up-count-v1", 2, "PASS_RESEARCH_ONLY"),
)


def _write_json(path: Path, payload: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _make_run(
    root: Path,
    *,
    model_id: str,
    dataset_id: str,
    samples: int,
    rights_scope: str,
    effective_rights_scope: str | None = None,
    spatial_metric_name: str = "game_l1",
) -> Round2Run:
    run = root / model_id / dataset_id
    run.mkdir(parents=True)
    shared = _write_json(
        run / "round2-shared-manifest.json",
        {
            "schema_version": 1,
            "round_id": "round-2-fixture",
            "sample_manifest_sha256": "a" * 64,
        },
    )
    _write_json(run / "round2-dataset-manifest.json", {"dataset_id": dataset_id})
    predictions = run / "predictions.csv"
    with predictions.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("sample_id", "ground_truth_count", "predicted_count"),
            lineterminator="\n",
        )
        writer.writeheader()
        for index in range(samples):
            writer.writerow(
                {
                    "sample_id": f"sample-{index}",
                    "ground_truth_count": 100 + index,
                    "predicted_count": 105 + index,
                }
            )
    metrics = _write_json(
        run / "metrics.json",
        {
            "expected_samples": samples,
            "recorded_samples": samples,
            "successful_samples": samples,
            "explicit_failures": 0,
            "mae": 5.0,
            "rmse": 5.0,
            "signed_bias": 5.0,
            "mape_reference": 5.0,
            "median_latency_ms": 70.0,
            "throughput_fps_batch1": 14.28,
            "peak_vram_mb": 4096.0,
            "spatial_metric_name": spatial_metric_name,
            "spatial_mean": (
                0.5 if spatial_metric_name == "localization_f1" else 10.0
            ),
        },
    )
    score = _write_json(
        run / "score.json", {"status": "PASS", "score": 100.0}
    )
    selection = _write_json(
        run / "selection_manifest.json",
        {"selected_count": 12, "shortfall": 0},
    )
    environment = _write_json(
        run / "environment-summary.json",
        {"git_commit": "b" * 40, "git_dirty": False},
    )
    rights = _write_json(
        run / "rights-decision.json",
        {
            "rights_scope": effective_rights_scope or rights_scope,
            "dataset_rights_scope": rights_scope,
            "dataset_id": dataset_id,
        },
    )
    sample_manifest = _write_json(
        run / "sample-manifest.json",
        {
            "dataset_id": dataset_id,
            "expected_samples": samples,
            "comparison_scope": "research_reference_only",
            "ranking_eligible": False,
        },
    )
    artifacts = (
        shared,
        run / "round2-dataset-manifest.json",
        predictions,
        metrics,
        selection,
        environment,
        rights,
        sample_manifest,
    )
    _write_json(
        run / "evidence-manifest.json",
        {
            "schema_version": 1,
            "protocol": {
                "dataset_id": dataset_id,
                "expected_samples": samples,
                "comparison_scope": "research_reference_only",
            },
            "artifacts": [artifact_reference(path, base_dir=run) for path in artifacts],
        },
    )
    return Round2Run(model_id, dataset_id, samples, rights_scope, run)


@pytest.fixture
def nine_runs(tmp_path: Path) -> tuple[Round2Run, ...]:
    return tuple(
        _make_run(
            tmp_path,
            model_id=model_id,
            dataset_id=dataset_id,
            samples=samples,
            rights_scope=rights_scope,
        )
        for model_id in MODELS
        for dataset_id, samples, rights_scope in DATASETS
    )


@pytest.fixture
def point_nine_runs(tmp_path: Path) -> tuple[Round2Run, ...]:
    return tuple(
        _make_run(
            tmp_path,
            model_id=model_id,
            dataset_id=dataset_id,
            samples=samples,
            rights_scope=rights_scope,
            effective_rights_scope=(
                "PASS_RESEARCH_ONLY"
                if model_id == "pet"
                else rights_scope
            ),
            spatial_metric_name=(
                "game_l1"
                if model_id == "steerer"
                else "localization_f1"
            ),
        )
        for model_id in POINT_MODELS
        for dataset_id, samples, rights_scope in DATASETS
    )


def test_comparison_refuses_missing_or_mixed_manifest_hashes(
    nine_runs: tuple[Round2Run, ...],
) -> None:
    changed = nine_runs[-1].path / "round2-shared-manifest.json"
    payload = json.loads(changed.read_text(encoding="utf-8"))
    payload["sample_manifest_sha256"] = "c" * 64
    _write_json(changed, payload)
    evidence = nine_runs[-1].path / "evidence-manifest.json"
    evidence_payload = json.loads(evidence.read_text(encoding="utf-8"))
    for reference in evidence_payload["artifacts"]:
        if reference["path"] == "round2-shared-manifest.json":
            reference["sha256"] = sha256_file(changed)
            reference["size_bytes"] = changed.stat().st_size
    _write_json(evidence, evidence_payload)

    with pytest.raises(ValueError, match="shared manifest"):
        build_round2_comparison(nine_runs)


def test_comparison_is_research_only_and_dataset_wise(
    nine_runs: tuple[Round2Run, ...],
) -> None:
    payload = build_round2_comparison(nine_runs)

    assert payload["rights_scope"] == "PASS_RESEARCH_ONLY"
    assert payload["comparison_scope"] == "research_reference_only"
    assert payload["ranking_eligible"] is False
    assert "pooled_commercial_score" not in payload
    assert "pooled_metrics" not in payload
    assert set(payload["datasets"]) == {
        "ucf-qnrf-kaggle-apache",
        "jhu-crowd-plus-v2",
        "up-count-v1",
    }
    assert set(payload["datasets"]["jhu-crowd-plus-v2"]["models"]) == set(MODELS)


def test_comparison_rehashes_referenced_artifacts(
    nine_runs: tuple[Round2Run, ...],
) -> None:
    target = nine_runs[0].path / "metrics.json"
    target.write_text(target.read_text(encoding="utf-8") + " ", encoding="utf-8")

    with pytest.raises(ValueError, match="artifact verification"):
        build_round2_comparison(nine_runs)


def test_comparison_write_is_atomic(
    nine_runs: tuple[Round2Run, ...], tmp_path: Path
) -> None:
    output = tmp_path / "comparison"

    paths = write_round2_comparison(output, nine_runs)

    assert paths["json"].is_file()
    assert paths["markdown"].is_file()
    assert not list(output.glob("*.tmp"))
    with pytest.raises(FileExistsError):
        write_round2_comparison(output, nine_runs)


def test_point_comparison_preserves_model_order_and_spatial_semantics(
    point_nine_runs: tuple[Round2Run, ...],
) -> None:
    payload = build_round2_comparison(
        point_nine_runs,
        model_order=POINT_MODELS,
    )

    assert payload["model_order"] == list(POINT_MODELS)
    assert payload["lanes"] == 9
    assert payload["rights_scope"] == "PASS_RESEARCH_ONLY"
    assert payload["models"]["pet"]["native_density_map"] is False
    assert payload["models"]["apgcc"]["native_density_map"] is False
    ucf_models = payload["datasets"]["ucf-qnrf-kaggle-apache"]["models"]
    assert ucf_models["steerer"]["spatial_metric_name"] == "game_l1"
    assert ucf_models["pet"]["spatial_metric_name"] == "localization_f1"
    assert ucf_models["apgcc"]["spatial_metric_name"] == "localization_f1"


def test_point_comparison_rejects_native_density_claim(
    point_nine_runs: tuple[Round2Run, ...],
) -> None:
    target_run = next(run for run in point_nine_runs if run.model_id == "pet")
    metrics_path = target_run.path / "metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    metrics["density_psnr"] = 30.0
    _write_json(metrics_path, metrics)
    evidence_path = target_run.path / "evidence-manifest.json"
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    for reference in evidence["artifacts"]:
        if reference["path"] == "metrics.json":
            reference["sha256"] = sha256_file(metrics_path)
            reference["size_bytes"] = metrics_path.stat().st_size
    _write_json(evidence_path, evidence)

    with pytest.raises(ValueError, match="native density"):
        build_round2_comparison(
            point_nine_runs,
            model_order=POINT_MODELS,
        )


def test_point_comparison_rejects_pet_rights_promotion(
    point_nine_runs: tuple[Round2Run, ...],
) -> None:
    target_run = next(
        run
        for run in point_nine_runs
        if run.model_id == "pet"
        and run.dataset_id == "ucf-qnrf-kaggle-apache"
    )
    rights_path = target_run.path / "rights-decision.json"
    rights = json.loads(rights_path.read_text(encoding="utf-8"))
    rights["rights_scope"] = "PASS_COMMERCIAL_CANDIDATE"
    _write_json(rights_path, rights)
    evidence_path = target_run.path / "evidence-manifest.json"
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    for reference in evidence["artifacts"]:
        if reference["path"] == "rights-decision.json":
            reference["sha256"] = sha256_file(rights_path)
            reference["size_bytes"] = rights_path.stat().st_size
    _write_json(evidence_path, evidence)

    with pytest.raises(ValueError, match="PET.*research-only"):
        build_round2_comparison(
            point_nine_runs,
            model_order=POINT_MODELS,
        )
