from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from droneai.integrity import sha256_file
from droneai.report_scenario_package import (
    MODEL_IDS,
    build_report_scenario_package,
)
from scripts.build_round1_scenario_inputs import main


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/reporting/round1_dual_scenarios.json"
SCENARIOS = {
    "moderate": {"sample_id": "img_0775", "ground_truth_count": 195.0},
    "high_density": {"sample_id": "img_0221", "ground_truth_count": 1762.0},
}


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _artifact(path: Path, content: bytes) -> dict[str, str]:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return {"path": path.as_posix(), "sha256": sha256_file(path)}


def _fixture(tmp_path: Path) -> tuple[Path, Path, Path]:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    results_root = tmp_path / "results"
    comparison = tmp_path / "comparison.json"
    _write_json(
        comparison,
        {
            "schema_version": 1,
            "dataset": {
                "dataset_id": config["dataset_id"],
                "split_id": config["split_id"],
                "split_role": config["split_role"],
                "expected_samples": config["expected_samples"],
            },
            "models": [
                {
                    "model_id": model_id,
                    "rights_scope": "PASS_RESEARCH_ONLY"
                    if model_id == "pet"
                    else "PASS_COMMERCIAL_CANDIDATE",
                }
                for model_id in MODEL_IDS
            ],
        },
    )
    shared = tmp_path / "artifacts"
    for model_id in MODEL_IDS:
        for scenario_id, spec in SCENARIOS.items():
            result_dir = results_root / config["result_paths"][model_id][scenario_id]
            refs = {
                "source_image": _artifact(
                    shared / scenario_id / "image.jpg", f"image-{scenario_id}".encode()
                ),
                "annotation": _artifact(
                    shared / scenario_id / "annotation.mat",
                    f"annotation-{scenario_id}".encode(),
                ),
                "checkpoint": _artifact(
                    shared / model_id / "checkpoint.pth", f"checkpoint-{model_id}".encode()
                ),
                "rights_decision": _artifact(
                    shared / model_id / "rights.json", f"rights-{model_id}".encode()
                ),
                "split_manifest": _artifact(
                    result_dir / "split-source-manifest.json",
                    f"split-{model_id}-{scenario_id}".encode(),
                ),
                "environment": _artifact(
                    result_dir / "environment-summary.json",
                    f"environment-{model_id}-{scenario_id}".encode(),
                ),
                "panel": _artifact(
                    result_dir / "one-sample-panel.png",
                    b"\x89PNG\r\n\x1a\n" + f"{model_id}-{scenario_id}".encode(),
                ),
            }
            gt = float(spec["ground_truth_count"])
            pred = gt + 5.0
            _write_json(
                result_dir / "result.json",
                {
                    "schema_version": 1,
                    "status": "PASS",
                    "gate": "one_sample",
                    "evaluation_scope": "research_comparison_only",
                    "sample_id": spec["sample_id"],
                    "dataset_id": config["dataset_id"],
                    "split_id": config["split_id"],
                    "split_role": config["split_role"],
                    "source_sha256": refs["source_image"]["sha256"],
                    "annotation_sha256": refs["annotation"]["sha256"],
                    "ground_truth_count": gt,
                    "predicted_count": pred,
                    "failure_state": None,
                    "checkpoint_sha256": refs["checkpoint"]["sha256"],
                    "rights_decision_sha256": refs["rights_decision"]["sha256"],
                    "split_manifest_sha256": refs["split_manifest"]["sha256"],
                    "environment_sha256": refs["environment"]["sha256"],
                    "rights_scope": "PASS_RESEARCH_ONLY"
                    if model_id == "pet"
                    else "PASS_COMMERCIAL_CANDIDATE",
                    "panel": {
                        "path": "one-sample-panel.png",
                        "sha256": refs["panel"]["sha256"],
                    },
                    "artifacts": refs,
                    "record": {
                        "sample_id": spec["sample_id"],
                        "ground_truth_count": gt,
                        "predicted_count": pred,
                        "absolute_error": 5.0,
                        "normalized_error": 5.0 / gt,
                        "latency_ms": 10.0,
                        "peak_vram_mb": 100.0,
                        "spatial_metric_name": "game_l1",
                        "spatial_metric_value": 2.0,
                        "failure_state": None,
                    },
                },
            )
    return comparison, results_root, CONFIG


def test_dual_scenario_config_is_exact() -> None:
    payload = json.loads(CONFIG.read_text(encoding="utf-8"))
    assert {
        key: {
            "sample_id": value["sample_id"],
            "ground_truth_count": value["ground_truth_count"],
        }
        for key, value in payload["scenarios"].items()
    } == SCENARIOS
    assert tuple(payload["model_order"]) == MODEL_IDS
    assert payload["dataset_id"] == "ucf-qnrf-kaggle-apache"
    assert payload["split_role"] == "validation"


def test_package_contains_six_models_for_both_scenarios(tmp_path: Path) -> None:
    comparison, results, config = _fixture(tmp_path)
    output = build_report_scenario_package(comparison, results, config, tmp_path / "package")
    manifest = json.loads((output / "assets-manifest.json").read_text(encoding="utf-8"))
    assert manifest["schema_version"] == 2
    assert set(manifest["scenarios"]) == set(SCENARIOS)
    for scenario in manifest["scenarios"].values():
        assert set(scenario["models"]) == set(MODEL_IDS)
        assert all(len(row["source_sha256"]) == 64 for row in scenario["models"].values())
        assert all((output / row["packaged_panel_path"]).is_file() for row in scenario["models"].values())
    assert len(manifest["assets"]) == 12


def test_package_rejects_changed_panel_hash(tmp_path: Path) -> None:
    comparison, results, config = _fixture(tmp_path)
    panel = results / "round1-report-scenarios-v1/steerer/img_0775/one-sample-panel.png"
    panel.write_bytes(b"changed")
    with pytest.raises(ValueError, match="panel SHA-256 mismatch"):
        build_report_scenario_package(comparison, results, config, tmp_path / "package")


def test_package_rejects_nonfinite_metric_before_copy(tmp_path: Path) -> None:
    comparison, results, config = _fixture(tmp_path)
    result_path = results / "round1-report-scenarios-v1/csrnet/img_0221/result.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    result["record"]["latency_ms"] = math.inf
    _write_json(result_path, result)
    with pytest.raises(ValueError, match="finite"):
        build_report_scenario_package(comparison, results, config, tmp_path / "package")
    assert not (tmp_path / "package").exists()


def test_cli_returns_structured_failure(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    code = main(
        [
            "--comparison", str(tmp_path / "missing.json"),
            "--results-root", str(tmp_path / "results"),
            "--config", str(CONFIG),
            "--output-dir", str(tmp_path / "package"),
        ]
    )
    error = json.loads(capsys.readouterr().err)
    assert code == 2
    assert error["status"] == "FAIL"

