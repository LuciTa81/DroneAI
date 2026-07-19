"""Build a hash-bound input package for the Round 1 dual-scenario report."""

from __future__ import annotations

import json
import math
import shutil
from pathlib import Path
from typing import Any

from .evaluation_artifacts import write_json
from .integrity import is_sha256, sha256_file


MODEL_IDS = ("steerer", "dm-count", "pet", "mpcount", "apgcc", "csrnet")
SCENARIO_IDS = ("moderate", "high_density")
ARTIFACT_HASH_FIELDS = {
    "source_image": "source_sha256",
    "annotation": "annotation_sha256",
    "checkpoint": "checkpoint_sha256",
    "rights_decision": "rights_decision_sha256",
    "split_manifest": "split_manifest_sha256",
    "environment": "environment_sha256",
}


def _read_object(path: str | Path, *, label: str) -> dict[str, Any]:
    target = Path(path)
    if not target.is_file():
        raise ValueError(f"{label} is missing: {target}")
    try:
        payload = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is not valid JSON: {target}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must be a JSON object: {target}")
    return payload


def _finite(value: object, *, label: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be finite") from exc
    if not math.isfinite(number):
        raise ValueError(f"{label} must be finite")
    return number


def _validate_config(config: dict[str, Any]) -> None:
    if config.get("schema_version") != 1:
        raise ValueError("scenario config schema_version must be 1")
    if tuple(config.get("model_order", ())) != MODEL_IDS:
        raise ValueError("scenario config model order is not frozen")
    scenarios = config.get("scenarios")
    if not isinstance(scenarios, dict) or tuple(scenarios) != SCENARIO_IDS:
        raise ValueError("scenario config must contain the two frozen scenarios")
    exact = {
        "moderate": ("img_0775", 195.0),
        "high_density": ("img_0221", 1762.0),
    }
    for scenario_id, (sample_id, count) in exact.items():
        item = scenarios.get(scenario_id)
        if not isinstance(item, dict) or (
            item.get("sample_id"), _finite(item.get("ground_truth_count"), label="ground truth")
        ) != (sample_id, count):
            raise ValueError(f"scenario config is not frozen: {scenario_id}")
    result_paths = config.get("result_paths")
    if not isinstance(result_paths, dict) or set(result_paths) != set(MODEL_IDS):
        raise ValueError("scenario result path matrix is incomplete")
    for model_id in MODEL_IDS:
        row = result_paths[model_id]
        if not isinstance(row, dict) or set(row) != set(SCENARIO_IDS):
            raise ValueError(f"scenario result paths are incomplete: {model_id}")
        if not all(isinstance(row[key], str) and row[key].strip() for key in SCENARIO_IDS):
            raise ValueError(f"scenario result path is invalid: {model_id}")


def _comparison_models(comparison: dict[str, Any], config: dict[str, Any]) -> dict[str, dict[str, Any]]:
    dataset = comparison.get("dataset")
    if not isinstance(dataset, dict) or (
        dataset.get("dataset_id"),
        dataset.get("split_id"),
        dataset.get("split_role"),
        dataset.get("expected_samples"),
    ) != (
        config.get("dataset_id"),
        config.get("split_id"),
        config.get("split_role"),
        config.get("expected_samples"),
    ):
        raise ValueError("comparison does not use the frozen validation split")
    raw_models = comparison.get("models")
    if not isinstance(raw_models, list):
        raise ValueError("comparison models must be a list")
    models = {
        str(item.get("model_id")): item
        for item in raw_models
        if isinstance(item, dict) and isinstance(item.get("model_id"), str)
    }
    if set(models) != set(MODEL_IDS) or len(raw_models) != len(MODEL_IDS):
        raise ValueError("comparison does not contain the frozen six models")
    return models


def _resolve_artifact(reference: object, result_dir: Path, *, label: str) -> tuple[Path, str]:
    if not isinstance(reference, dict):
        raise ValueError(f"artifact reference is missing: {label}")
    raw_path = reference.get("path")
    expected = reference.get("sha256")
    if not isinstance(raw_path, str) or not raw_path.strip() or not is_sha256(expected):
        raise ValueError(f"artifact reference is invalid: {label}")
    target = Path(raw_path)
    if not target.is_absolute():
        target = result_dir / target
    if not target.is_file():
        raise ValueError(f"artifact is missing: {label}: {target}")
    observed = sha256_file(target)
    if observed != str(expected).lower():
        if label == "panel":
            raise ValueError(f"panel SHA-256 mismatch: {target}")
        raise ValueError(f"artifact SHA-256 mismatch: {label}: {target}")
    return target, observed


def _validated_result(
    result_dir: Path,
    scenario_id: str,
    scenario: dict[str, Any],
    model_id: str,
    comparison_model: dict[str, Any],
    config: dict[str, Any],
) -> tuple[dict[str, Any], Path]:
    result_path = result_dir / "result.json"
    result = _read_object(result_path, label=f"{model_id}/{scenario_id} result")
    if result.get("schema_version") != 1 or result.get("status") != "PASS":
        raise ValueError(f"scenario result is not PASS: {model_id}/{scenario_id}")
    if result.get("gate") != "one_sample" or result.get("failure_state") is not None:
        raise ValueError(f"scenario result gate failed: {model_id}/{scenario_id}")
    if result.get("evaluation_scope") != "research_comparison_only":
        raise ValueError(f"scenario result scope is invalid: {model_id}/{scenario_id}")
    expected_identity = (
        scenario.get("sample_id"),
        config.get("dataset_id"),
        config.get("split_id"),
        config.get("split_role"),
    )
    observed_identity = (
        result.get("sample_id"),
        result.get("dataset_id"),
        result.get("split_id"),
        result.get("split_role"),
    )
    if observed_identity != expected_identity:
        raise ValueError(f"scenario result identity mismatch: {model_id}/{scenario_id}")
    gt = _finite(result.get("ground_truth_count"), label="ground_truth_count")
    if gt != float(scenario["ground_truth_count"]):
        raise ValueError(f"scenario ground truth mismatch: {model_id}/{scenario_id}")
    pred = _finite(result.get("predicted_count"), label="predicted_count")
    record = result.get("record")
    if not isinstance(record, dict) or record.get("failure_state") is not None:
        raise ValueError(f"scenario metric record failed: {model_id}/{scenario_id}")
    if (
        record.get("sample_id"),
        _finite(record.get("ground_truth_count"), label="record ground_truth_count"),
        _finite(record.get("predicted_count"), label="record predicted_count"),
    ) != (scenario["sample_id"], gt, pred):
        raise ValueError(f"scenario metric record identity mismatch: {model_id}/{scenario_id}")
    for field in (
        "absolute_error",
        "normalized_error",
        "latency_ms",
        "peak_vram_mb",
        "spatial_metric_value",
    ):
        _finite(record.get(field), label=field)
    if not isinstance(record.get("spatial_metric_name"), str) or not record["spatial_metric_name"]:
        raise ValueError("spatial_metric_name must be present")

    rights_scope = result.get("rights_scope", comparison_model.get("rights_scope"))
    if not isinstance(rights_scope, str) or not rights_scope.strip():
        raise ValueError(f"rights scope is missing: {model_id}/{scenario_id}")
    artifacts = result.get("artifacts")
    if not isinstance(artifacts, dict):
        raise ValueError(f"artifact bindings are missing: {model_id}/{scenario_id}")
    verified: dict[str, dict[str, str]] = {}
    for artifact_id, top_level_hash in ARTIFACT_HASH_FIELDS.items():
        path, observed_hash = _resolve_artifact(
            artifacts.get(artifact_id), result_dir, label=artifact_id
        )
        if result.get(top_level_hash) != observed_hash:
            raise ValueError(f"declared hash mismatch: {top_level_hash}")
        verified[artifact_id] = {"path": path.as_posix(), "sha256": observed_hash}

    panel = result.get("panel")
    if not isinstance(panel, dict):
        raise ValueError(f"panel declaration is missing: {model_id}/{scenario_id}")
    panel_path, panel_hash = _resolve_artifact(artifacts.get("panel"), result_dir, label="panel")
    declared_panel = Path(str(panel.get("path", "")))
    if not declared_panel.is_absolute():
        declared_panel = result_dir / declared_panel
    if declared_panel.resolve() != panel_path.resolve() or panel.get("sha256") != panel_hash:
        raise ValueError(f"panel SHA-256 mismatch: {panel_path}")

    row = {
        "model_id": model_id,
        "sample_id": result["sample_id"],
        "ground_truth_count": gt,
        "predicted_count": pred,
        "absolute_error": float(record["absolute_error"]),
        "normalized_error": float(record["normalized_error"]),
        "latency_ms": float(record["latency_ms"]),
        "peak_vram_mb": float(record["peak_vram_mb"]),
        "spatial_metric_name": record["spatial_metric_name"],
        "spatial_metric_value": float(record["spatial_metric_value"]),
        "rights_scope": rights_scope,
        "comparison_rights_scope": comparison_model.get("rights_scope"),
        "source_sha256": verified["source_image"]["sha256"],
        "annotation_sha256": verified["annotation"]["sha256"],
        "checkpoint_sha256": verified["checkpoint"]["sha256"],
        "rights_decision_sha256": verified["rights_decision"]["sha256"],
        "split_manifest_sha256": verified["split_manifest"]["sha256"],
        "environment_sha256": verified["environment"]["sha256"],
        "panel_sha256": panel_hash,
        "result_sha256": sha256_file(result_path),
    }
    return row, panel_path


def build_report_scenario_package(
    comparison_path: str | Path,
    results_root: str | Path,
    config_path: str | Path,
    output_dir: str | Path,
) -> Path:
    """Validate twelve one-sample results and copy their panels atomically."""

    comparison_source = Path(comparison_path).resolve()
    root = Path(results_root).resolve()
    config = _read_object(config_path, label="scenario config")
    comparison = _read_object(comparison_source, label="comparison")
    _validate_config(config)
    comparison_models = _comparison_models(comparison, config)

    output = Path(output_dir).resolve()
    temporary = output.parent / f".{output.name}.tmp"
    if output.exists() or temporary.exists():
        raise FileExistsError("scenario package output and temporary paths must not exist")

    scenarios: dict[str, Any] = {}
    pending_panels: list[tuple[Path, str, str, dict[str, Any]]] = []
    for scenario_id in SCENARIO_IDS:
        scenario = config["scenarios"][scenario_id]
        model_rows: dict[str, Any] = {}
        for model_id in MODEL_IDS:
            result_dir = root / config["result_paths"][model_id][scenario_id]
            row, panel_path = _validated_result(
                result_dir,
                scenario_id,
                scenario,
                model_id,
                comparison_models[model_id],
                config,
            )
            packaged_path = f"panels/{model_id}/{scenario_id}.png"
            row["packaged_panel_path"] = packaged_path
            model_rows[model_id] = row
            pending_panels.append((panel_path, packaged_path, row["panel_sha256"], row))
        scenarios[scenario_id] = {
            "sample_id": scenario["sample_id"],
            "ground_truth_count": float(scenario["ground_truth_count"]),
            "label": scenario["label"],
            "models": model_rows,
        }

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary.mkdir()
    assets: list[dict[str, str]] = []
    try:
        for source, packaged_path, expected_hash, _row in pending_panels:
            destination = temporary / packaged_path
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, destination)
            observed = sha256_file(destination)
            if observed != expected_hash:
                raise ValueError(f"copied panel SHA-256 mismatch: {packaged_path}")
            assets.append({"path": packaged_path, "sha256": observed})
        manifest = {
            "schema_version": 2,
            "status": "PASS_RESEARCH_ONLY",
            "comparison_sha256": sha256_file(comparison_source),
            "dataset": comparison["dataset"],
            "model_order": list(MODEL_IDS),
            "scenario_order": list(SCENARIO_IDS),
            "scenarios": scenarios,
            "assets": assets,
        }
        write_json(temporary / "assets-manifest.json", manifest)
        temporary.rename(output)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return output


__all__ = ["MODEL_IDS", "SCENARIO_IDS", "build_report_scenario_package"]
