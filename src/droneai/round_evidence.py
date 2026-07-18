from __future__ import annotations

import csv
import hashlib
import json
import math
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from droneai.integrity import is_sha256, sha256_file, verify_artifact_reference
from droneai.model_workflow import load_model_queue
from droneai.round_scorecard import ScorecardConfig, SUPPORTED_FAMILIES


@dataclass(frozen=True)
class ModelRunEvidence:
    model_id: str
    family: str
    rights_scope: str
    checkpoint_training_split_status: str
    comparison_scope: str
    run_dir: Path
    run_relative_path: str
    score: dict[str, object]
    metrics: dict[str, object]
    rights_decision: dict[str, object]
    aggregates: dict[str, float | None]
    capabilities: dict[str, bool]
    panels: tuple[str, ...]
    canonical_split_sha256: str
    input_artifacts: tuple[dict[str, str], ...]


def _read_object(path: Path, *, label: str) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid {label}: {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must be a JSON object: {path}")
    return payload


def _safe_target(root: Path, relative_path: str, *, label: str) -> Path:
    candidate = (root / relative_path).resolve()
    if not candidate.is_relative_to(root):
        raise ValueError(f"{label} escapes its evidence root: {relative_path}")
    return candidate


def _accepted(
    model: dict[str, object], kind: str, root: Path
) -> tuple[Path, dict[str, str]]:
    matches = [
        item
        for item in model["accepted_evidence"]
        if item["gate"] == "benchmark" and item["artifact_kind"] == kind
    ]
    if len(matches) != 1:
        raise ValueError(
            f"{model['model_id']} requires one accepted benchmark {kind}"
        )
    reference = matches[0]
    target = _safe_target(root, reference["path"], label="accepted evidence")
    if not target.is_file():
        raise ValueError(f"accepted evidence is missing: {reference['path']}")
    if sha256_file(target) != reference["sha256"]:
        raise ValueError(f"accepted evidence hash mismatch: {reference['path']}")
    return target, reference


def _anchored_evidence_hash(score: dict[str, object]) -> str:
    checks = score.get("checks")
    if not isinstance(checks, list):
        raise ValueError("score checks must be a list")
    anchors: set[str] = set()
    for check in checks:
        if not isinstance(check, dict):
            continue
        for token in str(check.get("evidence", "")).split(";"):
            token = token.strip()
            if token.startswith("evidence-manifest.json@"):
                digest = token.rsplit("@", 1)[1].lower()
                if not is_sha256(digest):
                    raise ValueError("score contains an invalid evidence manifest hash")
                anchors.add(digest)
    if len(anchors) != 1:
        raise ValueError("score must anchor exactly one evidence-manifest SHA-256")
    return anchors.pop()


def _finite(rows: list[dict[str, str]], field: str) -> list[float]:
    values: list[float] = []
    for row in rows:
        raw = row.get(field, "")
        if raw is None or not raw.strip():
            continue
        try:
            value = float(raw)
        except ValueError as exc:
            raise ValueError(f"invalid prediction field: {field}") from exc
        if not math.isfinite(value):
            raise ValueError(f"non-finite prediction field: {field}")
        values.append(value)
    return values


def _required_finite(value: object, *, field: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"required metric is unavailable: {field}") from exc
    if not math.isfinite(number):
        raise ValueError(f"required metric must be finite: {field}")
    return number


def _mean_or_none(rows: list[dict[str, str]], field: str) -> float | None:
    values = _finite(rows, field)
    return statistics.fmean(values) if values else None


def _relative_artifact(
    path: Path, root: Path, digest: str | None = None
) -> dict[str, str]:
    return {
        "path": path.relative_to(root).as_posix(),
        "sha256": digest or sha256_file(path),
    }


def _deduplicate_artifacts(
    artifacts: Iterable[dict[str, str]],
) -> tuple[dict[str, str], ...]:
    by_path: dict[str, str] = {}
    for artifact in artifacts:
        path = artifact["path"]
        digest = artifact["sha256"].lower()
        previous = by_path.setdefault(path, digest)
        if previous != digest:
            raise ValueError(f"conflicting artifact hashes: {path}")
    return tuple(
        {"path": path, "sha256": by_path[path]} for path in sorted(by_path)
    )


def _validate_predictions(
    rows: list[dict[str, str]], *, model_id: str, family: str, expected: int
) -> tuple[str, dict[str, bool]]:
    if len(rows) != expected:
        raise ValueError(f"incomplete prediction accounting: {model_id}")
    if any((row.get("failure_state") or "").strip() for row in rows):
        raise ValueError(f"incomplete prediction accounting: {model_id}")

    required_columns = (
        "sample_id",
        "ground_truth_count",
        "density_band",
        "condition_source_partition",
        "normalized_error",
        "zone_mae",
        "signed_error",
        "absolute_error",
        "latency_ms",
        "peak_vram_mb",
        "output_type",
    )
    if rows and any(column not in rows[0] for column in required_columns):
        raise ValueError(f"prediction schema is incomplete: {model_id}")
    for field in (
        "ground_truth_count",
        "normalized_error",
        "zone_mae",
        "signed_error",
        "absolute_error",
        "latency_ms",
        "peak_vram_mb",
    ):
        if len(_finite(rows, field)) != expected:
            raise ValueError(f"required prediction field is incomplete: {field}")

    sample_ids = [row["sample_id"].strip() for row in rows]
    if any(not sample_id for sample_id in sample_ids) or len(set(sample_ids)) != expected:
        raise ValueError(f"prediction sample IDs must be unique: {model_id}")
    bands = [row["density_band"].strip() for row in rows]
    if {band: bands.count(band) for band in set(bands)} != {
        "low": 12,
        "medium": 12,
        "high": 12,
    }:
        raise ValueError(f"density-band coverage must be 12/12/12: {model_id}")
    if any(
        row["condition_source_partition"].strip() != "official_validation"
        for row in rows
    ):
        raise ValueError(f"prediction source partition is not official validation: {model_id}")

    output_types = {row["output_type"].strip() for row in rows}
    expected_outputs = {
        "density": {"density"},
        "points": {"points"},
        "density_and_points": {"hybrid"},
    }[family]
    if output_types != expected_outputs:
        raise ValueError(f"model output type does not match family: {model_id}")

    density_values = _finite(rows, "game_l1")
    localization_values = _finite(rows, "localization_f1")
    capabilities = {
        "zone_aggregation": len(_finite(rows, "zone_mae")) == expected,
        "density_map": len(density_values) == expected
        and family in {"density", "density_and_points"},
        "point_localization": len(localization_values) == expected
        and family in {"points", "density_and_points"},
        "confidence": len(_finite(rows, "confidence")) == expected,
    }
    if family == "density" and not capabilities["density_map"]:
        raise ValueError(f"density output was not demonstrated: {model_id}")
    if family == "points" and not capabilities["point_localization"]:
        raise ValueError(f"point output was not demonstrated: {model_id}")
    if family == "density_and_points" and not (
        capabilities["density_map"] and capabilities["point_localization"]
    ):
        raise ValueError(f"hybrid outputs were not demonstrated: {model_id}")

    canonical = sorted(
        (
            row["sample_id"].strip(),
            _required_finite(row["ground_truth_count"], field="ground_truth_count"),
            row["density_band"].strip(),
            row["condition_source_partition"].strip(),
        )
        for row in rows
    )
    split_hash = hashlib.sha256(
        json.dumps(
            canonical, separators=(",", ":"), ensure_ascii=False
        ).encode("utf-8")
    ).hexdigest()
    return split_hash, capabilities


def load_verified_round(
    queue_path: str | Path,
    results_root: str | Path,
    config: ScorecardConfig,
) -> tuple[ModelRunEvidence, ...]:
    queue = load_model_queue(queue_path)
    root = Path(results_root).resolve()
    if queue["round_id"] != config.round_id:
        raise ValueError("queue round does not match scorecard")
    if tuple(model["model_id"] for model in queue["models"]) != config.required_model_ids:
        raise ValueError("queue model order does not match frozen scorecard")

    expected_identity = (
        config.dataset_id,
        config.split_id,
        config.split_role,
        config.expected_samples,
    )
    runs: list[ModelRunEvidence] = []
    for model in queue["models"]:
        model_id = str(model["model_id"])
        family = str(model["family"])
        observed_identity = (
            model["dataset_id"],
            model["split_id"],
            model["split_role"],
            model["expected_samples"],
        )
        if observed_identity != expected_identity:
            raise ValueError(f"frozen evaluation identity mismatch: {model_id}")
        if family not in SUPPORTED_FAMILIES:
            raise ValueError(f"unsupported model family: {family}")

        score_path, score_ref = _accepted(model, "score", root)
        metrics_path, metrics_ref = _accepted(model, "metrics", root)
        if score_path.parent != metrics_path.parent:
            raise ValueError(
                f"score and metrics must share one run directory: {model_id}"
            )
        run_dir = score_path.parent
        score = _read_object(score_path, label="accepted score")
        metrics = _read_object(metrics_path, label="accepted metrics")

        evidence_path = run_dir / "evidence-manifest.json"
        if not evidence_path.is_file() or sha256_file(evidence_path) != _anchored_evidence_hash(score):
            raise ValueError(f"evidence manifest hash mismatch: {model_id}")
        evidence = _read_object(evidence_path, label="evidence manifest")
        references = evidence.get("artifacts")
        if not isinstance(references, list):
            raise ValueError(f"evidence manifest artifacts must be a list: {model_id}")
        verified_by_path: dict[str, str] = {}
        result_artifacts: list[dict[str, str]] = [
            _relative_artifact(score_path, root, score_ref["sha256"]),
            _relative_artifact(metrics_path, root, metrics_ref["sha256"]),
            _relative_artifact(evidence_path, root),
        ]
        for reference in references:
            verified, message = verify_artifact_reference(reference, base_dir=run_dir)
            if not verified:
                raise ValueError(message)
            raw_path = str(reference["path"])
            target = _safe_target(run_dir, raw_path, label="manifest artifact")
            relative = target.relative_to(run_dir).as_posix()
            verified_by_path[relative] = str(reference["sha256"]).lower()
            result_artifacts.append(
                _relative_artifact(target, root, str(reference["sha256"]).lower())
            )

        required = {
            "predictions.csv",
            "selection_manifest.json",
            "metrics.json",
            "rights-decision.json",
        }
        if not required.issubset(verified_by_path):
            missing = sorted(required - set(verified_by_path))
            raise ValueError(f"evidence manifest is missing required artifacts: {missing}")

        predictions_path = run_dir / "predictions.csv"
        with predictions_path.open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        split_hash, capabilities = _validate_predictions(
            rows, model_id=model_id, family=family, expected=config.expected_samples
        )

        spatial_name = str(metrics.get("spatial_metric_name", ""))
        if family == "density" and spatial_name != "game_l1":
            raise ValueError(f"density family requires game_l1: {model_id}")
        if family == "points" and spatial_name != "localization_f1":
            raise ValueError(f"points family requires localization_f1: {model_id}")
        if family == "density_and_points" and spatial_name not in {
            "game_l1",
            "localization_f1",
        }:
            raise ValueError(f"hybrid family has an invalid spatial metric: {model_id}")

        band_bias = metrics.get("band_signed_bias")
        if not isinstance(band_bias, dict) or set(band_bias) != {"low", "medium", "high"}:
            raise ValueError(f"band signed bias is incomplete: {model_id}")
        band_values = {
            key: _required_finite(value, field=f"band_signed_bias.{key}")
            for key, value in band_bias.items()
        }
        mape = metrics.get("mape_reference")
        if mape is None:
            normalized_errors = _finite(rows, "normalized_error")
            mape = 100 * statistics.fmean(normalized_errors)
        aggregates: dict[str, float | None] = {
            "mean_zone_mae": statistics.fmean(_finite(rows, "zone_mae")),
            "spatial_metric_value": _required_finite(
                metrics.get("spatial_mean"), field="spatial_mean"
            ),
            "max_absolute_band_bias": max(abs(value) for value in band_values.values()),
            "mae": _required_finite(metrics.get("mae"), field="mae"),
            "rmse": _required_finite(metrics.get("rmse"), field="rmse"),
            "mape_reference": _required_finite(mape, field="mape_reference"),
            "signed_bias": _required_finite(
                metrics.get("signed_bias"), field="signed_bias"
            ),
            "high_band_bias": band_values["high"],
            "median_latency_ms": _required_finite(
                metrics.get("median_latency_ms"), field="median_latency_ms"
            ),
            "throughput_fps_batch1": _required_finite(
                metrics.get("throughput_fps_batch1"), field="throughput_fps_batch1"
            ),
            "peak_vram_mb": _required_finite(
                metrics.get("peak_vram_mb"), field="peak_vram_mb"
            ),
            "mean_game_l1": _mean_or_none(rows, "game_l1"),
            "density_psnr": _mean_or_none(rows, "density_psnr"),
            "density_ssim": _mean_or_none(rows, "density_ssim"),
            "localization_precision": _mean_or_none(rows, "localization_precision"),
            "localization_recall": _mean_or_none(rows, "localization_recall"),
            "localization_f1": _mean_or_none(rows, "localization_f1"),
            "localization_mean_distance": _mean_or_none(
                rows, "localization_mean_distance"
            ),
        }

        selection = _read_object(
            run_dir / "selection_manifest.json", label="selection manifest"
        )
        raw_panels = selection.get("panel_paths")
        if not isinstance(raw_panels, list) or any(
            not isinstance(panel, str) or not panel.strip() for panel in raw_panels
        ):
            raise ValueError(f"invalid curated panel paths: {model_id}")
        panels = tuple(panel.strip() for panel in raw_panels)
        if (
            selection.get("selected_count") != 12
            or selection.get("shortfall") != 0
            or len(panels) != 12
            or len(set(panels)) != 12
            or not set(panels).issubset(verified_by_path)
        ):
            raise ValueError(f"invalid curated panel selection: {model_id}")

        rights_decision = _read_object(
            run_dir / "rights-decision.json", label="rights decision"
        )
        if rights_decision.get("status") != model["rights_scope"]:
            raise ValueError(f"rights scope mismatch: {model_id}")

        runs.append(
            ModelRunEvidence(
                model_id=model_id,
                family=family,
                rights_scope=str(model["rights_scope"]),
                checkpoint_training_split_status=str(
                    model["checkpoint_training_split_status"]
                ),
                comparison_scope=str(model["comparison_scope"]),
                run_dir=run_dir,
                run_relative_path=run_dir.relative_to(root).as_posix(),
                score=score,
                metrics=metrics,
                rights_decision=rights_decision,
                aggregates=aggregates,
                capabilities=capabilities,
                panels=panels,
                canonical_split_sha256=split_hash,
                input_artifacts=_deduplicate_artifacts(result_artifacts),
            )
        )

    if len({run.canonical_split_sha256 for run in runs}) != 1:
        raise ValueError("canonical validation samples differ across model runs")
    return tuple(runs)
