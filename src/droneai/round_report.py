from __future__ import annotations

import csv
import io
import json
import shutil
from pathlib import Path

from droneai.evaluation_artifacts import enforce_review_budget, write_json
from droneai.integrity import is_sha256, sha256_file


_AGGREGATES = (
    "mae",
    "rmse",
    "mape_reference",
    "signed_bias",
    "high_band_bias",
    "max_absolute_band_bias",
    "mean_zone_mae",
    "spatial_metric_value",
    "median_latency_ms",
    "throughput_fps_batch1",
    "peak_vram_mb",
    "mean_game_l1",
    "density_psnr",
    "density_ssim",
    "localization_precision",
    "localization_recall",
    "localization_f1",
    "localization_mean_distance",
)
_CATEGORIES = (
    "zone_spatial",
    "count_stability",
    "runtime_resources",
    "operator_integration",
    "evidence_quality",
)
_CAPABILITIES = (
    "zone_aggregation",
    "density_map",
    "point_localization",
    "confidence",
)
_RIGHTS = (
    "code",
    "dataset",
    "pretrained_weights",
    "derived_weights",
    "deployment",
)


def _display(value: object, digits: int = 3) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _markdown(comparison: dict[str, object]) -> str:
    lines = [
        "# Round 1 fixed-CCTV technical comparison",
        "",
        "**Scope: PROVISIONAL_TECHNICAL_SHORTLIST; compatibility smoke; not an official ranking.**",
        "",
        "| Order | Model | Technical | Zone/spatial | Count | Runtime | Integration | Evidence | Rights | Deployment |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    for row in sorted(comparison["models"], key=lambda item: item["technical_order"]):
        category = row["categories"]
        lines.append(
            f"| {row['technical_order']} | {row['model_id']} | "
            f"{float(row['technical_score_display']):.2f} | "
            f"{float(category['zone_spatial']):.2f} | "
            f"{float(category['count_stability']):.2f} | "
            f"{float(category['runtime_resources']):.2f} | "
            f"{float(category['operator_integration']):.2f} | "
            f"{float(category['evidence_quality']):.2f} | "
            f"{row['rights_scope']} | {row['deployment_label']} |"
        )

    lines.extend(
        [
            "",
            "## Raw validation metrics",
            "",
            "| Model | MAE | RMSE | MAPE % | Bias | High bias | Zone MAE | Spatial | Latency ms | FPS | VRAM MB |",
            "|---|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|",
        ]
    )
    for row in comparison["models"]:
        metric = row["aggregates"]
        lines.append(
            f"| {row['model_id']} | {_display(metric['mae'])} | "
            f"{_display(metric['rmse'])} | {_display(metric['mape_reference'])} | "
            f"{_display(metric['signed_bias'])} | {_display(metric['high_band_bias'])} | "
            f"{_display(metric['mean_zone_mae'])} | {row['primary_spatial_metric']}="
            f"{_display(metric['spatial_metric_value'])} | "
            f"{_display(metric['median_latency_ms'])} | "
            f"{_display(metric['throughput_fps_batch1'])} | "
            f"{_display(metric['peak_vram_mb'])} |"
        )

    lines.extend(
        [
            "",
            "## Density, localization, and operator feature evidence",
            "",
            "| Model | GAME L1 | PSNR | SSIM | Loc precision | Loc recall | Loc F1 | Mean distance | Zone | Density | Points | Confidence |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---|---|---|---|",
        ]
    )
    for row in comparison["models"]:
        metric = row["aggregates"]
        capability = row["capabilities"]
        lines.append(
            f"| {row['model_id']} | {_display(metric['mean_game_l1'])} | "
            f"{_display(metric['density_psnr'])} | {_display(metric['density_ssim'])} | "
            f"{_display(metric['localization_precision'])} | "
            f"{_display(metric['localization_recall'])} | "
            f"{_display(metric['localization_f1'])} | "
            f"{_display(metric['localization_mean_distance'])} | "
            f"{capability['zone_aggregation']} | {capability['density_map']} | "
            f"{capability['point_localization']} | {capability['confidence']} |"
        )

    lines.extend(
        [
            "",
            "## Independent rights components",
            "",
            "| Model | Code | Dataset | Pretrained weight | Derived weight | Deployment | Overall |",
            "|---|---|---|---|---|---|---|",
        ]
    )
    for row in comparison["models"]:
        rights = row["rights_components"]
        lines.append(
            f"| {row['model_id']} | {rights['code']['status']} | "
            f"{rights['dataset']['status']} | {rights['pretrained_weights']['status']} | "
            f"{rights['derived_weights']['status']} | {rights['deployment']['status']} | "
            f"{row['deployment_label']} |"
        )

    lines.extend(["", "## Provisional technical shortlist", ""])
    lines.extend(
        f"{index}. `{model_id}`"
        for index, model_id in enumerate(
            comparison["shortlist"]["model_ids"], start=1
        )
    )
    lines.extend(["", "## Representative panels", ""])
    for row in comparison["models"]:
        links = ", ".join(
            f"[panel {index}]({path})"
            for index, path in enumerate(row["representative_panels"], start=1)
        )
        lines.append(f"- `{row['model_id']}`: {links}")
    lines.extend(["", "## Limitations", ""])
    lines.extend(f"- {item}" for item in comparison["limitations"])
    return "\n".join(lines) + "\n"


def _csv(comparison: dict[str, object]) -> str:
    fields = [
        "technical_order",
        "model_id",
        "family",
        "technical_score",
        "technical_score_display",
        *_CATEGORIES,
        *_AGGREGATES,
        "primary_spatial_metric",
        *(f"{name}_capability" for name in _CAPABILITIES),
        *(field for name in _RIGHTS for field in (f"{name}_component_id", f"{name}_rights")),
        "rights_scope",
        "deployment_label",
        "checkpoint_training_split_status",
        "comparison_scope",
    ]
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    for row in sorted(comparison["models"], key=lambda item: item["technical_order"]):
        flat: dict[str, object] = {
            "technical_order": row["technical_order"],
            "model_id": row["model_id"],
            "family": row["family"],
            "technical_score": row["technical_score"],
            "technical_score_display": row["technical_score_display"],
            "primary_spatial_metric": row["primary_spatial_metric"],
            "rights_scope": row["rights_scope"],
            "deployment_label": row["deployment_label"],
            "checkpoint_training_split_status": row[
                "checkpoint_training_split_status"
            ],
            "comparison_scope": row["comparison_scope"],
        }
        flat.update({name: row["categories"][name] for name in _CATEGORIES})
        flat.update({name: row["aggregates"].get(name) for name in _AGGREGATES})
        flat.update(
            {
                f"{name}_capability": row["capabilities"][name]
                for name in _CAPABILITIES
            }
        )
        for name in _RIGHTS:
            flat[f"{name}_component_id"] = row["rights_components"][name][
                "component_id"
            ]
            flat[f"{name}_rights"] = row["rights_components"][name]["status"]
        writer.writerow(flat)
    return buffer.getvalue()


def _validate_inputs(
    input_references: tuple[dict[str, str], ...]
) -> tuple[dict[str, str], ...]:
    normalized: list[dict[str, str]] = []
    seen: dict[str, str] = {}
    for reference in input_references:
        if not isinstance(reference, dict):
            raise ValueError("input reference must be an object")
        path = reference.get("path")
        digest = reference.get("sha256")
        if not isinstance(path, str) or not path.strip() or not is_sha256(digest):
            raise ValueError("input reference requires path and SHA-256")
        normalized_path = path.strip().replace("\\", "/")
        normalized_digest = str(digest).lower()
        previous = seen.setdefault(normalized_path, normalized_digest)
        if previous != normalized_digest:
            raise ValueError(f"input reference hash conflict: {normalized_path}")
    normalized.extend(
        {"path": path, "sha256": seen[path]} for path in sorted(seen)
    )
    return tuple(normalized)


def write_comparison_bundle(
    output_dir: str | Path,
    comparison: dict[str, object],
    scorecard_path: str | Path,
    *,
    input_references: tuple[dict[str, str], ...],
) -> Path:
    inputs = _validate_inputs(input_references)
    output = Path(output_dir)
    temporary = output.parent / f".{output.name}.tmp"
    if output.exists() or temporary.exists():
        raise FileExistsError(
            "comparison output and temporary path must not exist"
        )
    scorecard_source = Path(scorecard_path)
    scorecard = json.loads(scorecard_source.read_text(encoding="utf-8"))
    review_budget = int(scorecard["review_budget_bytes"])

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary.mkdir()
    try:
        write_json(temporary / "comparison.json", comparison)
        write_json(temporary / "scorecard-config.snapshot.json", scorecard)
        (temporary / "comparison.csv").write_text(
            _csv(comparison), encoding="utf-8", newline=""
        )
        (temporary / "comparison.md").write_text(
            _markdown(comparison), encoding="utf-8", newline=""
        )
        outputs = [
            {"path": path.name, "sha256": sha256_file(path)}
            for path in sorted(temporary.iterdir())
            if path.name != "evidence-manifest.json"
        ]
        write_json(
            temporary / "evidence-manifest.json",
            {
                "schema_version": 1,
                "inputs": list(inputs),
                "outputs": outputs,
            },
        )
        enforce_review_budget(temporary, limit_bytes=review_budget)
        temporary.rename(output)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return output

