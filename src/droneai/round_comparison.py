from __future__ import annotations

import re

from droneai.round_evidence import ModelRunEvidence
from droneai.round_scorecard import ScorecardConfig, score_technical


_COMMIT = re.compile(r"^[0-9a-f]{40}$")
_RIGHTS_COMPONENTS = (
    "code",
    "dataset",
    "pretrained_weights",
    "derived_weights",
    "deployment",
)


def deployment_label(scope: str) -> str:
    if scope == "PRODUCTION_APPROVED":
        return "PRODUCT_ELIGIBLE"
    if scope == "PASS_COMMERCIAL_CANDIDATE":
        return "PENDING_DUE_DILIGENCE"
    if scope == "PASS_RESEARCH_ONLY":
        return "RESEARCH_ONLY"
    if scope.startswith("BLOCKED"):
        return "BLOCKED"
    raise ValueError(f"unsupported rights scope: {scope}")


def _string_set(value: object, *, field: str) -> set[str]:
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise ValueError(f"rights decision {field} must be a string list")
    return set(value)


def _component_rights(
    decision: dict[str, object],
) -> dict[str, dict[str, str]]:
    component_ids = decision.get("component_ids")
    if not isinstance(component_ids, dict):
        raise ValueError("rights decision component_ids must be an object")
    ambiguous = _string_set(
        decision.get("ambiguous_components"), field="ambiguous_components"
    )
    restricted = _string_set(
        decision.get("restricted_components"), field="restricted_components"
    )
    known = set(_RIGHTS_COMPONENTS)
    if (ambiguous | restricted) - known:
        raise ValueError("rights decision contains an unknown component")

    result: dict[str, dict[str, str]] = {}
    for name in _RIGHTS_COMPONENTS:
        component_id = component_ids.get(name)
        if not isinstance(component_id, str) or not component_id.strip():
            raise ValueError(f"rights decision is missing component ID: {name}")
        if name in restricted:
            status = "RESTRICTED"
        elif name in ambiguous:
            status = "PENDING"
        else:
            status = "REVIEWED_NO_EXPLICIT_PROHIBITION"
        result[name] = {"component_id": component_id.strip(), "status": status}
    return result


def _required_technical_inputs(run: ModelRunEvidence) -> dict[str, float]:
    required = (
        "mean_zone_mae",
        "spatial_metric_value",
        "max_absolute_band_bias",
        "mae",
        "rmse",
        "mape_reference",
        "high_band_bias",
        "median_latency_ms",
        "peak_vram_mb",
    )
    values: dict[str, float] = {}
    for name in required:
        value = run.aggregates.get(name)
        if value is None:
            raise ValueError(
                f"required technical aggregate is unavailable: {run.model_id}.{name}"
            )
        values[name] = float(value)
    return values


def _row(run: ModelRunEvidence, config: ScorecardConfig) -> dict[str, object]:
    successful = int(run.metrics.get("successful_samples", -1))
    failures = int(run.metrics.get("explicit_failures", -1))
    scored = score_technical(
        family=run.family,
        spatial_metric_name=str(run.metrics.get("spatial_metric_name", "")),
        aggregates=_required_technical_inputs(run),
        capabilities=run.capabilities,
        checkpoint_training_split_status=run.checkpoint_training_split_status,
        coverage_complete=(
            successful == config.expected_samples and failures == 0
        ),
        artifacts_verified=True,
        config=config,
    )
    score = float(scored["technical_score"])
    return {
        "model_id": run.model_id,
        "family": run.family,
        "technical_score": score,
        "technical_score_display": round(score, 2),
        "categories": scored["categories"],
        "aggregates": run.aggregates,
        "primary_spatial_metric": str(run.metrics["spatial_metric_name"]),
        "capabilities": {
            name: "DEMONSTRATED" if demonstrated else "NOT_DEMONSTRATED"
            for name, demonstrated in run.capabilities.items()
        },
        "rights_scope": run.rights_scope,
        "deployment_label": deployment_label(run.rights_scope),
        "rights_components": _component_rights(run.rights_decision),
        "checkpoint_training_split_status": run.checkpoint_training_split_status,
        "comparison_scope": run.comparison_scope,
        "ranking_eligible": False,
        "representative_panels": [
            f"../{run.run_relative_path}/{panel}" for panel in run.panels
        ],
    }


def build_round_comparison(
    runs: tuple[ModelRunEvidence, ...],
    config: ScorecardConfig,
    *,
    git_commit: str,
) -> dict[str, object]:
    if not _COMMIT.fullmatch(git_commit):
        raise ValueError("comparison requires a full clean Git commit")
    if tuple(run.model_id for run in runs) != config.required_model_ids:
        raise ValueError("verified run order does not match scorecard")
    if not runs:
        raise ValueError("comparison requires verified model runs")
    if len({run.canonical_split_sha256 for run in runs}) != 1:
        raise ValueError("comparison runs do not share one canonical split")

    rows = [_row(run, config) for run in runs]
    ordered = sorted(
        rows,
        key=lambda row: (
            -float(row["technical_score"]),
            float(row["aggregates"]["mean_zone_mae"]),
            float(row["aggregates"]["mae"]),
            str(row["model_id"]),
        ),
    )
    for rank, row in enumerate(ordered, start=1):
        row["technical_order"] = rank
    shortlist = [
        str(row["model_id"]) for row in ordered[: config.shortlist_size]
    ]
    scopes = sorted({run.comparison_scope for run in runs})
    return {
        "schema_version": 1,
        "comparison_id": config.comparison_id,
        "round_id": config.round_id,
        "git_commit": git_commit,
        "dataset": {
            "dataset_id": config.dataset_id,
            "split_id": config.split_id,
            "split_role": config.split_role,
            "expected_samples": config.expected_samples,
            "canonical_split_sha256": runs[0].canonical_split_sha256,
        },
        "ranking_eligible": False,
        "comparison_scope": scopes[0] if len(scopes) == 1 else scopes,
        "models": rows,
        "shortlist": {
            "label": config.shortlist_label,
            "model_ids": shortlist,
            "tie_breakers": list(config.tie_breakers),
        },
        "limitations": [
            "Official checkpoints do not share one training protocol.",
            "This is a 36-image cross-domain operational screen, not an official ranking.",
            "Technical score does not grant code, dataset, weight, derived-weight, or deployment rights.",
            "Fine-tuning and test-split access are outside this comparison stage.",
        ],
    }
