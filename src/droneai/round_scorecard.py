from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping


SUPPORTED_FAMILIES = frozenset({"density", "points", "density_and_points"})
_CATEGORY_KEYS = frozenset(
    {
        "zone_spatial",
        "count_stability",
        "runtime_resources",
        "operator_integration",
        "evidence_quality",
    }
)
_ANCHOR_KEYS = frozenset(
    {
        "zone_mae_zero",
        "game_l1_zero",
        "localization_f1_full",
        "band_bias_zero",
        "mae_zero",
        "rmse_zero",
        "mape_zero_percent",
        "high_bias_zero",
        "latency_full_ms",
        "latency_zero_ms",
        "vram_full_mb",
        "vram_zero_mb",
    }
)
_CAPABILITY_KEYS = frozenset(
    {"zone_aggregation", "density_map", "point_localization", "confidence"}
)


@dataclass(frozen=True)
class ScorecardConfig:
    round_id: str
    comparison_id: str
    dataset_id: str
    split_id: str
    split_role: str
    expected_samples: int
    required_model_ids: tuple[str, ...]
    category_weights: dict[str, int]
    anchors: dict[str, float]
    capability_points: dict[str, int]
    shortlist_size: int
    tie_breakers: tuple[str, ...]
    shortlist_label: str
    ranking_eligible: bool
    review_budget_bytes: int


def clamp(value: float) -> float:
    if not math.isfinite(value):
        raise ValueError("score inputs must be finite")
    return min(1.0, max(0.0, value))


def piecewise(value: float, full: float, zero: float) -> float:
    if not all(math.isfinite(item) for item in (value, full, zero)):
        raise ValueError("piecewise requires finite values")
    if full >= zero:
        raise ValueError("piecewise requires full < zero")
    if value <= full:
        return 1.0
    if value >= zero:
        return 0.0
    return (zero - value) / (zero - full)


def _nonempty(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")
    return value.strip()


def load_scorecard(path: str | Path) -> ScorecardConfig:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise ValueError("scorecard schema_version must be 1")
    try:
        weights = {
            str(key): int(value)
            for key, value in dict(payload["category_weights"]).items()
        }
        anchors = {
            str(key): float(value) for key, value in dict(payload["anchors"]).items()
        }
        capabilities = {
            str(key): int(value)
            for key, value in dict(payload["capability_points"]).items()
        }
        model_ids = tuple(
            _nonempty(value, field="required_model_ids")
            for value in payload["required_model_ids"]
        )
        config = ScorecardConfig(
            round_id=_nonempty(payload.get("round_id"), field="round_id"),
            comparison_id=_nonempty(
                payload.get("comparison_id"), field="comparison_id"
            ),
            dataset_id=_nonempty(payload.get("dataset_id"), field="dataset_id"),
            split_id=_nonempty(payload.get("split_id"), field="split_id"),
            split_role=_nonempty(payload.get("split_role"), field="split_role"),
            expected_samples=int(payload["expected_samples"]),
            required_model_ids=model_ids,
            category_weights=weights,
            anchors=anchors,
            capability_points=capabilities,
            shortlist_size=int(payload["shortlist_size"]),
            tie_breakers=tuple(str(value) for value in payload["tie_breakers"]),
            shortlist_label=_nonempty(
                payload.get("shortlist_label"), field="shortlist_label"
            ),
            ranking_eligible=bool(payload["ranking_eligible"]),
            review_budget_bytes=int(payload["review_budget_bytes"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"invalid scorecard configuration: {exc}") from exc

    if set(config.category_weights) != _CATEGORY_KEYS:
        raise ValueError("scorecard category keys are invalid")
    if sum(config.category_weights.values()) != 100:
        raise ValueError("category weights must total 100")
    if any(value < 0 for value in config.category_weights.values()):
        raise ValueError("category weights must be non-negative")
    if set(config.anchors) != _ANCHOR_KEYS or any(
        not math.isfinite(value) or value <= 0 for value in config.anchors.values()
    ):
        raise ValueError("scorecard anchors must be named finite positive values")
    if set(config.capability_points) != _CAPABILITY_KEYS:
        raise ValueError("scorecard capability keys are invalid")
    if sum(config.capability_points.values()) != config.category_weights[
        "operator_integration"
    ]:
        raise ValueError("capability points must match operator integration weight")
    if config.split_role != "validation" or config.ranking_eligible:
        raise ValueError(
            "Round 1 scorecard must remain non-ranking validation evidence"
        )
    if config.expected_samples != 36:
        raise ValueError("Round 1 scorecard requires exactly 36 samples")
    if len(set(config.required_model_ids)) != len(config.required_model_ids):
        raise ValueError("required model IDs must be unique")
    if not 0 < config.shortlist_size <= len(config.required_model_ids):
        raise ValueError("shortlist size must fit the required model list")
    if config.review_budget_bytes != 25 * 1024 * 1024:
        raise ValueError("review budget must be exactly 25 MiB")
    return config


def _required_aggregate(aggregates: Mapping[str, float], name: str) -> float:
    try:
        value = float(aggregates[name])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"required aggregate is unavailable: {name}") from exc
    if not math.isfinite(value):
        raise ValueError(f"required aggregate must be finite: {name}")
    return value


def score_technical(
    *,
    family: str,
    spatial_metric_name: str,
    aggregates: Mapping[str, float],
    capabilities: Mapping[str, bool],
    checkpoint_training_split_status: str,
    coverage_complete: bool,
    artifacts_verified: bool,
    config: ScorecardConfig,
) -> dict[str, object]:
    if family not in SUPPORTED_FAMILIES:
        raise ValueError(f"unsupported model family: {family}")
    if family == "density" and spatial_metric_name != "game_l1":
        raise ValueError("density family requires game_l1 as its primary metric")
    if family == "points" and spatial_metric_name != "localization_f1":
        raise ValueError(
            "points family requires localization_f1 as its primary metric"
        )
    if spatial_metric_name not in {"game_l1", "localization_f1"}:
        raise ValueError(f"unsupported primary spatial metric: {spatial_metric_name}")
    unknown_capabilities = set(capabilities) - set(config.capability_points)
    if unknown_capabilities:
        raise ValueError(
            f"unsupported capability fields: {sorted(unknown_capabilities)}"
        )

    a = config.anchors
    zone = 20 * clamp(
        1 - _required_aggregate(aggregates, "mean_zone_mae") / a["zone_mae_zero"]
    )
    spatial_value = _required_aggregate(aggregates, "spatial_metric_value")
    if spatial_metric_name == "game_l1":
        spatial = 10 * clamp(1 - spatial_value / a["game_l1_zero"])
    else:
        spatial = 10 * clamp(spatial_value / a["localization_f1_full"])
    robustness = 5 * clamp(
        1
        - _required_aggregate(aggregates, "max_absolute_band_bias")
        / a["band_bias_zero"]
    )
    count = (
        10 * clamp(1 - _required_aggregate(aggregates, "mae") / a["mae_zero"])
        + 7
        * clamp(1 - _required_aggregate(aggregates, "rmse") / a["rmse_zero"])
        + 4
        * clamp(
            1
            - _required_aggregate(aggregates, "mape_reference")
            / a["mape_zero_percent"]
        )
        + 4
        * clamp(
            1
            - abs(_required_aggregate(aggregates, "high_band_bias"))
            / a["high_bias_zero"]
        )
    )
    runtime = 9 * piecewise(
        _required_aggregate(aggregates, "median_latency_ms"),
        a["latency_full_ms"],
        a["latency_zero_ms"],
    ) + 6 * piecewise(
        _required_aggregate(aggregates, "peak_vram_mb"),
        a["vram_full_mb"],
        a["vram_zero_mb"],
    )
    operator = sum(
        config.capability_points[name]
        for name in config.capability_points
        if capabilities.get(name) is True
    )
    evidence = (4 if coverage_complete else 0) + (4 if artifacts_verified else 0)
    if checkpoint_training_split_status == "VERIFIED_DISJOINT":
        evidence += 2

    categories = {
        "zone_spatial": zone + spatial + robustness,
        "count_stability": count,
        "runtime_resources": runtime,
        "operator_integration": float(operator),
        "evidence_quality": float(evidence),
    }
    for name, points in categories.items():
        if not 0 <= points <= config.category_weights[name]:
            raise ValueError(f"category score is outside its frozen weight: {name}")
    return {
        "categories": categories,
        "technical_score": sum(categories.values()),
    }
