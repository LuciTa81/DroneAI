from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable, Literal, Sequence

import numpy as np

from droneai.evaluation_contract import ScalarEvaluation
from droneai.integrity import is_sha256


@dataclass(frozen=True)
class Selection:
    sample_id: str
    category: str
    density_band: str
    reason: str


def _take_unique(
    candidates: Iterable[ScalarEvaluation],
    used: set[str],
) -> ScalarEvaluation | None:
    for row in candidates:
        if row.sample_id not in used:
            used.add(row.sample_id)
            return row
    return None


def select_review_samples(
    rows: Sequence[ScalarEvaluation],
    *,
    spatial_direction: Literal["minimize", "maximize"],
) -> tuple[Selection, ...]:
    successful = [
        row
        for row in rows
        if row.failure_state is None
        and row.predicted_count is not None
        and row.signed_error is not None
        and row.absolute_error is not None
    ]
    used: set[str] = set()

    spatial_rows = (
        row for row in successful if row.spatial_metric_value is not None
    )
    if spatial_direction == "minimize":
        spatial = sorted(
            spatial_rows,
            key=lambda row: (-float(row.spatial_metric_value), row.sample_id),
        )
    else:
        spatial = sorted(
            spatial_rows,
            key=lambda row: (float(row.spatial_metric_value), row.sample_id),
        )
    failure_candidates = {
        "worst spatial quality": _take_unique(spatial, used),
        "severe undercount": _take_unique(
            sorted(
                successful,
                key=lambda row: (float(row.signed_error), row.sample_id),
            ),
            used,
        ),
        "severe overcount": _take_unique(
            sorted(
                successful,
                key=lambda row: (-float(row.signed_error), row.sample_id),
            ),
            used,
        ),
    }
    selected = [
        Selection(chosen.sample_id, "failure", chosen.density_band, reason)
        for reason in (
            "worst spatial quality",
            "severe undercount",
            "severe overcount",
        )
        if (chosen := failure_candidates[reason]) is not None
    ]

    for band in ("low", "medium", "high"):
        band_rows = sorted(
            (row for row in successful if row.density_band == band),
            key=lambda row: (float(row.absolute_error), row.sample_id),
        )
        if not band_rows:
            continue
        errors = np.asarray(
            [float(row.absolute_error) for row in band_rows],
            dtype=np.float64,
        )
        for category, quantile in (
            ("best", 0.0),
            ("strong", 0.25),
            ("typical", 0.5),
        ):
            target = float(np.quantile(errors, quantile))
            candidates = sorted(
                band_rows,
                key=lambda row: (
                    abs(float(row.absolute_error) - target),
                    row.sample_id,
                ),
            )
            chosen = _take_unique(candidates, used)
            if chosen is not None:
                selected.append(
                    Selection(
                        chosen.sample_id,
                        category,
                        band,
                        f"nearest error quantile {quantile:.2f}",
                    )
                )

    return tuple(selected[:12])


def selection_manifest(
    selections: Sequence[Selection],
    *,
    predictions_sha256: str,
    spatial_direction: Literal["minimize", "maximize"],
    density_band_rules: Sequence[str],
    expected_panels: int = 12,
) -> dict[str, object]:
    if not is_sha256(predictions_sha256):
        raise ValueError("selection manifest requires predictions SHA-256")
    return {
        "schema_version": 1,
        "selector": "density-stratified-error-quantiles-v1",
        "expected_panels": expected_panels,
        "selected_count": len(selections),
        "shortfall": max(expected_panels - len(selections), 0),
        "shortfall_reason": (
            ""
            if len(selections) >= expected_panels
            else "insufficient unique eligible samples"
        ),
        "tie_breaker": "lexical sample_id",
        "spatial_direction": spatial_direction,
        "density_band_rules": list(density_band_rules),
        "predictions_sha256": predictions_sha256,
        "selections": [asdict(item) for item in selections],
    }
