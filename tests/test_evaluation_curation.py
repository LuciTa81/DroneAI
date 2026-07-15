from collections import Counter

from droneai.evaluation_contract import ScalarEvaluation
from droneai.evaluation_curation import select_review_samples, selection_manifest


def _row(
    sample_id: str,
    band: str,
    error: float,
    spatial: float | None,
) -> ScalarEvaluation:
    return ScalarEvaluation(
        sample_id,
        100.0,
        100.0 + error,
        error,
        abs(error),
        abs(error) / 100.0,
        band,
        1.0,
        2.0,
        "count",
        None,
        "game_l1" if spatial is not None else None,
        spatial,
    )


def _failure(sample_id: str, band: str, state: str) -> ScalarEvaluation:
    return ScalarEvaluation(
        sample_id,
        100.0,
        None,
        None,
        None,
        None,
        band,
        1.0,
        2.0,
        "count",
        state,
    )


def test_selector_returns_twelve_unique_reasoned_samples() -> None:
    rows = []
    for band in ("low", "medium", "high"):
        rows.extend(
            _row(
                f"{band}-{index:02d}",
                band,
                float(index - 5),
                float(index),
            )
            for index in range(12)
        )

    selected = select_review_samples(rows, spatial_direction="minimize")

    assert len(selected) == 12
    assert len({item.sample_id for item in selected}) == 12
    assert Counter(item.category for item in selected) == {
        "best": 3,
        "strong": 3,
        "typical": 3,
        "failure": 3,
    }
    assert all(item.reason for item in selected)


def test_selector_ties_are_resolved_by_sample_id_independent_of_input_order() -> None:
    rows = [
        _row("b", "low", 0.0, 0.0),
        _row("a", "low", 0.0, 0.0),
    ]

    forward = select_review_samples(rows, spatial_direction="minimize")
    reverse = select_review_samples(list(reversed(rows)), spatial_direction="minimize")

    assert forward == reverse
    assert forward[0].sample_id == "a"


def test_manifest_records_predictions_hash_and_shortfall() -> None:
    rows = [_row("only", "low", 0.0, 0.0)]
    selected = select_review_samples(rows, spatial_direction="minimize")

    manifest = selection_manifest(
        selected,
        predictions_sha256="a" * 64,
        spatial_direction="minimize",
        density_band_rules=("low: count < 10",),
        expected_panels=12,
    )

    assert manifest["predictions_sha256"] == "a" * 64
    assert manifest["selected_count"] == 1
    assert manifest["shortfall"] == 11
    assert manifest["shortfall_reason"] == "insufficient unique eligible samples"
    assert manifest["density_band_rules"] == ["low: count < 10"]


def test_maximize_spatial_metric_selects_lowest_value_as_worst() -> None:
    rows = [
        ScalarEvaluation(
            "good",
            1.0,
            1.0,
            0.0,
            0.0,
            0.0,
            "high",
            1.0,
            2.0,
            "points",
            None,
            "localization_f1",
            0.9,
        ),
        ScalarEvaluation(
            "bad",
            1.0,
            1.0,
            0.0,
            0.0,
            0.0,
            "high",
            1.0,
            2.0,
            "points",
            None,
            "localization_f1",
            0.1,
        ),
    ]

    selected = select_review_samples(rows, spatial_direction="maximize")

    assert any(
        item.sample_id == "bad" and item.reason == "worst spatial quality"
        for item in selected
    )


def test_unavailable_spatial_metric_produces_shortfall_without_fabrication() -> None:
    rows = []
    for band in ("low", "medium", "high"):
        rows.extend(
            _row(f"{band}-{index:02d}", band, float(index - 5), None)
            for index in range(12)
        )

    selected = select_review_samples(rows, spatial_direction="minimize")
    manifest = selection_manifest(
        selected,
        predictions_sha256="b" * 64,
        spatial_direction="minimize",
        density_band_rules=("low", "medium", "high"),
    )

    assert len(selected) == 11
    assert Counter(item.category for item in selected)["failure"] == 2
    assert all(item.reason != "worst spatial quality" for item in selected)
    assert manifest["shortfall"] == 1
    assert manifest["shortfall_reason"] == "insufficient unique eligible samples"


def test_explicit_failures_are_never_ranked_as_zero_count_successes() -> None:
    rows = [
        _failure("adapter-failure", "high", "cuda_out_of_memory"),
        _row("real-zero-count", "high", -100.0, None),
    ]

    selected = select_review_samples(rows, spatial_direction="minimize")
    manifest = selection_manifest(
        selected,
        predictions_sha256="c" * 64,
        spatial_direction="minimize",
        density_band_rules=("high",),
    )

    assert [item.sample_id for item in selected] == ["real-zero-count"]
    assert manifest["selected_count"] == 1
    assert manifest["shortfall"] == 11
    assert all(
        item["sample_id"] != "adapter-failure" for item in manifest["selections"]
    )


def test_selector_never_returns_more_than_twelve_samples() -> None:
    rows = [
        _row(f"sample-{index:03d}", "low", float(index - 50), float(index))
        for index in range(100)
    ]

    assert len(select_review_samples(rows, spatial_direction="minimize")) <= 12
