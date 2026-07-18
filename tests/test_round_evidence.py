from __future__ import annotations

import json
from pathlib import Path

import pytest

from droneai.round_evidence import load_verified_round
from tests.round_comparison_helpers import (
    make_round_fixture,
    reanchor_run,
    write_predictions,
)


def test_load_verified_round_accepts_six_matching_runs(tmp_path: Path) -> None:
    queue, results, config = make_round_fixture(tmp_path)

    runs = load_verified_round(queue, results, config)

    assert tuple(run.model_id for run in runs) == config.required_model_ids
    assert len({run.canonical_split_sha256 for run in runs}) == 1
    assert all(run.aggregates["mape_reference"] == pytest.approx(10.0) for run in runs)
    assert runs[0].capabilities == {
        "zone_aggregation": True,
        "density_map": True,
        "point_localization": False,
        "confidence": False,
    }
    assert runs[1].capabilities["density_map"] is True
    assert runs[1].capabilities["point_localization"] is True
    assert runs[2].capabilities["point_localization"] is True
    assert runs[2].aggregates["density_psnr"] is None
    assert runs[0].aggregates["localization_f1"] is None
    assert runs[-1].run_relative_path == "csrnet/run"
    assert len(runs[-1].panels) == 12


def test_nested_prediction_hash_change_is_rejected(tmp_path: Path) -> None:
    queue, results, config = make_round_fixture(tmp_path)
    (results / "dm-count/run/predictions.csv").write_text(
        "changed\n", encoding="utf-8"
    )

    with pytest.raises(ValueError, match="artifact hash mismatch"):
        load_verified_round(queue, results, config)


def test_changed_manifest_without_score_anchor_is_rejected(tmp_path: Path) -> None:
    queue, results, config = make_round_fixture(tmp_path)
    manifest = results / "dm-count/run/evidence-manifest.json"
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["schema_version"] = 99
    manifest.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="evidence manifest hash mismatch"):
        load_verified_round(queue, results, config)


def test_mixed_sample_identity_is_rejected_after_internal_reanchoring(
    tmp_path: Path,
) -> None:
    queue, results, config = make_round_fixture(tmp_path)
    write_predictions(
        results / "pet/run/predictions.csv",
        family="points",
        changed_sample="img_other",
    )
    reanchor_run(queue, results, "pet")

    with pytest.raises(ValueError, match="canonical validation samples differ"):
        load_verified_round(queue, results, config)


def test_test_role_is_rejected_before_artifact_loading(tmp_path: Path) -> None:
    queue, results, config = make_round_fixture(tmp_path)
    payload = json.loads(queue.read_text(encoding="utf-8"))
    payload["models"][0]["split_role"] = "test"
    queue.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="test split"):
        load_verified_round(queue, results, config)


def test_non_finite_required_prediction_value_is_rejected(tmp_path: Path) -> None:
    queue, results, config = make_round_fixture(tmp_path)
    target = results / "apgcc/run/predictions.csv"
    target.write_text(
        target.read_text(encoding="utf-8").replace("10.0", "nan", 1),
        encoding="utf-8",
    )
    reanchor_run(queue, results, "apgcc")

    with pytest.raises(ValueError, match="non-finite prediction field"):
        load_verified_round(queue, results, config)


def test_unknown_family_is_rejected(tmp_path: Path) -> None:
    queue, results, config = make_round_fixture(tmp_path)
    payload = json.loads(queue.read_text(encoding="utf-8"))
    payload["models"][3]["family"] = "detector"
    queue.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="unsupported model family"):
        load_verified_round(queue, results, config)
