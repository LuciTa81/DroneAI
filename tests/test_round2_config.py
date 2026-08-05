from __future__ import annotations

import json
from pathlib import Path

import pytest

from droneai.round2_config import (
    LEGACY_MODEL_IDS,
    POINT_MODEL_IDS,
    SUPPORTED_MODEL_IDS,
    load_round2_config,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG = REPO_ROOT / "configs/evaluation/round2_reference_benchmark.json"
POINT_CONFIG = (
    REPO_ROOT / "configs/evaluation/round2_point_reference_benchmark.json"
)


def test_round2_config_freezes_exact_matrix() -> None:
    config = load_round2_config(CONFIG)

    assert [model.model_id for model in config.models] == [
        "steerer",
        "dm-count",
        "mpcount",
    ]
    assert [
        (lane.dataset_id, lane.partition, lane.samples)
        for lane in config.datasets
    ] == [
        ("ucf-qnrf-kaggle-apache", "test", 334),
        ("jhu-crowd-plus-v2", "val", 500),
        ("up-count-v1", "val_test", 166),
    ]
    assert config.total_samples_per_model == 1000
    assert config.combined_rights_scope == "PASS_RESEARCH_ONLY"
    assert config.fine_tuning is False


def test_round2_point_config_freezes_exact_matrix() -> None:
    config = load_round2_config(POINT_CONFIG)

    assert tuple(model.model_id for model in config.models) == POINT_MODEL_IDS
    assert tuple(lane.samples for lane in config.datasets) == (334, 500, 166)
    assert config.total_samples_per_model == 1000
    assert config.combined_rights_scope == "PASS_RESEARCH_ONLY"
    assert config.fine_tuning is False


def test_round2_model_constants_preserve_both_approved_shortlists() -> None:
    assert LEGACY_MODEL_IDS == ("steerer", "dm-count", "mpcount")
    assert POINT_MODEL_IDS == ("steerer", "pet", "apgcc")
    assert SUPPORTED_MODEL_IDS == (
        "steerer",
        "dm-count",
        "mpcount",
        "pet",
        "apgcc",
    )


def test_round2_config_resolves_dataset_manifests_from_repository() -> None:
    config = load_round2_config(CONFIG)

    assert all(lane.manifest_path.is_file() for lane in config.datasets)
    assert [lane.rights_scope for lane in config.datasets] == [
        "PASS_COMMERCIAL_CANDIDATE",
        "PASS_RESEARCH_ONLY",
        "PASS_RESEARCH_ONLY",
    ]
    up_count = config.datasets[2]
    assert up_count.manifest_path.name == "up_count.round2_reference.json"
    manifest = json.loads(up_count.manifest_path.read_text(encoding="utf-8"))
    assert manifest["split"]["required_splits"] == ["val", "test"]
    assert manifest["rights"]["commercial_use"] is False


def test_round2_config_rejects_training_action(tmp_path: Path) -> None:
    payload = json.loads(CONFIG.read_text(encoding="utf-8"))
    payload["action"] = "fine_tuning"
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(
        ValueError, match="frozen_internal_reference_evaluation"
    ):
        load_round2_config(path)


@pytest.mark.parametrize(
    "model_ids",
    [
        ("steerer", "pet", "mpcount"),
        ("pet", "steerer", "apgcc"),
        ("steerer", "pet"),
        ("steerer", "pet", "apgcc", "dm-count"),
    ],
)
def test_round2_config_rejects_unapproved_model_matrix(
    tmp_path: Path,
    model_ids: tuple[str, ...],
) -> None:
    payload = json.loads(CONFIG.read_text(encoding="utf-8"))
    family = {
        "steerer": "density_and_points",
        "dm-count": "density",
        "mpcount": "density",
        "pet": "points",
        "apgcc": "points",
    }
    payload["models"] = [
        {
            "model_id": model_id,
            "family": family[model_id],
            "rights_scope": (
                "PASS_RESEARCH_ONLY"
                if model_id == "pet"
                else "PASS_COMMERCIAL_CANDIDATE"
            ),
        }
        for model_id in model_ids
    ]
    path = tmp_path / "bad-models.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="approved ordered model shortlist"):
        load_round2_config(path)
