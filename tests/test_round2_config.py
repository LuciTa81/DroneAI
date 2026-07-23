from __future__ import annotations

import json
from pathlib import Path

import pytest

from droneai.round2_config import load_round2_config


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG = REPO_ROOT / "configs/evaluation/round2_reference_benchmark.json"


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


def test_round2_config_resolves_dataset_manifests_from_repository() -> None:
    config = load_round2_config(CONFIG)

    assert all(lane.manifest_path.is_file() for lane in config.datasets)
    assert [lane.rights_scope for lane in config.datasets] == [
        "PASS_COMMERCIAL_CANDIDATE",
        "PASS_RESEARCH_ONLY",
        "PASS_RESEARCH_ONLY",
    ]


def test_round2_config_rejects_training_action(tmp_path: Path) -> None:
    payload = json.loads(CONFIG.read_text(encoding="utf-8"))
    payload["action"] = "fine_tuning"
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(
        ValueError, match="frozen_internal_reference_evaluation"
    ):
        load_round2_config(path)
