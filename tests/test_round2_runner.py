from __future__ import annotations

import json
from pathlib import Path

import pytest

from droneai.round2_config import DatasetLane
from droneai.round2_runner import (
    RuntimeModel,
    build_round2_adapter,
    build_round2_protocol,
    load_round2_runtime_config,
    write_scoped_rights_decision,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
POINT_RUNTIME = (
    REPO_ROOT / "configs/evaluation/round2_point_reference_home5090.json"
)


def _targets() -> dict[str, object]:
    return {
        "localization_radius": 16.0,
        "targets": {
            "mae_max": 1000.0,
            "rmse_max": 2000.0,
            "bias_max": 1000.0,
            "band_bias_max": 2000.0,
            "condition_bias_max": 2000.0,
            "latency_max_ms": 5000.0,
            "vram_max_mb": 32000.0,
            "spatial_metric_name": "game_l1",
            "spatial_direction": "minimize",
            "spatial_target": 5000.0,
            "required_density_bands": ["low", "medium", "high"],
            "density_band_rules": [
                "low: count < 500",
                "medium: 500 <= count < 1500",
                "high: count >= 1500",
            ],
            "required_condition_keys": ["source_partition"],
            "zone_warning_count": 50.0,
            "zone_critical_count": 150.0,
        },
    }


def _runtime(tmp_path: Path, model_id: str = "steerer") -> RuntimeModel:
    rights = tmp_path / "rights.json"
    rights.write_text("{}", encoding="utf-8")
    return RuntimeModel(
        model_id=model_id,
        model_config=tmp_path / "model-config.json",
        upstream_dir=tmp_path / "upstream",
        upstream_commit="a" * 40,
        checkpoint=tmp_path / "checkpoint.pth",
        checkpoint_sha256="b" * 64,
        rights_decision=rights,
        rights_manifest=tmp_path / "rights-manifest.json",
        device="cuda",
        long_side_cap=3072 if model_id == "steerer" else None,
        patch_size=3584 if model_id == "mpcount" else None,
    )


def test_mixed_rights_protocol_cannot_claim_ranking(tmp_path: Path) -> None:
    manifest = tmp_path / "dataset.json"
    manifest.write_text("{}", encoding="utf-8")
    lane = DatasetLane(
        dataset_id="jhu-crowd-plus-v2",
        partition="val",
        samples=500,
        rights_scope="PASS_RESEARCH_ONLY",
        manifest_path=manifest,
    )

    protocol = build_round2_protocol(
        "steerer",
        lane,
        runtime=_runtime(tmp_path),
        model_config=_targets(),
    )

    assert protocol.comparison_scope == "research_reference_only"
    assert protocol.checkpoint_training_split_status == "UNKNOWN"
    assert protocol.sealed_test_access_approved is False
    assert protocol.split_role == "validation"
    assert protocol.expected_samples == 500


def test_ucf_test_protocol_records_current_test_authorization(tmp_path: Path) -> None:
    manifest = tmp_path / "dataset.json"
    manifest.write_text("{}", encoding="utf-8")
    lane = DatasetLane(
        dataset_id="ucf-qnrf-kaggle-apache",
        partition="test",
        samples=334,
        rights_scope="PASS_COMMERCIAL_CANDIDATE",
        manifest_path=manifest,
    )

    protocol = build_round2_protocol(
        "dm-count",
        lane,
        runtime=_runtime(tmp_path, "dm-count"),
        model_config=_targets(),
    )

    assert protocol.split_role == "test"
    assert protocol.sealed_test_access_approved is True
    assert protocol.checkpoint_training_split_status == "VERIFIED_DISJOINT"
    assert protocol.comparison_scope == "research_reference_only"


def test_adapter_factory_passes_only_existing_adapter_arguments(tmp_path: Path) -> None:
    calls: list[dict[str, object]] = []

    class FakeAdapter:
        def __init__(self, **kwargs: object) -> None:
            calls.append(kwargs)

    runtime = _runtime(tmp_path, "mpcount")
    adapter = build_round2_adapter(
        "mpcount",
        runtime=runtime,
        adapter_types={"mpcount": FakeAdapter},
    )

    assert isinstance(adapter, FakeAdapter)
    assert calls == [
        {
            "upstream_dir": runtime.upstream_dir,
            "expected_upstream_commit": "a" * 40,
            "checkpoint_path": runtime.checkpoint,
            "checkpoint_sha256": "b" * 64,
            "device": "cuda",
            "patch_size": 3584,
        }
    ]


def test_runtime_config_rejects_non_round2_model(tmp_path: Path) -> None:
    path = tmp_path / "runtime.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "models": {"csrnet": {"upstream_dir": "/tmp/csrnet"}},
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="exactly"):
        load_round2_runtime_config(path)


def test_point_runtime_config_freezes_exact_model_order() -> None:
    runtimes = load_round2_runtime_config(POINT_RUNTIME)

    assert tuple(runtimes) == ("steerer", "pet", "apgcc")
    assert runtimes["pet"].long_side_cap == 1536
    assert runtimes["apgcc"].long_side_cap is None


def test_runtime_config_rejects_mixed_approved_shortlist(tmp_path: Path) -> None:
    payload = json.loads(
        (
            REPO_ROOT
            / "configs/evaluation/round2_reference_home5090.json"
        ).read_text(encoding="utf-8")
    )
    payload["models"]["pet"] = payload["models"].pop("dm-count")
    path = tmp_path / "mixed-runtime.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="approved ordered model shortlist"):
        load_round2_runtime_config(path)


def test_steerer_checkpoint_keeps_ucf_result_research_only(tmp_path: Path) -> None:
    manifest = tmp_path / "dataset.json"
    manifest.write_text("{}", encoding="utf-8")
    lane = DatasetLane(
        dataset_id="ucf-qnrf-kaggle-apache",
        partition="test",
        samples=334,
        rights_scope="PASS_COMMERCIAL_CANDIDATE",
        manifest_path=manifest,
    )
    runtime = _runtime(tmp_path)

    protocol = build_round2_protocol(
        "steerer",
        lane,
        runtime=runtime,
        model_config=_targets(),
    )

    path = write_scoped_rights_decision(
        tmp_path / "scoped.json", runtime=runtime, lane=lane
    )
    payload = json.loads(path.read_text(encoding="utf-8"))

    assert payload["dataset_rights_scope"] == "PASS_COMMERCIAL_CANDIDATE"
    assert payload["rights_scope"] == "PASS_RESEARCH_ONLY"
    assert protocol.split_verified is True
    assert protocol.leakage_free is True
    assert protocol.checkpoint_training_split_status == "UNKNOWN"
    assert protocol.comparison_scope == "research_reference_only"
