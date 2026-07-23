from __future__ import annotations

import json
from pathlib import Path

import pytest

from droneai.round2_config import DatasetLane
from droneai.round2_runner import (
    RuntimeModel,
    build_round2_adapter,
    build_round2_protocol,
    load_and_validate_model_config,
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
        long_side_cap=(
            3072
            if model_id == "steerer"
            else 1536 if model_id == "pet" else None
        ),
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


@pytest.mark.parametrize(
    ("model_id", "extra_kwargs"),
    [
        ("pet", {"long_side_cap": 1536}),
        ("apgcc", {}),
    ],
)
def test_point_adapter_factory_uses_existing_adapter_contract(
    tmp_path: Path,
    model_id: str,
    extra_kwargs: dict[str, object],
) -> None:
    calls: list[dict[str, object]] = []

    class FakeAdapter:
        def __init__(self, **kwargs: object) -> None:
            calls.append(kwargs)

    runtime = _runtime(tmp_path, model_id)
    adapter = build_round2_adapter(
        model_id,
        runtime=runtime,
        adapter_types={model_id: FakeAdapter},
    )

    assert isinstance(adapter, FakeAdapter)
    assert calls == [
        {
            "upstream_dir": runtime.upstream_dir,
            "expected_upstream_commit": "a" * 40,
            "checkpoint_path": runtime.checkpoint,
            "checkpoint_sha256": "b" * 64,
            "device": "cuda",
            **extra_kwargs,
        }
    ]


@pytest.mark.parametrize(
    (
        "model_id",
        "loader_path",
        "validator_path",
    ),
    [
        (
            "pet",
            "droneai.pet_smoke.load_pet_smoke_config",
            "droneai.pet_smoke.validate_pet_rights_decision",
        ),
        (
            "apgcc",
            "droneai.apgcc_smoke.load_apgcc_smoke_config",
            "droneai.apgcc_smoke.validate_apgcc_rights_decision",
        ),
    ],
)
def test_point_model_config_uses_accepted_loader_and_rights_validator(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    model_id: str,
    loader_path: str,
    validator_path: str,
) -> None:
    calls: list[tuple[object, ...]] = []
    runtime = _runtime(tmp_path, model_id)

    def fake_loader(path: Path) -> dict[str, object]:
        calls.append(("load", path))
        return {
            "candidate_id": f"{model_id}-candidate",
            "upstream_commit": "a" * 40,
        }

    def fake_validator(
        path: Path,
        *,
        manifest_path: Path,
        expected_candidate_id: str,
    ) -> dict[str, object]:
        calls.append(
            (
                "validate",
                path,
                manifest_path,
                expected_candidate_id,
            )
        )
        return {}

    monkeypatch.setattr(loader_path, fake_loader)
    monkeypatch.setattr(validator_path, fake_validator)

    config = load_and_validate_model_config(runtime)

    assert config["candidate_id"] == f"{model_id}-candidate"
    assert calls == [
        ("load", runtime.model_config),
        (
            "validate",
            runtime.rights_decision,
            runtime.rights_manifest,
            f"{model_id}-candidate",
        ),
    ]


@pytest.mark.parametrize(
    (
        "model_id",
        "dataset_id",
        "dataset_scope",
        "expected_checkpoint_status",
        "expected_rights_scope",
    ),
    [
        (
            "pet",
            "ucf-qnrf-kaggle-apache",
            "PASS_COMMERCIAL_CANDIDATE",
            "UNKNOWN",
            "PASS_RESEARCH_ONLY",
        ),
        (
            "apgcc",
            "ucf-qnrf-kaggle-apache",
            "PASS_COMMERCIAL_CANDIDATE",
            "VERIFIED_DISJOINT",
            "PASS_COMMERCIAL_CANDIDATE",
        ),
        (
            "apgcc",
            "jhu-crowd-plus-v2",
            "PASS_RESEARCH_ONLY",
            "VERIFIED_DISJOINT",
            "PASS_RESEARCH_ONLY",
        ),
        (
            "apgcc",
            "up-count-v1",
            "PASS_RESEARCH_ONLY",
            "VERIFIED_DISJOINT",
            "PASS_RESEARCH_ONLY",
        ),
    ],
)
def test_point_protocol_and_rights_are_model_and_dataset_scoped(
    tmp_path: Path,
    model_id: str,
    dataset_id: str,
    dataset_scope: str,
    expected_checkpoint_status: str,
    expected_rights_scope: str,
) -> None:
    manifest = tmp_path / f"{dataset_id}.json"
    manifest.write_text("{}", encoding="utf-8")
    lane = DatasetLane(
        dataset_id=dataset_id,
        partition=(
            "test" if dataset_id == "ucf-qnrf-kaggle-apache" else "val"
        ),
        samples=1,
        rights_scope=dataset_scope,
        manifest_path=manifest,
    )
    runtime = _runtime(tmp_path, model_id)
    point_targets = _targets()
    targets = point_targets["targets"]
    assert isinstance(targets, dict)
    targets["spatial_metric_name"] = "localization_f1"
    targets["spatial_direction"] = "maximize"
    targets["spatial_target"] = 0.0

    protocol = build_round2_protocol(
        model_id,
        lane,
        runtime=runtime,
        model_config=point_targets,
    )
    path = write_scoped_rights_decision(
        tmp_path / f"{model_id}-{dataset_id}-rights.json",
        runtime=runtime,
        lane=lane,
    )
    payload = json.loads(path.read_text(encoding="utf-8"))

    assert protocol.checkpoint_training_split_status == expected_checkpoint_status
    assert protocol.spatial_metric_name == "localization_f1"
    assert protocol.spatial_direction == "maximize"
    assert payload["rights_scope"] == expected_rights_scope
    assert payload["training"] is False
    assert payload["fine_tuning"] is False
    assert payload["production_approval"] is False
