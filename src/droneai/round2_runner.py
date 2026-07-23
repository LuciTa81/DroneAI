"""Dataset-neutral execution helpers for the frozen Round 2 benchmark."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping, cast

from droneai.evaluation_contract import ModelAdapter
from droneai.evaluation_runner import EvaluationProtocol
from droneai.integrity import is_sha256, sha256_file
from droneai.round2_config import (
    APPROVED_MODEL_MATRICES,
    SUPPORTED_MODEL_IDS,
    DatasetLane,
)


@dataclass(frozen=True)
class RuntimeModel:
    model_id: str
    model_config: Path
    upstream_dir: Path
    upstream_commit: str
    checkpoint: Path
    checkpoint_sha256: str
    rights_decision: Path
    rights_manifest: Path
    device: str
    long_side_cap: int | None = None
    patch_size: int | None = None

    def __post_init__(self) -> None:
        if self.model_id not in SUPPORTED_MODEL_IDS:
            raise ValueError(f"unsupported Round 2 model: {self.model_id}")
        if re.fullmatch(r"[0-9a-f]{40}", self.upstream_commit) is None:
            raise ValueError("runtime model requires a pinned upstream commit")
        if not is_sha256(self.checkpoint_sha256):
            raise ValueError("runtime model requires a checkpoint SHA-256")
        if self.device not in {"cpu", "cuda", "cuda:0"}:
            raise ValueError("runtime model device is invalid")
        for value in (self.long_side_cap, self.patch_size):
            if value is not None and (isinstance(value, bool) or value <= 0):
                raise ValueError("optional runtime dimensions must be positive")


def _repository_root(config_path: Path) -> Path:
    try:
        return config_path.resolve().parents[2]
    except IndexError as exc:
        raise ValueError("runtime config must live under configs/evaluation") from exc


def _resolved_path(value: object, *, repository_root: Path) -> Path:
    path = Path(str(value))
    if path.is_absolute():
        return path
    return (repository_root / path).resolve()


def load_round2_runtime_config(path: str | Path) -> dict[str, RuntimeModel]:
    config_path = Path(path).resolve()
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise ValueError("Round 2 runtime config requires schema_version=1")
    raw_models = payload.get("models")
    if (
        not isinstance(raw_models, dict)
        or tuple(raw_models) not in APPROVED_MODEL_MATRICES
    ):
        raise ValueError(
            "runtime config must contain exactly an approved ordered model shortlist"
        )
    root = _repository_root(config_path)
    result: dict[str, RuntimeModel] = {}
    required = {
        "model_config",
        "upstream_dir",
        "upstream_commit",
        "checkpoint",
        "checkpoint_sha256",
        "rights_decision",
        "rights_manifest",
        "device",
    }
    for model_id, raw in raw_models.items():
        if not isinstance(raw, dict) or not required <= set(raw):
            raise ValueError(f"runtime model fields are incomplete: {model_id}")
        result[model_id] = RuntimeModel(
            model_id=model_id,
            model_config=_resolved_path(raw["model_config"], repository_root=root),
            upstream_dir=_resolved_path(raw["upstream_dir"], repository_root=root),
            upstream_commit=str(raw["upstream_commit"]).lower(),
            checkpoint=_resolved_path(raw["checkpoint"], repository_root=root),
            checkpoint_sha256=str(raw["checkpoint_sha256"]).lower(),
            rights_decision=_resolved_path(raw["rights_decision"], repository_root=root),
            rights_manifest=_resolved_path(raw["rights_manifest"], repository_root=root),
            device=str(raw["device"]),
            long_side_cap=(
                int(raw["long_side_cap"]) if "long_side_cap" in raw else None
            ),
            patch_size=int(raw["patch_size"]) if "patch_size" in raw else None,
        )
    return result


def load_and_validate_model_config(runtime: RuntimeModel) -> dict[str, object]:
    """Validate the accepted Round 1 model config and its rights decision."""

    if runtime.model_id == "steerer":
        from droneai.steerer_smoke import (
            load_steerer_smoke_config,
            validate_steerer_rights_decision,
        )

        config = load_steerer_smoke_config(runtime.model_config)
        validate_steerer_rights_decision(
            runtime.rights_decision,
            manifest_path=runtime.rights_manifest,
            expected_candidate_id=str(config["candidate_id"]),
        )
    elif runtime.model_id == "dm-count":
        from droneai.dm_count_smoke import load_smoke_config, validate_rights_decision

        config = load_smoke_config(runtime.model_config)
        validate_rights_decision(
            runtime.rights_decision,
            manifest_path=runtime.rights_manifest,
            expected_candidate_id=str(config["candidate_id"]),
        )
    elif runtime.model_id == "mpcount":
        from droneai.mpcount_smoke import (
            load_mpcount_smoke_config,
            validate_mpcount_rights_decision,
        )

        config = load_mpcount_smoke_config(runtime.model_config)
        validate_mpcount_rights_decision(
            runtime.rights_decision,
            manifest_path=runtime.rights_manifest,
            expected_candidate_id=str(config["candidate_id"]),
        )
    else:  # pragma: no cover - RuntimeModel rejects this earlier
        raise ValueError(f"unsupported Round 2 model: {runtime.model_id}")
    if str(config.get("upstream_commit")) != runtime.upstream_commit:
        raise ValueError("runtime upstream commit differs from accepted model config")
    return config


def build_round2_adapter(
    model_id: str,
    *,
    runtime: RuntimeModel,
    adapter_types: Mapping[str, type[Any]] | None = None,
) -> ModelAdapter:
    if model_id != runtime.model_id:
        raise ValueError("requested model differs from runtime model identity")
    if adapter_types is None:
        from droneai.dm_count_adapter import DMCountAdapter
        from droneai.mpcount_adapter import MPCountAdapter
        from droneai.steerer_adapter import STEERERAdapter

        adapter_types = {
            "steerer": STEERERAdapter,
            "dm-count": DMCountAdapter,
            "mpcount": MPCountAdapter,
        }
    if model_id not in adapter_types:
        raise ValueError(f"unsupported Round 2 model: {model_id}")
    kwargs: dict[str, object] = {
        "upstream_dir": runtime.upstream_dir,
        "expected_upstream_commit": runtime.upstream_commit,
        "checkpoint_path": runtime.checkpoint,
        "checkpoint_sha256": runtime.checkpoint_sha256,
        "device": runtime.device,
    }
    if model_id == "steerer" and runtime.long_side_cap is not None:
        kwargs["long_side_cap"] = runtime.long_side_cap
    if model_id == "mpcount" and runtime.patch_size is not None:
        kwargs["patch_size"] = runtime.patch_size
    return cast(ModelAdapter, adapter_types[model_id](**kwargs))


class DatasetRightsAdapter:
    """Retain model inference while narrowing its brief to dataset rights."""

    def __init__(self, adapter: ModelAdapter, lane: DatasetLane) -> None:
        self._adapter = adapter
        self._lane = lane

    def predict(self, sample, *, retain_native: bool = False):
        return self._adapter.predict(sample, retain_native=retain_native)

    def brief(self):
        brief = self._adapter.brief()
        if self._lane.rights_scope != "PASS_RESEARCH_ONLY":
            return brief
        return replace(
            brief,
            rights_status="PASS_RESEARCH_ONLY",
            dataset_rights_status=(
                f"{self._lane.dataset_id} is PASS_RESEARCH_ONLY; "
                "frozen internal reference evaluation only"
            ),
            deployment_rights_status=(
                "blocked for this dataset lane; no training, derived weights, "
                "commercial deployment, or production approval"
            ),
        )


def scope_adapter_to_dataset(adapter: ModelAdapter, lane: DatasetLane) -> ModelAdapter:
    return cast(ModelAdapter, DatasetRightsAdapter(adapter, lane))


def _split_id(round_id: str, lane: DatasetLane) -> str:
    return (
        f"{round_id}-{lane.dataset_id.replace('_', '-')}-"
        f"{lane.partition.replace('_', '-')}"
    )


def _checkpoint_claim(model_id: str, lane: DatasetLane) -> tuple[str, str]:
    if model_id == "steerer":
        return (
            "UNKNOWN",
            "The official STEERER QNRF configuration names separate train.txt and "
            "test.txt partitions, but exact checkpoint membership is unaudited; "
            f"this {lane.dataset_id} result is descriptive research reference only.",
        )
    if model_id == "dm-count":
        return (
            "VERIFIED_DISJOINT",
            "The accepted DM-Count checkpoint uses the official UCF-QNRF Train "
            "protocol; UCF Test and the two external datasets are disjoint.",
        )
    if model_id == "mpcount":
        return (
            "VERIFIED_DISJOINT",
            "The accepted MPCount checkpoint was trained on ShanghaiTech-A, which "
            f"is disjoint from {lane.dataset_id}.",
        )
    raise ValueError(f"unsupported Round 2 model: {model_id}")


def build_round2_protocol(
    model_id: str,
    lane: DatasetLane,
    *,
    runtime: RuntimeModel,
    model_config: Mapping[str, object],
    round_id: str = "round-2-reference-benchmark-v1",
    expected_samples: int | None = None,
) -> EvaluationProtocol:
    if model_id != runtime.model_id:
        raise ValueError("protocol model differs from runtime model")
    targets = model_config.get("targets")
    if not isinstance(targets, dict):
        raise ValueError("accepted model config is missing targets")
    status, evidence = _checkpoint_claim(model_id, lane)
    if lane.partition == "test":
        split_role = "test"
    else:
        split_role = "validation"
    sample_count = lane.samples if expected_samples is None else expected_samples
    return EvaluationProtocol(
        run_id=f"{round_id}-{model_id}-{lane.dataset_id}",
        protocol_id=f"{round_id}-{model_id}-{lane.dataset_id}-{lane.partition}",
        dataset_id=lane.dataset_id,
        split_id=_split_id(round_id, lane),
        split_role=cast(Any, split_role),
        expected_samples=sample_count,
        split_verified=True,
        # Dataset/split isolation is verified independently from whether a third-party
        # checkpoint's exact training membership can be audited. The latter remains
        # UNKNOWN and therefore never becomes a held-out or ranking claim.
        leakage_free=True,
        checkpoint_training_split_status=status,
        comparison_scope="research_reference_only",
        checkpoint_split_evidence=evidence,
        sealed_test_access_approved=lane.partition == "test",
        require_clean_git=True,
        rights_decision_path=str(runtime.rights_decision),
        rights_decision_sha256=sha256_file(runtime.rights_decision),
        localization_radius=float(model_config["localization_radius"]),
        mae_max=float(targets["mae_max"]),
        rmse_max=float(targets["rmse_max"]),
        bias_max=float(targets["bias_max"]),
        band_bias_max=float(targets["band_bias_max"]),
        condition_bias_max=float(targets["condition_bias_max"]),
        latency_max_ms=float(targets["latency_max_ms"]),
        vram_max_mb=float(targets["vram_max_mb"]),
        spatial_metric_name=str(targets["spatial_metric_name"]),
        spatial_direction=cast(Any, str(targets["spatial_direction"])),
        spatial_target=float(targets["spatial_target"]),
        required_density_bands=tuple(
            str(value) for value in cast(list[object], targets["required_density_bands"])
        ),
        density_band_rules=tuple(
            str(value) for value in cast(list[object], targets["density_band_rules"])
        ),
        required_condition_keys=tuple(
            str(value) for value in cast(list[object], targets["required_condition_keys"])
        ),
        zone_warning_count=float(targets["zone_warning_count"]),
        zone_critical_count=float(targets["zone_critical_count"]),
    )


def write_scoped_rights_decision(
    path: str | Path,
    *,
    runtime: RuntimeModel,
    lane: DatasetLane,
) -> Path:
    target = Path(path)
    effective_scope = (
        "PASS_RESEARCH_ONLY"
        if lane.rights_scope == "PASS_RESEARCH_ONLY" or runtime.model_id == "steerer"
        else lane.rights_scope
    )
    payload = {
        "schema_version": 1,
        "action": "frozen_internal_reference_evaluation",
        "model_id": runtime.model_id,
        "dataset_id": lane.dataset_id,
        "rights_scope": effective_scope,
        "dataset_rights_scope": lane.rights_scope,
        "model_rights_decision": {
            "path": str(runtime.rights_decision),
            "sha256": sha256_file(runtime.rights_decision),
        },
        "dataset_manifest": {
            "path": str(lane.manifest_path),
            "sha256": sha256_file(lane.manifest_path),
        },
        "training": False,
        "fine_tuning": False,
        "derived_weight_use": False,
        "production_approval": False,
    }
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if target.read_text(encoding="utf-8") != text:
            raise FileExistsError(f"scoped rights decision differs: {target}")
        return target
    target.write_text(text, encoding="utf-8")
    return target


__all__ = [
    "DatasetRightsAdapter",
    "RuntimeModel",
    "build_round2_adapter",
    "build_round2_protocol",
    "load_and_validate_model_config",
    "load_round2_runtime_config",
    "scope_adapter_to_dataset",
    "write_scoped_rights_decision",
]
