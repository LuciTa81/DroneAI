"""Run one frozen model × dataset lane from the Round 2 reference matrix."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from dataclasses import asdict, replace
from pathlib import Path
from typing import Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from droneai.evaluation_artifacts import artifact_reference, write_json
from droneai.evaluation_metrics import evaluate_sample
from droneai.evaluation_panels import render_review_panel
from droneai.evaluation_runner import run_evaluation
from droneai.integrity import is_sha256, sha256_file
from droneai.model_brief import write_model_brief
from droneai.round2_config import (
    SUPPORTED_MODEL_IDS,
    DatasetLane,
    load_round2_config,
)
from droneai.round2_manifest import verify_round2_manifest
from droneai.round2_runner import (
    build_round2_adapter,
    build_round2_protocol,
    load_and_validate_model_config,
    load_round2_runtime_config,
    scope_adapter_to_dataset,
    write_scoped_rights_decision,
)
from droneai.runtime_probe import collect_environment


DEFAULT_RUNTIME_CONFIG = REPO_ROOT / "configs/evaluation/round2_reference_home5090.json"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a frozen Round 2 model × dataset reference lane."
    )
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--runtime-config", type=Path, default=DEFAULT_RUNTIME_CONFIG)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--ucf-root", type=Path)
    parser.add_argument("--jhu-root", type=Path)
    parser.add_argument("--up-count-root", type=Path)
    parser.add_argument("--model", choices=SUPPORTED_MODEL_IDS)
    parser.add_argument(
        "--dataset",
        choices=("ucf-qnrf-kaggle-apache", "jhu-crowd-plus-v2", "up-count-v1"),
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--one-sample", action="store_true")
    mode.add_argument("--resume", action="store_true")
    return parser


def _load_manifest(path: Path, *, expected_round_id: str) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise ValueError("Round 2 manifest requires schema_version=1")
    if payload.get("round_id") != expected_round_id:
        raise ValueError("Round 2 manifest round identity mismatch")
    digest = payload.get("sample_manifest_sha256")
    if not isinstance(digest, str) or not is_sha256(digest):
        raise ValueError("Round 2 manifest sample SHA-256 is invalid")
    return payload


def _dry_run_payload(config, manifest: dict[str, object]) -> dict[str, object]:
    return {
        "round_id": config.round_id,
        "action": config.action,
        "fine_tuning": config.fine_tuning,
        "combined_rights_scope": config.combined_rights_scope,
        "sample_manifest_sha256": manifest["sample_manifest_sha256"],
        "runs": [
            {
                "model_id": model.model_id,
                "dataset_id": lane.dataset_id,
                "partition": lane.partition,
                "samples": lane.samples,
                "dataset_rights_scope": lane.rights_scope,
                "comparison_scope": "research_reference_only",
                "training": False,
                "fine_tuning": False,
            }
            for model in config.models
            for lane in config.datasets
        ],
    }


def _required(value: object, name: str) -> object:
    if value is None:
        raise ValueError(f"{name} is required unless --dry-run is used")
    return value


def _dataset_roots(args: argparse.Namespace) -> dict[str, Path]:
    return {
        "ucf-qnrf-kaggle-apache": Path(_required(args.ucf_root, "--ucf-root")),
        "jhu-crowd-plus-v2": Path(_required(args.jhu_root, "--jhu-root")),
        "up-count-v1": Path(_required(args.up_count_root, "--up-count-root")),
    }


def _copy_immutable(source: Path, destination: Path) -> Path:
    content = source.read_bytes()
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if destination.read_bytes() != content:
            raise FileExistsError(f"provenance snapshot differs: {destination}")
        return destination
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    if temporary.exists():
        raise FileExistsError(f"provenance staging file exists: {temporary}")
    temporary.write_bytes(content)
    temporary.replace(destination)
    return destination


def _run_one_sample(*, adapter, sample, protocol, output_dir: Path, provenance) -> int:
    reserved = (
        "one-sample-result.json",
        "one-sample-panel.png",
        "model-brief.md",
        "environment-summary.json",
        "evidence-manifest.json",
    )
    if any((output_dir / name).exists() for name in reserved):
        raise FileExistsError("one-sample output already contains result artifacts")
    prediction = adapter.predict(sample, retain_native=True)
    record = evaluate_sample(
        sample,
        prediction,
        localization_radius=protocol.localization_radius,
    )
    if record.failure_state is not None:
        raise RuntimeError(f"one-sample prediction failed: {record.failure_state}")
    brief = adapter.brief()
    panel = render_review_panel(
        sample,
        prediction,
        record,
        output_dir / "one-sample-panel.png",
        model_id=brief.model_id,
        checkpoint_sha256=brief.checkpoint_sha256,
        zone_warning_count=protocol.zone_warning_count,
        zone_critical_count=protocol.zone_critical_count,
        category="round2_one_sample_preflight",
    )
    result = write_json(
        output_dir / "one-sample-result.json",
        {
            "schema_version": 1,
            "protocol": asdict(protocol),
            "record": asdict(record),
            "native_metadata": prediction.metadata,
            "native_output_type": prediction.output_type,
            "dataset_rights_scope": brief.rights_status,
            "training": False,
            "fine_tuning": False,
        },
    )
    brief_path = write_model_brief(output_dir / "model-brief.md", brief)
    environment = write_json(
        output_dir / "environment-summary.json", collect_environment(REPO_ROOT)
    )
    references = [
        artifact_reference(path, base_dir=output_dir)
        for path in (*provenance, result, panel, brief_path, environment)
    ]
    evidence = write_json(
        output_dir / "evidence-manifest.json",
        {
            "schema_version": 1,
            "sample_id": sample.sample_id,
            "artifacts": references,
        },
    )
    print(
        json.dumps(
            {
                "status": brief.rights_status,
                "model_id": brief.model_id,
                "dataset_id": sample.dataset_id,
                "sample_id": sample.sample_id,
                "ground_truth_count": record.ground_truth_count,
                "predicted_count": record.predicted_count,
                "absolute_error": record.absolute_error,
                "latency_ms": record.latency_ms,
                "peak_vram_mb": record.peak_vram_mb,
                "evidence_sha256": sha256_file(evidence),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    config = load_round2_config(args.config)
    runtimes = load_round2_runtime_config(args.runtime_config)
    expected_models = tuple(model.model_id for model in config.models)
    runtime_models = tuple(runtimes)
    if runtime_models != expected_models:
        raise ValueError(
            "runtime models must exactly match benchmark models: "
            f"expected={expected_models!r}, actual={runtime_models!r}"
        )
    manifest_path = args.manifest.resolve()
    manifest = _load_manifest(manifest_path, expected_round_id=config.round_id)
    if args.dry_run:
        print(json.dumps(_dry_run_payload(config, manifest), indent=2, sort_keys=True))
        return 0

    model_id = str(_required(args.model, "--model"))
    dataset_id = str(_required(args.dataset, "--dataset"))
    output_dir = Path(_required(args.output_dir, "--output-dir")).resolve()
    roots = _dataset_roots(args)
    samples = verify_round2_manifest(manifest, config, roots)
    lane = next((value for value in config.datasets if value.dataset_id == dataset_id), None)
    if lane is None:
        raise ValueError(f"dataset is not in Round 2 config: {dataset_id}")
    lane_samples = tuple(sample for sample in samples if sample.dataset_id == dataset_id)
    if len(lane_samples) != lane.samples:
        raise ValueError("verified manifest lane count differs from config")

    runtime = runtimes[model_id]
    model_config = load_and_validate_model_config(runtime)
    output_dir.mkdir(parents=True, exist_ok=True)
    shared_manifest = _copy_immutable(
        manifest_path, output_dir / "round2-shared-manifest.json"
    )
    dataset_manifest = _copy_immutable(
        lane.manifest_path, output_dir / "round2-dataset-manifest.json"
    )
    scoped_rights = write_scoped_rights_decision(
        output_dir / "round2-rights-scope.json", runtime=runtime, lane=lane
    )
    scoped_runtime = replace(runtime, rights_decision=scoped_rights)
    adapter = scope_adapter_to_dataset(
        build_round2_adapter(model_id, runtime=runtime), lane
    )
    protocol = build_round2_protocol(
        model_id,
        lane,
        runtime=scoped_runtime,
        model_config=model_config,
        round_id=config.round_id,
        expected_samples=1 if args.one_sample else lane.samples,
    )
    provenance = (shared_manifest, dataset_manifest, scoped_rights)
    os.chdir(REPO_ROOT)
    if args.one_sample:
        first = min(lane_samples, key=lambda sample: sample.sample_id)
        return _run_one_sample(
            adapter=adapter,
            sample=first,
            protocol=protocol,
            output_dir=output_dir,
            provenance=provenance,
        )
    report = run_evaluation(
        adapter=adapter,
        samples=lane_samples,
        protocol=protocol,
        output_dir=output_dir,
        provenance_artifacts=provenance,
        resume=args.resume,
    )
    print(report.to_markdown())
    return 0 if report.is_success else 2


if __name__ == "__main__":
    raise SystemExit(main())
