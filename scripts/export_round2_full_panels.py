"""Export one accepted Round 2 lane as a complete resumable PNG panel set."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import uuid
from pathlib import Path
from typing import Callable, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from droneai.evaluation_contract import EvaluationSample, ScalarEvaluation
from droneai.evaluation_metrics import evaluate_sample
from droneai.evaluation_panels import render_review_panel
from droneai.full_panel_export import (
    AcceptedPrediction,
    PanelExportIdentity,
    PanelProgressLedger,
    load_accepted_predictions,
    panel_filename,
    verify_fresh_record,
)
from droneai.integrity import is_sha256, sha256_file
from droneai.round2_config import SUPPORTED_MODEL_IDS, load_round2_config
from droneai.round2_manifest import verify_round2_manifest
from droneai.round2_runner import (
    build_round2_adapter,
    build_round2_protocol,
    load_and_validate_model_config,
    load_round2_runtime_config,
    scope_adapter_to_dataset,
)
from droneai.runtime_probe import collect_environment

_DATASETS = (
    "ucf-qnrf-kaggle-apache",
    "jhu-crowd-plus-v2",
    "up-count-v1",
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export all native Round 2 result panels for one frozen lane."
    )
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--runtime-config", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--ucf-root", required=True, type=Path)
    parser.add_argument("--jhu-root", required=True, type=Path)
    parser.add_argument("--up-count-root", required=True, type=Path)
    parser.add_argument("--model", required=True, choices=SUPPORTED_MODEL_IDS)
    parser.add_argument("--dataset", required=True, choices=_DATASETS)
    parser.add_argument("--result-lane", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--max-samples", type=int)
    return parser


def validate_output_boundary(
    *,
    result_lane: str | Path,
    output_dir: str | Path,
) -> None:
    accepted = Path(result_lane).resolve()
    output = Path(output_dir).resolve()
    try:
        output.relative_to(accepted)
    except ValueError:
        return
    raise ValueError("full-panel output must stay outside the accepted result lane")


def _write_json_new(path: Path, payload: Mapping[str, object]) -> Path:
    content = (
        json.dumps(
            payload,
            indent=2,
            ensure_ascii=False,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(content, encoding="utf-8", newline="\n")
    try:
        if path.exists():
            raise FileExistsError(f"export summary already exists: {path}")
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return path


def _load_existing_summary(
    path: Path,
    *,
    expected_panels: int,
    identity: PanelExportIdentity,
) -> dict[str, object] | None:
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if (
        not isinstance(payload, dict)
        or payload.get("schema_version") != 1
        or payload.get("panel_count") != expected_panels
        or payload.get("model_id") != identity.model_id
        or payload.get("dataset_id") != identity.dataset_id
        or payload.get("checkpoint_sha256") != identity.checkpoint_sha256
        or payload.get("shared_manifest_sha256")
        != identity.shared_manifest_sha256
    ):
        raise ValueError("existing export summary differs from requested lane")
    return payload


def export_lane(
    *,
    adapter,
    samples: Sequence[EvaluationSample],
    protocol,
    accepted_predictions: Mapping[str, AcceptedPrediction],
    identity: PanelExportIdentity,
    output_dir: str | Path,
    max_samples: int | None = None,
    evaluate: Callable[..., ScalarEvaluation] = evaluate_sample,
    render_panel: Callable[..., Path] = render_review_panel,
    environment: Mapping[str, object] | None = None,
) -> dict[str, object]:
    ordered = tuple(samples)
    if not ordered:
        raise ValueError("full-panel export requires at least one sample")
    sample_ids = tuple(sample.sample_id for sample in ordered)
    if len(set(sample_ids)) != len(sample_ids):
        raise ValueError("full-panel export sample identities are duplicated")
    if set(sample_ids) != set(accepted_predictions):
        raise ValueError(
            "accepted prediction sample identities differ from verified samples"
        )
    if max_samples is not None:
        if isinstance(max_samples, bool) or max_samples <= 0:
            raise ValueError("max_samples must be positive")
        ordered = ordered[:max_samples]

    output = Path(output_dir).resolve()
    panels = output / "panels"
    panels.mkdir(parents=True, exist_ok=True)
    ledger = PanelProgressLedger.open(
        output / "progress.jsonl",
        identity=identity,
        output_root=output,
    )
    target_ids = tuple(sample.sample_id for sample in ordered)
    completed = set(ledger.completed_sample_ids())
    if not completed <= set(target_ids):
        raise ValueError("progress ledger contains a sample outside this export")
    summary_path = output / "export-summary.json"
    existing = _load_existing_summary(
        summary_path,
        expected_panels=len(ordered),
        identity=identity,
    )
    if existing is not None:
        if completed != set(target_ids):
            raise ValueError("complete export summary has incomplete progress")
        return existing

    started = time.perf_counter()
    for sample in ordered:
        if sample.sample_id in completed:
            continue
        accepted = accepted_predictions[sample.sample_id]
        prediction = adapter.predict(sample, retain_native=True)
        fresh = evaluate(
            sample,
            prediction,
            localization_radius=protocol.localization_radius,
        )
        verify_fresh_record(accepted, fresh)
        final_panel = panels / panel_filename(sample.sample_id)
        temporary = panels / (
            f".{final_panel.stem}.{uuid.uuid4().hex}.tmp.png"
        )
        try:
            render_panel(
                sample,
                prediction,
                fresh,
                temporary,
                model_id=identity.model_id,
                checkpoint_sha256=identity.checkpoint_sha256,
                zone_warning_count=protocol.zone_warning_count,
                zone_critical_count=protocol.zone_critical_count,
                category="company_share_full",
            )
            if final_panel.exists():
                if sha256_file(final_panel) != sha256_file(temporary):
                    raise FileExistsError(
                        f"untracked final panel differs: {final_panel}"
                    )
            else:
                temporary.replace(final_panel)
            ledger.append_completed(
                source_sha256=sample.source_sha256,
                accepted=accepted,
                fresh=fresh,
                panel_path=final_panel,
            )
            completed.add(sample.sample_id)
        finally:
            if temporary.exists():
                temporary.unlink()

    if completed != set(target_ids):
        raise RuntimeError("full-panel export ended with incomplete progress")
    panel_paths = tuple(sorted(panels.glob("*.png")))
    if len(panel_paths) != len(ordered):
        raise RuntimeError("full-panel directory count differs from export target")
    elapsed = time.perf_counter() - started
    payload: dict[str, object] = {
        "schema_version": 1,
        "status": "PASS_RESEARCH_ONLY",
        "model_id": identity.model_id,
        "dataset_id": identity.dataset_id,
        "panel_count": len(panel_paths),
        "sample_count": len(ordered),
        "count_mismatches": 0,
        "explicit_failures": 0,
        "training": False,
        "fine_tuning": False,
        "calibration": False,
        "git_commit": identity.git_commit,
        "upstream_commit": identity.upstream_commit,
        "checkpoint_sha256": identity.checkpoint_sha256,
        "shared_manifest_sha256": identity.shared_manifest_sha256,
        "progress_sha256": sha256_file(output / "progress.jsonl"),
        "panel_bytes": sum(path.stat().st_size for path in panel_paths),
        "elapsed_seconds_this_invocation": elapsed,
        "environment": dict(environment or {}),
    }
    _write_json_new(summary_path, payload)
    return payload


def _load_manifest_identity(
    path: Path,
    *,
    expected_round_id: str,
) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if (
        not isinstance(payload, dict)
        or payload.get("schema_version") != 1
        or payload.get("round_id") != expected_round_id
    ):
        raise ValueError("Round 2 shared manifest identity is invalid")
    digest = payload.get("sample_manifest_sha256")
    if not is_sha256(digest):
        raise ValueError("Round 2 shared manifest SHA-256 is invalid")
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.max_samples is not None and args.max_samples <= 0:
        raise ValueError("--max-samples must be positive")
    validate_output_boundary(
        result_lane=args.result_lane,
        output_dir=args.output_dir,
    )
    config = load_round2_config(args.config)
    runtimes = load_round2_runtime_config(args.runtime_config)
    expected_models = tuple(model.model_id for model in config.models)
    if tuple(runtimes) != expected_models:
        raise ValueError("runtime models must exactly match benchmark models")
    if args.model not in runtimes:
        raise ValueError("requested model is not in the Round 2 runtime")
    lane = next(
        (
            candidate
            for candidate in config.datasets
            if candidate.dataset_id == args.dataset
        ),
        None,
    )
    if lane is None:
        raise ValueError("requested dataset is not in the Round 2 config")
    roots = {
        "ucf-qnrf-kaggle-apache": args.ucf_root,
        "jhu-crowd-plus-v2": args.jhu_root,
        "up-count-v1": args.up_count_root,
    }
    manifest_path = args.manifest.resolve()
    manifest = _load_manifest_identity(
        manifest_path,
        expected_round_id=config.round_id,
    )
    all_samples = verify_round2_manifest(manifest, config, roots)
    samples = tuple(
        sample for sample in all_samples if sample.dataset_id == args.dataset
    )
    if len(samples) != lane.samples:
        raise ValueError("verified sample count differs from the dataset lane")

    result_lane = args.result_lane.resolve()
    required_result_files = (
        "predictions.csv",
        "metrics.json",
        "score.json",
        "evidence-manifest.json",
    )
    for name in required_result_files:
        if not (result_lane / name).is_file():
            raise FileNotFoundError(f"accepted result artifact is missing: {name}")
    accepted = load_accepted_predictions(result_lane / "predictions.csv")

    runtime = runtimes[args.model]
    model_config = load_and_validate_model_config(runtime)
    adapter = scope_adapter_to_dataset(
        build_round2_adapter(args.model, runtime=runtime),
        lane,
    )
    protocol = build_round2_protocol(
        args.model,
        lane,
        runtime=runtime,
        model_config=model_config,
        round_id=config.round_id,
    )
    environment = collect_environment(REPO_ROOT)
    git_commit = environment.get("git_commit")
    if environment.get("git_dirty") is not False:
        raise ValueError("full-panel export requires a clean DroneAI repository")
    if not isinstance(git_commit, str):
        raise ValueError("full-panel export requires the DroneAI Git commit")
    identity = PanelExportIdentity(
        model_id=args.model,
        dataset_id=args.dataset,
        git_commit=git_commit,
        upstream_commit=runtime.upstream_commit,
        checkpoint_sha256=runtime.checkpoint_sha256,
        shared_manifest_sha256=str(manifest["sample_manifest_sha256"]),
    )
    os.chdir(REPO_ROOT)
    summary = export_lane(
        adapter=adapter,
        samples=samples,
        protocol=protocol,
        accepted_predictions=accepted,
        identity=identity,
        output_dir=args.output_dir,
        max_samples=args.max_samples,
        environment=environment,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
