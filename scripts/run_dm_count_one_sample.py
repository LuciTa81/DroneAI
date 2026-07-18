"""Run one explicit frozen DM-Count validation sample for report evidence."""

from __future__ import annotations

import argparse
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Sequence, cast


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from droneai.dm_count_adapter import DMCountAdapter
from droneai.dm_count_smoke import (
    load_smoke_config,
    prepare_smoke_samples,
    validate_official_split_paths,
    validate_rights_decision,
    write_split_source_manifest,
)
from droneai.evaluation_artifacts import write_json
from droneai.evaluation_metrics import evaluate_sample
from droneai.evaluation_panels import render_review_panel
from droneai.integrity import sha256_file
from droneai.model_brief import write_model_brief
from droneai.one_sample_evidence import build_artifact_bindings, select_sample
from droneai.runtime_probe import collect_environment


_select_sample = select_sample


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run one explicit frozen DM-Count validation sample")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--train-root", required=True, type=Path)
    parser.add_argument("--upstream-dir", required=True, type=Path)
    parser.add_argument("--train-list", required=True, type=Path)
    parser.add_argument("--validation-list", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--rights-decision", required=True, type=Path)
    parser.add_argument("--rights-manifest", required=True, type=Path)
    parser.add_argument("--sample-id", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--device", required=True, choices=("cuda", "cuda:0"))
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    output = args.output_dir.resolve()
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError(f"output must be new or empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    paths = {
        key: value.resolve()
        for key, value in {
            "config": args.config,
            "train_root": args.train_root,
            "upstream": args.upstream_dir,
            "train_list": args.train_list,
            "validation_list": args.validation_list,
            "checkpoint": args.checkpoint,
            "rights": args.rights_decision,
            "manifest": args.rights_manifest,
        }.items()
    }
    config = load_smoke_config(paths["config"])
    validate_rights_decision(
        paths["rights"],
        manifest_path=paths["manifest"],
        expected_candidate_id=str(config["candidate_id"]),
    )
    validate_official_split_paths(
        upstream_dir=paths["upstream"],
        train_list_path=paths["train_list"],
        validation_list_path=paths["validation_list"],
    )
    prepared = prepare_smoke_samples(
        config=config,
        train_root=paths["train_root"],
        train_list_path=paths["train_list"],
        validation_list_path=paths["validation_list"],
    )
    sample = _select_sample(prepared.samples, args.sample_id)
    split_manifest = write_split_source_manifest(
        output / "split-source-manifest.json",
        prepared=prepared,
        train_list_path=paths["train_list"],
        validation_list_path=paths["validation_list"],
    )
    adapter = DMCountAdapter(
        upstream_dir=paths["upstream"],
        expected_upstream_commit=str(config["upstream_commit"]),
        checkpoint_path=paths["checkpoint"],
        checkpoint_sha256=args.checkpoint_sha256,
        device=args.device,
    )
    brief = adapter.brief()
    brief.require_full_run_approval()
    prediction = adapter.predict(sample, retain_native=True)
    record = evaluate_sample(
        sample, prediction, localization_radius=float(config["localization_radius"])
    )
    write_model_brief(output / "model-brief.md", brief)
    environment_path = write_json(output / "environment-summary.json", collect_environment(ROOT))
    panel_path = None
    if prediction.failure_state is None:
        targets = cast(dict[str, object], config["targets"])
        panel_path = render_review_panel(
            sample,
            prediction,
            record,
            output / "one-sample-panel.png",
            model_id=brief.model_id,
            checkpoint_sha256=brief.checkpoint_sha256,
            zone_warning_count=float(targets["zone_warning_count"]),
            zone_critical_count=float(targets["zone_critical_count"]),
            category="report_scenario",
        )
    artifacts = None
    if panel_path is not None:
        artifacts = build_artifact_bindings(
            sample=sample,
            split_manifest_path=split_manifest,
            checkpoint_path=paths["checkpoint"],
            checkpoint_sha256=brief.checkpoint_sha256,
            rights_decision_path=paths["rights"],
            environment_path=environment_path,
            panel_path=panel_path,
        )
    result = {
        "schema_version": 1,
        "status": "PASS" if prediction.failure_state is None else "FAIL",
        "gate": "one_sample",
        "evaluation_scope": "research_comparison_only",
        "comparison_scope": str(config["comparison_scope"]),
        "sample_id": sample.sample_id,
        "dataset_id": sample.dataset_id,
        "split_id": sample.split_id,
        "split_role": str(config["split_role"]),
        "seed": int(config["seed"]),
        "source_sha256": sample.source_sha256,
        "annotation_sha256": sample.annotation_sha256,
        "ground_truth_count": sample.ground_truth_count,
        "predicted_count": prediction.predicted_count,
        "failure_state": prediction.failure_state,
        "record": asdict(record),
        "prediction_metadata": prediction.metadata,
        "parameter_count": brief.parameter_count,
        "upstream_commit": brief.upstream_commit,
        "checkpoint_sha256": brief.checkpoint_sha256,
        "rights_scope": brief.rights_status,
        "rights_decision_sha256": sha256_file(paths["rights"]),
        "rights_manifest_sha256": sha256_file(paths["manifest"]),
        "split_manifest_sha256": sha256_file(split_manifest),
        "environment_sha256": sha256_file(environment_path),
        "panel": None if panel_path is None else {"path": panel_path.name, "sha256": sha256_file(panel_path)},
        "artifacts": artifacts,
    }
    write_json(output / "result.json", result)
    print(
        f"DM-Count one-sample {result['status']}: sample={sample.sample_id} "
        f"gt={sample.ground_truth_count} pred={prediction.predicted_count}"
    )
    return 0 if prediction.failure_state is None else 2


if __name__ == "__main__":
    raise SystemExit(main())
