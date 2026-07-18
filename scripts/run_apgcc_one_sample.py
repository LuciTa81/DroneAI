"""Run one frozen APGCC validation sample before the 36-image benchmark."""

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Sequence, cast

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from droneai.apgcc_adapter import APGCCAdapter
from droneai.apgcc_smoke import (
    load_apgcc_smoke_config,
    prepare_smoke_samples,
    validate_apgcc_rights_decision,
    validate_apgcc_split_upstream,
    write_split_source_manifest,
)
from droneai.evaluation_artifacts import write_json
from droneai.evaluation_contract import NativePrediction
from droneai.evaluation_metrics import evaluate_sample
from droneai.evaluation_panels import render_review_panel
from droneai.integrity import sha256_file
from droneai.model_brief import write_model_brief
from droneai.runtime_probe import collect_environment


def compatibility_checks(
    prediction: NativePrediction,
) -> tuple[dict[str, bool], float]:
    success = prediction.failure_state is None and prediction.predicted_count is not None
    width = prediction.metadata.get("original_width")
    height = prediction.metadata.get("original_height")
    point_output = bool(success and prediction.output_type == "points")
    points_valid = bool(
        point_output
        and isinstance(width, int)
        and isinstance(height, int)
        and width > 0
        and height > 0
        and all(
            math.isfinite(x)
            and math.isfinite(y)
            and 0 <= x < width
            and 0 <= y < height
            for x, y in prediction.points
        )
    )
    confidences_valid = bool(
        point_output
        and len(prediction.point_confidences) == len(prediction.points)
        and all(
            math.isfinite(value) and 0.0 <= value <= 1.0
            for value in prediction.point_confidences
        )
    )
    count_matches = bool(
        point_output
        and math.isclose(
            float(prediction.predicted_count),
            float(len(prediction.points)),
            rel_tol=0.0,
            abs_tol=1e-9,
        )
    )
    checkpoint_strict = bool(
        success
        and prediction.metadata.get("missing_checkpoint_key_count") == 0
        and prediction.metadata.get("unexpected_checkpoint_key_count") == 0
    )
    runtime_valid = bool(
        success
        and math.isfinite(prediction.latency_ms)
        and prediction.latency_ms > 0
        and math.isfinite(prediction.peak_vram_mb)
        and prediction.peak_vram_mb >= 0
    )
    checks = {
        "successful_forward": bool(success),
        "native_point_output": point_output,
        "points_finite_in_bounds": points_valid,
        "point_confidences_aligned": confidences_valid,
        "point_count_matches_output": count_matches,
        "checkpoint_loaded_strictly": checkpoint_strict,
        "runtime_metrics_finite": runtime_valid,
    }
    score = 100.0 * sum(checks.values()) / len(checks)
    return checks, score


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run one frozen APGCC validation sample.")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--train-root", required=True, type=Path)
    parser.add_argument("--upstream-dir", required=True, type=Path)
    parser.add_argument("--split-upstream-dir", required=True, type=Path)
    parser.add_argument("--train-list", required=True, type=Path)
    parser.add_argument("--validation-list", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--rights-decision", required=True, type=Path)
    parser.add_argument("--rights-manifest", required=True, type=Path)
    parser.add_argument("--sample-id")
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
        name: value.resolve()
        for name, value in {
            "config": args.config,
            "train_root": args.train_root,
            "upstream": args.upstream_dir,
            "split_upstream": args.split_upstream_dir,
            "train_list": args.train_list,
            "validation_list": args.validation_list,
            "checkpoint": args.checkpoint,
            "rights": args.rights_decision,
            "manifest": args.rights_manifest,
        }.items()
    }
    config = load_apgcc_smoke_config(paths["config"])
    validate_apgcc_rights_decision(
        paths["rights"],
        manifest_path=paths["manifest"],
        expected_candidate_id=str(config["candidate_id"]),
    )
    validate_apgcc_split_upstream(
        split_upstream_dir=paths["split_upstream"],
        train_list_path=paths["train_list"],
        validation_list_path=paths["validation_list"],
        expected_commit=str(config["split_upstream_commit"]),
    )
    prepared = prepare_smoke_samples(
        config=config,
        train_root=paths["train_root"],
        train_list_path=paths["train_list"],
        validation_list_path=paths["validation_list"],
    )
    candidates = sorted(prepared.samples, key=lambda sample: sample.sample_id)
    if args.sample_id:
        matches = [sample for sample in candidates if sample.sample_id == args.sample_id]
        if len(matches) != 1:
            raise ValueError(f"sample-id is not in the frozen split: {args.sample_id}")
        sample = matches[0]
    else:
        medium = [
            sample
            for sample in candidates
            if sample.condition_tags.get("density_band") == "medium"
        ]
        if not medium:
            raise RuntimeError("frozen split has no medium-density sample")
        sample = medium[0]

    split_manifest = write_split_source_manifest(
        output / "split-source-manifest.json",
        prepared=prepared,
        train_list_path=paths["train_list"],
        validation_list_path=paths["validation_list"],
        model_upstream_commit=str(config["upstream_commit"]),
        split_upstream_commit=str(config["split_upstream_commit"]),
    )
    adapter = APGCCAdapter(
        upstream_dir=paths["upstream"],
        expected_upstream_commit=str(config["upstream_commit"]),
        checkpoint_path=paths["checkpoint"],
        checkpoint_sha256=args.checkpoint_sha256,
        device=args.device,
    )
    brief = adapter.brief()
    brief.require_full_run_approval()
    prediction = adapter.predict(sample, retain_native=True)
    checks, technical_score = compatibility_checks(prediction)
    record = evaluate_sample(
        sample,
        prediction,
        localization_radius=float(config["localization_radius"]),
    )
    write_model_brief(output / "model-brief.md", brief)
    environment_path = write_json(
        output / "environment-summary.json", collect_environment(ROOT)
    )
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
            category="one_sample_compatibility",
        )
    passed = prediction.failure_state is None and technical_score == 100.0
    result = {
        "schema_version": 1,
        "status": "PASS" if passed else "FAIL",
        "technical_score": technical_score,
        "gate": "one_sample",
        "evaluation_scope": "research_comparison_only",
        "comparison_scope": str(config["comparison_scope"]),
        "ranking_eligible": False,
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
        "point_confidence_count": len(prediction.point_confidences),
        "compatibility_checks": checks,
        "parameter_count": brief.parameter_count,
        "upstream_commit": brief.upstream_commit,
        "checkpoint_sha256": brief.checkpoint_sha256,
        "rights_scope": brief.rights_status,
        "rights_decision_sha256": sha256_file(paths["rights"]),
        "rights_manifest_sha256": sha256_file(paths["manifest"]),
        "split_manifest_sha256": sha256_file(split_manifest),
        "environment_sha256": sha256_file(environment_path),
        "panel": None
        if panel_path is None
        else {"path": panel_path.name, "sha256": sha256_file(panel_path)},
    }
    write_json(output / "result.json", result)
    print(
        f"APGCC one-sample {result['status']}: sample={sample.sample_id} "
        f"gt={sample.ground_truth_count} pred={prediction.predicted_count}"
    )
    return 0 if passed else 2


if __name__ == "__main__":
    sys.exit(main())
