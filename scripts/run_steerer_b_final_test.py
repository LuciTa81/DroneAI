"""Run the one-time validation-selected STEERER B final UCF-QNRF Test."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from droneai.evaluation_artifacts import write_json
from droneai.evaluation_runner import run_evaluation
from droneai.integrity import sha256_file
from droneai.steerer_adapter import STEERERAdapter
from droneai.steerer_final_test import (
    build_b_final_test_protocol,
    finalize_b_final_test,
    load_b_final_test_profile,
    validate_checkpoint_manifest,
    verify_checkpoint_files,
)
from droneai.steerer_training_profile import load_training_profile
from droneai.ucf_qnrf import index_ucf_qnrf_test, prepare_evaluation_sample


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the frozen three-checkpoint STEERER B official Test334 exactly once."
    )
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--test-root", required=True, type=Path)
    parser.add_argument("--upstream-dir", required=True, type=Path)
    parser.add_argument("--checkpoint-root", required=True, type=Path)
    parser.add_argument("--checkpoint-manifest", required=True, type=Path)
    parser.add_argument("--training-profile", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--device", required=True, choices=("cuda",))
    return parser


def _new_output(path: Path) -> Path:
    output = path.resolve()
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError(f"B final-Test output must be new or empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    return output


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    profile = load_b_final_test_profile(args.config)
    training_profile = load_training_profile(args.training_profile)
    if training_profile.success_scope != profile.rights_scope:
        raise PermissionError("training profile rights scope differs from final-Test profile")
    validate_checkpoint_manifest(profile, args.checkpoint_manifest)
    checkpoint_paths = verify_checkpoint_files(profile, args.checkpoint_root)
    records = index_ucf_qnrf_test(args.test_root)
    if len(records) != profile.expected_samples:
        raise ValueError(
            "official Test population mismatch: "
            f"expected={profile.expected_samples} observed={len(records)}"
        )
    samples = tuple(
        prepare_evaluation_sample(
            record,
            split_id=profile.test_split_id,
            source_partition="official_test",
        )
        for record in records
    )
    output = _new_output(args.output_dir)
    selection_path = write_json(
        output / "checkpoint-selection.json",
        {
            "schema_version": 1,
            "run_id": profile.run_id,
            "selection_basis": profile.selection_basis,
            "result_label": profile.result_label,
            "test_metrics_observed": False,
            "post_test_training_allowed": profile.post_test_training_allowed,
            "checkpoints": [
                {
                    "role": row.role,
                    "epoch": row.epoch,
                    "filename": row.filename,
                    "sha256": row.sha256,
                }
                for row in profile.checkpoints
            ],
        },
    )
    inventory_path = write_json(
        output / "test-source-inventory.json",
        {
            "schema_version": 1,
            "dataset_id": profile.dataset_id,
            "split_id": profile.test_split_id,
            "sample_count": len(records),
            "samples": [
                {
                    "sample_id": row.sample_id,
                    "image_sha256": row.image_sha256,
                    "annotation_sha256": row.annotation_sha256,
                    "count": row.count,
                    "width": row.width,
                    "height": row.height,
                }
                for row in records
            ],
        },
    )
    shared_provenance = {
        "schema_version": 1,
        "profile_sha256": sha256_file(args.config),
        "training_profile_sha256": sha256_file(args.training_profile),
        "checkpoint_manifest_sha256": sha256_file(args.checkpoint_manifest),
        "selection_sha256": sha256_file(selection_path),
        "test_inventory_sha256": sha256_file(inventory_path),
    }
    reports = []
    for checkpoint, checkpoint_path in zip(profile.checkpoints, checkpoint_paths):
        role_root = output / checkpoint.role
        role_root.mkdir()
        provenance_path = write_json(
            role_root / "b-final-test-provenance.json",
            {
                **shared_provenance,
                "checkpoint_role": checkpoint.role,
                "checkpoint_epoch": checkpoint.epoch,
                "checkpoint_sha256": checkpoint.sha256,
            },
        )
        adapter = STEERERAdapter(
            upstream_dir=args.upstream_dir,
            expected_upstream_commit=profile.upstream_commit,
            checkpoint_path=checkpoint_path,
            checkpoint_sha256=checkpoint.sha256,
            device=args.device,
            checkpoint_origin="project_training",
        )
        protocol = build_b_final_test_protocol(
            profile,
            checkpoint=checkpoint,
            rights_path=args.training_profile,
        )
        report = run_evaluation(
            adapter=adapter,
            samples=samples,
            protocol=protocol,
            output_dir=role_root,
            provenance_artifacts=(provenance_path,),
        )
        reports.append(report)
    comparison_path, closure_path = finalize_b_final_test(output, profile=profile)
    print(
        json.dumps(
            {
                "status": "completed",
                "run_id": profile.run_id,
                "roles": len(reports),
                "samples_per_role": profile.expected_samples,
                "comparison_path": str(comparison_path),
                "closure_path": str(closure_path),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
