"""Run STEERER A-lane G1 with the frozen official checkpoint and Test334."""

from __future__ import annotations

import argparse
import json
import os
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
from droneai.steerer_official_reproduction import (
    build_g1_protocol,
    evaluate_g1_compatibility,
    load_official_reproduction_profile,
    validate_g1_rights_decision,
    validate_official_checkpoint_manifest,
    verify_official_checkpoint_file,
)
from droneai.ucf_qnrf import index_ucf_qnrf_test, prepare_evaluation_sample


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the immutable STEERER A G1 official-checkpoint compatibility "
            "evaluation on all 334 official UCF-QNRF Test images."
        )
    )
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--test-root", required=True, type=Path)
    parser.add_argument("--upstream-dir", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--checkpoint-manifest", required=True, type=Path)
    parser.add_argument("--rights-decision", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--device", required=True, choices=("cuda",))
    return parser


def _new_output(path: Path) -> Path:
    output = path.resolve()
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError(f"G1 output must be new or empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    return output


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    profile = load_official_reproduction_profile(args.config)
    manifest = validate_official_checkpoint_manifest(
        profile, args.checkpoint_manifest
    )
    checkpoint = verify_official_checkpoint_file(profile, args.checkpoint)
    validate_g1_rights_decision(args.rights_decision)
    records = index_ucf_qnrf_test(args.test_root)
    if len(records) != profile.test_samples:
        raise ValueError(
            "official Test population mismatch: "
            f"expected={profile.test_samples} observed={len(records)}"
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
    source_inventory = write_json(
        output / "test-source-inventory.json",
        {
            "schema_version": 1,
            "dataset_id": profile.dataset_id,
            "split_id": profile.test_split_id,
            "sample_count": len(records),
            "samples": [
                {
                    "sample_id": record.sample_id,
                    "image_sha256": record.image_sha256,
                    "annotation_sha256": record.annotation_sha256,
                    "count": record.count,
                    "width": record.width,
                    "height": record.height,
                }
                for record in records
            ],
        },
    )
    provenance = write_json(
        output / "g1-provenance.json",
        {
            "schema_version": 1,
            "run_id": profile.run_id,
            "result_label": profile.result_label,
            "selection_basis": profile.selection_basis,
            "official_weight_scope": profile.official_weight_scope,
            "profile_sha256": sha256_file(args.config),
            "checkpoint_manifest_sha256": sha256_file(args.checkpoint_manifest),
            "checkpoint_sha256": profile.checkpoint.sha256,
            "rights_decision_sha256": sha256_file(args.rights_decision),
            "test_inventory_sha256": sha256_file(source_inventory),
            "artifact_manifest_allowed_scope": manifest["allowed_scope"],
            "long_training_authorized_before_g1": False,
        },
    )
    adapter = STEERERAdapter(
        upstream_dir=args.upstream_dir,
        expected_upstream_commit=profile.upstream_commit,
        checkpoint_path=checkpoint,
        checkpoint_sha256=profile.checkpoint.sha256,
        device=args.device,
        checkpoint_origin="research_checkpoint",
    )
    protocol = build_g1_protocol(profile, rights_path=args.rights_decision)
    os.chdir(REPO_ROOT)
    report = run_evaluation(
        adapter=adapter,
        samples=samples,
        protocol=protocol,
        output_dir=output,
        provenance_artifacts=(source_inventory, provenance),
    )
    metrics = json.loads((output / "metrics.json").read_text(encoding="utf-8"))
    compatibility = evaluate_g1_compatibility(profile, metrics)
    gate_path = write_json(output / "compatibility-gate.json", compatibility)
    print(
        json.dumps(
            {
                "run_id": profile.run_id,
                "evaluation_status": report.status,
                "compatibility_status": compatibility["status"],
                "training_authorized": compatibility["training_authorized"],
                "compatibility_gate": str(gate_path),
            },
            sort_keys=True,
        )
    )
    return 0 if report.is_success and compatibility["training_authorized"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
