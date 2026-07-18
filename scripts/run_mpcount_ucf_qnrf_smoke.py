"""Run the frozen 36-image MPCount UCF-QNRF validation benchmark."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from droneai.evaluation_runner import run_evaluation
from droneai.mpcount_adapter import MPCountAdapter
from droneai.mpcount_smoke import (
    build_mpcount_protocol,
    load_mpcount_smoke_config,
    prepare_smoke_samples,
    validate_mpcount_rights_decision,
    validate_mpcount_split_upstream,
    write_split_source_manifest,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the frozen 36-image official UCF-QNRF validation benchmark "
            "with the deterministic MPCount checkpoint."
        )
    )
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
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--device", required=True, choices=("cuda", "cuda:0"))
    parser.add_argument("--patch-size", type=int, default=3584)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        if not output_dir.is_dir() or any(output_dir.iterdir()):
            raise FileExistsError(f"output must be new or empty: {output_dir}")
    else:
        output_dir.mkdir(parents=True)

    paths = {
        "config": args.config.resolve(),
        "train_root": args.train_root.resolve(),
        "upstream_dir": args.upstream_dir.resolve(),
        "split_upstream_dir": args.split_upstream_dir.resolve(),
        "train_list": args.train_list.resolve(),
        "validation_list": args.validation_list.resolve(),
        "checkpoint": args.checkpoint.resolve(),
        "rights_decision": args.rights_decision.resolve(),
        "rights_manifest": args.rights_manifest.resolve(),
    }
    config = load_mpcount_smoke_config(paths["config"])
    validate_mpcount_rights_decision(
        paths["rights_decision"],
        manifest_path=paths["rights_manifest"],
        expected_candidate_id=str(config["candidate_id"]),
    )
    validate_mpcount_split_upstream(
        split_upstream_dir=paths["split_upstream_dir"],
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
    expected_samples = int(config["expected_samples"])
    if expected_samples != 36 or len(prepared.samples) != 36:
        raise ValueError(
            "MPCount benchmark requires exactly 36 frozen validation samples; "
            f"config={expected_samples}, prepared={len(prepared.samples)}"
        )

    adapter = MPCountAdapter(
        upstream_dir=paths["upstream_dir"],
        expected_upstream_commit=str(config["upstream_commit"]),
        checkpoint_path=paths["checkpoint"],
        checkpoint_sha256=args.checkpoint_sha256,
        device=args.device,
        patch_size=args.patch_size,
    )
    split_manifest = write_split_source_manifest(
        output_dir / "split-source-manifest.json",
        prepared=prepared,
        train_list_path=paths["train_list"],
        validation_list_path=paths["validation_list"],
        model_upstream_commit=str(config["upstream_commit"]),
        split_upstream_commit=str(config["split_upstream_commit"]),
    )
    protocol = build_mpcount_protocol(
        config,
        rights_decision_path=paths["rights_decision"],
        rights_manifest_path=paths["rights_manifest"],
        split_verified=prepared.split_verified,
    )
    os.chdir(REPO_ROOT)
    report = run_evaluation(
        adapter=adapter,
        samples=prepared.samples,
        protocol=protocol,
        output_dir=output_dir,
        provenance_artifacts=(split_manifest,),
    )
    print(report.to_markdown())
    return 0 if report.is_success else 2


if __name__ == "__main__":
    sys.exit(main())
