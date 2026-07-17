"""Run official PET on the frozen 36-image UCF-QNRF validation subset."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from droneai.evaluation_runner import run_evaluation
from droneai.pet_adapter import PETAdapter
from droneai.pet_smoke import (
    build_pet_protocol,
    load_pet_smoke_config,
    prepare_smoke_samples,
    validate_pet_rights_decision,
    validate_pet_split_upstream,
    write_split_source_manifest,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run frozen PET UCF-QNRF validation smoke.")
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
    config = load_pet_smoke_config(paths["config"])
    validate_pet_rights_decision(
        paths["rights"],
        manifest_path=paths["manifest"],
        expected_candidate_id=str(config["candidate_id"]),
    )
    validate_pet_split_upstream(
        split_upstream_dir=paths["split_upstream"],
        train_list_path=paths["train_list"],
        validation_list_path=paths["validation_list"],
        expected_commit=str(config["split_upstream_commit"]),
    )
    adapter = PETAdapter(
        upstream_dir=paths["upstream"],
        expected_upstream_commit=str(config["upstream_commit"]),
        checkpoint_path=paths["checkpoint"],
        checkpoint_sha256=args.checkpoint_sha256,
        device=args.device,
    )
    prepared = prepare_smoke_samples(
        config=config,
        train_root=paths["train_root"],
        train_list_path=paths["train_list"],
        validation_list_path=paths["validation_list"],
    )
    provenance = write_split_source_manifest(
        output / "split-source-manifest.json",
        prepared=prepared,
        train_list_path=paths["train_list"],
        validation_list_path=paths["validation_list"],
        model_upstream_commit=str(config["upstream_commit"]),
        split_upstream_commit=str(config["split_upstream_commit"]),
    )
    protocol = build_pet_protocol(
        config,
        rights_decision_path=paths["rights"],
        rights_manifest_path=paths["manifest"],
        split_verified=prepared.split_verified,
    )
    os.chdir(ROOT)
    report = run_evaluation(
        adapter=adapter,
        samples=prepared.samples,
        protocol=protocol,
        output_dir=output,
        provenance_artifacts=(provenance,),
    )
    print(report.to_markdown())
    return 0 if report.is_success else 2


if __name__ == "__main__":
    sys.exit(main())
