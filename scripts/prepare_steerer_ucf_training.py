"""Prepare the deterministic, training-only STEERER UCF-QNRF split."""

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

from droneai.integrity import sha256_file
from droneai.steerer_training_data import prepare_training_dataset
from droneai.steerer_training_split import build_training_split
from droneai.steerer_training_profile import load_training_profile
from droneai.ucf_qnrf import index_ucf_qnrf_train


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create a deterministic STEERER train/validation split from UCF-QNRF training data."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--train-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        profile = load_training_profile(args.config)
        records = index_ucf_qnrf_train(args.train_root)
        if len(records) != profile.dataset_population:
            raise ValueError(
                "indexed training population must equal approved profile population: "
                f"expected={profile.dataset_population} observed={len(records)}"
            )
        split = build_training_split(
            records, seed=profile.seed, validation_count=profile.validation_count
        )
        if (len(split.train_ids), len(split.validation_ids)) != (
            profile.train_count,
            profile.validation_count,
        ):
            raise ValueError(
                "prepared split must equal approved profile counts: "
                f"expected={profile.train_count}/{profile.validation_count} "
                f"observed={len(split.train_ids)}/{len(split.validation_ids)}"
            )
        manifest_path = prepare_training_dataset(records, split, output_root=args.output_root)
    except (FileExistsError, FileNotFoundError, PermissionError, ValueError) as error:
        print(json.dumps({"status": "gate_violation", "error": str(error)}), file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "status": "prepared",
                "sample_count": len(records),
                "train_count": len(split.train_ids),
                "validation_count": len(split.validation_ids),
                "manifest_path": str(manifest_path),
                "manifest_sha256": sha256_file(manifest_path),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
