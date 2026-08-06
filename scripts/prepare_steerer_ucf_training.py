"""Prepare the deterministic, training-only STEERER UCF-QNRF split."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from droneai.steerer_training_split import build_training_split, write_training_split
from droneai.ucf_qnrf import index_ucf_qnrf_train


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create a deterministic STEERER train/validation split from UCF-QNRF training data."
    )
    parser.add_argument("--train-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=3035)
    parser.add_argument("--validation-count", type=int, default=240)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    records = index_ucf_qnrf_train(args.train_root)
    split = build_training_split(
        records, seed=args.seed, validation_count=args.validation_count
    )
    write_training_split(args.output_dir, split)
    print(f"Prepared {len(split.train_ids)} train and {len(split.validation_ids)} validation IDs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
