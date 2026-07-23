"""Prepare official JHU-CROWD++ Validation for the Stage 1 gate."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from droneai.jhu_crowd import prepare_jhu_validation


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Normalize JHU-CROWD++ Validation without editing source files."
    )
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--validation-root", required=True, type=Path)
    parser.add_argument("--expected-samples", type=int, default=500)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    count = prepare_jhu_validation(
        dataset_root=args.dataset_root,
        validation_root=args.validation_root,
        expected_samples=args.expected_samples,
    )
    print(f"Prepared {count} JHU-CROWD++ validation samples")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
