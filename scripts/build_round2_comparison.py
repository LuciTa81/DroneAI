"""Build the compact, dataset-wise Round 2 comparison package."""

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
from droneai.round2_comparison import Round2Run, write_round2_comparison
from droneai.round2_config import load_round2_config


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Verify nine Round 2 runs and write dataset-wise comparison files."
    )
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--results-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    config = load_round2_config(args.config)
    root = args.results_root.resolve()
    runs = tuple(
        Round2Run(
            model_id=model.model_id,
            dataset_id=lane.dataset_id,
            expected_samples=lane.samples,
            dataset_rights_scope=lane.rights_scope,
            path=root / model.model_id / lane.dataset_id,
        )
        for model in config.models
        for lane in config.datasets
    )
    paths = write_round2_comparison(args.output_dir.resolve(), runs)
    print(
        json.dumps(
            {
                name: {"path": str(path), "sha256": sha256_file(path)}
                for name, path in paths.items()
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
