"""Prepare the official UCF-QNRF Test partition for STEERER A G1."""

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
from droneai.steerer_official_data import prepare_official_test_partition
from droneai.steerer_official_reproduction import load_official_reproduction_profile


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare Test334 exactly as pinned STEERER prepare_QNRF.py."
    )
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--raw-test-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    profile = load_official_reproduction_profile(args.config)
    output = args.output_dir.resolve()
    if output != Path(profile.g1_prepared_test_root).resolve():
        raise ValueError("prepared Test output differs from the isolated A profile")
    manifest = prepare_official_test_partition(
        args.raw_test_root,
        output_root=output,
        expected_samples=profile.test_samples,
    )
    print(
        json.dumps(
            {
                "status": "prepared",
                "samples": profile.test_samples,
                "output_root": str(output),
                "manifest": str(manifest),
                "manifest_sha256": sha256_file(manifest),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
