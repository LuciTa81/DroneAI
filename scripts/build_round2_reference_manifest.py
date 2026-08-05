"""Build or verify the immutable Round 2 1,000-sample manifest."""

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
from droneai.round2_config import load_round2_config
from droneai.round2_manifest import verify_round2_manifest, write_round2_manifest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build the immutable Round 2 reference sample manifest"
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--ucf-root", type=Path, required=True)
    parser.add_argument("--jhu-root", type=Path, required=True)
    parser.add_argument("--up-count-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--verify-only", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    config = load_round2_config(args.config)
    roots = {
        "ucf-qnrf-kaggle-apache": args.ucf_root.resolve(),
        "jhu-crowd-plus-v2": args.jhu_root.resolve(),
        "up-count-v1": args.up_count_root.resolve(),
    }
    if args.verify_only:
        if not args.output.is_file():
            raise FileNotFoundError(f"Round 2 manifest missing: {args.output}")
        payload = json.loads(args.output.read_text(encoding="utf-8"))
        samples = verify_round2_manifest(payload, config, roots)
        digest = sha256_file(args.output)
    else:
        digest = write_round2_manifest(
            args.output,
            config=config,
            dataset_roots=roots,
        )
        samples = verify_round2_manifest(
            json.loads(args.output.read_text(encoding="utf-8")), config, roots
        )
    print(
        json.dumps(
            {
                "path": args.output.resolve().as_posix(),
                "sha256": digest,
                "samples": len(samples),
                "rights_scope": config.combined_rights_scope,
                "fine_tuning": config.fine_tuning,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
