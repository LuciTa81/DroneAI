"""Build one Google Drive-ready company share ZIP from approved evidence."""

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

from droneai.company_share_package import (
    build_company_share,
    load_company_share_spec,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build the hash-verified DroneAI company share ZIP."
    )
    parser.add_argument("--spec", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    spec_path = args.spec.resolve()
    spec = load_company_share_spec(spec_path)
    report = build_company_share(spec)
    print(
        json.dumps(
            report.as_dict(),
            indent=2,
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
