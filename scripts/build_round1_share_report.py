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
from droneai.round1_share_report import PAGE_COUNT, build_round1_share_pdf


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build the Korean Round 1 shareable technical report"
    )
    parser.add_argument("--comparison", required=True, type=Path)
    parser.add_argument("--assets-manifest", required=True, type=Path)
    parser.add_argument("--content", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--report-date", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        output = build_round1_share_pdf(
            args.comparison,
            args.assets_manifest,
            args.content,
            args.output,
            report_date=args.report_date,
        )
        print(
            json.dumps(
                {
                    "status": "PASS_RESEARCH_ONLY",
                    "output": str(output),
                    "sha256": sha256_file(output),
                    "pages": PAGE_COUNT,
                },
                ensure_ascii=False,
                sort_keys=True,
            )
        )
        return 0
    except Exception as exc:
        print(
            json.dumps(
                {
                    "status": "FAIL",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                },
                ensure_ascii=False,
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
