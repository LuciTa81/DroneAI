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

from droneai.pdf_input_package import build_pdf_input_package


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Package verified Round 1 panels for the private PDF"
    )
    parser.add_argument("--comparison", required=True, type=Path)
    parser.add_argument("--results-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        output = build_pdf_input_package(
            args.comparison, args.results_root, args.output_dir
        )
        print(
            json.dumps(
                {"status": "PASS_RESEARCH_ONLY", "output_dir": str(output)},
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
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
