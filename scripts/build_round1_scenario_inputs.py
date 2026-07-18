"""Package verified one-sample scenario panels for the Round 1 share report."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from droneai.report_scenario_package import build_report_scenario_package


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build the verified dual-scenario report package")
    parser.add_argument("--comparison", required=True, type=Path)
    parser.add_argument("--results-root", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        output = build_report_scenario_package(
            args.comparison, args.results_root, args.config, args.output_dir
        )
        print(json.dumps({"status": "PASS_RESEARCH_ONLY", "output_dir": str(output)}, sort_keys=True))
        return 0
    except Exception as exc:
        print(
            json.dumps(
                {"status": "FAIL", "error_type": type(exc).__name__, "error": str(exc)},
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
