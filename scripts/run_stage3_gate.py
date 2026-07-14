"""Score persisted DM-Count reproduction evidence."""

from __future__ import annotations

import argparse
from pathlib import Path

from droneai.stage3 import run_stage3_gate


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Stage 3 DM-Count gate")
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    report = run_stage3_gate(evidence_path=args.evidence, output_dir=args.output_dir)
    print(report.to_markdown())
    raise SystemExit(0 if report.is_success else 1)


if __name__ == "__main__":
    main()
