"""Classify a complete model/data/weight/deployment rights bundle."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from droneai.stage3c import run_stage3c


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the DroneAI Stage 3C rights gate")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = run_stage3c(manifest_path=args.manifest, output_dir=args.output_dir)
    print(result.decision.to_markdown())
    print(result.report.to_markdown())
    raise SystemExit(0 if result.report.is_success else 2)


if __name__ == "__main__":
    main()
