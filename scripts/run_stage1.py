from __future__ import annotations

import argparse

from droneai.stage1 import run_stage1


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the DroneAI Stage 1 data gate")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    report = run_stage1(
        manifest_path=args.manifest,
        dataset_root=args.dataset_root,
        output_dir=args.output_dir,
    )
    print(report.to_markdown())
    if report.status != "PASS":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
