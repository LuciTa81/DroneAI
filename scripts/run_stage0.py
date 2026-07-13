from __future__ import annotations

import argparse

from droneai.stage0 import DEFAULT_SEEDS, run_stage0


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the DroneAI Stage 0 score gate")
    parser.add_argument("--drive-root", required=True)
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--output-dir")
    parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    args = parser.parse_args()

    report = run_stage0(
        drive_root=args.drive_root,
        repo_root=args.repo_root,
        output_dir=args.output_dir,
        seeds=args.seeds,
    )
    print(report.to_markdown())
    if report.status != "PASS":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
