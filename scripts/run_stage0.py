from __future__ import annotations

import argparse
from pathlib import Path

from droneai.runtime_profile import load_runtime_profile
from droneai.stage0 import DEFAULT_SEEDS, run_stage0


DEFAULT_HOME_PROFILE = Path(__file__).resolve().parents[1] / "configs" / "runtime" / "home5090_docker.json"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the DroneAI Stage 0 score gate")
    backend = parser.add_mutually_exclusive_group()
    backend.add_argument("--profile")
    backend.add_argument("--drive-root", help="Legacy Colab/Drive compatibility input")
    parser.add_argument("--repo-root")
    parser.add_argument("--output-dir")
    parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    args = parser.parse_args()

    if args.drive_root:
        report = run_stage0(
            drive_root=args.drive_root,
            repo_root=args.repo_root or ".",
            output_dir=args.output_dir,
            seeds=args.seeds,
        )
    else:
        profile = load_runtime_profile(args.profile or DEFAULT_HOME_PROFILE)
        report = run_stage0(
            profile=profile,
            repo_root=args.repo_root or profile.repo_root,
            output_dir=args.output_dir,
            seeds=args.seeds,
        )
    print(report.to_markdown())
    if report.status != "PASS":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
