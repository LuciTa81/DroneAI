from __future__ import annotations

import argparse

from droneai.stage2 import run_stage2


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the DroneAI Stage 2 CSRNet smoke gate")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--device")
    args = parser.parse_args()
    report = run_stage2(
        output_dir=args.output_dir,
        steps=args.steps,
        learning_rate=args.learning_rate,
        device_name=args.device,
    )
    print(report.to_markdown())
    if report.status != "PASS":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
