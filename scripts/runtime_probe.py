from __future__ import annotations

import argparse

from droneai.runtime_probe import write_environment_report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        default="reports/runs/runtime-probe/environment.json",
    )
    args = parser.parse_args()
    target = write_environment_report(args.output)
    print(target)


if __name__ == "__main__":
    main()
