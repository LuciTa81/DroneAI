from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from droneai.integrity import sha256_file
from droneai.round_comparison import build_round_comparison
from droneai.round_evidence import load_verified_round
from droneai.round_report import write_comparison_bundle
from droneai.round_scorecard import load_scorecard


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build the Round 1 fixed-CCTV comparison"
    )
    parser.add_argument("--queue", required=True, type=Path)
    parser.add_argument("--results-root", required=True, type=Path)
    parser.add_argument("--scorecard", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser


def _git_identity() -> str:
    status = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    if status.strip():
        raise ValueError("comparison generation requires a clean Git worktree")
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _input_references(
    *,
    queue: Path,
    scorecard: Path,
    results_root: Path,
    runs: tuple[object, ...],
) -> tuple[dict[str, str], ...]:
    references = [
        {"path": queue.resolve().as_posix(), "sha256": sha256_file(queue)},
        {
            "path": scorecard.resolve().as_posix(),
            "sha256": sha256_file(scorecard),
        },
    ]
    root_label = results_root.resolve().as_posix().rstrip("/")
    for run in runs:
        references.extend(
            {
                "path": f"{root_label}/{reference['path']}",
                "sha256": reference["sha256"],
            }
            for reference in run.input_artifacts
        )
    by_path: dict[str, str] = {}
    for reference in references:
        previous = by_path.setdefault(reference["path"], reference["sha256"])
        if previous != reference["sha256"]:
            raise ValueError(f"input reference hash conflict: {reference['path']}")
    return tuple(
        {"path": path, "sha256": by_path[path]} for path in sorted(by_path)
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        config = load_scorecard(args.scorecard)
        runs = load_verified_round(args.queue, args.results_root, config)
        comparison = build_round_comparison(
            runs, config, git_commit=_git_identity()
        )
        inputs = _input_references(
            queue=args.queue,
            scorecard=args.scorecard,
            results_root=args.results_root,
            runs=runs,
        )
        output = write_comparison_bundle(
            args.output_dir,
            comparison,
            args.scorecard,
            input_references=inputs,
        )
        print(
            json.dumps(
                {
                    "status": "PASS_RESEARCH_ONLY",
                    "output_dir": str(output),
                    "shortlist": comparison["shortlist"],
                },
                sort_keys=True,
            )
        )
        return 0
    except Exception as exc:
        print(
            json.dumps(
                {
                    "schema_version": 1,
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
