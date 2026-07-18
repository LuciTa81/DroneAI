"""Report the next safe DroneAI model-evaluation action from verified evidence."""

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

from droneai.model_workflow import build_workflow_status


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Inspect the DroneAI model queue")
    subparsers = parser.add_subparsers(dest="command", required=True)
    status = subparsers.add_parser("status", help="verify evidence and report the next action")
    status.add_argument("--queue", required=True, type=Path)
    status.add_argument("--results-root", required=True, type=Path)
    status.add_argument("--format", choices=("json", "markdown"), default="json")
    return parser


def _markdown(status: dict[str, object]) -> str:
    dataset = status["dataset"]
    validation = status["validation"]
    lines = [
        "# DroneAI model workflow status",
        "",
        f"- Current stage: `{status['current_stage']}`",
        f"- Active model: `{status['active_model']}`",
        f"- Model family: `{status['model_family']}`",
        (
            f"- Dataset/split: `{dataset['dataset_id']}` / "
            f"`{dataset['split_id']}`"
        ),
        (
            f"- Validation: `{validation['expected_samples']}` "
            f"{dataset['split_role']} samples; "
            f"fine-tuning=`{validation['fine_tuning']}`"
        ),
        f"- Metrics: `{', '.join(validation['metrics'])}`",
        f"- Rights scope: `{status['rights_scope']}`",
        f"- Evaluation scope: `{status['evaluation_scope']}`",
        (
            f"- Comparison: `{status['comparison_scope']}`; checkpoint split "
            f"`{status['checkpoint_training_split_status']}`; ranking eligible "
            f"`{status['ranking_eligible']}`"
        ),
        f"- Checkpoint split evidence: {status['checkpoint_split_evidence']}",
        f"- Completed gates: `{', '.join(status['completed_gates'])}`",
        f"- Next action: `{status['next_action']}`",
        f"- Approval required: `{status['approval_required']}`",
        "",
        "| Model | Family | Queue | Rights | Comparison | Checkpoint split | Ranking | Verified gates |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for entry in status["roadmap"]:
        lines.append(
            "| {model_id} | {family} | {queue_state} | {rights_scope} | {comparison_scope} | {checkpoint_training_split_status} | {ranking_eligible} | {gates} |".format(
                **entry,
                gates=", ".join(entry["verified_gates"]),
            )
        )
    return "\n".join(lines) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        status = build_workflow_status(args.queue, args.results_root)
    except Exception as exc:
        error = {
            "schema_version": 1,
            "status": "FAIL",
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
        print(
            json.dumps(error, ensure_ascii=False, sort_keys=True, separators=(",", ":")),
            file=sys.stderr,
        )
        return 2

    if args.format == "markdown":
        print(_markdown(status), end="")
    else:
        print(
            json.dumps(
                status,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
