"""Write a small, hash-verified STEERER stage review bundle from existing evidence."""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from droneai.steerer_training_gate import (
    REQUIRED_ARTIFACTS,
    score_training_stage,
    validate_training_evidence,
    verify_referenced_artifacts,
)


_OUTPUT_NAMES = frozenset(
    {"score.json", "score.md", "metrics.json", "environment.json", "lineage.json", "README.md"}
)


def _reject_nonfinite(value: str) -> None:
    raise ValueError(f"strict JSON forbids non-finite constant {value}")


def load_strict_json(path: str | Path) -> Any:
    """Read JSON without accepting JavaScript's non-standard NaN/Infinity tokens."""

    try:
        return json.loads(Path(path).read_text(encoding="utf-8"), parse_constant=_reject_nonfinite)
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid JSON evidence: {path}") from error


def _assert_json_safe(value: Any) -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("result JSON cannot contain NaN or Infinity")
    if isinstance(value, Mapping):
        for item in value.values():
            _assert_json_safe(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _assert_json_safe(item)


def _json_bytes(payload: Any) -> bytes:
    _assert_json_safe(payload)
    return (json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")


def _write_new(path: Path, content: str | bytes) -> None:
    data = content.encode("utf-8") if isinstance(content, str) else content
    with path.open("xb") as stream:
        stream.write(data)
        stream.flush()


def _validate_output_directory(output_dir: Path) -> None:
    if output_dir.is_symlink():
        raise ValueError("stage result directory cannot be a symlink")
    if output_dir.exists():
        raise FileExistsError("stage result directory must be new and empty")


def _require_verified_artifacts(evidence: Any, *, stage: str) -> None:
    observations = verify_referenced_artifacts(evidence, stage=stage)
    failures = [observation for observation in observations.values() if not observation.startswith("verified:")]
    if failures:
        raise ValueError("; ".join(failures))


def _bundle_payloads(evidence: Mapping[str, Any], *, stage: str) -> dict[str, str | bytes]:
    report = score_training_stage(evidence, stage=stage)
    provenance = evidence["provenance"]
    artifacts = evidence["artifacts"]
    assert isinstance(provenance, Mapping) and isinstance(artifacts, Mapping)
    lineage = {
        "schema_version": 1,
        "run_id": evidence["run_id"],
        "stage": stage,
        "git_commit": provenance["git_commit"],
        "upstream_commit": provenance["upstream_commit"],
        "runtime": {
            "container_image_digest": provenance["container_image_digest"],
            "python": provenance["python"],
            "torch": provenance["torch"],
            "cuda": provenance["cuda"],
            "gpu": provenance["gpu"],
        },
        "environment": artifacts["environment"],
        "dataset_inventory": artifacts["dataset_inventory"],
        "split_files": {
            "train": artifacts["train_split"],
            "validation": artifacts["validation_split"],
        },
        "backbone": artifacts["backbone"],
        "input_checkpoint": artifacts["input_checkpoint"],
        "output_checkpoint": artifacts["output_checkpoint"],
        "config": artifacts["config"],
        "rights_decision": artifacts["rights_decision"],
    }
    metrics = {"schema_version": 1, "run_id": evidence["run_id"], "stage": stage, "metrics": evidence["metrics"]}
    environment = {
        "schema_version": 1,
        "run_id": evidence["run_id"],
        "stage": stage,
        "runtime": lineage["runtime"],
        "environment_evidence": artifacts["environment"],
    }
    readme = "\n".join(
        [
            f"# STEERER UCF-QNRF {stage} training evidence",
            "",
            f"- Status: `{report.status}`",
            "- Decision scope: `PASS_COMMERCIAL_CANDIDATE` (not production approval)",
            "- This bundle only scores existing non-Test training evidence; it does not run training or evaluation.",
            "- `lineage.json` records the hash-verified input lineage.",
            "",
        ]
    )
    return {
        "score.json": _json_bytes(report.to_dict()),
        "score.md": report.to_markdown(),
        "metrics.json": _json_bytes(metrics),
        "environment.json": _json_bytes(environment),
        "lineage.json": _json_bytes(lineage),
        "README.md": readme,
    }


def write_stage_result(evidence: Any, *, stage: str, output_dir: str | Path) -> Path:
    """Verify all input bytes before creating an empty result directory."""

    validated = validate_training_evidence(evidence, stage=stage)
    _require_verified_artifacts(validated, stage=stage)
    payloads = _bundle_payloads(validated, stage=stage)
    if set(payloads) != _OUTPUT_NAMES:
        raise RuntimeError("stage bundle file set is incomplete")
    target = Path(output_dir)
    _validate_output_directory(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}.", dir=target.parent))
    try:
        for name in sorted(_OUTPUT_NAMES):
            _write_new(staging / name, payloads[name])
        _validate_output_directory(target)
        os.replace(staging, target)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return target


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Score existing STEERER validation evidence without training or Test access."
    )
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--stage", choices=("T0", "T1", "T5", "T50"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        evidence = load_strict_json(args.evidence)
        write_stage_result(evidence, stage=args.stage, output_dir=args.output_dir)
    except (FileExistsError, FileNotFoundError, OSError, TypeError, ValueError) as error:
        print(json.dumps({"status": "gate_violation", "error": str(error)}), file=sys.stderr)
        return 2
    print(json.dumps({"status": "scored", "stage": args.stage, "output_dir": str(args.output_dir)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
