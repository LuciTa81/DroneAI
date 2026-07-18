from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "run_model_workflow.py"


def _write(path: Path, payload: object) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fixture(tmp_path: Path) -> tuple[Path, Path]:
    results = tmp_path / "results"
    evidence_path = results / "steerer/one/result.json"
    digest = _write(evidence_path, {"status": "PASS"})
    queue = tmp_path / "queue.json"
    queue.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "round_id": "round-1-static-ucf-qnrf",
                "active_model": "steerer",
                "models": [
                    {
                        "model_id": "steerer",
                        "family": "density_and_points",
                        "queue_state": "active",
                        "rights_scope": "PASS_COMMERCIAL_CANDIDATE",
                        "dataset_id": "ucf-qnrf-kaggle-apache",
                        "split_id": "frozen-validation-36",
                        "split_role": "validation",
                        "expected_samples": 36,
                        "checkpoint_training_split_status": "UNKNOWN",
                        "comparison_scope": "compatibility_smoke",
                        "checkpoint_split_evidence": (
                            "public checkpoint training membership is not published"
                        ),
                        "gate_order": ["one_sample", "benchmark"],
                        "accepted_evidence": [
                            {
                                "gate": "one_sample",
                                "artifact_kind": "one_sample_result",
                                "path": "steerer/one/result.json",
                                "sha256": digest,
                            }
                        ],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    return queue, results


def _run(queue: Path, results: Path, *, output_format: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "status",
            "--queue",
            str(queue),
            "--results-root",
            str(results),
            "--format",
            output_format,
        ],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )


def test_status_json_is_compact_and_complete(tmp_path: Path) -> None:
    queue, results = _fixture(tmp_path)

    completed = _run(queue, results, output_format="json")

    assert completed.returncode == 0, completed.stderr
    status = json.loads(completed.stdout)
    assert status["active_model"] == "steerer"
    assert status["current_stage"] == "round-1-static-ucf-qnrf"
    assert status["next_action"] == "benchmark"
    assert status["rights_scope"] == "PASS_COMMERCIAL_CANDIDATE"
    assert status["dataset"]["dataset_id"] == "ucf-qnrf-kaggle-apache"
    assert status["validation"]["expected_samples"] == 36
    assert status["checkpoint_training_split_status"] == "UNKNOWN"
    assert status["comparison_scope"] == "compatibility_smoke"
    assert status["ranking_eligible"] is False
    assert status["roadmap"][0]["model_id"] == "steerer"
    assert len(completed.stdout.encode("utf-8")) < 4096


def test_status_markdown_exposes_review_fields(tmp_path: Path) -> None:
    queue, results = _fixture(tmp_path)

    completed = _run(queue, results, output_format="markdown")

    assert completed.returncode == 0, completed.stderr
    for expected in (
        "Active model: `steerer`",
        "Current stage: `round-1-static-ucf-qnrf`",
        "Dataset/split: `ucf-qnrf-kaggle-apache` / `frozen-validation-36`",
        "Validation: `36` validation samples; fine-tuning=`False`",
        "Rights scope: `PASS_COMMERCIAL_CANDIDATE`",
        "Comparison: `compatibility_smoke`; checkpoint split `UNKNOWN`; ranking eligible `False`",
        "Next action: `benchmark`",
        "Approval required: `True`",
        "| steerer | density_and_points | active |",
    ):
        assert expected in completed.stdout


def test_invalid_evidence_returns_exit_two(tmp_path: Path) -> None:
    queue, results = _fixture(tmp_path)
    (results / "steerer/one/result.json").write_text("changed", encoding="utf-8")

    completed = _run(queue, results, output_format="json")

    assert completed.returncode == 2
    error = json.loads(completed.stderr)
    assert error["status"] == "FAIL"
    assert error["error_type"] == "ValueError"
    assert "evidence SHA-256 mismatch" in error["error"]
