from __future__ import annotations

import json
from pathlib import Path

from scripts.run_round2_reference import _parser, main


CONFIG = Path("configs/evaluation/round2_reference_benchmark.json")


def test_dry_run_lists_nine_dataset_runs_without_inference(
    tmp_path: Path, capsys
) -> None:
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "round_id": "round-2-reference-benchmark-v1",
                "fine_tuning": False,
                "expected_samples_per_model": 1000,
                "sample_manifest_sha256": "a" * 64,
                "samples": [],
            }
        ),
        encoding="utf-8",
    )

    assert main(
        ["--config", str(CONFIG), "--manifest", str(manifest), "--dry-run"]
    ) == 0
    payload = json.loads(capsys.readouterr().out)

    assert len(payload["runs"]) == 9
    assert payload["fine_tuning"] is False
    assert payload["sample_manifest_sha256"] == "a" * 64


def test_cli_exposes_one_sample_resume_and_dataset_roots() -> None:
    help_text = _parser().format_help()

    assert "--one-sample" in help_text
    assert "--resume" in help_text
    assert "--ucf-root" in help_text
    assert "--jhu-root" in help_text
    assert "--up-count-root" in help_text
