from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from droneai.round_comparison import build_round_comparison
from droneai.round_evidence import load_verified_round
from droneai.round_report import write_comparison_bundle
from tests.round_comparison_helpers import SCORECARD_PATH, make_round_fixture


def _comparison(tmp_path: Path) -> dict[str, object]:
    queue, results, config = make_round_fixture(tmp_path)
    runs = load_verified_round(queue, results, config)
    return build_round_comparison(runs, config, git_commit="a" * 40)


def test_writer_creates_exact_small_bundle(tmp_path: Path) -> None:
    output = tmp_path / "comparison"
    written = write_comparison_bundle(
        output,
        _comparison(tmp_path),
        SCORECARD_PATH,
        input_references=(
            {"path": "source/metrics.json", "sha256": "a" * 64},
        ),
    )

    assert written == output
    assert {path.name for path in output.iterdir()} == {
        "comparison.json",
        "comparison.csv",
        "comparison.md",
        "evidence-manifest.json",
        "scorecard-config.snapshot.json",
    }
    manifest = json.loads(
        (output / "evidence-manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["inputs"][0]["sha256"] == "a" * 64
    assert {item["path"] for item in manifest["outputs"]} == {
        "comparison.json",
        "comparison.csv",
        "comparison.md",
        "scorecard-config.snapshot.json",
    }
    with (output / "comparison.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 6
    assert {
        "density_psnr",
        "density_ssim",
        "localization_f1",
        "localization_mean_distance",
        "density_map_capability",
        "point_localization_capability",
    }.issubset(rows[0])
    markdown = (output / "comparison.md").read_text(encoding="utf-8")
    assert "not an official ranking" in markdown
    assert "Independent rights components" in markdown
    assert "Representative panels" in markdown


def test_writer_failure_leaves_no_final_or_temporary_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "comparison"

    def fail_budget(*_args: object, **_kwargs: object) -> None:
        raise ValueError("over budget")

    monkeypatch.setattr("droneai.round_report.enforce_review_budget", fail_budget)
    with pytest.raises(ValueError, match="over budget"):
        write_comparison_bundle(
            output,
            _comparison(tmp_path),
            SCORECARD_PATH,
            input_references=(),
        )
    assert not output.exists()
    assert not (tmp_path / ".comparison.tmp").exists()


def test_writer_refuses_existing_output_and_invalid_input_hash(tmp_path: Path) -> None:
    comparison = _comparison(tmp_path)
    output = tmp_path / "comparison"
    output.mkdir()
    with pytest.raises(FileExistsError):
        write_comparison_bundle(
            output, comparison, SCORECARD_PATH, input_references=()
        )

    with pytest.raises(ValueError, match="input reference"):
        write_comparison_bundle(
            tmp_path / "other",
            comparison,
            SCORECARD_PATH,
            input_references=({"path": "source", "sha256": "short"},),
        )
    assert not (tmp_path / "other").exists()

