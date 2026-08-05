from pathlib import Path

import pytest

import scripts.build_round2_comparison as comparison_cli
from scripts.build_round2_comparison import _parser


def test_comparison_cli_exposes_config_results_and_output() -> None:
    help_text = _parser().format_help()

    assert "--config" in help_text
    assert "--results-root" in help_text
    assert "--output-dir" in help_text


def test_comparison_cli_passes_config_model_order(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    calls: list[tuple[str, ...]] = []

    def fake_write(
        output_dir: Path,
        runs: tuple[object, ...],
        *,
        model_order: tuple[str, ...],
    ) -> dict[str, Path]:
        calls.append(model_order)
        output_dir.mkdir(parents=True)
        json_path = output_dir / "comparison.json"
        markdown_path = output_dir / "comparison.md"
        json_path.write_text("{}", encoding="utf-8")
        markdown_path.write_text("# comparison\n", encoding="utf-8")
        return {"json": json_path, "markdown": markdown_path}

    monkeypatch.setattr(comparison_cli, "write_round2_comparison", fake_write)

    result = comparison_cli.main(
        [
            "--config",
            "configs/evaluation/round2_point_reference_benchmark.json",
            "--results-root",
            str(tmp_path / "results"),
            "--output-dir",
            str(tmp_path / "comparison"),
        ]
    )

    assert result == 0
    assert calls == [("steerer", "pet", "apgcc")]
