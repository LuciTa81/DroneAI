from scripts.build_round2_comparison import _parser


def test_comparison_cli_exposes_config_results_and_output() -> None:
    help_text = _parser().format_help()

    assert "--config" in help_text
    assert "--results-root" in help_text
    assert "--output-dir" in help_text
