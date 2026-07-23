from scripts.prepare_jhu_crowd import _parser


def test_jhu_prepare_cli_exposes_roots_and_expected_count() -> None:
    help_text = _parser().format_help()

    assert "--dataset-root" in help_text
    assert "--validation-root" in help_text
    assert "--expected-samples" in help_text
