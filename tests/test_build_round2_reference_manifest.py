from scripts.build_round2_reference_manifest import _parser


def test_round2_manifest_cli_exposes_verify_only() -> None:
    help_text = _parser().format_help()

    assert "--verify-only" in help_text
    assert "--ucf-root" in help_text
    assert "--jhu-root" in help_text
    assert "--up-count-root" in help_text
