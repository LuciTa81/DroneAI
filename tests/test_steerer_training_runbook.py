from pathlib import Path


RUNBOOK = Path("docs/runbooks/steerer-ucf-qnrf-training-home5090.md")


def test_runbook_contains_all_operational_commands_and_approval_stops() -> None:
    text = RUNBOOK.read_text(encoding="utf-8")
    for token in (
        "home5090-pop",
        "T0",
        "T1",
        "T5",
        "T50",
        "resume",
        "score.json",
        "rsync",
    ):
        assert token in text
    assert "Do not run T5 before user approval" in text
    assert "UCF-QNRF Test remains sealed" in text
    assert "docker inspect --format '{{.Image}}' crowd-jupyter" in text
    assert "--container-image-digest" in text
    assert "steerer-qnrf-imagenet-20260806-t0-b" in text
    assert "steerer-qnrf-imagenet-20260806-t1-b" in text
    assert 'result["metrics_path"]' in text
    assert 'result["environment_path"]' in text
