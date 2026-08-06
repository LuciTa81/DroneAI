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
        "T800",
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
    assert "steerer-qnrf-imagenet-20260806-t1-e" in text
    assert "T1-c" in text
    assert "T1-d" in text
    assert 'result["metrics_path"]' in text
    assert 'result["environment_path"]' in text


def test_runbook_contains_exact_t800_launch_recovery_and_score_controls() -> None:
    text = RUNBOOK.read_text(encoding="utf-8")

    for token in (
        "--stage T800 --approved-stage T800",
        "milestone-005.pth",
        "2ce6800bcbbdaace881e5f2ecc9f481616a67b9b96254232b7e8e5285ceeafc4",
        "status.json",
        "run.lock",
        "metrics.t800.json",
        "milestone-800.pth",
        "best-mae.pth",
        "best-rmse.pth",
        "checkpoint-manifest.json",
        "sha256sum",
        "tmux",
    ):
        assert token in text
    assert "After a restart" in text
    assert "--resume \"$CHECKPOINT_ROOT/last.pth\"" in text
    assert "Do not access the official UCF-QNRF Test" in text
