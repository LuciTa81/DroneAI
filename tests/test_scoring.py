from droneai.scoring import CheckResult, score_stage


def test_blocker_cannot_be_hidden_by_high_score() -> None:
    report = score_stage(
        stage_id="test",
        stage_name="test gate",
        threshold=80,
        checks=[
            CheckResult("quality", "model", "quality", 90, True),
            CheckResult("integrity", "data", "integrity", 10, False, blocker=True),
        ],
    )
    assert report.score == 90
    assert report.status == "BLOCKED"
    assert report.to_dict()["failed_blockers"] == ["integrity"]


def test_threshold_controls_non_blocking_review() -> None:
    report = score_stage(
        stage_id="test",
        stage_name="test gate",
        threshold=85,
        checks=[
            CheckResult("a", "one", "a", 80, True),
            CheckResult("b", "two", "b", 20, False),
        ],
    )
    assert report.status == "REVIEW"


def test_weights_must_total_one_hundred() -> None:
    try:
        score_stage(
            stage_id="test",
            stage_name="bad weights",
            threshold=80,
            checks=[CheckResult("a", "one", "a", 99, True)],
        )
    except ValueError as exc:
        assert "total 100" in str(exc)
    else:
        raise AssertionError("invalid weights were accepted")
