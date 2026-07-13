from droneai.runtime_probe import collect_environment


def test_runtime_probe_has_required_fields() -> None:
    report = collect_environment()
    assert report["python"]
    assert report["platform"]
    assert "torch" in report
    assert "tensorflow" in report
