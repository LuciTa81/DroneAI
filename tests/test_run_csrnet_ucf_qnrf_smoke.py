from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import run_csrnet_ucf_qnrf_smoke as runner


REQUIRED_DESTINATIONS = {
    "config",
    "train_root",
    "upstream_dir",
    "split_upstream_dir",
    "train_list",
    "validation_list",
    "checkpoint",
    "checkpoint_sha256",
    "rights_decision",
    "rights_manifest",
    "output_dir",
    "device",
}
FORBIDDEN_DESTINATIONS = {
    "test_root",
    "download",
    "download_url",
    "url",
    "train",
    "epoch",
    "epochs",
    "optimizer",
    "fine_tune",
    "finetune",
    "resume",
    "sample_id",
}
CSRNET_PINNED_COMMIT = "ed29d895989c188cb913a9503721271c6cf1ab1f"
SPLIT_PINNED_COMMIT = "cc5f2132e0d1328909f31b6d665b8e0b15c30467"
NEGATIVE_DENSITY_POLICY = "clip_zero_preserve_raw_audit"


def valid_args(*, output_dir: Path) -> list[str]:
    root = output_dir.parent
    return [
        "--config",
        str(root / "config.json"),
        "--train-root",
        str(root / "train"),
        "--upstream-dir",
        str(root / "upstream"),
        "--split-upstream-dir",
        str(root / "split-upstream"),
        "--train-list",
        str(root / "train.txt"),
        "--validation-list",
        str(root / "validation.txt"),
        "--checkpoint",
        str(root / "checkpoint.pth"),
        "--checkpoint-sha256",
        "a" * 64,
        "--rights-decision",
        str(root / "rights-decision.json"),
        "--rights-manifest",
        str(root / "rights-manifest.json"),
        "--output-dir",
        str(output_dir),
        "--device",
        "cuda",
    ]


def test_parser_exposes_only_frozen_non_test_non_training_inputs() -> None:
    actions = {
        action.dest: action
        for action in runner._parser()._actions
        if action.dest != "help"
    }

    assert set(actions) == REQUIRED_DESTINATIONS
    assert not FORBIDDEN_DESTINATIONS.intersection(actions)
    assert all(action.required for action in actions.values())


def test_nonempty_output_directory_is_rejected_before_work(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "run"
    output.mkdir()
    keep = output / "keep.txt"
    keep.write_text("do not overwrite", encoding="utf-8")
    monkeypatch.setattr(
        runner,
        "load_csrnet_smoke_config",
        lambda _path: pytest.fail("config work started before safety check"),
    )

    with pytest.raises(FileExistsError, match="new or empty"):
        runner.main(valid_args(output_dir=output))

    assert keep.read_text(encoding="utf-8") == "do not overwrite"


@pytest.mark.parametrize(("is_success", "expected_status"), [(True, 0), (False, 2)])
def test_fake_36_sample_run_delegates_to_common_harness(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    is_success: bool,
    expected_status: int,
) -> None:
    output = tmp_path / "run"
    calls: dict[str, object] = {}
    config = {
        "upstream_commit": CSRNET_PINNED_COMMIT,
        "split_upstream_commit": SPLIT_PINNED_COMMIT,
        "candidate_id": "csrnet-independent-shha-to-ucf-qnrf-research-comparison",
        "expected_samples": 36,
        "negative_density_policy": NEGATIVE_DENSITY_POLICY,
    }
    prepared = SimpleNamespace(
        samples=tuple(f"sample-{index:02d}" for index in range(36)),
        split_verified=True,
    )
    protocol = object()
    report = SimpleNamespace(is_success=is_success, to_markdown=lambda: "# report")

    monkeypatch.setattr(runner, "load_csrnet_smoke_config", lambda _path: config)
    monkeypatch.setattr(
        runner,
        "validate_csrnet_rights_decision",
        lambda *args, **kwargs: calls.setdefault("rights", (args, kwargs)),
    )
    monkeypatch.setattr(
        runner,
        "validate_csrnet_upstreams",
        lambda **kwargs: calls.setdefault("upstreams", kwargs),
    )
    monkeypatch.setattr(
        runner,
        "prepare_smoke_samples",
        lambda **kwargs: calls.setdefault("prepare", kwargs) and prepared,
    )

    def fake_manifest(path: Path, **kwargs: object) -> Path:
        calls["manifest"] = kwargs
        path.write_text("{}\n", encoding="utf-8")
        return path

    monkeypatch.setattr(runner, "write_split_source_manifest", fake_manifest)

    class FakeAdapter:
        def __init__(self, **kwargs: object) -> None:
            calls["adapter"] = kwargs

    monkeypatch.setattr(runner, "CSRNetAdapter", FakeAdapter)
    monkeypatch.setattr(
        runner,
        "build_csrnet_protocol",
        lambda *args, **kwargs: calls.setdefault("protocol", (args, kwargs))
        and protocol,
    )
    monkeypatch.setattr(
        runner,
        "run_evaluation",
        lambda **kwargs: calls.setdefault("evaluation", kwargs) and report,
    )
    monkeypatch.setattr(runner.os, "chdir", lambda path: calls.setdefault("cwd", path))

    assert runner.main(valid_args(output_dir=output)) == expected_status
    assert capsys.readouterr().out == "# report\n"
    assert calls["adapter"] == {
        "upstream_dir": (tmp_path / "upstream").resolve(),
        "expected_upstream_commit": CSRNET_PINNED_COMMIT,
        "checkpoint_path": (tmp_path / "checkpoint.pth").resolve(),
        "checkpoint_sha256": "a" * 64,
        "device": "cuda",
        "negative_density_policy": NEGATIVE_DENSITY_POLICY,
    }
    assert calls["evaluation"]["samples"] == prepared.samples
    assert calls["evaluation"]["protocol"] is protocol
    assert calls["evaluation"]["output_dir"] == output.resolve()
    assert calls["evaluation"]["provenance_artifacts"] == (
        output.resolve() / "split-source-manifest.json",
    )


def test_wrong_sample_count_stops_before_adapter_and_evaluation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "run"
    config = {
        "upstream_commit": CSRNET_PINNED_COMMIT,
        "split_upstream_commit": SPLIT_PINNED_COMMIT,
        "candidate_id": "csrnet-independent-shha-to-ucf-qnrf-research-comparison",
        "expected_samples": 36,
        "negative_density_policy": NEGATIVE_DENSITY_POLICY,
    }
    monkeypatch.setattr(runner, "load_csrnet_smoke_config", lambda _path: config)
    monkeypatch.setattr(runner, "validate_csrnet_rights_decision", lambda *a, **k: None)
    monkeypatch.setattr(runner, "validate_csrnet_upstreams", lambda **kwargs: None)
    monkeypatch.setattr(
        runner,
        "prepare_smoke_samples",
        lambda **kwargs: SimpleNamespace(samples=("only-one",), split_verified=True),
    )
    monkeypatch.setattr(
        runner,
        "CSRNetAdapter",
        lambda **kwargs: pytest.fail("adapter constructed after sample-count failure"),
    )

    with pytest.raises(ValueError, match="exactly 36"):
        runner.main(valid_args(output_dir=output))
