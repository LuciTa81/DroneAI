from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import run_steerer_ucf_qnrf_smoke as runner


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
}
STEERER_PINNED_COMMIT = "5b1854dbc2d280f2326d67c65515d8baf9083810"
SPLIT_UPSTREAM_PINNED_COMMIT = "cc5f2132e0d1328909f31b6d665b8e0b15c30467"


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
        "cpu",
    ]


def test_parser_exposes_only_explicit_non_test_non_training_inputs() -> None:
    actions = {
        action.dest: action
        for action in runner._parser()._actions
        if action.dest != "help"
    }

    assert set(actions) == REQUIRED_DESTINATIONS
    assert not FORBIDDEN_DESTINATIONS.intersection(actions)
    assert all(action.required for action in actions.values())


def test_nonempty_output_directory_is_rejected_before_work(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "run"
    output.mkdir()
    keep = output / "keep.txt"
    keep.write_text("do not overwrite", encoding="utf-8")
    monkeypatch.setattr(
        runner,
        "load_steerer_smoke_config",
        lambda _path: pytest.fail("config work started before the safety check"),
    )

    with pytest.raises(FileExistsError, match="new or empty"):
        runner.main(valid_args(output_dir=output))

    assert keep.read_text(encoding="utf-8") == "do not overwrite"


@pytest.mark.parametrize(("is_success", "expected_status"), [(True, 0), (False, 2)])
def test_fake_end_to_end_returns_common_report_status(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    is_success: bool,
    expected_status: int,
) -> None:
    output = tmp_path / "run"
    calls: dict[str, object] = {}
    config = {
        "upstream_commit": STEERER_PINNED_COMMIT,
        "split_upstream_commit": SPLIT_UPSTREAM_PINNED_COMMIT,
        "candidate_id": "steerer-official-ucf-qnrf-research-comparison",
    }
    prepared = SimpleNamespace(samples=("fake-sample",), split_verified=True)
    protocol = object()
    report = SimpleNamespace(is_success=is_success, to_markdown=lambda: "# fake report")

    monkeypatch.setattr(runner, "load_steerer_smoke_config", lambda path: config)
    monkeypatch.setattr(
        runner,
        "validate_steerer_rights_decision",
        lambda *args, **kwargs: calls.setdefault(
            "rights_kwargs", (args, kwargs)
        ),
    )
    monkeypatch.setattr(
        runner,
        "validate_steerer_split_upstream",
        lambda **kwargs: calls.setdefault("split_upstream", kwargs),
    )
    monkeypatch.setattr(
        runner,
        "prepare_smoke_samples",
        lambda **kwargs: calls.setdefault("prepared_kwargs", kwargs) and prepared,
    )

    def fake_write_manifest(path: Path, **kwargs: object) -> Path:
        calls["manifest_kwargs"] = kwargs
        path.write_text("{}\n", encoding="utf-8")
        return path

    monkeypatch.setattr(runner, "write_split_source_manifest", fake_write_manifest)

    class FakeAdapter:
        def __init__(self, **kwargs: object) -> None:
            calls["adapter_kwargs"] = kwargs

    monkeypatch.setattr(runner, "STEERERAdapter", FakeAdapter)
    monkeypatch.setattr(
        runner,
        "build_steerer_protocol",
        lambda *args, **kwargs: calls.setdefault(
            "protocol_kwargs", (args, kwargs)
        )
        and protocol,
    )
    monkeypatch.setattr(
        runner,
        "run_evaluation",
        lambda **kwargs: calls.setdefault("evaluation_kwargs", kwargs) and report,
    )
    monkeypatch.setattr(runner.os, "chdir", lambda path: calls.setdefault("cwd", path))

    status = runner.main(valid_args(output_dir=output))

    assert status == expected_status
    assert capsys.readouterr().out == "# fake report\n"
    assert output.is_dir()
    assert calls["adapter_kwargs"] == {
        "upstream_dir": (tmp_path / "upstream").resolve(),
        "expected_upstream_commit": STEERER_PINNED_COMMIT,
        "checkpoint_path": (tmp_path / "checkpoint.pth").resolve(),
        "checkpoint_sha256": "a" * 64,
        "device": "cpu",
    }
    assert calls["split_upstream"] == {
        "split_upstream_dir": (tmp_path / "split-upstream").resolve(),
        "train_list_path": (tmp_path / "train.txt").resolve(),
        "validation_list_path": (tmp_path / "validation.txt").resolve(),
        "expected_commit": SPLIT_UPSTREAM_PINNED_COMMIT,
    }
    assert calls["manifest_kwargs"] == {
        "prepared": prepared,
        "train_list_path": (tmp_path / "train.txt").resolve(),
        "validation_list_path": (tmp_path / "validation.txt").resolve(),
        "model_upstream_commit": STEERER_PINNED_COMMIT,
        "split_upstream_commit": SPLIT_UPSTREAM_PINNED_COMMIT,
    }
    assert calls["rights_kwargs"] == (
        ((tmp_path / "rights-decision.json").resolve(),),
        {
            "manifest_path": (tmp_path / "rights-manifest.json").resolve(),
            "expected_candidate_id": "steerer-official-ucf-qnrf-research-comparison",
        },
    )
    evaluation_kwargs = calls["evaluation_kwargs"]
    assert isinstance(evaluation_kwargs, dict)
    assert evaluation_kwargs["samples"] == prepared.samples
    assert evaluation_kwargs["protocol"] is protocol
    assert evaluation_kwargs["output_dir"] == output.resolve()
    assert evaluation_kwargs["provenance_artifacts"] == (
        output.resolve() / "split-source-manifest.json",
    )


def test_rights_exceptions_are_not_converted_to_gate_failures(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "run"
    monkeypatch.setattr(
        runner,
        "load_steerer_smoke_config",
        lambda _path: {
            "candidate_id": "steerer-official-ucf-qnrf-research-comparison",
        },
    )
    monkeypatch.setattr(
        runner,
        "validate_steerer_rights_decision",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            PermissionError("rights denied")
        ),
    )

    with pytest.raises(PermissionError, match="rights denied"):
        runner.main(valid_args(output_dir=output))


def test_split_upstream_failure_stops_before_sample_or_adapter_work(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "run"
    monkeypatch.setattr(
        runner,
        "load_steerer_smoke_config",
        lambda _path: {
            "upstream_commit": STEERER_PINNED_COMMIT,
            "split_upstream_commit": SPLIT_UPSTREAM_PINNED_COMMIT,
            "candidate_id": "steerer-official-ucf-qnrf-research-comparison",
        },
    )
    monkeypatch.setattr(
        runner, "validate_steerer_rights_decision", lambda *args, **kwargs: None
    )
    monkeypatch.setattr(
        runner,
        "validate_steerer_split_upstream",
        lambda **kwargs: (_ for _ in ()).throw(ValueError("split checkout dirty")),
    )
    monkeypatch.setattr(
        runner,
        "prepare_smoke_samples",
        lambda **kwargs: pytest.fail("sample indexing ran after split failure"),
    )
    monkeypatch.setattr(
        runner,
        "STEERERAdapter",
        lambda **kwargs: pytest.fail("adapter constructed after split failure"),
    )

    with pytest.raises(ValueError, match="split checkout dirty"):
        runner.main(valid_args(output_dir=output))


def test_dependency_overlay_pins_only_the_seventeen_model_and_evaluation_dependencies() -> None:
    overlay = Path("requirements/models/steerer-home5090.txt")

    assert overlay.read_text(encoding="utf-8").splitlines() == [
        "mmcv==1.7.2",
        "addict==2.4.0",
        "packaging==25.0",
        "Pillow==11.3.0",
        "PyYAML==6.0.2",
        "yapf==0.43.0",
        "opencv-python-headless==4.12.0.88",
        "timm==1.0.19",
        "einops==0.8.1",
        "dict_recursive_update==1.0.1",
        "yacs==0.1.8",
        "tensorboardX==2.6.2.2",
        "fvcore==0.1.5.post20221221",
        "scikit-image==0.26.0",
        "ImageIO==2.37.3",
        "tifffile==2026.7.14",
        "lazy_loader==0.4",
    ]
