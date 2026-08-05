from __future__ import annotations

import json
from pathlib import Path

from scripts.build_company_share_package import _parser, main


def test_parser_requires_explicit_spec_path() -> None:
    actions = {action.dest for action in _parser()._actions}

    assert "spec" in actions
    assert "dataset_root" not in actions
    assert "checkpoint_root" not in actions


def test_main_loads_spec_and_prints_delivery_json(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    path = tmp_path / "spec.json"
    path.write_text(json.dumps({"schema_version": 1}), encoding="utf-8")
    calls: list[Path] = []

    class Report:
        def as_dict(self):
            return {"status": "PASS_RESEARCH_ONLY", "files": 10}

    def fake_load(value: Path):
        calls.append(value)
        return object()

    monkeypatch.setattr(
        "scripts.build_company_share_package.load_company_share_spec",
        fake_load,
    )
    monkeypatch.setattr(
        "scripts.build_company_share_package.build_company_share",
        lambda _spec: Report(),
    )

    assert main(["--spec", str(path)]) == 0
    assert calls == [path.resolve()]
    assert json.loads(capsys.readouterr().out)["status"] == "PASS_RESEARCH_ONLY"
