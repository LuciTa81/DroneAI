"""Run and verify the six-model by two-scenario report matrix on home5090."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from droneai.evaluation_artifacts import write_json
from droneai.integrity import sha256_file


MODEL_ORDER = ("steerer", "dm-count", "pet", "mpcount", "apgcc", "csrnet")
SCENARIO_ORDER = ("moderate", "high_density")


@dataclass(frozen=True)
class ScenarioCommand:
    model_id: str
    scenario_id: str
    sample_id: str
    output_dir: Path
    argv: tuple[str, ...]


def _load_config(config: str | Path | dict[str, Any]) -> dict[str, Any]:
    if isinstance(config, dict):
        return config
    target = Path(config)
    try:
        payload = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"scenario runner config is invalid: {target}") from exc
    if not isinstance(payload, dict):
        raise ValueError("scenario runner config must be an object")
    return payload


def build_commands(config: str | Path | dict[str, Any], output_root: Path) -> list[ScenarioCommand]:
    payload = _load_config(config)
    if payload.get("schema_version") != 1 or tuple(payload.get("model_order", ())) != MODEL_ORDER:
        raise ValueError("scenario runner model order is not frozen")
    scenarios = payload.get("scenarios")
    if not isinstance(scenarios, list) or tuple(
        item.get("scenario_id") if isinstance(item, dict) else None for item in scenarios
    ) != SCENARIO_ORDER:
        raise ValueError("scenario runner scenarios are not frozen")
    if tuple(
        item.get("sample_id") if isinstance(item, dict) else None for item in scenarios
    ) != ("img_0775", "img_0221"):
        raise ValueError("scenario runner sample IDs are not frozen")
    models = payload.get("models")
    if not isinstance(models, dict) or set(models) != set(MODEL_ORDER):
        raise ValueError("scenario runner model profiles are incomplete")

    commands: list[ScenarioCommand] = []
    for model_id in MODEL_ORDER:
        profile = models[model_id]
        if not isinstance(profile, dict):
            raise ValueError(f"scenario runner profile is invalid: {model_id}")
        python = profile.get("python")
        runner = profile.get("runner")
        arguments = profile.get("arguments")
        if not isinstance(python, str) or not isinstance(runner, str) or not isinstance(arguments, dict):
            raise ValueError(f"scenario runner profile is incomplete: {model_id}")
        if any(key in arguments for key in ("test-root", "train", "fine-tune", "url", "download")):
            raise ValueError(f"scenario runner has a forbidden argument: {model_id}")
        checkpoint_hash = arguments.get("checkpoint-sha256")
        if not isinstance(checkpoint_hash, str) or len(checkpoint_hash) != 64:
            raise ValueError(f"scenario runner checkpoint hash is invalid: {model_id}")
        for scenario in scenarios:
            scenario_id = str(scenario["scenario_id"])
            sample_id = str(scenario["sample_id"])
            result_dir = output_root / model_id / sample_id
            argv: list[str] = [python, runner]
            for key, value in arguments.items():
                argv.extend((f"--{key}", str(value)))
            argv.extend(("--sample-id", sample_id, "--output-dir", str(result_dir)))
            commands.append(
                ScenarioCommand(
                    model_id=model_id,
                    scenario_id=scenario_id,
                    sample_id=sample_id,
                    output_dir=result_dir,
                    argv=tuple(argv),
                )
            )
    return commands


def _command_sha256(argv: tuple[str, ...]) -> str:
    encoded = json.dumps(list(argv), ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _validate_result(command: ScenarioCommand) -> dict[str, Any]:
    result_path = command.output_dir / "result.json"
    panel_path = command.output_dir / "one-sample-panel.png"
    try:
        result = json.loads(result_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"scenario result is invalid: {command.model_id}/{command.sample_id}") from exc
    if not isinstance(result, dict) or result.get("status") != "PASS":
        raise ValueError(f"scenario result did not pass: {command.model_id}/{command.sample_id}")
    if (
        result.get("sample_id"),
        result.get("split_role"),
        result.get("evaluation_scope"),
        result.get("failure_state"),
    ) != (command.sample_id, "validation", "research_comparison_only", None):
        raise ValueError(f"scenario result identity failed: {command.model_id}/{command.sample_id}")
    panel = result.get("panel")
    if not isinstance(panel, dict) or panel.get("sha256") != sha256_file(panel_path):
        raise ValueError(f"scenario panel hash failed: {command.model_id}/{command.sample_id}")
    return {
        "model_id": command.model_id,
        "scenario_id": command.scenario_id,
        "sample_id": command.sample_id,
        "command_sha256": _command_sha256(command.argv),
        "result_path": result_path.as_posix(),
        "result_sha256": sha256_file(result_path),
        "panel_path": panel_path.as_posix(),
        "panel_sha256": sha256_file(panel_path),
    }


def execute(commands: Sequence[ScenarioCommand], *, dry_run: bool) -> list[dict[str, Any]]:
    if dry_run:
        return [
            {
                "model_id": command.model_id,
                "scenario_id": command.scenario_id,
                "sample_id": command.sample_id,
                "argv": list(command.argv),
                "command_sha256": _command_sha256(command.argv),
            }
            for command in commands
        ]
    for command in commands:
        if command.output_dir.exists() and (
            not command.output_dir.is_dir() or any(command.output_dir.iterdir())
        ):
            raise FileExistsError(f"output must be new or empty: {command.output_dir}")
    completed: list[dict[str, Any]] = []
    for command in commands:
        subprocess.run(command.argv, cwd=ROOT, check=True)
        completed.append(_validate_result(command))
    return completed


def verify(commands: Sequence[ScenarioCommand], manifest_path: Path) -> list[dict[str, Any]]:
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"scenario run manifest is invalid: {manifest_path}") from exc
    rows = [_validate_result(command) for command in commands]
    if not isinstance(manifest, dict) or manifest.get("status") != "PASS_RESEARCH_ONLY":
        raise ValueError("scenario run manifest did not pass")
    if manifest.get("results") != rows:
        raise ValueError("scenario run manifest differs from verified artifacts")
    return rows


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the frozen Round 1 report scenario matrix")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--verify-only", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    commands = build_commands(args.config, args.output_root.resolve())
    manifest_path = args.output_root.resolve() / "scenario-run-manifest.json"
    if args.verify_only:
        rows = verify(commands, manifest_path)
    else:
        rows = execute(commands, dry_run=args.dry_run)
        if not args.dry_run:
            args.output_root.resolve().mkdir(parents=True, exist_ok=True)
            write_json(
                manifest_path,
                {
                    "schema_version": 1,
                    "status": "PASS_RESEARCH_ONLY",
                    "model_order": list(MODEL_ORDER),
                    "scenario_order": list(SCENARIO_ORDER),
                    "results": rows,
                },
            )
    print(json.dumps({"status": "PASS_RESEARCH_ONLY", "count": len(rows), "results": rows}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
