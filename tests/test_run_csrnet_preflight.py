from __future__ import annotations

import importlib.util
import json
import subprocess
from pathlib import Path
from typing import Sequence

import pytest

from droneai.stage3c import manifest_semantic_sha256


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/run_csrnet_preflight.py"
UPSTREAM_COMMIT = "ed29d895989c188cb913a9503721271c6cf1ab1f"


def _preflight():
    assert SCRIPT.is_file()
    spec = importlib.util.spec_from_file_location("run_csrnet_preflight", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _artifact(path: Path, payload: bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return path


def _json(path: Path, payload: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


class Runner:
    def __init__(self, *, inference_executed: bool = False) -> None:
        self.inference_executed = inference_executed
        self.calls: list[tuple[str, ...]] = []

    def __call__(self, command: Sequence[str]) -> subprocess.CompletedProcess[str]:
        call = tuple(str(part) for part in command)
        self.calls.append(call)
        if call[0] == "git":
            is_repo = Path(call[2]).name == "DroneAI"
            if call[-2:] == ("status", "--porcelain"):
                return subprocess.CompletedProcess(call, 0, "", "")
            if call[-2:] == ("rev-parse", "HEAD"):
                commit = "d" * 40 if is_repo else UPSTREAM_COMMIT
                return subprocess.CompletedProcess(call, 0, commit + "\n", "")
        if "-c" in call:
            payload = {
                "python": "3.12.3",
                "torch": "2.9.0a0+nv25.10",
                "torchvision": "0.24.0a0",
                "cuda_available": True,
                "cuda_runtime": "13.0",
                "gpu": "NVIDIA GeForce RTX 5090",
                "matmul": 32.0,
                "state_keys": 34,
                "all_finite": True,
                "missing_keys": [],
                "unexpected_keys": [],
                "parameter_count": 16263489,
                "convolution_layers": 17,
                "dilated_convolution_layers": 6,
                "output_stride": 8,
                "inference_executed": self.inference_executed,
            }
            return subprocess.CompletedProcess(call, 0, json.dumps(payload), "")
        raise AssertionError(f"unexpected command: {call}")


def _fixtures(tmp_path: Path, module):
    raw = _artifact(tmp_path / "raw.pth", b"raw-checkpoint")
    safe = _artifact(tmp_path / "safe.pth", b"safe-state-dict")
    model_profile = _json(
        tmp_path / "model.json",
        {
            "checkpoint": {
                "raw_sha256": module.sha256_file(raw),
                "raw_size_bytes": raw.stat().st_size,
                "safe_state_dict_sha256": module.sha256_file(safe),
                "safe_state_dict_size_bytes": safe.stat().st_size,
            }
        },
    )
    manifest = {
        "schema_version": 1,
        "candidate_id": "csrnet-candidate",
        "intended_use": "commercial_product_rnd",
        "components": [],
    }
    rights_manifest = _json(tmp_path / "manifest.json", manifest)
    rights_decision = _json(
        tmp_path / "rights.json",
        {
            "status": "PASS_COMMERCIAL_CANDIDATE",
            "candidate_id": "csrnet-candidate",
            "manifest_semantic_sha256": manifest_semantic_sha256(manifest),
            "allowed_actions": ["research_checkpoint_evaluation"],
        },
    )
    return {
        "repo_path": tmp_path / "DroneAI",
        "upstream_path": tmp_path / "CSRNet-pytorch",
        "expected_upstream_commit": UPSTREAM_COMMIT,
        "raw_checkpoint_path": raw,
        "safe_checkpoint_path": safe,
        "dataset_config_path": _artifact(tmp_path / "dataset.json", b"dataset"),
        "split_manifest_path": _artifact(tmp_path / "split.json", b"split"),
        "sample_manifest_path": _artifact(tmp_path / "samples.json", b"samples"),
        "model_profile_path": model_profile,
        "runtime_profile_path": _artifact(tmp_path / "runtime.json", b"runtime"),
        "rights_decision_path": rights_decision,
        "rights_manifest_path": rights_manifest,
    }


def test_csrnet_preflight_hashes_assets_and_strictly_maps_without_inference(
    tmp_path: Path,
) -> None:
    module = _preflight()
    runner = Runner()

    result = module.capture_preflight(**_fixtures(tmp_path, module), runner=runner)

    assert result["status"] == "PASS"
    assert result["model_id"] == "csrnet"
    assert result["inference_executed"] is False
    assert result["upstream"]["commit"] == UPSTREAM_COMMIT
    assert result["checkpoint"]["raw"]["sha256"] == module.sha256_file(
        tmp_path / "raw.pth"
    )
    assert result["checkpoint"]["safe_state_dict"]["sha256"] == module.sha256_file(
        tmp_path / "safe.pth"
    )
    assert result["environment"]["missing_keys"] == []
    assert result["environment"]["unexpected_keys"] == []
    probe_call = next(call for call in runner.calls if "-c" in call)
    probe = probe_call[probe_call.index("-c") + 1]
    assert "weights_only=True" in probe
    assert "strict=True" in probe
    assert ".forward(" not in probe


def test_csrnet_preflight_rejects_checkpoint_hash_mismatch(tmp_path: Path) -> None:
    module = _preflight()
    fixtures = _fixtures(tmp_path, module)
    profile = json.loads(
        Path(fixtures["model_profile_path"]).read_text(encoding="utf-8")
    )
    profile["checkpoint"]["raw_sha256"] = "0" * 64
    Path(fixtures["model_profile_path"]).write_text(
        json.dumps(profile), encoding="utf-8"
    )

    with pytest.raises(ValueError, match="raw checkpoint identity mismatch"):
        module.capture_preflight(**fixtures, runner=Runner())


def test_csrnet_preflight_rejects_probe_that_ran_inference(tmp_path: Path) -> None:
    module = _preflight()
    with pytest.raises(ValueError, match="must not execute inference"):
        module.capture_preflight(
            **_fixtures(tmp_path, module), runner=Runner(inference_executed=True)
        )
