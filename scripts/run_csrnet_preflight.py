"""Verify CSRNet identities, strict checkpoint mapping and CUDA without inference."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from droneai.stage3c import manifest_semantic_sha256


Runner = Callable[[Sequence[str]], subprocess.CompletedProcess[str]]


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _run(command: Sequence[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [str(part) for part in command],
        check=False,
        capture_output=True,
        text=True,
    )


def _git_output(runner: Runner, repository: Path, *arguments: str) -> str:
    result = runner(("git", "-C", str(repository), *arguments))
    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "no output").strip()
        raise RuntimeError(f"git {' '.join(arguments)} failed: {detail}")
    return result.stdout.strip()


def _identity(path: str | Path) -> dict[str, object]:
    target = Path(path)
    if not target.is_file():
        raise FileNotFoundError(target)
    return {
        "path": str(target),
        "sha256": sha256_file(target),
        "size_bytes": target.stat().st_size,
    }


def _json_object(path: str | Path) -> dict[str, object]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def _probe_environment(
    *,
    python: Path,
    source_root: Path,
    safe_checkpoint: Path,
    runner: Runner,
) -> dict[str, object]:
    code = (
        "import json,platform,sys,torch,torchvision\n"
        "sys.path.insert(0,sys.argv[1])\n"
        "from droneai.csrnet import CSRNet,architecture_summary\n"
        "state=torch.load(sys.argv[2],map_location='cpu',weights_only=True)\n"
        "model=CSRNet()\n"
        "loaded=model.load_state_dict(state,strict=True)\n"
        "available=torch.cuda.is_available()\n"
        "matrix=torch.ones((32,32),device='cuda') if available else None\n"
        "matmul=float((matrix@matrix)[0,0].item()) if available else None\n"
        "summary=architecture_summary(model)\n"
        "print(json.dumps({"
        "'python':platform.python_version(),"
        "'torch':torch.__version__,"
        "'torchvision':torchvision.__version__,"
        "'cuda_available':available,"
        "'cuda_runtime':torch.version.cuda,"
        "'gpu':torch.cuda.get_device_name(0) if available else None,"
        "'matmul':matmul,"
        "'state_keys':len(state),"
        "'all_finite':all(bool(torch.isfinite(value).all()) for value in state.values()),"
        "'missing_keys':list(loaded.missing_keys),"
        "'unexpected_keys':list(loaded.unexpected_keys),"
        "'parameter_count':summary['parameters'],"
        "'convolution_layers':summary['convolution_layers'],"
        "'dilated_convolution_layers':summary['dilated_convolution_layers'],"
        "'output_stride':summary['output_stride'],"
        "'inference_executed':False"
        "}))\n"
    )
    result = runner(
        (str(python), "-c", code, str(source_root), str(safe_checkpoint))
    )
    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "no output").strip()
        raise RuntimeError(f"CSRNet environment probe failed: {detail}")
    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError("CSRNet environment probe returned invalid JSON") from exc
    if not isinstance(payload, dict):
        raise RuntimeError("CSRNet environment probe must return a JSON object")
    if payload.get("inference_executed") is not False:
        raise ValueError("preflight environment probe must not execute inference")
    expected = {
        "cuda_available": True,
        "matmul": 32.0,
        "state_keys": 34,
        "all_finite": True,
        "missing_keys": [],
        "unexpected_keys": [],
        "parameter_count": 16263489,
        "convolution_layers": 17,
        "dilated_convolution_layers": 6,
        "output_stride": 8,
    }
    mismatches = [
        f"{key}: expected {value!r}, found {payload.get(key)!r}"
        for key, value in expected.items()
        if payload.get(key) != value
    ]
    if mismatches:
        raise ValueError("CSRNet environment mismatch: " + " | ".join(mismatches))
    return payload


def capture_preflight(
    *,
    repo_path: str | Path,
    upstream_path: str | Path,
    expected_upstream_commit: str,
    raw_checkpoint_path: str | Path,
    safe_checkpoint_path: str | Path,
    dataset_config_path: str | Path,
    split_manifest_path: str | Path,
    sample_manifest_path: str | Path,
    model_profile_path: str | Path,
    runtime_profile_path: str | Path,
    rights_decision_path: str | Path,
    rights_manifest_path: str | Path,
    runner: Runner | None = None,
) -> dict[str, object]:
    command_runner = runner or _run
    repo = Path(repo_path)
    upstream = Path(upstream_path)

    repo_dirty = _git_output(command_runner, repo, "status", "--porcelain")
    if repo_dirty:
        raise ValueError(f"DroneAI repository is dirty: {repo_dirty}")
    git_commit = _git_output(command_runner, repo, "rev-parse", "HEAD")
    upstream_dirty = _git_output(
        command_runner, upstream, "status", "--porcelain"
    )
    if upstream_dirty:
        raise ValueError(f"upstream repository is dirty: {upstream_dirty}")
    upstream_commit = _git_output(command_runner, upstream, "rev-parse", "HEAD")
    if upstream_commit != expected_upstream_commit:
        raise ValueError(
            "upstream commit mismatch: "
            f"expected {expected_upstream_commit}, found {upstream_commit}"
        )

    model_profile = _json_object(model_profile_path)
    checkpoint_profile = model_profile.get("checkpoint")
    if not isinstance(checkpoint_profile, dict):
        raise ValueError("model profile must contain checkpoint identities")
    raw = _identity(raw_checkpoint_path)
    safe = _identity(safe_checkpoint_path)
    if (
        raw["sha256"] != checkpoint_profile.get("raw_sha256")
        or raw["size_bytes"] != checkpoint_profile.get("raw_size_bytes")
    ):
        raise ValueError("raw checkpoint identity mismatch")
    if (
        safe["sha256"] != checkpoint_profile.get("safe_state_dict_sha256")
        or safe["size_bytes"] != checkpoint_profile.get("safe_state_dict_size_bytes")
    ):
        raise ValueError("safe checkpoint identity mismatch")

    rights_manifest = _json_object(rights_manifest_path)
    rights_decision = _json_object(rights_decision_path)
    if rights_decision.get("status") != "PASS_COMMERCIAL_CANDIDATE":
        raise PermissionError("CSRNet rights decision must be PASS_COMMERCIAL_CANDIDATE")
    actions = rights_decision.get("allowed_actions")
    if not isinstance(actions, list) or "research_checkpoint_evaluation" not in actions:
        raise PermissionError(
            "CSRNet rights decision must allow research_checkpoint_evaluation"
        )
    if rights_decision.get("candidate_id") != rights_manifest.get("candidate_id"):
        raise ValueError("CSRNet rights candidate identity mismatch")
    if rights_decision.get("manifest_semantic_sha256") != manifest_semantic_sha256(
        rights_manifest
    ):
        raise ValueError("CSRNet rights manifest semantic hash mismatch")

    environment = _probe_environment(
        python=Path(sys.executable),
        source_root=repo / "src",
        safe_checkpoint=Path(safe_checkpoint_path),
        runner=command_runner,
    )
    identities = {
        "dataset_config": _identity(dataset_config_path),
        "split_manifest": _identity(split_manifest_path),
        "sample_manifest": _identity(sample_manifest_path),
        "model_profile": _identity(model_profile_path),
        "runtime_profile": _identity(runtime_profile_path),
        "rights_decision": _identity(rights_decision_path),
        "rights_manifest": _identity(rights_manifest_path),
    }
    return {
        "schema_version": 1,
        "status": "PASS",
        "gate": "preflight",
        "model_id": "csrnet",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "inference_executed": False,
        "git": {"path": str(repo), "commit": git_commit, "clean": True},
        "upstream": {
            "path": str(upstream),
            "commit": upstream_commit,
            "clean": True,
            "executed": False,
        },
        "checkpoint": {
            "raw": raw,
            "safe_state_dict": safe,
            "mapping": {
                "strict": True,
                "state_keys": environment["state_keys"],
                "missing_keys": environment["missing_keys"],
                "unexpected_keys": environment["unexpected_keys"],
                "all_finite": environment["all_finite"],
            },
            "conversion": checkpoint_profile.get("conversion"),
        },
        "environment": environment,
        **identities,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Verify CSRNet identities, CUDA and strict mapping without inference."
    )
    parser.add_argument("--repo", required=True, type=Path)
    parser.add_argument("--upstream", required=True, type=Path)
    parser.add_argument("--expected-upstream-commit", required=True)
    parser.add_argument("--raw-checkpoint", required=True, type=Path)
    parser.add_argument("--safe-checkpoint", required=True, type=Path)
    parser.add_argument("--dataset-config", required=True, type=Path)
    parser.add_argument("--split-manifest", required=True, type=Path)
    parser.add_argument("--sample-manifest", required=True, type=Path)
    parser.add_argument("--model-profile", required=True, type=Path)
    parser.add_argument("--runtime-profile", required=True, type=Path)
    parser.add_argument("--rights-decision", required=True, type=Path)
    parser.add_argument("--rights-manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        payload = capture_preflight(
            repo_path=args.repo,
            upstream_path=args.upstream,
            expected_upstream_commit=args.expected_upstream_commit,
            raw_checkpoint_path=args.raw_checkpoint,
            safe_checkpoint_path=args.safe_checkpoint,
            dataset_config_path=args.dataset_config,
            split_manifest_path=args.split_manifest,
            sample_manifest_path=args.sample_manifest,
            model_profile_path=args.model_profile,
            runtime_profile_path=args.runtime_profile,
            rights_decision_path=args.rights_decision,
            rights_manifest_path=args.rights_manifest,
        )
        args.output.parent.mkdir(parents=True, exist_ok=False)
        args.output.write_text(
            json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
        return 0
    except Exception as exc:
        print(
            json.dumps(
                {
                    "schema_version": 1,
                    "status": "FAIL",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                },
                ensure_ascii=False,
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    sys.exit(main())
