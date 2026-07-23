from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import cast

from droneai.evaluation_contract import OutputType, ScalarEvaluation
from droneai.integrity import is_sha256, sha256_file

_OUTPUT_TYPES = {"count", "density", "points", "hybrid"}
_SAFE_SAMPLE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")
_GIT_COMMIT = re.compile(r"[0-9a-f]{40}")


@dataclass(frozen=True)
class AcceptedPrediction:
    sample_id: str
    predicted_count: float
    output_type: OutputType

    def __post_init__(self) -> None:
        if not self.sample_id:
            raise ValueError("accepted prediction requires a sample identity")
        if (
            not math.isfinite(self.predicted_count)
            or self.predicted_count < 0
        ):
            raise ValueError(
                "accepted predicted count must be finite and non-negative"
            )
        if self.output_type not in _OUTPUT_TYPES:
            raise ValueError("accepted prediction output type is invalid")


@dataclass(frozen=True)
class PanelExportIdentity:
    model_id: str
    dataset_id: str
    git_commit: str
    upstream_commit: str
    checkpoint_sha256: str
    shared_manifest_sha256: str

    def __post_init__(self) -> None:
        if not self.model_id or not self.dataset_id:
            raise ValueError("panel export model and dataset identities are required")
        for name, value in (
            ("git commit", self.git_commit),
            ("upstream commit", self.upstream_commit),
        ):
            if _GIT_COMMIT.fullmatch(value) is None:
                raise ValueError(f"panel export {name} is invalid")
        for name, value in (
            ("checkpoint", self.checkpoint_sha256),
            ("shared manifest", self.shared_manifest_sha256),
        ):
            if not is_sha256(value):
                raise ValueError(f"panel export {name} SHA-256 is invalid")


def _identity_sha256(identity: PanelExportIdentity) -> str:
    encoded = json.dumps(
        asdict(identity),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def load_accepted_predictions(path: str | Path) -> dict[str, AcceptedPrediction]:
    source = Path(path)
    with source.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {
            "sample_id",
            "predicted_count",
            "output_type",
            "failure_state",
        }
        if reader.fieldnames is None or not required <= set(reader.fieldnames):
            raise ValueError("accepted predictions CSV fields are incomplete")
        accepted: dict[str, AcceptedPrediction] = {}
        for row in reader:
            sample_id = str(row["sample_id"]).strip()
            if sample_id in accepted:
                raise ValueError(
                    f"duplicate accepted prediction sample: {sample_id}"
                )
            failure = str(row["failure_state"] or "").strip()
            if failure:
                raise ValueError(
                    f"accepted prediction contains a failure: {sample_id}"
                )
            try:
                predicted_count = float(row["predicted_count"])
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"accepted predicted count is invalid: {sample_id}"
                ) from exc
            output_type = str(row["output_type"]).strip()
            prediction = AcceptedPrediction(
                sample_id=sample_id,
                predicted_count=predicted_count,
                output_type=cast(OutputType, output_type),
            )
            accepted[sample_id] = prediction
    if not accepted:
        raise ValueError("accepted predictions CSV is empty")
    return accepted


def verify_fresh_record(
    accepted: AcceptedPrediction,
    fresh: ScalarEvaluation,
) -> None:
    if fresh.sample_id != accepted.sample_id:
        raise ValueError("fresh sample identity differs from accepted prediction")
    if fresh.failure_state is not None or fresh.predicted_count is None:
        raise ValueError("fresh prediction contains an explicit failure")
    if fresh.output_type != accepted.output_type:
        raise ValueError("fresh output type differs from accepted prediction")
    if not math.isfinite(fresh.predicted_count) or fresh.predicted_count < 0:
        raise ValueError("fresh predicted count must be finite and non-negative")
    if not math.isclose(
        fresh.predicted_count,
        accepted.predicted_count,
        rel_tol=0.0,
        abs_tol=1e-6,
    ):
        raise ValueError("fresh result differs from accepted predicted count")


def panel_filename(sample_id: str) -> str:
    if (
        _SAFE_SAMPLE_ID.fullmatch(sample_id) is None
        or sample_id in {".", ".."}
    ):
        raise ValueError("sample identity is unsafe for a panel path")
    digest = hashlib.sha256(sample_id.encode("utf-8")).hexdigest()[:16]
    return f"{sample_id}-{digest}.png"


class PanelProgressLedger:
    def __init__(
        self,
        *,
        path: Path,
        identity: PanelExportIdentity,
        output_root: Path,
        completed: dict[str, dict[str, object]],
    ) -> None:
        self.path = path
        self.identity = identity
        self.output_root = output_root
        self._identity_sha256 = _identity_sha256(identity)
        self._completed = completed

    @classmethod
    def open(
        cls,
        path: str | Path,
        *,
        identity: PanelExportIdentity,
        output_root: str | Path,
    ) -> PanelProgressLedger:
        ledger_path = Path(path).resolve()
        root = Path(output_root).resolve()
        root.mkdir(parents=True, exist_ok=True)
        try:
            ledger_path.relative_to(root)
        except ValueError as exc:
            raise ValueError("progress ledger must stay inside output root") from exc

        identity_digest = _identity_sha256(identity)
        completed: dict[str, dict[str, object]] = {}
        if not ledger_path.exists():
            ledger_path.parent.mkdir(parents=True, exist_ok=True)
            header = {
                "schema_version": 1,
                "kind": "identity",
                "identity": asdict(identity),
                "identity_sha256": identity_digest,
            }
            with ledger_path.open("x", encoding="utf-8", newline="\n") as handle:
                handle.write(
                    json.dumps(
                        header,
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                    )
                    + "\n"
                )
                handle.flush()
                os.fsync(handle.fileno())
        else:
            rows: list[dict[str, object]] = []
            with ledger_path.open("r", encoding="utf-8") as handle:
                for line_number, line in enumerate(handle, start=1):
                    if not line.strip():
                        raise ValueError(
                            f"progress ledger contains a blank row: {line_number}"
                        )
                    payload = json.loads(line)
                    if not isinstance(payload, dict):
                        raise ValueError("progress ledger row must be an object")
                    rows.append(payload)
            if not rows or rows[0].get("kind") != "identity":
                raise ValueError("progress ledger identity header is missing")
            if (
                rows[0].get("identity") != asdict(identity)
                or rows[0].get("identity_sha256") != identity_digest
            ):
                raise ValueError("progress ledger identity differs from requested run")
            for row in rows[1:]:
                if row.get("kind") != "completed":
                    raise ValueError("progress ledger row kind is invalid")
                if row.get("identity_sha256") != identity_digest:
                    raise ValueError("completed progress identity is invalid")
                sample_id = str(row.get("sample_id", ""))
                if sample_id in completed:
                    raise ValueError(
                        f"duplicate completed progress sample: {sample_id}"
                    )
                relative = Path(str(row.get("panel_path", "")))
                if relative.is_absolute() or ".." in relative.parts:
                    raise ValueError("completed panel path is unsafe")
                panel = (root / relative).resolve()
                try:
                    panel.relative_to(root)
                except ValueError as exc:
                    raise ValueError(
                        "completed panel must stay inside output root"
                    ) from exc
                if not panel.is_file():
                    raise FileNotFoundError(f"completed panel is missing: {panel}")
                if sha256_file(panel) != row.get("panel_sha256"):
                    raise ValueError(
                        f"completed panel SHA-256 mismatch: {sample_id}"
                    )
                completed[sample_id] = row
        return cls(
            path=ledger_path,
            identity=identity,
            output_root=root,
            completed=completed,
        )

    def completed_sample_ids(self) -> tuple[str, ...]:
        return tuple(self._completed)

    def append_completed(
        self,
        *,
        source_sha256: str,
        accepted: AcceptedPrediction,
        fresh: ScalarEvaluation,
        panel_path: str | Path,
    ) -> None:
        if not is_sha256(source_sha256):
            raise ValueError("completed sample source SHA-256 is invalid")
        verify_fresh_record(accepted, fresh)
        if accepted.sample_id in self._completed:
            raise ValueError(
                f"panel progress sample is already complete: {accepted.sample_id}"
            )
        panel = Path(panel_path).resolve()
        try:
            relative = panel.relative_to(self.output_root)
        except ValueError as exc:
            raise ValueError("panel must stay inside output root") from exc
        if relative.name != panel_filename(accepted.sample_id):
            raise ValueError("panel filename differs from the sample identity")
        if not panel.is_file():
            raise FileNotFoundError(f"completed panel is missing: {panel}")
        row: dict[str, object] = {
            "schema_version": 1,
            "kind": "completed",
            "identity_sha256": self._identity_sha256,
            "sample_id": accepted.sample_id,
            "source_sha256": source_sha256,
            "accepted_predicted_count": accepted.predicted_count,
            "fresh_predicted_count": fresh.predicted_count,
            "output_type": fresh.output_type,
            "panel_path": relative.as_posix(),
            "panel_sha256": sha256_file(panel),
            "latency_ms": fresh.latency_ms,
        }
        with self.path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(
                json.dumps(
                    row,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
                + "\n"
            )
            handle.flush()
            os.fsync(handle.fileno())
        self._completed[accepted.sample_id] = row
