"""Append-only, identity-bound progress for resumable evaluation runs."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

from droneai.evaluation_contract import ScalarEvaluation
from droneai.integrity import is_sha256


@dataclass(frozen=True)
class ProgressEntry:
    record: ScalarEvaluation
    native_metadata: dict[str, object]
    fingerprint: str


def _canonical_line(payload: object) -> bytes:
    return (
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")


def _sample_ids_sha256(sample_ids: Sequence[str]) -> str:
    payload = json.dumps(
        list(sample_ids), ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _write_fsynced(path: Path, payload: bytes, *, exclusive: bool) -> None:
    flags = os.O_WRONLY | os.O_CREAT
    flags |= os.O_EXCL if exclusive else os.O_APPEND
    descriptor = os.open(path, flags, 0o600)
    try:
        os.write(descriptor, payload)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


class ProgressLedger:
    def __init__(
        self,
        *,
        path: Path,
        identity_sha256: str,
        ordered_sample_ids: tuple[str, ...],
        entries: list[ProgressEntry],
    ) -> None:
        self.path = path
        self.identity_sha256 = identity_sha256
        self.ordered_sample_ids = ordered_sample_ids
        self._entries = entries

    @classmethod
    def open(
        cls,
        path: str | Path,
        identity_sha256: str,
        ordered_sample_ids: Sequence[str],
    ) -> "ProgressLedger":
        target = Path(path)
        sample_ids = tuple(ordered_sample_ids)
        if not is_sha256(identity_sha256):
            raise ValueError("progress identity must be a SHA-256")
        if (
            not sample_ids
            or len(sample_ids) != len(set(sample_ids))
            or any(not value for value in sample_ids)
        ):
            raise ValueError("progress requires unique ordered sample IDs")
        expected_ids_sha256 = _sample_ids_sha256(sample_ids)
        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            _write_fsynced(
                target,
                _canonical_line(
                    {
                        "type": "header",
                        "schema_version": 1,
                        "identity_sha256": identity_sha256,
                        "sample_ids_sha256": expected_ids_sha256,
                        "ordered_sample_ids": list(sample_ids),
                    }
                ),
                exclusive=True,
            )
            return cls(
                path=target,
                identity_sha256=identity_sha256,
                ordered_sample_ids=sample_ids,
                entries=[],
            )
        lines = target.read_text(encoding="utf-8").splitlines()
        if not lines:
            raise ValueError("progress ledger is empty")
        try:
            header = json.loads(lines[0])
        except json.JSONDecodeError as exc:
            raise ValueError("progress header is not valid JSON") from exc
        if (
            header.get("type") != "header"
            or header.get("schema_version") != 1
            or header.get("identity_sha256") != identity_sha256
        ):
            raise ValueError("progress identity differs from the current run")
        if (
            header.get("sample_ids_sha256") != expected_ids_sha256
            or tuple(header.get("ordered_sample_ids") or ()) != sample_ids
        ):
            raise ValueError("progress sample order differs from the current run")
        entries: list[ProgressEntry] = []
        for line_number, line in enumerate(lines[1:], start=2):
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"progress line {line_number} is not valid JSON"
                ) from exc
            if payload.get("type") != "sample":
                raise ValueError(f"progress line {line_number} has invalid type")
            record_payload = payload.get("record")
            metadata = payload.get("native_metadata")
            fingerprint = payload.get("fingerprint")
            if not isinstance(record_payload, dict) or not isinstance(metadata, dict):
                raise ValueError(f"progress line {line_number} has invalid payload")
            if not isinstance(fingerprint, str) or not is_sha256(fingerprint):
                raise ValueError(f"progress line {line_number} has invalid fingerprint")
            try:
                record = ScalarEvaluation(**record_payload)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"progress line {line_number} has invalid record"
                ) from exc
            expected_id = sample_ids[len(entries)] if len(entries) < len(sample_ids) else None
            if record.sample_id != expected_id:
                raise ValueError("progress records are not a canonical prefix")
            entries.append(
                ProgressEntry(
                    record=record,
                    native_metadata=dict(metadata),
                    fingerprint=fingerprint,
                )
            )
        return cls(
            path=target,
            identity_sha256=identity_sha256,
            ordered_sample_ids=sample_ids,
            entries=entries,
        )

    def append(
        self,
        record: ScalarEvaluation,
        *,
        native_metadata: dict[str, object],
        fingerprint: str,
    ) -> None:
        if len(self._entries) >= len(self.ordered_sample_ids):
            raise ValueError("progress ledger is already complete")
        expected_id = self.ordered_sample_ids[len(self._entries)]
        if record.sample_id != expected_id:
            raise ValueError(
                f"progress next sample must be {expected_id}, observed {record.sample_id}"
            )
        if not is_sha256(fingerprint):
            raise ValueError("progress fingerprint must be a SHA-256")
        entry = ProgressEntry(
            record=record,
            native_metadata=dict(native_metadata),
            fingerprint=fingerprint,
        )
        _write_fsynced(
            self.path,
            _canonical_line(
                {
                    "type": "sample",
                    "record": asdict(record),
                    "native_metadata": entry.native_metadata,
                    "fingerprint": entry.fingerprint,
                }
            ),
            exclusive=False,
        )
        self._entries.append(entry)

    def completed_entries(self) -> tuple[ProgressEntry, ...]:
        return tuple(self._entries)

    def completed_records(self) -> tuple[ScalarEvaluation, ...]:
        return tuple(entry.record for entry in self._entries)
