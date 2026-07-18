from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


MODEL_IDS = ("dm-count", "steerer", "pet", "mpcount", "apgcc", "csrnet")
_FORBIDDEN_CLAIMS = ("PRODUCTION_APPROVED", "배포 승인 완료")


@dataclass(frozen=True)
class ModelContent:
    model_id: str
    paper: str
    family: str
    backbone: str
    structure: tuple[str, ...]
    input: str
    native_output: str
    count_method: str
    strengths: tuple[str, ...]
    limitations: tuple[str, ...]
    cctv_interpretation: str
    reference_source: str


@dataclass(frozen=True)
class ReportContent:
    title: str
    subtitle: str
    models: tuple[ModelContent, ...]
    conclusion: tuple[str, ...]
    roadmap: tuple[str, ...]


def _text(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"report content field must be non-empty: {field}")
    return value.strip()


def _text_list(value: object, *, field: str, minimum: int) -> tuple[str, ...]:
    if not isinstance(value, list) or len(value) < minimum:
        raise ValueError(f"report content list is incomplete: {field}")
    return tuple(_text(item, field=field) for item in value)


def _reject_forbidden_claims(payload: object) -> None:
    serialized = json.dumps(payload, ensure_ascii=False)
    if any(claim in serialized for claim in _FORBIDDEN_CLAIMS):
        raise ValueError("report content cannot claim production approval")


def load_report_content(path: str | Path) -> ReportContent:
    source = Path(path)
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid report content: {source}") from exc
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise ValueError("report content schema_version must be 1")
    _reject_forbidden_claims(payload)
    raw_models = payload.get("models")
    if not isinstance(raw_models, list):
        raise ValueError("report content models must be a list")
    observed_ids = tuple(
        item.get("model_id") if isinstance(item, dict) else None for item in raw_models
    )
    if observed_ids != MODEL_IDS:
        raise ValueError("report content must contain the frozen six model IDs")

    models: list[ModelContent] = []
    for index, raw in enumerate(raw_models):
        if not isinstance(raw, dict):
            raise ValueError(f"report model record must be an object: {index}")
        prefix = f"models[{index}]"
        models.append(
            ModelContent(
                model_id=_text(raw.get("model_id"), field=f"{prefix}.model_id"),
                paper=_text(raw.get("paper"), field=f"{prefix}.paper"),
                family=_text(raw.get("family"), field=f"{prefix}.family"),
                backbone=_text(raw.get("backbone"), field=f"{prefix}.backbone"),
                structure=_text_list(
                    raw.get("structure"), field=f"{prefix}.structure", minimum=2
                ),
                input=_text(raw.get("input"), field=f"{prefix}.input"),
                native_output=_text(
                    raw.get("native_output"), field=f"{prefix}.native_output"
                ),
                count_method=_text(
                    raw.get("count_method"), field=f"{prefix}.count_method"
                ),
                strengths=_text_list(
                    raw.get("strengths"), field=f"{prefix}.strengths", minimum=2
                ),
                limitations=_text_list(
                    raw.get("limitations"), field=f"{prefix}.limitations", minimum=2
                ),
                cctv_interpretation=_text(
                    raw.get("cctv_interpretation"),
                    field=f"{prefix}.cctv_interpretation",
                ),
                reference_source=_text(
                    raw.get("reference_source"),
                    field=f"{prefix}.reference_source",
                ),
            )
        )

    return ReportContent(
        title=_text(payload.get("title"), field="title"),
        subtitle=_text(payload.get("subtitle"), field="subtitle"),
        models=tuple(models),
        conclusion=_text_list(
            payload.get("conclusion"), field="conclusion", minimum=3
        ),
        roadmap=_text_list(payload.get("roadmap"), field="roadmap", minimum=5),
    )

