from __future__ import annotations

import json
import os
import shutil
import stat
import uuid
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

from droneai.integrity import sha256_file

ROOT_DIRECTORIES = (
    "결과해석",
    "STEERER",
    "PET",
    "APGCC",
    "MPCount",
    "DMCount",
    "CSRNet",
)
ROUND2_DISPLAY = {
    "steerer": "STEERER",
    "pet": "PET",
    "apgcc": "APGCC",
}
ROUND1_DISPLAYS = ("MPCount", "DMCount", "CSRNet")
DATASET_DISPLAY = {
    "ucf-qnrf-kaggle-apache": "UCF-QNRF_334",
    "jhu-crowd-plus-v2": "JHU-CROWD++_500",
    "up-count-v1": "UP-COUNT_166",
}
_FORBIDDEN_SUFFIXES = {
    ".pth",
    ".pt",
    ".ckpt",
    ".npy",
    ".npz",
    ".tar",
    ".zip",
    ".whl",
}
_ALLOWED_SUFFIXES = {
    ".png",
    ".json",
    ".jsonl",
    ".csv",
    ".md",
    ".txt",
    ".log",
    ".pdf",
    ".py",
    ".toml",
    ".yaml",
    ".yml",
    ".ini",
    ".cfg",
}
_FORBIDDEN_COMPONENTS = {
    "datasets",
    "checkpoints",
    ".venv",
    ".venvs",
    "attempts",
    "partial",
    "__pycache__",
    ".git",
}
_REPARSE_POINT = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)


@dataclass
class CompanyShareSpec:
    staging_root: Path
    zip_path: Path
    report_pdf: Path
    full_panels_root: Path
    round2_results_root: Path
    round2_expected: dict[str, dict[str, int]]
    round1_result_roots: dict[str, Path]
    model_code_files: dict[str, tuple[Path, ...]]
    interpretation_files: tuple[Path, ...] = ()
    extra_result_roots: dict[str, tuple[Path, ...]] | None = None

    def __post_init__(self) -> None:
        self.staging_root = self.staging_root.resolve()
        self.zip_path = self.zip_path.resolve()
        self.report_pdf = self.report_pdf.resolve()
        self.full_panels_root = self.full_panels_root.resolve()
        self.round2_results_root = self.round2_results_root.resolve()
        self.round1_result_roots = {
            key: value.resolve() for key, value in self.round1_result_roots.items()
        }
        self.model_code_files = {
            key: tuple(value.resolve() for value in values)
            for key, values in self.model_code_files.items()
        }
        self.interpretation_files = tuple(
            value.resolve() for value in self.interpretation_files
        )
        self.extra_result_roots = {
            key: tuple(value.resolve() for value in values)
            for key, values in (self.extra_result_roots or {}).items()
        }
        if set(self.round2_expected) != set(ROUND2_DISPLAY):
            raise ValueError("package requires STEERER, PET, and APGCC panel maps")
        if set(self.round1_result_roots) != set(ROUND1_DISPLAYS):
            raise ValueError("package requires the three declared Round 1 models")
        if set(self.model_code_files) != set(ROOT_DIRECTORIES[1:]):
            raise ValueError("package requires model-code files for all six models")
        for model, lanes in self.round2_expected.items():
            if not lanes:
                raise ValueError(f"Round 2 expected lanes are empty: {model}")
            for dataset_id, count in lanes.items():
                if dataset_id not in DATASET_DISPLAY:
                    raise ValueError(f"unsupported package dataset: {dataset_id}")
                if isinstance(count, bool) or count <= 0:
                    raise ValueError("Round 2 expected panel count must be positive")


@dataclass(frozen=True)
class PackageReport:
    scope: str
    staging_root: Path
    share_root: Path
    zip_path: Path
    sidecar_path: Path
    round2_panel_count: int
    file_count: int
    uncompressed_bytes: int
    zip_bytes: int
    zip_sha256: str
    content_manifest_sha256: str

    def as_dict(self) -> dict[str, object]:
        return {
            "status": self.scope,
            "staging_root": str(self.staging_root),
            "share_root": str(self.share_root),
            "zip_path": str(self.zip_path),
            "sidecar_path": str(self.sidecar_path),
            "round2_panel_count": self.round2_panel_count,
            "file_count": self.file_count,
            "uncompressed_bytes": self.uncompressed_bytes,
            "zip_bytes": self.zip_bytes,
            "zip_sha256": self.zip_sha256,
            "content_manifest_sha256": self.content_manifest_sha256,
        }


def _is_linklike(path: Path) -> bool:
    metadata = os.lstat(path)
    return bool(
        getattr(metadata, "st_file_attributes", 0) & _REPARSE_POINT
    ) or path.is_symlink()


def _validate_source(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"package source file is missing: {path}")
    if _is_linklike(path):
        raise ValueError(f"package source cannot be a link: {path}")
    lower_parts = {part.lower() for part in path.parts}
    forbidden_parts = lower_parts & _FORBIDDEN_COMPONENTS
    if forbidden_parts:
        raise ValueError(
            f"package source contains a forbidden path component: {path}"
        )
    suffix = path.suffix.lower()
    if suffix in _FORBIDDEN_SUFFIXES:
        raise ValueError(f"package source has a forbidden suffix: {path}")
    if suffix not in _ALLOWED_SUFFIXES and not path.name.upper().startswith(
        "LICENSE"
    ):
        raise ValueError(f"package source type is not allowlisted: {path}")


def _copy_file(source: Path, destination: Path) -> Path:
    _validate_source(source)
    if destination.exists():
        raise FileExistsError(f"package destination already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination, follow_symlinks=False)
    return destination


def _copy_tree(source: Path, destination: Path) -> list[Path]:
    if not source.is_dir():
        raise FileNotFoundError(f"package source directory is missing: {source}")
    if _is_linklike(source):
        raise ValueError(f"package source cannot be a link: {source}")
    copied: list[Path] = []
    for path in sorted(source.rglob("*")):
        relative = path.relative_to(source)
        if any(part.lower() in _FORBIDDEN_COMPONENTS for part in relative.parts):
            raise ValueError(
                f"package source contains a forbidden path component: {path}"
            )
        if path.is_dir():
            if _is_linklike(path):
                raise ValueError(f"package source cannot be a link: {path}")
            continue
        copied.append(_copy_file(path, destination / relative))
    if not copied:
        raise ValueError(f"package source directory contains no files: {source}")
    return copied


def _write_text_new(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(text)
    return path


def _write_json_new(path: Path, payload: object) -> Path:
    return _write_text_new(
        path,
        json.dumps(
            payload,
            indent=2,
            ensure_ascii=False,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
    )


def _model_code_readme(display: str, files: Sequence[Path]) -> str:
    listed = "\n".join(f"- `{path.name}`" for path in files)
    return (
        f"# {display} 모델 코드\n\n"
        "이 폴더에는 DroneAI 통합·재현에 필요한 코드와 설정만 포함됩니다.\n"
        "데이터셋, pretrained checkpoint, 파생 weight, 가상환경은 포함하지 "
        "않습니다.\n\n"
        "## 포함 파일\n\n"
        f"{listed}\n"
    )


def _top_readme(panel_count: int) -> str:
    return f"""# DroneAI 회사 공유용 연구 결과

- 범위: `PASS_RESEARCH_ONLY`
- Round 2 전체 출력 PNG: `{panel_count:,}`장
- 학습: 하지 않음
- 파인튜닝: 하지 않음
- 캘리브레이션: 하지 않음

`predictions.csv`와 `metrics.json`이 수치 결과의 기준입니다. PNG는 동일
checkpoint를 다시 실행해 기존 count와 `1e-6` 이내에서 일치한 연구용
시각화입니다. 이 패키지는 상용 사용, weight 재배포, 또는 제품 배포
승인을 의미하지 않습니다.
"""


def _rights_readme() -> str:
    return """# 라이선스 및 사용 범위

전체 패키지 판정은 `PASS_RESEARCH_ONLY`입니다.

- STEERER: checkpoint 권리 추가 확인 필요
- PET: 공식 코드/checkpoint의 academic-use 제한
- APGCC: 코드 MIT, 공개 checkpoint의 별도 상용 조건 미확인
- JHU-CROWD++ 및 UP-COUNT: 연구 비교용
- UCF-QNRF 데이터 표기와 각 pretrained checkpoint 권리는 별개

원본 데이터셋, checkpoint, derived weight는 포함하지 않았습니다.
"""


def _write_content_manifest(
    share_root: Path,
    *,
    excluded: set[Path],
    destination: Path,
) -> tuple[Path, int, int]:
    files = tuple(
        path
        for path in sorted(share_root.rglob("*"))
        if path.is_file() and path.resolve() not in excluded
    )
    lines = [
        f"{sha256_file(path)}  {path.relative_to(share_root).as_posix()}"
        for path in files
    ]
    _write_text_new(destination, "\n".join(lines) + "\n")
    return destination, len(files), sum(path.stat().st_size for path in files)


def _write_deterministic_zip(share_root: Path, destination: Path) -> Path:
    temporary = destination.with_name(
        f".{destination.name}.{uuid.uuid4().hex}.tmp"
    )
    with zipfile.ZipFile(
        temporary,
        mode="x",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=6,
    ) as archive:
        for source in sorted(path for path in share_root.rglob("*") if path.is_file()):
            if _is_linklike(source):
                raise ValueError(f"ZIP source cannot be a link: {source}")
            member = f"{share_root.name}/{source.relative_to(share_root).as_posix()}"
            if member.startswith("/") or ".." in Path(member).parts:
                raise ValueError(f"unsafe ZIP member: {member}")
            info = zipfile.ZipInfo(member, date_time=(2026, 7, 23, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            archive.writestr(info, source.read_bytes(), compresslevel=6)
    if destination.exists():
        raise FileExistsError(f"company-share ZIP already exists: {destination}")
    temporary.replace(destination)
    return destination


def build_company_share(spec: CompanyShareSpec) -> PackageReport:
    if spec.staging_root.exists():
        raise FileExistsError(
            f"company-share staging root already exists: {spec.staging_root}"
        )
    if spec.zip_path.exists():
        raise FileExistsError(f"company-share ZIP already exists: {spec.zip_path}")
    sidecar = Path(f"{spec.zip_path}.sha256")
    if sidecar.exists():
        raise FileExistsError(f"company-share ZIP sidecar already exists: {sidecar}")
    _validate_source(spec.report_pdf)

    temporary_root = spec.staging_root.with_name(
        f".{spec.staging_root.name}.{uuid.uuid4().hex}.tmp"
    )
    temporary_root.mkdir(parents=True)
    share = temporary_root / "Drone AI"
    share.mkdir()
    roots = {name: share / name for name in ROOT_DIRECTORIES}
    for path in roots.values():
        path.mkdir()

    round2_panels = 0
    for model_id, lanes in spec.round2_expected.items():
        display = ROUND2_DISPLAY[model_id]
        for dataset_id, expected in lanes.items():
            dataset_dir = roots[display] / DATASET_DISPLAY[dataset_id]
            panel_source = (
                spec.full_panels_root / model_id / dataset_id / "panels"
            )
            panels = tuple(sorted(panel_source.glob("*.png")))
            if len(panels) != expected:
                raise ValueError(
                    f"Round 2 panel count differs: {model_id}/{dataset_id}; "
                    f"expected={expected}, actual={len(panels)}"
                )
            for panel in panels:
                _copy_file(panel, dataset_dir / "결과이미지_전체" / panel.name)
            result_lane = spec.round2_results_root / model_id / dataset_id
            _copy_tree(result_lane, dataset_dir / "검증자료")
            export_summary = (
                spec.full_panels_root / model_id / dataset_id / "export-summary.json"
            )
            _copy_file(export_summary, dataset_dir / "export-summary.json")
            _write_text_new(
                dataset_dir / "README.md",
                (
                    f"# {display} / {dataset_id}\n\n"
                    f"- 전체 출력 PNG: `{expected}`장\n"
                    "- 수치 기준: `검증자료/predictions.csv`\n"
                    "- 범위: `PASS_RESEARCH_ONLY`\n"
                ),
            )
            round2_panels += expected

    for display, source in spec.round1_result_roots.items():
        destination = roots[display] / "UCF-QNRF_Validation_36"
        _copy_tree(source, destination / "검증자료")
        _write_text_new(
            destination / "00_검증범위.md",
            (
                f"# {display} Round 1 검증 범위\n\n"
                "UCF-QNRF validation 36장 compatibility smoke 결과입니다. "
                "Round 2의 모델당 1,000장 결과와 평가 규모가 다르며 공식 "
                "순위 또는 제품 승인을 의미하지 않습니다.\n"
            ),
        )

    for display, sources in spec.model_code_files.items():
        code_root = roots[display] / "모델코드"
        _write_text_new(
            code_root / "README.md",
            _model_code_readme(display, sources),
        )
        for source in sources:
            _copy_file(source, code_root / source.name)

    interpretation = roots["결과해석"]
    _copy_file(
        spec.report_pdf,
        interpretation / "Round1_공통장면_모델비교_공유용.pdf",
    )
    for index, source in enumerate(spec.interpretation_files, start=1):
        _copy_file(
            source,
            interpretation / "비교자료" / f"{index:02d}_{source.name}",
        )
    for display, sources in (spec.extra_result_roots or {}).items():
        if display not in roots or display == "결과해석":
            raise ValueError(f"extra result model is invalid: {display}")
        for source in sources:
            _copy_tree(source, roots[display] / "추가결과" / source.name)

    _write_text_new(
        interpretation / "00_먼저읽기.md",
        _top_readme(round2_panels),
    )
    _write_text_new(
        interpretation / "라이선스_및_사용범위.md",
        _rights_readme(),
    )
    validation = interpretation / "패키지_검증정보"
    content_manifest = validation / "files.sha256"
    package_manifest = validation / "package-manifest.json"
    manifest, content_count, content_bytes = _write_content_manifest(
        share,
        excluded={content_manifest.resolve(), package_manifest.resolve()},
        destination=content_manifest,
    )
    content_manifest_sha256 = sha256_file(manifest)
    package_payload = {
        "schema_version": 1,
        "scope": "PASS_RESEARCH_ONLY",
        "round2_panel_count": round2_panels,
        "round1_models": ["dm-count", "mpcount", "csrnet"],
        "excluded_assets": [
            "datasets",
            "checkpoints",
            "weights",
            "raw_density",
            "virtual_environments",
        ],
        "content_file_count": content_count,
        "content_bytes": content_bytes,
        "content_manifest_sha256": content_manifest_sha256,
    }
    _write_json_new(package_manifest, package_payload)

    if spec.staging_root.exists():
        raise FileExistsError(
            f"company-share staging root appeared during build: {spec.staging_root}"
        )
    temporary_root.replace(spec.staging_root)
    final_share = spec.staging_root / "Drone AI"
    spec.zip_path.parent.mkdir(parents=True, exist_ok=True)
    _write_deterministic_zip(final_share, spec.zip_path)
    zip_sha256 = sha256_file(spec.zip_path)
    _write_text_new(sidecar, f"{zip_sha256}  {spec.zip_path.name}\n")
    all_files = tuple(path for path in final_share.rglob("*") if path.is_file())
    return PackageReport(
        scope="PASS_RESEARCH_ONLY",
        staging_root=spec.staging_root,
        share_root=final_share,
        zip_path=spec.zip_path,
        sidecar_path=sidecar,
        round2_panel_count=round2_panels,
        file_count=len(all_files),
        uncompressed_bytes=sum(path.stat().st_size for path in all_files),
        zip_bytes=spec.zip_path.stat().st_size,
        zip_sha256=zip_sha256,
        content_manifest_sha256=content_manifest_sha256,
    )


def _resolve(base: Path, value: object) -> Path:
    path = Path(str(value))
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def load_company_share_spec(path: str | Path) -> CompanyShareSpec:
    source = Path(path).resolve()
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise ValueError("company-share spec requires schema_version=1")
    required = {
        "staging_root",
        "zip_path",
        "report_pdf",
        "full_panels_root",
        "round2_results_root",
        "round2_expected",
        "round1_result_roots",
        "model_code_files",
    }
    if not required <= set(payload):
        raise ValueError("company-share spec fields are incomplete")
    base = source.parent
    raw_expected = payload["round2_expected"]
    raw_round1 = payload["round1_result_roots"]
    raw_code = payload["model_code_files"]
    raw_interpretation = payload.get("interpretation_files", [])
    raw_extras = payload.get("extra_result_roots", {})
    if (
        not isinstance(raw_expected, dict)
        or not isinstance(raw_round1, dict)
        or not isinstance(raw_code, dict)
        or not isinstance(raw_interpretation, list)
        or not isinstance(raw_extras, dict)
    ):
        raise ValueError("company-share spec collection fields are invalid")
    return CompanyShareSpec(
        staging_root=_resolve(base, payload["staging_root"]),
        zip_path=_resolve(base, payload["zip_path"]),
        report_pdf=_resolve(base, payload["report_pdf"]),
        full_panels_root=_resolve(base, payload["full_panels_root"]),
        round2_results_root=_resolve(base, payload["round2_results_root"]),
        round2_expected={
            str(model): {
                str(dataset): int(count)
                for dataset, count in lanes.items()
            }
            for model, lanes in raw_expected.items()
            if isinstance(lanes, dict)
        },
        round1_result_roots={
            str(model): _resolve(base, value)
            for model, value in raw_round1.items()
        },
        model_code_files={
            str(model): tuple(_resolve(base, value) for value in values)
            for model, values in raw_code.items()
            if isinstance(values, list)
        },
        interpretation_files=tuple(
            _resolve(base, value) for value in raw_interpretation
        ),
        extra_result_roots={
            str(model): tuple(_resolve(base, value) for value in values)
            for model, values in raw_extras.items()
            if isinstance(values, list)
        },
    )
