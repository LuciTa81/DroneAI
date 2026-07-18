from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Callable

from pypdf import PdfReader
from reportlab.lib.colors import HexColor, white
from reportlab.lib.pagesizes import A4, landscape
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas as canvas_module
from reportlab.lib.utils import ImageReader

from droneai.integrity import sha256_file
from droneai.pdf_report_content import ModelContent, ReportContent, load_report_content


PAGE_SIZE = landscape(A4)
PAGE_COUNT = 15
WIDTH, HEIGHT = PAGE_SIZE
MODEL_IDS = ("dm-count", "steerer", "pet", "mpcount", "apgcc", "csrnet")
DISPLAY_NAMES = {
    "dm-count": "DM-Count",
    "steerer": "STEERER",
    "pet": "PET",
    "mpcount": "MPCount",
    "apgcc": "APGCC",
    "csrnet": "CSRNet",
}

NAVY = HexColor("#102A43")
BLUE = HexColor("#1F5D8F")
CYAN = HexColor("#18A6B8")
AMBER = HexColor("#E5A63B")
RED = HexColor("#C94C4C")
GREEN = HexColor("#2D8C6F")
INK = HexColor("#1F2933")
MUTED = HexColor("#627D98")
LINE = HexColor("#D8E2EA")
PAPER = HexColor("#F7F9FB")
PANEL = white

_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def register_korean_fonts() -> tuple[str, str]:
    regular_name = "DroneAI-Regular"
    bold_name = "DroneAI-Bold"
    registered = set(pdfmetrics.getRegisteredFontNames())
    if regular_name in registered and bold_name in registered:
        return regular_name, bold_name
    candidates = (
        (
            Path("C:/Windows/Fonts/malgun.ttf"),
            Path("C:/Windows/Fonts/malgunbd.ttf"),
        ),
        (
            Path("C:/Windows/Fonts/NotoSansKR-Regular.ttf"),
            Path("C:/Windows/Fonts/NotoSansKR-Bold.ttf"),
        ),
    )
    for regular, bold in candidates:
        if regular.is_file() and bold.is_file():
            pdfmetrics.registerFont(TTFont(regular_name, str(regular)))
            pdfmetrics.registerFont(TTFont(bold_name, str(bold)))
            return regular_name, bold_name
    raise FileNotFoundError("Korean regular/bold font pair is unavailable")


def _read_object(path: Path, *, label: str) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid {label}: {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must be a JSON object")
    return payload


def _wrap(text: str, width: float, font: str, size: float) -> list[str]:
    lines: list[str] = []
    current = ""
    for word in text.replace("\n", " \n ").split(" "):
        if word == "\n":
            if current:
                lines.append(current)
                current = ""
            continue
        candidate = word if not current else f"{current} {word}"
        if pdfmetrics.stringWidth(candidate, font, size) <= width:
            current = candidate
            continue
        if current:
            lines.append(current)
            current = ""
        fragment = ""
        for character in word:
            candidate = fragment + character
            if fragment and pdfmetrics.stringWidth(candidate, font, size) > width:
                lines.append(fragment)
                fragment = character
            else:
                fragment = candidate
        current = fragment
    if current:
        lines.append(current)
    return lines or [""]


def draw_wrapped(
    canvas: canvas_module.Canvas,
    text: str,
    x: float,
    y: float,
    width: float,
    *,
    font: str,
    size: float,
    leading: float,
    color=INK,
    max_lines: int | None = None,
) -> float:
    lines = _wrap(text, width, font, size)
    if max_lines is not None and len(lines) > max_lines:
        lines = lines[:max_lines]
        final = lines[-1]
        while final and pdfmetrics.stringWidth(final + "...", font, size) > width:
            final = final[:-1]
        lines[-1] = final + "..."
    canvas.setFont(font, size)
    canvas.setFillColor(color)
    for line in lines:
        canvas.drawString(x, y, line)
        y -= leading
    return y


def _background(canvas: canvas_module.Canvas) -> None:
    canvas.setFillColor(PAPER)
    canvas.rect(0, 0, WIDTH, HEIGHT, fill=1, stroke=0)


def draw_footer(
    canvas: canvas_module.Canvas,
    page_number: int,
    *,
    regular_font: str,
) -> None:
    canvas.setStrokeColor(LINE)
    canvas.line(34, 28, WIDTH - 34, 28)
    canvas.setFont(regular_font, 7.5)
    canvas.setFillColor(MUTED)
    canvas.drawString(34, 16, "DroneAI Round 1 | UCF-QNRF 36-image validation")
    canvas.drawRightString(WIDTH - 34, 16, f"{page_number} / {PAGE_COUNT}")


def _title(
    canvas: canvas_module.Canvas,
    title: str,
    subtitle: str,
    *,
    bold_font: str,
    regular_font: str,
) -> None:
    canvas.setFillColor(NAVY)
    canvas.setFont(bold_font, 22)
    canvas.drawString(34, HEIGHT - 48, title)
    canvas.setFillColor(MUTED)
    canvas.setFont(regular_font, 9)
    canvas.drawString(35, HEIGHT - 65, subtitle)
    canvas.setStrokeColor(CYAN)
    canvas.setLineWidth(2.5)
    canvas.line(34, HEIGHT - 76, WIDTH - 34, HEIGHT - 76)


def _card(
    canvas: canvas_module.Canvas,
    x: float,
    y: float,
    width: float,
    height: float,
    *,
    fill=PANEL,
    stroke=LINE,
    radius: float = 7,
) -> None:
    canvas.setFillColor(fill)
    canvas.setStrokeColor(stroke)
    canvas.setLineWidth(0.7)
    canvas.roundRect(x, y, width, height, radius, fill=1, stroke=1)


def draw_metric_card(
    canvas: canvas_module.Canvas,
    title: str,
    value: str,
    x: float,
    y: float,
    width: float,
    height: float,
    *,
    regular_font: str,
    bold_font: str,
    accent=CYAN,
) -> None:
    _card(canvas, x, y, width, height)
    canvas.setFillColor(accent)
    canvas.rect(x, y, 4, height, fill=1, stroke=0)
    canvas.setFont(regular_font, 7.5)
    canvas.setFillColor(MUTED)
    canvas.drawString(x + 12, y + height - 16, title)
    canvas.setFont(bold_font, 14)
    canvas.setFillColor(NAVY)
    canvas.drawString(x + 12, y + 12, value)


def draw_panel(
    canvas: canvas_module.Canvas,
    image_path: Path,
    caption: str,
    x: float,
    y: float,
    width: float,
    height: float,
    *,
    regular_font: str,
) -> None:
    _card(canvas, x, y, width, height, fill=white)
    reader = ImageReader(str(image_path))
    image_width, image_height = reader.getSize()
    available_height = height - 30
    scale = min((width - 12) / image_width, available_height / image_height)
    drawn_width = image_width * scale
    drawn_height = image_height * scale
    image_x = x + (width - drawn_width) / 2
    image_y = y + 26 + (available_height - drawn_height) / 2
    canvas.drawImage(
        reader,
        image_x,
        image_y,
        width=drawn_width,
        height=drawn_height,
        preserveAspectRatio=True,
        mask="auto",
    )
    draw_wrapped(
        canvas,
        caption,
        x + 8,
        y + 17,
        width - 16,
        font=regular_font,
        size=6.5,
        leading=7.5,
        color=MUTED,
        max_lines=2,
    )


def _fmt(value: object, digits: int = 2) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, (int, float)):
        number = float(value)
        if not math.isfinite(number):
            return "N/A"
        return f"{number:.{digits}f}"
    return str(value)


def _panel_caption(panel: dict[str, object]) -> str:
    spatial_name = str(panel.get("spatial_metric_name") or "N/A")
    spatial_value = _fmt(panel.get("spatial_metric_value"), 2)
    return (
        f"{DISPLAY_NAMES[str(panel['model_id'])]} | sample={panel['sample_id']} | "
        f"band={panel['density_band']} | GT={_fmt(panel['ground_truth_count'], 0)} | "
        f"Pred={_fmt(panel['predicted_count'], 1)} | {spatial_name}={spatial_value}"
    )


def _validate_inputs(
    comparison_path: Path,
    assets_manifest_path: Path,
) -> tuple[dict[str, object], dict[str, object]]:
    comparison = _read_object(comparison_path, label="comparison")
    dataset = comparison.get("dataset")
    shortlist = comparison.get("shortlist")
    models = comparison.get("models")
    if (
        comparison.get("ranking_eligible") is not False
        or not isinstance(dataset, dict)
        or dataset.get("dataset_id") != "ucf-qnrf-kaggle-apache"
        or dataset.get("split_role") != "validation"
        or dataset.get("expected_samples") != 36
        or not isinstance(shortlist, dict)
        or shortlist.get("label") != "PROVISIONAL_TECHNICAL_SHORTLIST"
        or shortlist.get("model_ids") != ["steerer", "dm-count", "pet"]
        or not isinstance(models, list)
        or tuple(row.get("model_id") for row in models) != MODEL_IDS
    ):
        raise ValueError("comparison does not match the frozen Round 1 report")

    assets = _read_object(assets_manifest_path, label="assets manifest")
    if (
        assets.get("schema_version") != 1
        or assets.get("comparison_sha256") != sha256_file(comparison_path)
    ):
        raise ValueError("assets manifest comparison hash mismatch")
    raw_assets = assets.get("assets")
    if not isinstance(raw_assets, list) or not raw_assets:
        raise ValueError("assets manifest contains no images")
    base = assets_manifest_path.parent
    verified_paths: set[str] = set()
    for item in raw_assets:
        if not isinstance(item, dict):
            raise ValueError("asset reference must be an object")
        raw_path = item.get("packaged_path")
        expected = item.get("packaged_sha256")
        if not isinstance(raw_path, str) or not isinstance(expected, str):
            raise ValueError("asset reference path/hash is missing")
        target = (base / raw_path).resolve()
        if not target.is_relative_to(base.resolve()) or not target.is_file():
            raise ValueError(f"asset is missing: {raw_path}")
        if sha256_file(target) != expected:
            raise ValueError(f"asset hash mismatch: {raw_path}")
        verified_paths.add(raw_path)

    raw_model_assets = assets.get("models")
    comparisons = assets.get("comparisons")
    if not isinstance(raw_model_assets, dict) or tuple(raw_model_assets) != MODEL_IDS:
        raise ValueError("assets manifest model order is invalid")
    for model_id in MODEL_IDS:
        model = raw_model_assets[model_id]
        if not isinstance(model, dict):
            raise ValueError(f"model assets are invalid: {model_id}")
        for role in ("lower_error", "limitation"):
            panel = model.get(role)
            if not isinstance(panel, dict) or panel.get("packaged_path") not in verified_paths:
                raise ValueError(f"model panel is unavailable: {model_id}.{role}")
    if not isinstance(comparisons, dict):
        raise ValueError("comparison panel groups are invalid")
    expected_groups = {
        "density": ("dm-count", "steerer", "mpcount", "csrnet"),
        "points": ("steerer", "pet", "apgcc"),
    }
    for group_name, expected_models in expected_groups.items():
        group = comparisons.get(group_name)
        if not isinstance(group, dict) or tuple(group) != expected_models:
            raise ValueError(f"comparison panel group is invalid: {group_name}")
        sample_ids = set()
        for model_id, panel in group.items():
            if not isinstance(panel, dict) or panel.get("packaged_path") not in verified_paths:
                raise ValueError(f"comparison panel is unavailable: {group_name}.{model_id}")
            sample_ids.add(panel.get("sample_id"))
        if len(sample_ids) != 1 or comparisons.get(f"{group_name}_sample_id") not in sample_ids:
            raise ValueError(f"comparison sample identity differs: {group_name}")
    return comparison, assets


def _rights_summary(row: dict[str, object]) -> str:
    return f"{row['rights_scope']} / {row['deployment_label']}"


def _draw_bullets(
    canvas: canvas_module.Canvas,
    items: tuple[str, ...] | list[str],
    x: float,
    y: float,
    width: float,
    *,
    regular_font: str,
    size: float = 8.4,
    leading: float = 11,
    max_lines: int = 3,
) -> float:
    for item in items:
        canvas.setFillColor(CYAN)
        canvas.circle(x + 3, y + 3, 2, fill=1, stroke=0)
        y = draw_wrapped(
            canvas,
            item,
            x + 12,
            y,
            width - 12,
            font=regular_font,
            size=size,
            leading=leading,
            max_lines=max_lines,
        ) - 3
    return y


def _page_cover(c, ctx) -> None:
    regular, bold = ctx["fonts"]
    c.setFillColor(NAVY)
    c.rect(0, 0, WIDTH, HEIGHT, fill=1, stroke=0)
    c.setFillColor(CYAN)
    c.rect(0, 0, 16, HEIGHT, fill=1, stroke=0)
    c.setFont(bold, 15)
    c.setFillColor(CYAN)
    c.drawString(58, HEIGHT - 82, "DroneAI / PRIVATE REVIEW DRAFT")
    c.setFont(bold, 31)
    c.setFillColor(white)
    y = draw_wrapped(
        c,
        ctx["content"].title,
        58,
        HEIGHT - 150,
        660,
        font=bold,
        size=31,
        leading=39,
        color=white,
        max_lines=2,
    )
    c.setFont(regular, 14)
    c.setFillColor(HexColor("#B8D5E5"))
    c.drawString(60, y - 14, ctx["content"].subtitle)
    _card(c, 58, 108, 720, 116, fill=HexColor("#173F5F"), stroke=HexColor("#2A668D"))
    c.setFillColor(AMBER)
    c.setFont(bold, 12)
    c.drawString(78, 196, "ONE-LINE CONCLUSION")
    draw_wrapped(
        c,
        "STEERER와 DM-Count를 고정형 CCTV 제품 검증 후보로 우선하며, PET은 기술 비교 3위이지만 연구 전용으로 유지한다.",
        78,
        170,
        670,
        font=bold,
        size=15,
        leading=22,
        color=white,
        max_lines=3,
    )
    c.setFillColor(HexColor("#B8D5E5"))
    c.setFont(regular, 9)
    c.drawString(60, 67, f"Report date: {ctx['report_date']}  |  36-image validation  |  No fine-tuning")


def _page_executive(c, ctx) -> None:
    regular, bold = ctx["fonts"]
    _background(c)
    _title(c, "Executive conclusion", "기술 우선순위와 제품 판단은 서로 다른 축이다.", bold_font=bold, regular_font=regular)
    rows = {row["model_id"]: row for row in ctx["comparison"]["models"]}
    shortlist = ("steerer", "dm-count", "pet")
    colors = (CYAN, BLUE, AMBER)
    for index, (model_id, accent) in enumerate(zip(shortlist, colors, strict=True)):
        x = 34 + index * 263
        _card(c, x, 318, 244, 168)
        c.setFillColor(accent)
        c.rect(x, 454, 244, 32, fill=1, stroke=0)
        c.setFont(bold, 13)
        c.setFillColor(white)
        c.drawString(x + 12, 464, f"#{index + 1}  {DISPLAY_NAMES[model_id]}")
        row = rows[model_id]
        c.setFont(bold, 27)
        c.setFillColor(NAVY)
        c.drawString(x + 14, 410, _fmt(row["technical_score_display"], 2))
        c.setFont(regular, 8)
        c.setFillColor(MUTED)
        c.drawString(x + 15, 394, "operational score / 100")
        draw_wrapped(c, _rights_summary(row), x + 14, 370, 214, font=regular, size=8, leading=10, max_lines=2)
        recommendation = {
            "steerer": "기술 1위. Density와 point를 함께 제공. Weight 실사 후 CCTV calibration 우선.",
            "dm-count": "Density baseline과 zone 집계에 적합. 파생 weight와 deployment 확인 필요.",
            "pet": "기술 비교군으로 유효하지만 code와 checkpoint가 연구 전용.",
        }[model_id]
        draw_wrapped(c, recommendation, x + 14, 347, 214, font=regular, size=8, leading=10, max_lines=3)
    _card(c, 34, 92, 507, 192)
    c.setFont(bold, 14)
    c.setFillColor(NAVY)
    c.drawString(52, 256, "Decision")
    _draw_bullets(c, list(ctx["content"].conclusion), 52, 232, 470, regular_font=regular, size=8.7, leading=11, max_lines=2)
    _card(c, 558, 92, 250, 192, fill=HexColor("#FFF7E8"), stroke=HexColor("#F1CF8B"))
    c.setFillColor(RED)
    c.setFont(bold, 13)
    c.drawString(576, 256, "Interpretation guardrail")
    _draw_bullets(
        c,
        [
            "PROVISIONAL_TECHNICAL_SHORTLIST",
            "공식 모델 순위가 아님",
            "36장 cross-domain validation",
            "기술 점수는 정확도 백분율이 아님",
            "상용 배포 승인이 아님",
        ],
        576,
        226,
        214,
        regular_font=regular,
        size=8.6,
        leading=11,
        max_lines=2,
    )


def _page_problem(c, ctx) -> None:
    regular, bold = ctx["fonts"]
    _background(c)
    _title(c, "고정형 CCTV 관제 문제 정의", "Count 하나보다 zone density, 추세, overlap ownership이 중요하다.", bold_font=bold, regular_font=regular)
    _card(c, 34, 92, 348, 404)
    c.setFont(bold, 14)
    c.setFillColor(NAVY)
    c.drawString(52, 466, "운영 목표")
    _draw_bullets(
        c,
        [
            "스테이지와 통로를 실제 안전 구역 ROI로 나누고 셀별 밀집도를 표시",
            "정확한 절대 인원보다 위험 구역, 상승 추세, 관제사의 확인 우선순위를 제공",
            "고정 카메라가 만드는 안정적인 시계열과 반복 가능한 calibration 활용",
            "복수 CCTV에서는 겹친 영역을 한 카메라의 ownership으로 귀속해 중복 count 방지",
        ],
        52,
        435,
        310,
        regular_font=regular,
        size=10,
        leading=14,
        max_lines=3,
    )
    c.setFont(bold, 12)
    c.setFillColor(RED)
    c.drawString(52, 234, "현재 기술 질문")
    _draw_bullets(
        c,
        [
            "Density map이 실제 zone 면적과 일관되게 연결되는가?",
            "Point output이 고밀도와 가림에서도 무너지지 않는가?",
            "카메라 경계와 overlap에서 count를 어떻게 합산할 것인가?",
        ],
        52,
        208,
        310,
        regular_font=regular,
        size=9,
        leading=12,
        max_lines=2,
    )
    _card(c, 400, 92, 408, 404)
    c.setFont(bold, 14)
    c.setFillColor(NAVY)
    c.drawString(420, 466, "CCTV zone and merge concept")
    stage_x, stage_y, stage_w, stage_h = 436, 168, 336, 234
    c.setFillColor(HexColor("#E8EEF3"))
    c.setStrokeColor(NAVY)
    c.roundRect(stage_x, stage_y, stage_w, stage_h, 8, fill=1, stroke=1)
    zone_colors = (HexColor("#BFE7DC"), HexColor("#F9DFA5"), HexColor("#F3B6B6"), HexColor("#C5D8EA"))
    labels = ("ZONE A", "OVERLAP", "ZONE B", "AISLE")
    zones = (
        (stage_x + 14, stage_y + 72, 130, 138),
        (stage_x + 132, stage_y + 72, 72, 138),
        (stage_x + 192, stage_y + 72, 130, 138),
        (stage_x + 14, stage_y + 18, 308, 42),
    )
    for color, label, (x, y, w, h) in zip(zone_colors, labels, zones, strict=True):
        c.setFillColor(color)
        c.setStrokeColor(white)
        c.rect(x, y, w, h, fill=1, stroke=1)
        c.setFillColor(NAVY)
        c.setFont(bold, 8)
        c.drawCentredString(x + w / 2, y + h / 2, label)
    c.setFillColor(BLUE)
    c.circle(stage_x + 75, stage_y + 224, 13, fill=1, stroke=0)
    c.circle(stage_x + 260, stage_y + 224, 13, fill=1, stroke=0)
    c.setFillColor(white)
    c.setFont(bold, 7)
    c.drawCentredString(stage_x + 75, stage_y + 221, "CAM 1")
    c.drawCentredString(stage_x + 260, stage_y + 221, "CAM 2")
    draw_wrapped(c, "Overlap은 사전 calibration으로 ownership을 고정하고, 인접 카메라 간 합산 시 한쪽만 count한다.", 430, 138, 350, font=regular, size=9, leading=12, max_lines=3)


def _page_evaluation(c, ctx) -> None:
    regular, bold = ctx["fonts"]
    _background(c)
    _title(c, "검증 설계와 하네스", "동일 validation 표본, 기능별 metric, 권리 독립 축을 고정했다.", bold_font=bold, regular_font=regular)
    cards = (
        ("Dataset", "UCF-QNRF Kaggle Apache"),
        ("Split", "validation / 36 images"),
        ("Density bands", "low 12 / medium 12 / high 12"),
        ("Inference", "batch 1 / RTX 5090"),
    )
    for index, (label, value) in enumerate(cards):
        draw_metric_card(c, label, value, 34 + index * 194, 430, 180, 58, regular_font=regular, bold_font=bold, accent=(CYAN, BLUE, AMBER, GREEN)[index])
    _card(c, 34, 102, 362, 302)
    c.setFont(bold, 14)
    c.setFillColor(NAVY)
    c.drawString(52, 374, "100-point operational score")
    weights = (
        ("Zone and spatial", 35, CYAN),
        ("Count stability", 25, BLUE),
        ("Runtime and resources", 15, GREEN),
        ("Operator integration", 15, AMBER),
        ("Evidence quality", 10, MUTED),
    )
    y = 338
    for label, value, color in weights:
        c.setFillColor(LINE)
        c.roundRect(54, y, 260, 15, 7, fill=1, stroke=0)
        c.setFillColor(color)
        c.roundRect(54, y, 260 * value / 35, 15, 7, fill=1, stroke=0)
        c.setFont(regular, 8)
        c.setFillColor(INK)
        c.drawString(54, y + 21, label)
        c.setFont(bold, 9)
        c.drawRightString(370, y + 3, f"{value} pt")
        y -= 49
    _card(c, 414, 102, 394, 302)
    c.setFont(bold, 14)
    c.setFillColor(NAVY)
    c.drawString(432, 374, "입력과 출력별 평가")
    rows = (
        ("공통", "RGB image", "MAE, RMSE, MAPE, bias, latency, FPS, VRAM"),
        ("Density", "density map", "sum count, GAME L1, zone MAE, PSNR, SSIM"),
        ("Points", "point set", "point count, precision, recall, F1, distance"),
        ("Hybrid", "density + points", "density와 localization을 모두 입증한 경우만 기능 점수"),
    )
    y = 338
    for family, output, metrics in rows:
        c.setFillColor(HexColor("#EDF3F7"))
        c.roundRect(432, y - 44, 356, 52, 5, fill=1, stroke=0)
        c.setFont(bold, 9)
        c.setFillColor(BLUE)
        c.drawString(444, y - 2, family)
        c.setFont(regular, 8)
        c.setFillColor(INK)
        c.drawString(510, y - 2, output)
        draw_wrapped(c, metrics, 444, y - 19, 330, font=regular, size=7.5, leading=9, max_lines=2)
        y -= 66
    c.setFont(bold, 9)
    c.setFillColor(RED)
    c.drawString(432, 118, "Rights rule: 기술 점수는 code, dataset, weight, deployment 권리와 독립적으로 계산")


def _page_overall(c, ctx) -> None:
    regular, bold = ctx["fonts"]
    _background(c)
    _title(c, "전체 비교 결과", "100-point operational harness score - 공식 모델 순위가 아님", bold_font=bold, regular_font=regular)
    ordered = sorted(ctx["comparison"]["models"], key=lambda row: row["technical_order"])
    _card(c, 34, 104, 420, 384)
    c.setFont(bold, 13)
    c.setFillColor(NAVY)
    c.drawString(52, 462, "Technical score")
    y = 424
    for row in ordered:
        score = float(row["technical_score"])
        c.setFont(bold, 9)
        c.setFillColor(INK)
        c.drawString(54, y + 4, DISPLAY_NAMES[row["model_id"]])
        c.setFillColor(LINE)
        c.roundRect(140, y, 250, 15, 6, fill=1, stroke=0)
        c.setFillColor(CYAN if row["model_id"] in {"steerer", "dm-count"} else AMBER if row["model_id"] == "pet" else BLUE)
        c.roundRect(140, y, 250 * score / 100, 15, 6, fill=1, stroke=0)
        c.setFont(bold, 9)
        c.setFillColor(NAVY)
        c.drawRightString(430, y + 4, _fmt(score, 2))
        y -= 49
    _card(c, 470, 104, 338, 384)
    headers = ("Model", "MAE", "Zone", "FPS", "VRAM")
    x_positions = (486, 576, 628, 684, 735)
    c.setFillColor(NAVY)
    c.rect(482, 440, 310, 30, fill=1, stroke=0)
    c.setFont(bold, 8)
    c.setFillColor(white)
    for x, header in zip(x_positions, headers, strict=True):
        c.drawString(x, 451, header)
    y = 414
    for index, row in enumerate(ordered):
        if index % 2 == 0:
            c.setFillColor(HexColor("#EDF3F7"))
            c.rect(482, y - 9, 310, 32, fill=1, stroke=0)
        aggregate = row["aggregates"]
        values = (
            DISPLAY_NAMES[row["model_id"]],
            _fmt(aggregate["mae"], 1),
            _fmt(aggregate["mean_zone_mae"], 1),
            _fmt(aggregate["throughput_fps_batch1"], 1),
            f"{float(aggregate['peak_vram_mb']) / 1000:.1f}G",
        )
        c.setFont(regular, 8)
        c.setFillColor(INK)
        for x, value in zip(x_positions, values, strict=True):
            c.drawString(x, y, value)
        y -= 43
    c.setFillColor(RED)
    c.setFont(bold, 8.5)
    draw_wrapped(c, "주의: 서로 다른 학습 protocol과 checkpoint를 사용한 36장 cross-domain 운영 screen이다.", 486, 148, 294, font=bold, size=8.5, leading=11, color=RED, max_lines=3)


def _model_page(c, ctx, model_id: str) -> None:
    regular, bold = ctx["fonts"]
    _background(c)
    content: ModelContent = ctx["content_by_id"][model_id]
    row = ctx["rows_by_id"][model_id]
    _title(c, DISPLAY_NAMES[model_id], f"{content.paper} | {content.family}", bold_font=bold, regular_font=regular)
    aggregate = row["aggregates"]
    metrics = (
        ("Technical", _fmt(row["technical_score_display"], 2)),
        ("MAE / RMSE", f"{_fmt(aggregate['mae'], 1)} / {_fmt(aggregate['rmse'], 1)}"),
        ("Zone MAE", _fmt(aggregate["mean_zone_mae"], 1)),
        ("FPS / VRAM", f"{_fmt(aggregate['throughput_fps_batch1'], 1)} / {float(aggregate['peak_vram_mb']) / 1000:.1f}G"),
    )
    for index, (label, value) in enumerate(metrics):
        draw_metric_card(c, label, value, 34 + index * 120, 432, 110, 55, regular_font=regular, bold_font=bold, accent=(CYAN, BLUE, AMBER, GREEN)[index])
    _card(c, 34, 92, 376, 326)
    c.setFont(bold, 11)
    c.setFillColor(NAVY)
    c.drawString(50, 394, "Architecture and I/O")
    y = 374
    y = draw_wrapped(c, f"Backbone: {content.backbone}", 50, y, 344, font=regular, size=8, leading=10, max_lines=2) - 3
    y = draw_wrapped(c, f"Input: {content.input}", 50, y, 344, font=regular, size=8, leading=10, max_lines=2) - 3
    y = draw_wrapped(c, f"Output: {content.native_output}", 50, y, 344, font=regular, size=8, leading=10, max_lines=2) - 3
    y = draw_wrapped(c, f"Count: {content.count_method}", 50, y, 344, font=regular, size=8, leading=10, max_lines=2) - 6
    c.setFont(bold, 10)
    c.setFillColor(BLUE)
    c.drawString(50, y, "Major blocks")
    y = _draw_bullets(c, list(content.structure), 50, y - 18, 344, regular_font=regular, size=7.5, leading=9, max_lines=1) - 4
    c.setFont(bold, 10)
    c.setFillColor(GREEN)
    c.drawString(50, y, "Strengths")
    y = _draw_bullets(c, list(content.strengths), 50, y - 17, 344, regular_font=regular, size=7.4, leading=9, max_lines=1) - 4
    c.setFont(bold, 10)
    c.setFillColor(RED)
    c.drawString(50, y, "Limitations")
    y = _draw_bullets(c, list(content.limitations), 50, y - 17, 344, regular_font=regular, size=7.4, leading=9, max_lines=1) - 4
    c.setFont(bold, 8)
    c.setFillColor(NAVY)
    draw_wrapped(c, f"CCTV: {content.cctv_interpretation}", 50, max(y, 105), 344, font=bold, size=7.5, leading=9, max_lines=3)
    assets = ctx["assets"]["models"][model_id]
    draw_panel(c, ctx["asset_root"] / assets["lower_error"]["packaged_path"], "Lower-error example | " + _panel_caption(assets["lower_error"]), 428, 265, 380, 153, regular_font=regular)
    draw_panel(c, ctx["asset_root"] / assets["limitation"]["packaged_path"], "Limitation example | " + _panel_caption(assets["limitation"]), 428, 96, 380, 153, regular_font=regular)
    c.setFont(regular, 7)
    c.setFillColor(MUTED)
    c.drawRightString(808, 82, _rights_summary(row))


def _comparison_page(c, ctx, *, group_name: str) -> None:
    regular, bold = ctx["fonts"]
    _background(c)
    if group_name == "density":
        title = "Density-family 동일 장면 비교"
        subtitle = "DM-Count, STEERER, MPCount, CSRNet | 같은 selected sample"
        height = 91
        gap = 8
    else:
        title = "Point/Hybrid 동일 장면 비교"
        subtitle = "STEERER, PET, APGCC | 같은 selected sample"
        height = 121
        gap = 13
    _title(c, title, subtitle, bold_font=bold, regular_font=regular)
    group = ctx["assets"]["comparisons"][group_name]
    y = 401 if group_name == "density" else 352
    for model_id, panel in group.items():
        draw_panel(c, ctx["asset_root"] / panel["packaged_path"], _panel_caption(panel), 34, y, 774, height, regular_font=regular)
        y -= height + gap
    c.setFillColor(RED)
    c.setFont(bold, 8)
    c.drawString(35, 78, "시각화는 모델별 native output 차이를 보여주며, 단독으로 정확도나 배포 적합성을 판정하지 않는다.")


def _page_rights(c, ctx) -> None:
    regular, bold = ctx["fonts"]
    _background(c)
    _title(c, "상용화 권리와 기술 판단", "Apache 코드 또는 데이터 표시는 weight와 deployment 권리를 자동 승인하지 않는다.", bold_font=bold, regular_font=regular)
    rows = sorted(ctx["comparison"]["models"], key=lambda row: row["technical_order"])
    columns = ("Model", "Code", "Dataset", "Pretrained", "Derived", "Deploy", "Overall")
    widths = (92, 105, 105, 112, 105, 105, 118)
    x0, y0 = 34, 430
    x = x0
    c.setFillColor(NAVY)
    c.rect(x0, y0, sum(widths), 34, fill=1, stroke=0)
    c.setFillColor(white)
    c.setFont(bold, 7.5)
    for width, label in zip(widths, columns, strict=True):
        c.drawCentredString(x + width / 2, y0 + 12, label)
        x += width
    status_labels = {
        "REVIEWED_NO_EXPLICIT_PROHIBITION": "Reviewed / no explicit ban",
        "PENDING": "Pending",
        "RESTRICTED": "Restricted",
    }
    y = y0 - 48
    for index, row in enumerate(rows):
        c.setFillColor(white if index % 2 else HexColor("#EDF3F7"))
        c.rect(x0, y, sum(widths), 42, fill=1, stroke=0)
        values = [DISPLAY_NAMES[row["model_id"]]]
        rights = row["rights_components"]
        for key in ("code", "dataset", "pretrained_weights", "derived_weights", "deployment"):
            values.append(status_labels[rights[key]["status"]])
        values.append(row["deployment_label"])
        x = x0
        for col_index, (width, value) in enumerate(zip(widths, values, strict=True)):
            color = RED if "Restricted" in value or value == "RESEARCH_ONLY" else AMBER if "Pending" in value or value == "PENDING_DUE_DILIGENCE" else INK
            c.setFillColor(color)
            c.setFont(bold if col_index in {0, 6} else regular, 6.8)
            lines = _wrap(str(value), width - 8, bold if col_index in {0, 6} else regular, 6.8)[:2]
            for line_index, line in enumerate(lines):
                c.drawCentredString(x + width / 2, y + 24 - line_index * 8, line)
            x += width
        y -= 48
    _card(c, 34, 88, 774, 68, fill=HexColor("#FFF7E8"), stroke=HexColor("#F1CF8B"))
    c.setFillColor(RED)
    c.setFont(bold, 9)
    c.drawString(50, 132, "Product interpretation")
    draw_wrapped(c, "STEERER와 DM-Count는 PENDING_DUE_DILIGENCE이며 제품 검증 후보일 뿐이다. PET은 code와 pretrained weight가 restricted이므로 연구 비교군으로만 유지한다.", 50, 113, 740, font=regular, size=8.5, leading=11, max_lines=3)


def _page_conclusion(c, ctx) -> None:
    regular, bold = ctx["fonts"]
    _background(c)
    _title(c, "결론과 다음 로드맵", "Fine-tuning보다 먼저 고정형 CCTV calibration과 field validation이 필요하다.", bold_font=bold, regular_font=regular)
    _card(c, 34, 286, 376, 202)
    c.setFont(bold, 14)
    c.setFillColor(NAVY)
    c.drawString(52, 458, "Final decision")
    _draw_bullets(c, list(ctx["content"].conclusion), 52, 430, 340, regular_font=regular, size=8.5, leading=11, max_lines=2)
    _card(c, 428, 286, 380, 202)
    c.setFont(bold, 14)
    c.setFillColor(NAVY)
    c.drawString(446, 458, "Ordered roadmap")
    y = 430
    for index, item in enumerate(ctx["content"].roadmap, start=1):
        c.setFillColor(CYAN if index < 5 else AMBER)
        c.circle(456, y + 3, 9, fill=1, stroke=0)
        c.setFillColor(white)
        c.setFont(bold, 7)
        c.drawCentredString(456, y, str(index))
        y = draw_wrapped(c, item, 472, y, 316, font=regular, size=8, leading=10, max_lines=2) - 5
    _card(c, 34, 92, 774, 174)
    c.setFont(bold, 12)
    c.setFillColor(NAVY)
    c.drawString(52, 240, "Evidence and references")
    dataset = ctx["comparison"]["dataset"]
    evidence_lines = [
        f"Dataset / split: {dataset['dataset_id']} / {dataset['split_id']} / {dataset['split_role']} / 36 images",
        f"Canonical split SHA-256: {dataset['canonical_split_sha256']}",
        f"Comparison JSON SHA-256: {sha256_file(ctx['comparison_path'])}",
    ]
    y = 220
    for line in evidence_lines:
        y = draw_wrapped(c, line, 52, y, 738, font=regular, size=7.3, leading=9, max_lines=2) - 2
    references = [
        f"{DISPLAY_NAMES[model.model_id]}: {model.paper}"
        for model in ctx["content"].models
    ]
    for index, reference in enumerate(references):
        column = index % 2
        row = index // 2
        draw_wrapped(c, reference, 52 + column * 374, 174 - row * 26, 350, font=regular, size=6.7, leading=8, max_lines=2)


def build_round1_pdf(
    comparison_path: str | Path,
    assets_manifest_path: str | Path,
    content_path: str | Path,
    output_path: str | Path,
    *,
    report_date: str,
) -> Path:
    if not _DATE.fullmatch(report_date):
        raise ValueError("report_date must use YYYY-MM-DD")
    comparison_source = Path(comparison_path).resolve()
    assets_source = Path(assets_manifest_path).resolve()
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.parent / f".{output.name}.tmp.pdf"
    if output.exists() or temporary.exists():
        raise FileExistsError("PDF output and temporary path must not exist")
    comparison, assets = _validate_inputs(comparison_source, assets_source)
    content = load_report_content(content_path)
    fonts = register_korean_fonts()
    rows_by_id = {row["model_id"]: row for row in comparison["models"]}
    content_by_id = {model.model_id: model for model in content.models}
    ctx = {
        "comparison": comparison,
        "comparison_path": comparison_source,
        "assets": assets,
        "asset_root": assets_source.parent,
        "content": content,
        "content_by_id": content_by_id,
        "rows_by_id": rows_by_id,
        "fonts": fonts,
        "report_date": report_date,
    }
    renderers: list[Callable[[canvas_module.Canvas, dict[str, object]], None]] = [
        _page_cover,
        _page_executive,
        _page_problem,
        _page_evaluation,
        _page_overall,
        lambda c, context: _model_page(c, context, "steerer"),
        lambda c, context: _model_page(c, context, "dm-count"),
        lambda c, context: _model_page(c, context, "pet"),
        lambda c, context: _model_page(c, context, "mpcount"),
        lambda c, context: _model_page(c, context, "apgcc"),
        lambda c, context: _model_page(c, context, "csrnet"),
        lambda c, context: _comparison_page(c, context, group_name="density"),
        lambda c, context: _comparison_page(c, context, group_name="points"),
        _page_rights,
        _page_conclusion,
    ]
    try:
        pdf = canvas_module.Canvas(
            str(temporary), pagesize=PAGE_SIZE, pageCompression=1
        )
        pdf.setTitle(content.title)
        pdf.setAuthor("DroneAI")
        pdf.setSubject("Private Round 1 fixed-CCTV model analysis")
        for page_number, renderer in enumerate(renderers, start=1):
            renderer(pdf, ctx)
            if page_number != 1:
                draw_footer(pdf, page_number, regular_font=fonts[0])
            pdf.showPage()
        pdf.save()
        reader = PdfReader(temporary)
        if len(reader.pages) != PAGE_COUNT:
            raise ValueError(f"PDF must contain exactly {PAGE_COUNT} pages")
        temporary.rename(output)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return output

