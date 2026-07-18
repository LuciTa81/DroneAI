from __future__ import annotations

import re
from pathlib import Path
from typing import Callable

from pypdf import PdfReader
from reportlab.lib.colors import HexColor, white
from reportlab.lib.pagesizes import A4
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas as canvas_module

from droneai.integrity import sha256_file
from droneai.pdf_report_content import ModelContent, load_report_content
from droneai.round1_pdf_report import (
    BLUE,
    CYAN,
    DISPLAY_NAMES,
    GREEN,
    INK,
    LINE,
    MODEL_IDS,
    MUTED,
    NAVY,
    PAPER,
    RED,
    _fmt,
    _panel_caption,
    _validate_inputs,
    draw_wrapped,
    register_korean_fonts,
)


PAGE_SIZE = A4
PAGE_COUNT = 12
WIDTH, HEIGHT = PAGE_SIZE
MARGIN = 42
CONTENT_WIDTH = WIDTH - MARGIN * 2
BODY_SIZE = 10.5
BODY_LEADING = 15.0
TABLE_HEADER_SIZE = 8.5
TABLE_BODY_SIZE = 8.5
CAPTION_SIZE = 8.5
CAPTION_LEADING = 11.0
HEADER_FOOTER_SIZE = 8.0
MODEL_TABLE_HEADER_HEIGHT = 42.0
MODEL_TABLE_BODY_HEIGHT = 126.0
CANONICAL_SPLIT_SHA256 = (
    "da1d947aad73d45be573d52a6462fa8022948fba2b1026a3bfff76b11a4b5c67"
)
SPLIT_ID = "dm-count-upstream-qnrf-val-smoke-36-v1"
_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")

SOFT_BLUE = HexColor("#EAF2F7")
SOFT_CYAN = HexColor("#E7F5F7")
SOFT_GRAY = HexColor("#F2F5F7")
AMBER = HexColor("#D99122")

MODEL_OUTPUT_NOTES = {
    "dm-count": "출력 해석(DM-Count): density map 합이 count이며 zone별 합으로 구역 밀집도를 읽는다.",
    "steerer": "출력 해석(STEERER): density 합으로 count하고 point를 zone 위치 후보로 함께 활용한다.",
    "pet": "출력 해석(PET): confidence를 통과한 point 개수가 count이며 좌표를 zone에 직접 배정한다.",
    "mpcount": "출력 해석(MPCount): full-resolution density 합으로 count하고 zone별 질량을 비교한다.",
    "apgcc": "출력 해석(APGCC): confidence point 개수가 count이며 위치 오차와 누락을 함께 확인한다.",
    "csrnet": "출력 해석(CSRNet): 비음수 density 합이 count이며 흐린 분포와 zone 오차를 확인한다.",
}
MODEL_PAGE_ORDER = ("steerer", "dm-count", "pet", "mpcount", "apgcc", "csrnet")
MODEL_PAGE_ASSETS = {
    "steerer": ("points", "steerer"),
    "dm-count": ("density", "dm-count"),
    "pet": ("points", "pet"),
    "mpcount": ("density", "mpcount"),
    "apgcc": ("points", "apgcc"),
    "csrnet": ("density", "csrnet"),
}
MODEL_PAGE_SUMMARIES = {
    "steerer": (
        "Density heatmap과 point 후보를 함께 제공해 zone 밀집도와 위치 확인에 활용한다.",
        "Round 1 최저 MAE·Zone MAE",
        "고해상도 연산으로 7.7 FPS",
    ),
    "dm-count": (
        "Density heatmap을 zone별로 합산하는 CCTV baseline 후보이다.",
        "질량 보존형 heatmap과 14.4 FPS",
        "고밀도·scale 변화에서 undercount 가능",
    ),
    "pet": (
        "Point 좌표를 zone에 직접 배정해 구역별 인원을 계산한다.",
        "위치 기반 집계와 17.2 FPS",
        "작은 사람·가림·domain shift에 민감",
    ),
    "mpcount": (
        "Full-resolution density를 zone별로 합산해 일반화 성능을 본다.",
        "Single-domain generalization 설계",
        "13.0 GB VRAM과 비교적 큰 count 오차",
    ),
    "apgcc": (
        "Point 좌표를 zone에 직접 배정해 위치 기반 인원을 계산한다.",
        "Point localization과 14.8 FPS",
        "ShanghaiTech-A checkpoint domain shift",
    ),
    "csrnet": (
        "Operational density를 zone별로 합산하는 고전 baseline이다.",
        "단순한 density heatmap과 zone aggregation",
        "가장 큰 MAE·RMSE와 20.3 GB VRAM",
    ),
}


def readability_contract() -> dict[str, float]:
    return {
        "body_size": BODY_SIZE,
        "body_leading": BODY_LEADING,
        "table_header_size": TABLE_HEADER_SIZE,
        "table_body_size": TABLE_BODY_SIZE,
        "caption_size": CAPTION_SIZE,
        "caption_leading": CAPTION_LEADING,
        "header_footer_size": HEADER_FOOTER_SIZE,
    }


def report_page_contract() -> tuple[tuple[str, ...], ...]:
    return (
        ("cover",),
        ("summary",),
        ("evaluation",),
        ("steerer",),
        ("dm-count",),
        ("pet",),
        ("mpcount",),
        ("apgcc",),
        ("csrnet",),
        ("performance",),
        ("rights",),
        ("conclusion",),
    )


def model_table_height_contract() -> dict[str, float]:
    return {"header": MODEL_TABLE_HEADER_HEIGHT, "body": MODEL_TABLE_BODY_HEIGHT}


def model_output_note(model_id: str) -> str:
    try:
        return MODEL_OUTPUT_NOTES[model_id]
    except KeyError as exc:
        raise ValueError(f"unknown model output note: {model_id}") from exc


def model_page_asset_contract() -> dict[str, tuple[str, str]]:
    return dict(MODEL_PAGE_ASSETS)


def _background(canvas: canvas_module.Canvas) -> None:
    canvas.setFillColor(PAPER)
    canvas.rect(0, 0, WIDTH, HEIGHT, fill=1, stroke=0)


def draw_report_header(
    canvas: canvas_module.Canvas,
    section: str,
    *,
    regular_font: str,
) -> None:
    canvas.setFont(regular_font, HEADER_FOOTER_SIZE)
    canvas.setFillColor(MUTED)
    canvas.drawString(MARGIN, HEIGHT - 24, "DroneAI | 고정형 CCTV 군중 밀집도 모델 비교")
    canvas.drawRightString(WIDTH - MARGIN, HEIGHT - 24, section)
    canvas.setStrokeColor(LINE)
    canvas.setLineWidth(0.7)
    canvas.line(MARGIN, HEIGHT - 32, WIDTH - MARGIN, HEIGHT - 32)


def draw_report_footer(
    canvas: canvas_module.Canvas,
    page_number: int,
    *,
    regular_font: str,
) -> None:
    canvas.setStrokeColor(LINE)
    canvas.setLineWidth(0.7)
    canvas.line(MARGIN, 35, WIDTH - MARGIN, 35)
    canvas.setFont(regular_font, HEADER_FOOTER_SIZE)
    canvas.setFillColor(MUTED)
    canvas.drawString(MARGIN, 22, "UCF-QNRF validation 36장 | cross-domain compatibility smoke")
    canvas.drawRightString(WIDTH - MARGIN, 22, f"{page_number} / {PAGE_COUNT}")


def draw_section_title(
    canvas: canvas_module.Canvas,
    number: str,
    title: str,
    subtitle: str,
    *,
    regular_font: str,
    bold_font: str,
) -> None:
    canvas.setFont(bold_font, 10)
    canvas.setFillColor(CYAN)
    canvas.drawString(MARGIN, HEIGHT - 61, number)
    canvas.setFont(bold_font, 20)
    canvas.setFillColor(NAVY)
    canvas.drawString(MARGIN, HEIGHT - 88, title)
    canvas.setFont(regular_font, BODY_SIZE)
    canvas.setFillColor(MUTED)
    canvas.drawString(MARGIN, HEIGHT - 106, subtitle)
    canvas.setStrokeColor(CYAN)
    canvas.setLineWidth(1.5)
    canvas.line(MARGIN, HEIGHT - 116, WIDTH - MARGIN, HEIGHT - 116)


def draw_report_table(
    canvas: canvas_module.Canvas,
    rows: list[list[str]],
    column_widths: tuple[float, ...],
    x: float,
    y: float,
    *,
    row_height: float,
    header_row_height: float | None = None,
    fonts: tuple[str, str],
) -> float:
    if not rows or any(len(row) != len(column_widths) for row in rows):
        raise ValueError("report table dimensions are invalid")
    regular, bold = fonts
    total_width = sum(column_widths)
    for row_index, row in enumerate(rows):
        current_height = (
            header_row_height
            if row_index == 0 and header_row_height is not None
            else row_height
        )
        row_y = y - current_height
        if row_index == 0:
            canvas.setFillColor(NAVY)
            canvas.rect(x, row_y, total_width, current_height, fill=1, stroke=0)
        else:
            canvas.setFillColor(white if row_index % 2 else SOFT_BLUE)
            canvas.rect(x, row_y, total_width, current_height, fill=1, stroke=0)
        cell_x = x
        for width, value in zip(column_widths, row, strict=True):
            canvas.setStrokeColor(LINE)
            canvas.setLineWidth(0.4)
            canvas.rect(cell_x, row_y, width, current_height, fill=0, stroke=1)
            draw_wrapped(
                canvas,
                value,
                cell_x + 4,
                row_y + current_height - 11,
                width - 8,
                font=bold if row_index == 0 else regular,
                size=TABLE_HEADER_SIZE if row_index == 0 else TABLE_BODY_SIZE,
                leading=11,
                color=white if row_index == 0 else INK,
                max_lines=max(1, int((current_height - 9) // 11)),
            )
            cell_x += width
        y = row_y
    return y


def draw_figure(
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
    canvas.setFillColor(white)
    canvas.setStrokeColor(LINE)
    canvas.setLineWidth(0.6)
    canvas.rect(x, y, width, height, fill=1, stroke=1)
    reader = ImageReader(str(image_path))
    image_width, image_height = reader.getSize()
    caption_height = 50
    available_width = width - 10
    available_height = height - caption_height - 8
    scale = min(available_width / image_width, available_height / image_height)
    drawn_width = image_width * scale
    drawn_height = image_height * scale
    canvas.drawImage(
        reader,
        x + (width - drawn_width) / 2,
        y + caption_height + (available_height - drawn_height) / 2,
        width=drawn_width,
        height=drawn_height,
        preserveAspectRatio=True,
        mask="auto",
    )
    draw_wrapped(
        canvas,
        caption,
        x + 6,
        y + 23,
        width - 12,
        font=regular_font,
        size=CAPTION_SIZE,
        leading=CAPTION_LEADING,
        color=MUTED,
        max_lines=3,
    )


def _section_label(canvas, text: str, x: float, y: float, *, bold_font: str) -> None:
    canvas.setFont(bold_font, 12)
    canvas.setFillColor(BLUE)
    canvas.drawString(x, y, text)
    canvas.setStrokeColor(CYAN)
    canvas.setLineWidth(1)
    canvas.line(x, y - 5, WIDTH - MARGIN, y - 5)


def _bullets(
    canvas,
    items: tuple[str, ...] | list[str],
    x: float,
    y: float,
    width: float,
    *,
    regular_font: str,
    bold_font: str,
    leading: float = 15,
) -> float:
    for index, item in enumerate(items, start=1):
        canvas.setFillColor(CYAN)
        canvas.circle(x + 5, y + 2, 5, fill=1, stroke=0)
        canvas.setFont(bold_font, 6.5)
        canvas.setFillColor(white)
        canvas.drawCentredString(x + 5, y, str(index))
        y = draw_wrapped(
            canvas,
            item,
            x + 17,
            y + 4,
            width - 17,
            font=regular_font,
            size=BODY_SIZE,
            leading=BODY_LEADING,
            max_lines=3,
        ) - max(0, leading - BODY_LEADING)
    return y


def _page_cover(c, ctx) -> None:
    regular, bold = ctx["fonts"]
    _background(c)
    c.setFillColor(NAVY)
    c.rect(0, 0, 16, HEIGHT, fill=1, stroke=0)
    c.setFont(bold, 10.5)
    c.setFillColor(CYAN)
    c.drawString(MARGIN, HEIGHT - 80, "TECHNICAL REPORT | ROUND 1")
    c.setFont(bold, 27)
    c.setFillColor(NAVY)
    c.drawString(MARGIN, HEIGHT - 132, "고정형 CCTV 군중 밀집도")
    c.drawString(MARGIN, HEIGHT - 169, "모델 비교 보고서")
    c.setFont(regular, BODY_SIZE)
    c.setFillColor(MUTED)
    c.drawString(MARGIN, HEIGHT - 200, "6개 crowd-counting 모델의 구조, 출력, 성능 및 적용성 검토")
    c.setStrokeColor(CYAN)
    c.setLineWidth(2)
    c.line(MARGIN, HEIGHT - 221, WIDTH - MARGIN, HEIGHT - 221)
    meta = (
        ("검증 데이터", "UCF-QNRF validation 36장"),
        ("실행 조건", "batch 1 / fine-tuning 없음"),
        ("비교 범위", "official checkpoint cross-domain compatibility smoke"),
        ("작성일", ctx["report_date"]),
    )
    y = HEIGHT - 285
    for label, value in meta:
        c.setFont(bold, 10)
        c.setFillColor(BLUE)
        c.drawString(MARGIN, y, label)
        c.setFont(regular, BODY_SIZE)
        c.setFillColor(INK)
        c.drawString(MARGIN + 90, y, value)
        y -= 28
    c.setFillColor(SOFT_CYAN)
    c.setStrokeColor(CYAN)
    c.roundRect(MARGIN, 135, CONTENT_WIDTH, 104, 4, fill=1, stroke=1)
    c.setFont(bold, 10.5)
    c.setFillColor(BLUE)
    c.drawString(MARGIN + 16, 213, "핵심 결론")
    draw_wrapped(
        c,
        "제품 검증: STEERER·DM-Count. PET는 연구 비교군으로 유지한다.",
        MARGIN + 16,
        187,
        CONTENT_WIDTH - 32,
        font=bold,
        size=12,
        leading=17,
        color=NAVY,
        max_lines=3,
    )
    c.setFont(regular, HEADER_FOOTER_SIZE)
    c.setFillColor(MUTED)
    c.drawString(MARGIN, 70, "공유용 검토본 | 공식 모델 순위가 아님 | 제품 배포 승인 문서가 아님")


def _page_summary(c, ctx) -> None:
    regular, bold = ctx["fonts"]
    draw_section_title(c, "01", "요약 및 핵심 결론", "기술 비교 결과와 해석 범위를 먼저 제시한다.", regular_font=regular, bold_font=bold)
    _section_label(c, "1.1 주요 판단", MARGIN, 690, bold_font=bold)
    findings = (
        "STEERER는 36장 비교에서 기술 점수 83.18로 가장 높은 운영 적합성을 보였고, density와 localization 출력을 함께 제공했다.",
        "DM-Count는 기술 점수 74.06과 14.38 FPS를 기록했으며, 해석 가능한 density heatmap과 구역별 집계에 적합했다.",
        "PET는 기술적으로 3번째 후보이나 code와 pretrained weight가 제한되어 연구 비교군으로만 유지한다.",
        "현재 수치는 공식 pretrained checkpoint를 다른 validation domain에서 비교한 compatibility smoke 결과다.",
        "현 결과만으로 제품 배포를 승인할 수 없으며 CCTV calibration과 현장 검증, 권리 확인이 별도로 필요하다.",
    )
    y = _bullets(c, findings, MARGIN, 660, CONTENT_WIDTH, regular_font=regular, bold_font=bold, leading=19)
    _section_label(c, "1.2 해석 주의", MARGIN, y - 5, bold_font=bold)
    caution = (
        "기술 점수는 count, zone/spatial, runtime, 관제 활용성, 증거 품질을 합산한 운영 목적 점수다. 정확도 백분율이나 공식 모델 순위가 아니다.",
        "MAE와 RMSE는 낮을수록 좋지만, CCTV 관제에서는 zone 오차와 출력 형태, 지연시간, VRAM도 함께 판단해야 한다.",
        "PASS_COMMERCIAL_CANDIDATE는 명시적 금지 문구를 찾지 못한 후보 상태이며 생산·배포 승인이 아니다.",
    )
    _bullets(c, caution, MARGIN, y - 35, CONTENT_WIDTH, regular_font=regular, bold_font=bold, leading=20)


def _page_evaluation(c, ctx) -> None:
    regular, bold = ctx["fonts"]
    draw_section_title(c, "02", "검증 목적·데이터·조건", "동일 표본과 기능별 metric을 고정해 출력과 운영 적합성을 비교했다.", regular_font=regular, bold_font=bold)
    c.setFont(bold, 9)
    c.setFillColor(NAVY)
    c.drawString(MARGIN, 690, "표 2-1. 고정된 평가 조건")
    rows = [
        ["항목", "설정", "해석"],
        ["Dataset", "UCF-QNRF Kaggle Apache", "라벨이 있는 crowd-counting 데이터"],
        ["Split", "validation 36장", "low 12 / medium 12 / high 12"],
        ["Runtime", "RTX 5090 / batch 1", "latency, FPS, peak VRAM 측정"],
        ["Training", "fine-tuning 없음", "공식 checkpoint의 교차 도메인 적합성 확인"],
        ["Scope", "cross-domain compatibility smoke", "공식 모델 순위가 아님"],
    ]
    y = draw_report_table(c, rows, (90, 170, 251), MARGIN, 675, row_height=45, fonts=ctx["fonts"])
    _section_label(c, "2.1 공통 지표", MARGIN, y - 28, bold_font=bold)
    draw_wrapped(c, "MAE, RMSE, signed bias, MAPE 참고치, median latency, batch-1 FPS, peak VRAM을 모든 모델에 공통 적용했다.", MARGIN, y - 52, CONTENT_WIDTH, font=regular, size=BODY_SIZE, leading=BODY_LEADING, max_lines=3)
    _section_label(c, "2.2 출력 계열별 지표", MARGIN, y - 98, bold_font=bold)
    families = [
        ["계열", "Native output", "추가 평가"],
        ["Density", "density map", "sum count, GAME L1, zone MAE, PSNR, SSIM"],
        ["Points", "point set", "precision, recall, F1, mean distance"],
        ["Hybrid", "density + points", "density와 localization을 각각 입증"],
    ]
    draw_report_table(c, families, (90, 150, 271), MARGIN, y - 114, row_height=44, fonts=ctx["fonts"])


def _page_models_group(
    c,
    ctx,
    *,
    number: str,
    title: str,
    subtitle: str,
    model_ids: tuple[str, ...],
    table_label: str,
) -> None:
    regular, bold = ctx["fonts"]
    draw_section_title(c, number, title, subtitle, regular_font=regular, bold_font=bold)
    c.setFont(bold, 9)
    c.setFillColor(NAVY)
    c.drawString(MARGIN, 690, table_label)
    rows = [["Model", "계열 / Backbone", "Native output / Count", "고정형 CCTV 활용 판단"]]
    content_by_id = ctx["content_by_id"]
    for model_id in model_ids:
        item: ModelContent = content_by_id[model_id]
        rows.append([
            DISPLAY_NAMES[model_id],
            f"{item.family}\n{item.backbone}",
            f"{item.native_output}\nCount: {item.count_method}",
            item.cctv_interpretation,
        ])
    y = draw_report_table(
        c,
        rows,
        (70, 135, 135, 171),
        MARGIN,
        675,
        row_height=MODEL_TABLE_BODY_HEIGHT,
        header_row_height=MODEL_TABLE_HEADER_HEIGHT,
        fonts=ctx["fonts"],
    )
    summary = (
        "Density 계열은 heatmap 질량을 zone별로 합산하고, point 계열은 예측 좌표를 zone에 직접 할당한다. "
        "Hybrid 계열은 두 출력을 함께 제공해 관제 설명력과 위치 기반 후속 판단을 보완한다."
    )
    draw_wrapped(c, summary, MARGIN, y - 28, CONTENT_WIDTH, font=regular, size=BODY_SIZE, leading=BODY_LEADING, max_lines=4)


def _spatial_text(row: dict[str, object]) -> str:
    aggregate = row["aggregates"]
    if row["model_id"] in {"pet", "apgcc"}:
        return f"F1 {_fmt(aggregate['localization_f1'], 2)}"
    if row["model_id"] == "steerer":
        return f"Zone {_fmt(aggregate['mean_zone_mae'], 1)} / F1 {_fmt(aggregate['localization_f1'], 2)}"
    return f"Zone {_fmt(aggregate['mean_zone_mae'], 1)}"


def _draw_model_metrics(c, row: dict[str, object], *, fonts: tuple[str, str]) -> None:
    regular, bold = fonts
    aggregate = row["aggregates"]
    metrics = (
        ("기술 점수", _fmt(row["technical_score"], 2)),
        ("MAE", _fmt(aggregate["mae"], 1)),
        ("RMSE", _fmt(aggregate["rmse"], 1)),
        ("공간 지표", _spatial_text(row)),
        ("FPS", _fmt(aggregate["throughput_fps_batch1"], 1)),
        ("VRAM", f"{float(aggregate['peak_vram_mb']) / 1024:.1f} GB"),
    )
    width = CONTENT_WIDTH / len(metrics)
    for index, (label, value) in enumerate(metrics):
        x = MARGIN + index * width
        c.setFillColor(white if index % 2 == 0 else SOFT_BLUE)
        c.setStrokeColor(LINE)
        c.rect(x, 635, width, 54, fill=1, stroke=1)
        c.setFont(bold, 8.5)
        c.setFillColor(BLUE)
        c.drawString(x + 7, 672, label)
        draw_wrapped(
            c,
            value,
            x + 7,
            652,
            width - 14,
            font=regular,
            size=9.5,
            leading=11,
            max_lines=2,
        )


def _draw_model_detail_card(
    c,
    *,
    x: float,
    y: float,
    width: float,
    height: float,
    title: str,
    entries: tuple[tuple[str, str, int], ...],
    fonts: tuple[str, str],
) -> None:
    regular, bold = fonts
    c.setFillColor(white)
    c.setStrokeColor(LINE)
    c.roundRect(x, y, width, height, 4, fill=1, stroke=1)
    c.setFont(bold, 11)
    c.setFillColor(NAVY)
    c.drawString(x + 12, y + height - 22, title)
    cursor = y + height - 43
    for label, value, max_lines in entries:
        c.setFont(bold, 8.5)
        c.setFillColor(BLUE)
        c.drawString(x + 12, cursor, label)
        cursor = draw_wrapped(
            c,
            value,
            x + 72,
            cursor,
            width - 84,
            font=regular,
            size=BODY_SIZE,
            leading=BODY_LEADING,
            max_lines=max_lines,
        ) - 7


def _page_model_detail(
    c,
    ctx,
    *,
    model_id: str,
    number: str,
) -> None:
    regular, bold = ctx["fonts"]
    item: ModelContent = ctx["content_by_id"][model_id]
    row = ctx["comparison_by_id"][model_id]
    draw_section_title(
        c,
        number,
        f"{DISPLAY_NAMES[model_id]} 모델 상세",
        f"{item.family} | {item.backbone}",
        regular_font=regular,
        bold_font=bold,
    )
    _draw_model_metrics(c, row, fonts=ctx["fonts"])

    family, asset_model_id = MODEL_PAGE_ASSETS[model_id]
    panel = ctx["assets"]["comparisons"][family][asset_model_id]
    caption = (
        f"그림 {int(number)}-1. {_panel_caption(panel)}\n"
        f"{model_output_note(model_id)}"
    )
    draw_figure(
        c,
        ctx["asset_root"] / panel["packaged_path"],
        caption,
        MARGIN,
        350,
        CONTENT_WIDTH,
        245,
        regular_font=regular,
    )

    card_gap = 12
    card_width = (CONTENT_WIDTH - card_gap) / 2
    _draw_model_detail_card(
        c,
        x=MARGIN,
        y=72,
        width=card_width,
        height=246,
        title="구조·입출력",
        entries=(
            ("입력", item.input, 3),
            ("출력", item.native_output, 3),
            ("Count", item.count_method, 3),
        ),
        fonts=ctx["fonts"],
    )
    rights_text = (
        "PASS_RESEARCH_ONLY / 제품 lane 제외"
        if model_id == "pet"
        else f"{row['rights_scope']} / 배포 권리 확인 필요"
    )
    cctv_summary, strength_summary, limitation_summary = MODEL_PAGE_SUMMARIES[
        model_id
    ]
    _draw_model_detail_card(
        c,
        x=MARGIN + card_width + card_gap,
        y=72,
        width=card_width,
        height=246,
        title="관제 활용·제약·권리",
        entries=(
            ("활용", cctv_summary, 3),
            ("장점", strength_summary, 2),
            ("제약", limitation_summary, 2),
            ("권리", rights_text, 2),
        ),
        fonts=ctx["fonts"],
    )


def _page_performance(c, ctx) -> None:
    regular, bold = ctx["fonts"]
    draw_section_title(c, "09", "성능 비교와 해석", "Count 정확도, 공간 출력, 속도와 자원 사용량을 함께 본다.", regular_font=regular, bold_font=bold)
    c.setFont(bold, 9)
    c.setFillColor(NAVY)
    c.drawString(MARGIN, 690, "표 9-1. UCF-QNRF validation 36장 비교 결과")
    rows = [["Model", "기술 점수", "MAE", "RMSE", "공간 지표", "FPS", "VRAM"]]
    ordered = sorted(ctx["comparison"]["models"], key=lambda row: row["technical_order"])
    for row in ordered:
        aggregate = row["aggregates"]
        rows.append([
            DISPLAY_NAMES[row["model_id"]],
            _fmt(row["technical_score"], 2),
            _fmt(aggregate["mae"], 1),
            _fmt(aggregate["rmse"], 1),
            _spatial_text(row),
            _fmt(aggregate["throughput_fps_batch1"], 1),
            f"{float(aggregate['peak_vram_mb']) / 1024:.1f} GB",
        ])
    y = draw_report_table(c, rows, (70, 70, 61, 61, 105, 60, 84), MARGIN, 675, row_height=52, fonts=ctx["fonts"])
    _section_label(c, "9.1 결과 해석", MARGIN, y - 28, bold_font=bold)
    analysis = (
        "STEERER는 MAE 71.6, RMSE 98.0, zone MAE 19.8, localization F1 0.80으로 출력 기능과 공간 정확도의 균형이 가장 좋았다.",
        "DM-Count는 MAE 154.3이지만 14.4 FPS와 8.8 GB 수준의 VRAM으로 density 기반 CCTV baseline 후보가 된다.",
        "PET는 MAE 76.4와 17.2 FPS를 기록했으나 연구 전용 권리 범위 때문에 제품 검증 후보에서 분리한다.",
        "CSRNet은 비교 baseline으로 의미가 있으나 MAE 332.3, zone MAE 91.1, 약 20.4 GB VRAM으로 현 조건의 우선순위는 낮다.",
    )
    _bullets(c, analysis, MARGIN, y - 58, CONTENT_WIDTH, regular_font=regular, bold_font=bold, leading=18)
    c.setFont(regular, HEADER_FOOTER_SIZE)
    c.setFillColor(RED)
    c.drawString(MARGIN, 62, "주: 기술 점수는 운영 목적의 합성 점수이며 정확도 백분율이나 공식 benchmark 순위가 아니다.")


def _page_output_group(
    c,
    ctx,
    *,
    number: str,
    title: str,
    subtitle: str,
    family: str,
    model_ids: tuple[str, ...],
    figure_labels: tuple[str, ...],
    narrative: str,
) -> None:
    regular, bold = ctx["fonts"]
    draw_section_title(c, number, title, subtitle, regular_font=regular, bold_font=bold)
    group = ctx["assets"]["comparisons"][family]
    if len(model_ids) == 1:
        positions = ((MARGIN, 335, CONTENT_WIDTH, 300),)
        narrative_y = 295
    else:
        positions = (
            (MARGIN, 430, CONTENT_WIDTH, 205),
            (MARGIN, 195, CONTENT_WIDTH, 205),
        )
        narrative_y = 157
    for model_id, label, (x, y, width, height) in zip(
        model_ids, figure_labels, positions, strict=True
    ):
        panel = group[model_id]
        caption = (
            f"{label}. {_panel_caption(panel)}\n"
            f"{model_output_note(model_id)}"
        )
        draw_figure(
            c,
            ctx["asset_root"] / panel["packaged_path"],
            caption,
            x,
            y,
            width,
            height,
            regular_font=regular,
        )
    draw_wrapped(c, narrative, MARGIN, narrative_y, CONTENT_WIDTH, font=regular, size=BODY_SIZE, leading=BODY_LEADING, max_lines=5)


def _status_short(status: str) -> str:
    return {
        "REVIEWED_NO_EXPLICIT_PROHIBITION": "검토 / 명시 금지 없음",
        "PENDING": "확인 필요",
        "RESTRICTED": "제한",
    }.get(status, status)


def _page_rights(c, ctx) -> None:
    regular, bold = ctx["fonts"]
    draw_section_title(c, "10", "상용화 권리 및 CCTV 적용 판단", "기술 성능과 component-level 권리를 독립적으로 판단한다.", regular_font=regular, bold_font=bold)
    c.setFont(bold, 9)
    c.setFillColor(NAVY)
    c.drawString(MARGIN, 690, "표 10-1. 코드·데이터·weight·deployment 권리 상태")
    rows = [["Model", "Code", "Dataset", "Pretrained", "Derived", "Deploy", "판단"]]
    ordered = sorted(ctx["comparison"]["models"], key=lambda row: row["technical_order"])
    for row in ordered:
        rights = row["rights_components"]
        rows.append([
            DISPLAY_NAMES[row["model_id"]],
            _status_short(rights["code"]["status"]),
            _status_short(rights["dataset"]["status"]),
            _status_short(rights["pretrained_weights"]["status"]),
            _status_short(rights["derived_weights"]["status"]),
            _status_short(rights["deployment"]["status"]),
            "연구 비교" if row["model_id"] == "pet" else "후보 / 확인 필요",
        ])
    y = draw_report_table(c, rows, (50, 70, 70, 74, 67, 67, 113), MARGIN, 675, row_height=52, fonts=ctx["fonts"])
    _section_label(c, "10.1 고정형 CCTV 적용", MARGIN, y - 28, bold_font=bold)
    guidance = (
        "카메라마다 ROI, perspective, 실제 zone 면적과 density threshold를 사전 calibration한다.",
        "인접 CCTV overlap에는 ownership mask를 고정해 한쪽 카메라만 count하도록 하고, 경계 통과는 시간 기반 보정 규칙으로 검증한다.",
        "STEERER와 DM-Count는 제품 검증 후보지만 pretrained/derived weight와 deployment 권리가 모두 확인되기 전에는 배포하지 않는다.",
        "PET는 code와 pretrained weight 제한 때문에 연구 비교군으로 유지한다.",
    )
    _bullets(c, guidance, MARGIN, y - 58, CONTENT_WIDTH, regular_font=regular, bold_font=bold, leading=18)


def _page_conclusion(c, ctx) -> None:
    regular, bold = ctx["fonts"]
    draw_section_title(c, "11", "최종 추천과 다음 단계", "Fine-tuning보다 먼저 CCTV calibration과 field validation을 진행한다.", regular_font=regular, bold_font=bold)
    _section_label(c, "11.1 최종 추천", MARGIN, 690, bold_font=bold)
    recommendation = (
        "제품 검증 1순위: STEERER - 낮은 count/zone 오차와 density+point 출력의 조합.",
        "제품 검증 2순위: DM-Count - 해석 가능한 density heatmap과 비교적 높은 처리 속도.",
        "연구 비교군: PET - count와 localization은 유효하지만 권리 제한으로 제품 lane에서 제외.",
        "MPCount, APGCC, CSRNet은 현재 cross-domain 결과에서 보조 비교군으로 유지.",
    )
    y = _bullets(c, recommendation, MARGIN, 660, CONTENT_WIDTH, regular_font=regular, bold_font=bold, leading=19)
    _section_label(c, "11.2 실행 순서", MARGIN, y - 5, bold_font=bold)
    roadmap = (
        "고정형 CCTV별 ROI, perspective, zone 면적과 density threshold calibration",
        "여러 CCTV의 overlap ownership과 중복 count 제거 규칙 검증",
        "실제 행사 또는 유사 CCTV 영상에서 시간대·조도·거리·고밀도 변화 평가",
        "고정형 CCTV 검증 후 UP-COUNT 또는 DroneCrowd 추가 domain 평가",
        "component-level 권리가 적격인 최종 1개 모델만 마지막 단계에서 fine-tuning 검토",
    )
    y = _bullets(c, roadmap, MARGIN, y - 35, CONTENT_WIDTH, regular_font=regular, bold_font=bold, leading=18)
    _section_label(c, "11.3 증거와 범위", MARGIN, y - 5, bold_font=bold)
    dataset = ctx["comparison"]["dataset"]
    evidence = (
        f"Dataset / split: {dataset['dataset_id']} / {dataset['split_id']} / validation 36장",
        f"Canonical split SHA-256: {dataset['canonical_split_sha256']}",
        f"Comparison JSON SHA-256: {sha256_file(ctx['comparison_path'])}",
        "결론 범위: PASS_RESEARCH_ONLY 공유용 검토 | 공식 모델 순위가 아님 | 제품 배포 승인 아님",
    )
    for line in evidence:
        y = draw_wrapped(c, line, MARGIN, y - 20, CONTENT_WIDTH, font=regular, size=BODY_SIZE, leading=BODY_LEADING, color=MUTED, max_lines=2)


def _validate_share_scope(comparison: dict[str, object], assets: dict[str, object]) -> None:
    dataset = comparison["dataset"]
    if (
        dataset.get("split_id") != SPLIT_ID
        or dataset.get("canonical_split_sha256") != CANONICAL_SPLIT_SHA256
    ):
        raise ValueError("comparison split identity is invalid for the share report")
    comparisons = assets["comparisons"]
    if (
        comparisons.get("density_sample_id") != "img_0097"
        or comparisons.get("points_sample_id") != "img_0062"
    ):
        raise ValueError("share report comparison samples are invalid")


def build_round1_share_pdf(
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
    _validate_share_scope(comparison, assets)
    content = load_report_content(content_path)
    fonts = register_korean_fonts()
    ctx = {
        "comparison": comparison,
        "comparison_path": comparison_source,
        "assets": assets,
        "asset_root": assets_source.parent,
        "content": content,
        "content_by_id": {item.model_id: item for item in content.models},
        "comparison_by_id": {
            item["model_id"]: item for item in comparison["models"]
        },
        "fonts": fonts,
        "report_date": report_date,
    }
    renderers: list[tuple[str, Callable[[canvas_module.Canvas, dict[str, object]], None]]] = [
        ("표지", _page_cover),
        ("요약 및 핵심 결론", _page_summary),
        ("검증 목적·데이터·조건", _page_evaluation),
        (
            "STEERER 모델 상세",
            lambda c, ctx: _page_model_detail(c, ctx, model_id="steerer", number="03"),
        ),
        (
            "DM-Count 모델 상세",
            lambda c, ctx: _page_model_detail(c, ctx, model_id="dm-count", number="04"),
        ),
        (
            "PET 모델 상세",
            lambda c, ctx: _page_model_detail(c, ctx, model_id="pet", number="05"),
        ),
        (
            "MPCount 모델 상세",
            lambda c, ctx: _page_model_detail(c, ctx, model_id="mpcount", number="06"),
        ),
        (
            "APGCC 모델 상세",
            lambda c, ctx: _page_model_detail(c, ctx, model_id="apgcc", number="07"),
        ),
        (
            "CSRNet 모델 상세",
            lambda c, ctx: _page_model_detail(c, ctx, model_id="csrnet", number="08"),
        ),
        ("성능 비교와 해석", _page_performance),
        ("상용화 권리 및 CCTV 적용", _page_rights),
        ("최종 추천과 다음 단계", _page_conclusion),
    ]
    try:
        pdf = canvas_module.Canvas(str(temporary), pagesize=PAGE_SIZE, pageCompression=1)
        pdf.setTitle("DroneAI 고정형 CCTV 군중 밀집도 모델 비교 보고서")
        pdf.setAuthor("DroneAI")
        pdf.setSubject("Round 1 shareable fixed-CCTV technical report")
        for page_number, (section, renderer) in enumerate(renderers, start=1):
            _background(pdf)
            if page_number != 1:
                draw_report_header(pdf, section, regular_font=fonts[0])
            renderer(pdf, ctx)
            if page_number != 1:
                draw_report_footer(pdf, page_number, regular_font=fonts[0])
            pdf.showPage()
        pdf.save()
        reader = PdfReader(temporary)
        if len(reader.pages) != PAGE_COUNT:
            raise ValueError(f"PDF must contain exactly {PAGE_COUNT} pages")
        for page in reader.pages:
            if float(page.mediabox.height) <= float(page.mediabox.width):
                raise ValueError("PDF page must be A4 portrait")
        temporary.rename(output)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return output
