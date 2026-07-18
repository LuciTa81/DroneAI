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
PAGE_COUNT = 9
WIDTH, HEIGHT = PAGE_SIZE
MARGIN = 42
CONTENT_WIDTH = WIDTH - MARGIN * 2
CANONICAL_SPLIT_SHA256 = (
    "da1d947aad73d45be573d52a6462fa8022948fba2b1026a3bfff76b11a4b5c67"
)
SPLIT_ID = "dm-count-upstream-qnrf-val-smoke-36-v1"
_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")

SOFT_BLUE = HexColor("#EAF2F7")
SOFT_CYAN = HexColor("#E7F5F7")
SOFT_GRAY = HexColor("#F2F5F7")
AMBER = HexColor("#D99122")


def _background(canvas: canvas_module.Canvas) -> None:
    canvas.setFillColor(PAPER)
    canvas.rect(0, 0, WIDTH, HEIGHT, fill=1, stroke=0)


def draw_report_header(
    canvas: canvas_module.Canvas,
    section: str,
    *,
    regular_font: str,
) -> None:
    canvas.setFont(regular_font, 7.5)
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
    canvas.setFont(regular_font, 7)
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
    canvas.setFont(regular_font, 8.5)
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
    fonts: tuple[str, str],
) -> float:
    if not rows or any(len(row) != len(column_widths) for row in rows):
        raise ValueError("report table dimensions are invalid")
    regular, bold = fonts
    total_width = sum(column_widths)
    for row_index, row in enumerate(rows):
        row_y = y - row_height
        if row_index == 0:
            canvas.setFillColor(NAVY)
            canvas.rect(x, row_y, total_width, row_height, fill=1, stroke=0)
        else:
            canvas.setFillColor(white if row_index % 2 else SOFT_BLUE)
            canvas.rect(x, row_y, total_width, row_height, fill=1, stroke=0)
        cell_x = x
        for width, value in zip(column_widths, row, strict=True):
            canvas.setStrokeColor(LINE)
            canvas.setLineWidth(0.4)
            canvas.rect(cell_x, row_y, width, row_height, fill=0, stroke=1)
            draw_wrapped(
                canvas,
                value,
                cell_x + 4,
                row_y + row_height - 11,
                width - 8,
                font=bold if row_index == 0 else regular,
                size=6.3 if row_index == 0 else 6.5,
                leading=8,
                color=white if row_index == 0 else INK,
                max_lines=max(1, int((row_height - 7) // 8)),
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
    caption_height = 33
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
        size=6.3,
        leading=7.5,
        color=MUTED,
        max_lines=3,
    )


def _section_label(canvas, text: str, x: float, y: float, *, bold_font: str) -> None:
    canvas.setFont(bold_font, 10)
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
            size=8.7,
            leading=12,
            max_lines=3,
        ) - (leading - 12)
    return y


def _page_cover(c, ctx) -> None:
    regular, bold = ctx["fonts"]
    _background(c)
    c.setFillColor(NAVY)
    c.rect(0, 0, 16, HEIGHT, fill=1, stroke=0)
    c.setFont(bold, 9)
    c.setFillColor(CYAN)
    c.drawString(MARGIN, HEIGHT - 80, "TECHNICAL REPORT | ROUND 1")
    c.setFont(bold, 27)
    c.setFillColor(NAVY)
    c.drawString(MARGIN, HEIGHT - 132, "고정형 CCTV 군중 밀집도")
    c.drawString(MARGIN, HEIGHT - 169, "모델 비교 보고서")
    c.setFont(regular, 11)
    c.setFillColor(MUTED)
    c.drawString(MARGIN, HEIGHT - 200, "6개 crowd-counting 모델의 구조, 출력, 성능 및 적용성 검토")
    c.setStrokeColor(CYAN)
    c.setLineWidth(2)
    c.line(MARGIN, HEIGHT - 221, WIDTH - MARGIN, HEIGHT - 221)
    meta = (
        ("검증 데이터", "UCF-QNRF validation 36장"),
        ("실행 환경", "RTX 5090 / batch 1 / fine-tuning 없음"),
        ("비교 범위", "official checkpoint cross-domain compatibility smoke"),
        ("작성일", ctx["report_date"]),
    )
    y = HEIGHT - 285
    for label, value in meta:
        c.setFont(bold, 8.5)
        c.setFillColor(BLUE)
        c.drawString(MARGIN, y, label)
        c.setFont(regular, 9.5)
        c.setFillColor(INK)
        c.drawString(MARGIN + 90, y, value)
        y -= 28
    c.setFillColor(SOFT_CYAN)
    c.setStrokeColor(CYAN)
    c.roundRect(MARGIN, 135, CONTENT_WIDTH, 104, 4, fill=1, stroke=1)
    c.setFont(bold, 9)
    c.setFillColor(BLUE)
    c.drawString(MARGIN + 16, 213, "핵심 결론")
    draw_wrapped(
        c,
        "STEERER와 DM-Count를 고정형 CCTV 제품 검증 후보로 우선하며, PET는 연구 비교군으로 유지한다.",
        MARGIN + 16,
        187,
        CONTENT_WIDTH - 32,
        font=bold,
        size=12,
        leading=17,
        color=NAVY,
        max_lines=3,
    )
    c.setFont(regular, 7.5)
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
    c.setFont(bold, 8)
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
    y = draw_report_table(c, rows, (90, 170, 251), MARGIN, 675, row_height=39, fonts=ctx["fonts"])
    _section_label(c, "2.1 공통 지표", MARGIN, y - 28, bold_font=bold)
    draw_wrapped(c, "MAE, RMSE, signed bias, MAPE 참고치, median latency, batch-1 FPS, peak VRAM을 모든 모델에 공통 적용했다.", MARGIN, y - 52, CONTENT_WIDTH, font=regular, size=9, leading=13, max_lines=3)
    _section_label(c, "2.2 출력 계열별 지표", MARGIN, y - 98, bold_font=bold)
    families = [
        ["계열", "Native output", "추가 평가"],
        ["Density", "density map", "sum count, GAME L1, zone MAE, PSNR, SSIM"],
        ["Points", "point set", "precision, recall, F1, mean distance"],
        ["Hybrid", "density + points", "density와 localization을 각각 입증"],
    ]
    draw_report_table(c, families, (90, 150, 271), MARGIN, y - 114, row_height=40, fonts=ctx["fonts"])


def _page_models(c, ctx) -> None:
    regular, bold = ctx["fonts"]
    draw_section_title(c, "03", "모델 구조와 입출력 비교", "모델 이름보다 native output과 관제 연결 방식을 중심으로 비교한다.", regular_font=regular, bold_font=bold)
    c.setFont(bold, 8)
    c.setFillColor(NAVY)
    c.drawString(MARGIN, 690, "표 3-1. 6개 모델의 구조·출력·CCTV 활용")
    rows = [["Model", "계열", "Backbone", "Native output", "Count 방식", "CCTV 활용"]]
    content_by_id = ctx["content_by_id"]
    order = ("steerer", "dm-count", "pet", "mpcount", "apgcc", "csrnet")
    for model_id in order:
        item: ModelContent = content_by_id[model_id]
        rows.append([
            DISPLAY_NAMES[model_id],
            item.family,
            item.backbone,
            item.native_output,
            item.count_method,
            item.cctv_interpretation,
        ])
    draw_report_table(c, rows, (55, 55, 90, 102, 82, 127), MARGIN, 675, row_height=73, fonts=ctx["fonts"])
    draw_wrapped(c, "STEERER는 density와 point를 함께 제공하는 hybrid 후보다. DM-Count·MPCount·CSRNet은 density 기반 zone aggregation에, PET·APGCC는 위치 기반 zone count에 직접 연결할 수 있다.", MARGIN, 112, CONTENT_WIDTH, font=regular, size=8.5, leading=12, max_lines=4)


def _spatial_text(row: dict[str, object]) -> str:
    aggregate = row["aggregates"]
    if row["model_id"] in {"pet", "apgcc"}:
        return f"F1 {_fmt(aggregate['localization_f1'], 2)}"
    if row["model_id"] == "steerer":
        return f"Zone {_fmt(aggregate['mean_zone_mae'], 1)} / F1 {_fmt(aggregate['localization_f1'], 2)}"
    return f"Zone {_fmt(aggregate['mean_zone_mae'], 1)}"


def _page_performance(c, ctx) -> None:
    regular, bold = ctx["fonts"]
    draw_section_title(c, "04", "성능 비교와 해석", "Count 정확도, 공간 출력, 속도와 자원 사용량을 함께 본다.", regular_font=regular, bold_font=bold)
    c.setFont(bold, 8)
    c.setFillColor(NAVY)
    c.drawString(MARGIN, 690, "표 4-1. UCF-QNRF validation 36장 비교 결과")
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
    y = draw_report_table(c, rows, (70, 70, 61, 61, 105, 60, 84), MARGIN, 675, row_height=42, fonts=ctx["fonts"])
    _section_label(c, "4.1 결과 해석", MARGIN, y - 28, bold_font=bold)
    analysis = (
        "STEERER는 MAE 71.6, RMSE 98.0, zone MAE 19.8, localization F1 0.80으로 출력 기능과 공간 정확도의 균형이 가장 좋았다.",
        "DM-Count는 MAE 154.3이지만 14.4 FPS와 8.8 GB 수준의 VRAM으로 density 기반 CCTV baseline 후보가 된다.",
        "PET는 MAE 76.4와 17.2 FPS를 기록했으나 연구 전용 권리 범위 때문에 제품 검증 후보에서 분리한다.",
        "CSRNet은 비교 baseline으로 의미가 있으나 MAE 332.3, zone MAE 91.1, 약 20.4 GB VRAM으로 현 조건의 우선순위는 낮다.",
    )
    _bullets(c, analysis, MARGIN, y - 58, CONTENT_WIDTH, regular_font=regular, bold_font=bold, leading=18)
    c.setFont(regular, 7)
    c.setFillColor(RED)
    c.drawString(MARGIN, 62, "주: 기술 점수는 운영 목적의 합성 점수이며 정확도 백분율이나 공식 benchmark 순위가 아니다.")


def _page_density(c, ctx) -> None:
    regular, bold = ctx["fonts"]
    draw_section_title(c, "05", "Density 계열 출력 사례", "동일 장면 img_0097에서 heatmap과 zone 집계 형태를 비교한다.", regular_font=regular, bold_font=bold)
    group = ctx["assets"]["comparisons"]["density"]
    layout = (
        ("dm-count", MARGIN, 420, "그림 5-1"),
        ("steerer", 303, 420, "그림 5-2"),
        ("mpcount", MARGIN, 188, "그림 5-3"),
        ("csrnet", 303, 188, "그림 5-4"),
    )
    for model_id, x, y, label in layout:
        panel = group[model_id]
        draw_figure(c, ctx["asset_root"] / panel["packaged_path"], f"{label}. {_panel_caption(panel)}", x, y, 250, 205, regular_font=regular)
    draw_wrapped(c, "Density map은 전체 질량의 합으로 count를 만들고, 사전 정의한 CCTV zone 안의 질량을 합산해 구역별 밀집도를 계산할 수 있다. 동일 장면에서도 분포의 선명도와 under/over-count 형태가 다르므로 현장 perspective calibration이 필요하다.", MARGIN, 158, CONTENT_WIDTH, font=regular, size=8.5, leading=12, max_lines=4)


def _page_points(c, ctx) -> None:
    regular, bold = ctx["fonts"]
    draw_section_title(c, "06", "Point/Hybrid 계열 출력 사례", "동일 장면 img_0062에서 위치 후보와 zone 연결성을 비교한다.", regular_font=regular, bold_font=bold)
    group = ctx["assets"]["comparisons"]["points"]
    top = (("steerer", MARGIN, "그림 6-1"), ("pet", 303, "그림 6-2"))
    for model_id, x, label in top:
        panel = group[model_id]
        draw_figure(c, ctx["asset_root"] / panel["packaged_path"], f"{label}. {_panel_caption(panel)}", x, 425, 250, 200, regular_font=regular)
    panel = group["apgcc"]
    draw_figure(c, ctx["asset_root"] / panel["packaged_path"], f"그림 6-3. {_panel_caption(panel)}", MARGIN, 202, CONTENT_WIDTH, 190, regular_font=regular)
    draw_wrapped(c, "Point 출력은 예측 위치 개수로 count를 만들고 좌표를 zone에 직접 할당할 수 있다. 다만 고밀도·가림·tiny person에서 누락과 중복이 발생할 수 있다. STEERER는 density와 point를 함께 제공하며, PET는 연구 비교군, APGCC는 상용 권리 확인 전 후보 상태다.", MARGIN, 170, CONTENT_WIDTH, font=regular, size=8.5, leading=12, max_lines=4)


def _status_short(status: str) -> str:
    return {
        "REVIEWED_NO_EXPLICIT_PROHIBITION": "검토 / 명시 금지 없음",
        "PENDING": "확인 필요",
        "RESTRICTED": "제한",
    }.get(status, status)


def _page_rights(c, ctx) -> None:
    regular, bold = ctx["fonts"]
    draw_section_title(c, "07", "상용화 권리 및 CCTV 적용 판단", "기술 성능과 component-level 권리를 독립적으로 판단한다.", regular_font=regular, bold_font=bold)
    c.setFont(bold, 8)
    c.setFillColor(NAVY)
    c.drawString(MARGIN, 690, "표 7-1. 코드·데이터·weight·deployment 권리 상태")
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
    y = draw_report_table(c, rows, (50, 70, 70, 74, 67, 67, 113), MARGIN, 675, row_height=47, fonts=ctx["fonts"])
    _section_label(c, "7.1 고정형 CCTV 적용", MARGIN, y - 28, bold_font=bold)
    guidance = (
        "카메라마다 ROI, perspective, 실제 zone 면적과 density threshold를 사전 calibration한다.",
        "인접 CCTV overlap에는 ownership mask를 고정해 한쪽 카메라만 count하도록 하고, 경계 통과는 시간 기반 보정 규칙으로 검증한다.",
        "STEERER와 DM-Count는 제품 검증 후보지만 pretrained/derived weight와 deployment 권리가 모두 확인되기 전에는 배포하지 않는다.",
        "PET는 code와 pretrained weight 제한 때문에 연구 비교군으로 유지한다.",
    )
    _bullets(c, guidance, MARGIN, y - 58, CONTENT_WIDTH, regular_font=regular, bold_font=bold, leading=18)


def _page_conclusion(c, ctx) -> None:
    regular, bold = ctx["fonts"]
    draw_section_title(c, "08", "최종 추천과 다음 단계", "Fine-tuning보다 먼저 CCTV calibration과 field validation을 진행한다.", regular_font=regular, bold_font=bold)
    _section_label(c, "8.1 최종 추천", MARGIN, 690, bold_font=bold)
    recommendation = (
        "제품 검증 1순위: STEERER - 낮은 count/zone 오차와 density+point 출력의 조합.",
        "제품 검증 2순위: DM-Count - 해석 가능한 density heatmap과 비교적 높은 처리 속도.",
        "연구 비교군: PET - count와 localization은 유효하지만 권리 제한으로 제품 lane에서 제외.",
        "MPCount, APGCC, CSRNet은 현재 cross-domain 결과에서 보조 비교군으로 유지.",
    )
    y = _bullets(c, recommendation, MARGIN, 660, CONTENT_WIDTH, regular_font=regular, bold_font=bold, leading=19)
    _section_label(c, "8.2 실행 순서", MARGIN, y - 5, bold_font=bold)
    roadmap = (
        "고정형 CCTV별 ROI, perspective, zone 면적과 density threshold calibration",
        "여러 CCTV의 overlap ownership과 중복 count 제거 규칙 검증",
        "실제 행사 또는 유사 CCTV 영상에서 시간대·조도·거리·고밀도 변화 평가",
        "고정형 CCTV 검증 후 UP-COUNT 또는 DroneCrowd 추가 domain 평가",
        "component-level 권리가 적격인 최종 1개 모델만 마지막 단계에서 fine-tuning 검토",
    )
    y = _bullets(c, roadmap, MARGIN, y - 35, CONTENT_WIDTH, regular_font=regular, bold_font=bold, leading=18)
    _section_label(c, "8.3 증거와 범위", MARGIN, y - 5, bold_font=bold)
    dataset = ctx["comparison"]["dataset"]
    evidence = (
        f"Dataset / split: {dataset['dataset_id']} / {dataset['split_id']} / validation 36장",
        f"Canonical split SHA-256: {dataset['canonical_split_sha256']}",
        f"Comparison JSON SHA-256: {sha256_file(ctx['comparison_path'])}",
        "결론 범위: PASS_RESEARCH_ONLY 공유용 검토 | 공식 모델 순위가 아님 | 제품 배포 승인 아님",
    )
    for line in evidence:
        y = draw_wrapped(c, line, MARGIN, y - 20, CONTENT_WIDTH, font=regular, size=7.2, leading=10, color=MUTED, max_lines=2)


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
        "fonts": fonts,
        "report_date": report_date,
    }
    renderers: list[tuple[str, Callable[[canvas_module.Canvas, dict[str, object]], None]]] = [
        ("표지", _page_cover),
        ("요약 및 핵심 결론", _page_summary),
        ("검증 목적·데이터·조건", _page_evaluation),
        ("모델 구조와 입출력 비교", _page_models),
        ("성능 비교와 해석", _page_performance),
        ("Density 계열 출력", _page_density),
        ("Point/Hybrid 계열 출력", _page_points),
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
