from __future__ import annotations

import math
from pathlib import Path
from typing import Literal

import numpy as np
from PIL import Image, ImageDraw, PngImagePlugin

from droneai.evaluation_contract import (
    EvaluationSample,
    NativePrediction,
    ScalarEvaluation,
)

SpatialKind = Literal["density", "points"]


def fit_source(
    image: Image.Image,
    size: tuple[int, int],
) -> tuple[Image.Image, float, float, float]:
    scale = min(size[0] / image.width, size[1] / image.height)
    fitted_size = (
        max(1, round(image.width * scale)),
        max(1, round(image.height * scale)),
    )
    fitted = image.resize(fitted_size, Image.Resampling.LANCZOS)
    offset_x = (size[0] - fitted_size[0]) / 2
    offset_y = (size[1] - fitted_size[1]) / 2
    canvas = Image.new("RGB", size, "black")
    canvas.paste(fitted, (round(offset_x), round(offset_y)))
    return canvas, scale, offset_x, offset_y


def render_heatmap(
    values: np.ndarray,
    size: tuple[int, int],
    *,
    content_box: tuple[int, int, int, int],
) -> Image.Image:
    normalized = np.asarray(values, dtype=np.float32)
    scale = float(normalized.max()) or 1.0
    unit = np.clip(normalized / scale, 0.0, 1.0)
    rgb = np.stack((unit, np.sqrt(unit), 1.0 - unit), axis=2)
    left, top, right, bottom = content_box
    heatmap = Image.fromarray(np.uint8(rgb * 255), "RGB").resize(
        (right - left, bottom - top),
        Image.Resampling.BILINEAR,
    )
    canvas = Image.new("RGB", size, "black")
    canvas.paste(heatmap, (left, top))
    return canvas


def draw_points(
    base: Image.Image,
    points: tuple[tuple[float, float], ...],
    *,
    scale: float,
    offset_x: float,
    offset_y: float,
    color: str,
) -> Image.Image:
    output = base.copy()
    draw = ImageDraw.Draw(output)
    for x, y in points:
        screen_x = offset_x + x * scale
        screen_y = offset_y + y * scale
        draw.ellipse(
            (
                screen_x - 3,
                screen_y - 3,
                screen_x + 3,
                screen_y + 3,
            ),
            outline=color,
            width=2,
        )
    return output


def zone_counts(
    sample: EvaluationSample,
    prediction: NativePrediction,
) -> dict[str, float]:
    spatial_kind = _native_spatial_kind(prediction)
    if spatial_kind is None:
        raise ValueError("zone counts unavailable without native spatial output")

    counts: dict[str, float] = {}
    if spatial_kind == "density":
        if prediction.density is None:
            raise ValueError("declared density output is unavailable")
        density = np.asarray(prediction.density)
        map_height, map_width = density.shape
        for zone in sample.zones:
            x0 = round(zone.x0 * map_width / sample.width)
            x1 = round(zone.x1 * map_width / sample.width)
            y0 = round(zone.y0 * map_height / sample.height)
            y1 = round(zone.y1 * map_height / sample.height)
            counts[zone.zone_id] = float(density[y0:y1, x0:x1].sum())
        return counts

    if spatial_kind == "points":
        for zone in sample.zones:
            counts[zone.zone_id] = float(
                sum(zone.contains(x, y) for x, y in prediction.points)
            )
        return counts

    raise AssertionError("unreachable native spatial kind")


def _native_spatial_kind(prediction: NativePrediction) -> SpatialKind | None:
    if prediction.failure_state is not None or prediction.output_type == "count":
        return None
    if prediction.output_type == "density":
        return "density"
    if prediction.output_type == "points":
        return "points"
    if prediction.density is not None:
        return "density"
    if prediction.points:
        return "points"
    return None


def _unavailable_view(size: tuple[int, int], message: str) -> Image.Image:
    output = Image.new("RGB", size, "black")
    ImageDraw.Draw(output).text((8, size[1] // 2), message, fill="white")
    return output


def _format_optional(value: float | None, places: int) -> str:
    if value is None:
        return "unavailable"
    return f"{value:.{places}f}"


def _require_matching_number(
    name: str,
    record_value: float | None,
    source_value: float | None,
) -> None:
    if record_value is None or source_value is None:
        if record_value is not None or source_value is not None:
            raise ValueError(f"{name} mismatch")
        return
    if not math.isfinite(record_value) or not math.isfinite(source_value):
        raise ValueError(f"{name} must be finite")
    if not math.isclose(
        record_value,
        source_value,
        rel_tol=1e-9,
        abs_tol=1e-9,
    ):
        raise ValueError(f"{name} mismatch")


def _require_finite_number(name: str, value: float) -> None:
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")


def _validate_panel_inputs(
    sample: EvaluationSample,
    prediction: NativePrediction,
    record: ScalarEvaluation,
) -> None:
    if prediction.sample_id != sample.sample_id or record.sample_id != sample.sample_id:
        raise ValueError("sample_id mismatch")
    if record.output_type != prediction.output_type:
        raise ValueError("output_type mismatch")
    if record.failure_state != prediction.failure_state:
        raise ValueError("failure_state mismatch")
    _require_matching_number(
        "ground_truth_count",
        record.ground_truth_count,
        sample.ground_truth_count,
    )
    _require_matching_number(
        "predicted_count",
        record.predicted_count,
        prediction.predicted_count,
    )
    # Panels rerun selected samples to retain native spatial output. Runtime
    # values naturally vary between passes, so display and validate the frozen
    # first-pass record rather than requiring equality with the retained rerun.
    _require_finite_number("latency_ms", record.latency_ms)
    _require_finite_number("peak_vram_mb", record.peak_vram_mb)


def render_review_panel(
    sample: EvaluationSample,
    prediction: NativePrediction,
    record: ScalarEvaluation,
    output_path: str | Path,
    *,
    model_id: str,
    checkpoint_sha256: str,
    zone_warning_count: float,
    zone_critical_count: float,
    category: str | None = None,
) -> Path:
    _validate_panel_inputs(sample, prediction, record)
    target = Path(output_path)
    if target.suffix.lower() != ".png":
        raise ValueError("output path must end with .png")
    if target.resolve(strict=False) == sample.image_path.resolve(strict=False):
        raise ValueError("output path resolves to source image")
    if target.exists():
        raise FileExistsError(f"panel target already exists: {target}")

    column_size = (320, 240)
    with Image.open(sample.image_path) as source_handle:
        source, image_scale, offset_x, offset_y = fit_source(
            source_handle.convert("RGB"),
            column_size,
        )
    content_box = (
        round(offset_x),
        round(offset_y),
        round(offset_x + sample.width * image_scale),
        round(offset_y + sample.height * image_scale),
    )
    prediction_spatial_kind = _native_spatial_kind(prediction)

    point_ground_truth_preferred = prediction.output_type == "points" or (
        prediction.output_type == "hybrid"
        and prediction_spatial_kind == "points"
    )
    if point_ground_truth_preferred and sample.has_point_annotations:
        ground_truth = draw_points(
            source,
            sample.ground_truth_points,
            scale=image_scale,
            offset_x=offset_x,
            offset_y=offset_y,
            color="lime",
        )
        ground_truth_label = "ground-truth points"
    elif sample.ground_truth_density is not None:
        ground_truth = render_heatmap(
            sample.ground_truth_density,
            column_size,
            content_box=content_box,
        )
        ground_truth_label = (
            "derived ground-truth density"
            if sample.has_point_annotations
            else "ground-truth density"
        )
    elif sample.has_point_annotations:
        ground_truth = draw_points(
            source,
            sample.ground_truth_points,
            scale=image_scale,
            offset_x=offset_x,
            offset_y=offset_y,
            color="lime",
        )
        ground_truth_label = "ground-truth points"
    else:
        ground_truth_label = "ground-truth spatial output unavailable"
        ground_truth = _unavailable_view(column_size, ground_truth_label)

    if prediction.failure_state is not None:
        prediction_label = f"prediction unavailable: {prediction.failure_state}"
        predicted = _unavailable_view(column_size, prediction_label)
    elif prediction_spatial_kind == "density":
        if prediction.density is None:
            raise ValueError("declared density output is unavailable")
        predicted = render_heatmap(
            prediction.density,
            column_size,
            content_box=content_box,
        )
        prediction_label = (
            "native predicted density (hybrid)"
            if prediction.output_type == "hybrid"
            else "native predicted density"
        )
    elif prediction_spatial_kind == "points":
        predicted = draw_points(
            source,
            prediction.points,
            scale=image_scale,
            offset_x=offset_x,
            offset_y=offset_y,
            color="red",
        )
        prediction_label = (
            "native predicted points (hybrid)"
            if prediction.output_type == "hybrid"
            else "native predicted points"
        )
    else:
        prediction_label = "native spatial output unavailable"
        predicted = _unavailable_view(column_size, prediction_label)

    operator = source.copy()
    operator_draw = ImageDraw.Draw(operator)
    has_native_spatial_output = prediction_spatial_kind is not None
    if not sample.zones:
        counts = None
        operator_label = "operator zones unavailable: none declared"
    elif not has_native_spatial_output:
        counts = None
        operator_label = "operator zones unavailable"
    else:
        counts = zone_counts(sample, prediction)
        operator_label = (
            f"operator zones (hybrid {prediction_spatial_kind})"
            if prediction.output_type == "hybrid"
            else "operator zones"
        )
    if counts is None:
        for zone in sample.zones:
            box = (
                offset_x + zone.x0 * image_scale,
                offset_y + zone.y0 * image_scale,
                offset_x + zone.x1 * image_scale,
                offset_y + zone.y1 * image_scale,
            )
            operator_draw.rectangle(box, outline="white", width=2)
            operator_draw.text(
                (box[0] + 3, box[1] + 3),
                f"{zone.zone_id}: unavailable",
                fill="white",
            )
    else:
        for zone in sample.zones:
            count = counts[zone.zone_id]
            if count >= zone_critical_count:
                risk, color = "red", "red"
            elif count >= zone_warning_count:
                risk, color = "yellow", "yellow"
            else:
                risk, color = "green", "lime"
            box = (
                offset_x + zone.x0 * image_scale,
                offset_y + zone.y0 * image_scale,
                offset_x + zone.x1 * image_scale,
                offset_y + zone.y1 * image_scale,
            )
            operator_draw.rectangle(box, outline=color, width=2)
            operator_draw.text(
                (box[0] + 3, box[1] + 3),
                f"{zone.zone_id}: {count:.1f} {risk}",
                fill=color,
            )

    canvas = Image.new("RGB", (4 * column_size[0], 365), "white")
    columns = (
        (source, "input"),
        (ground_truth, ground_truth_label),
        (predicted, prediction_label),
        (operator, operator_label),
    )
    draw = ImageDraw.Draw(canvas)
    for index, (image, label) in enumerate(columns):
        x = index * column_size[0]
        canvas.paste(image, (x, 30))
        draw.text((x + 4, 8), label, fill="black")

    spatial = (
        "spatial=unavailable"
        if record.spatial_metric_name is None
        or record.spatial_metric_value is None
        else f"{record.spatial_metric_name}={record.spatial_metric_value:.3f}"
    )
    confidence = (
        "confidence=unavailable"
        if prediction.confidence is None
        else f"confidence={prediction.confidence:.3f}"
    )
    failure = prediction.failure_state or "none"
    identity_line = (
        f"model={model_id} dataset={sample.dataset_id} "
        f"checkpoint={checkpoint_sha256[:12]} sample={sample.sample_id} "
        f"input={sample.width}x{sample.height}"
    )
    count_error_line = (
        f"category={category or 'unavailable'} density_band={record.density_band} "
        f"gt={sample.ground_truth_count:.2f} "
        f"pred={_format_optional(prediction.predicted_count, 2)} "
        f"signed_error={_format_optional(record.signed_error, 2)} "
        f"abs_error={_format_optional(record.absolute_error, 2)} "
        f"normalized_error={_format_optional(record.normalized_error, 3)}"
    )
    runtime_line = (
        f"output_type={prediction.output_type} {spatial} {confidence} "
        f"failure={failure} latency_ms={record.latency_ms:.2f} "
        f"peak_vram_mb={record.peak_vram_mb:.1f}"
    )
    annotation_lines = (identity_line, count_error_line, runtime_line)
    for y, line in zip((280, 305, 330), annotation_lines):
        draw.text((4, y), line, fill="black")

    target.parent.mkdir(parents=True, exist_ok=True)
    png_info = PngImagePlugin.PngInfo()
    png_info.add_text("panel_columns", " | ".join(label for _, label in columns))
    png_info.add_text("panel_annotations", "\n".join(annotation_lines))
    try:
        with target.open("xb") as target_handle:
            canvas.save(
                target_handle,
                format="PNG",
                optimize=True,
                pnginfo=png_info,
            )
    except FileExistsError as error:
        raise FileExistsError(f"panel target already exists: {target}") from error
    return target
