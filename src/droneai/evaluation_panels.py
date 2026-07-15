from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, PngImagePlugin

from droneai.evaluation_contract import (
    EvaluationSample,
    NativePrediction,
    ScalarEvaluation,
)


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
    if prediction.failure_state is not None:
        raise ValueError("zone counts unavailable for failed prediction")

    counts: dict[str, float] = {}
    if prediction.density is not None:
        density = np.asarray(prediction.density)
        map_height, map_width = density.shape
        for zone in sample.zones:
            x0 = round(zone.x0 * map_width / sample.width)
            x1 = round(zone.x1 * map_width / sample.width)
            y0 = round(zone.y0 * map_height / sample.height)
            y1 = round(zone.y1 * map_height / sample.height)
            counts[zone.zone_id] = float(density[y0:y1, x0:x1].sum())
        return counts

    if prediction.output_type == "points" or prediction.points:
        for zone in sample.zones:
            counts[zone.zone_id] = float(
                sum(zone.contains(x, y) for x, y in prediction.points)
            )
        return counts

    raise ValueError("zone counts unavailable without native spatial output")


def _unavailable_view(size: tuple[int, int], message: str) -> Image.Image:
    output = Image.new("RGB", size, "black")
    ImageDraw.Draw(output).text((8, size[1] // 2), message, fill="white")
    return output


def _format_optional(value: float | None, places: int) -> str:
    if value is None:
        return "unavailable"
    return f"{value:.{places}f}"


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

    if sample.ground_truth_density is not None:
        ground_truth = render_heatmap(
            sample.ground_truth_density,
            column_size,
            content_box=content_box,
        )
        ground_truth_label = "ground-truth density"
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
    elif prediction.density is not None:
        predicted = render_heatmap(
            prediction.density,
            column_size,
            content_box=content_box,
        )
        prediction_label = "native predicted density"
    elif prediction.output_type == "points" or prediction.points:
        predicted = draw_points(
            source,
            prediction.points,
            scale=image_scale,
            offset_x=offset_x,
            offset_y=offset_y,
            color="red",
        )
        prediction_label = "native predicted points"
    else:
        prediction_label = "native spatial output unavailable"
        predicted = _unavailable_view(column_size, prediction_label)

    operator = source.copy()
    operator_draw = ImageDraw.Draw(operator)
    has_native_spatial_output = prediction.failure_state is None and (
        prediction.density is not None
        or prediction.output_type == "points"
        or bool(prediction.points)
    )
    if not sample.zones:
        counts = None
        operator_label = "operator zones unavailable: none declared"
    elif not has_native_spatial_output:
        counts = None
        operator_label = "operator zones unavailable"
    else:
        counts = zone_counts(sample, prediction)
        operator_label = "operator zones"
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

    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    png_info = PngImagePlugin.PngInfo()
    png_info.add_text("panel_columns", " | ".join(label for _, label in columns))
    png_info.add_text("panel_annotations", "\n".join(annotation_lines))
    canvas.save(target, format="PNG", optimize=True, pnginfo=png_info)
    return target
