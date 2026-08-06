from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


RESIDUAL_VECTOR_SCALE = 12.0


def render_training_residual_bubble_chart(
    table_width_mm: float,
    table_height_mm: float,
    training_report: Mapping[str, Any],
    *,
    output_size: tuple[int, int] = (1640, 790),
) -> np.ndarray:
    """Render training residuals as a table-coordinate bubble chart.

    Position is the training target, bubble area grows with residual magnitude,
    and the three colors match the operator thresholds used in the reference
    view: <2 mm, 2-5 mm, and >=5 mm.
    """

    width, height = max(1000, int(output_size[0])), max(560, int(output_size[1]))
    canvas = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    title_font = _chart_font(30, bold=True)
    axis_font = _chart_font(23, bold=True)
    tick_font = _chart_font(21)
    legend_font = _chart_font(20)

    left, right = 170, width - 55
    top, bottom = 150, height - 162
    x_max = max(650.0, np.ceil(float(table_width_mm) / 650.0) * 650.0)
    y_max = max(400.0, np.ceil(float(table_height_mm) / 400.0) * 400.0)
    x_ticks = np.linspace(0.0, x_max, 5)
    y_ticks = np.linspace(0.0, y_max, 5)

    ink = (31, 35, 38)
    muted = (132, 136, 139)
    grid = (226, 228, 230)
    samples = list(training_report.get("samples", []))
    draw.text((50, 54), f"{len(samples)}个训练点的空间残差", font=title_font, fill=ink)
    for value in y_ticks:
        y = _chart_y(value, y_max, top, bottom)
        draw.line((left, y, right, y), fill=grid, width=2)
        label = f"{int(round(value)):,}"
        bbox = draw.textbbox((0, 0), label, font=tick_font)
        draw.text((left - 16 - (bbox[2] - bbox[0]), y - 14), label, font=tick_font, fill=muted)
    for value in x_ticks:
        x = _chart_x(value, x_max, left, right)
        label = f"{int(round(value))}"
        bbox = draw.textbbox((0, 0), label, font=tick_font)
        draw.text((x - (bbox[2] - bbox[0]) / 2, bottom + 14), label, font=tick_font, fill=muted)

    x_label = "台面 X (mm)"
    x_bbox = draw.textbbox((0, 0), x_label, font=axis_font)
    draw.text(((width - (x_bbox[2] - x_bbox[0])) / 2, bottom + 48), x_label, font=axis_font, fill=ink)
    _draw_rotated_axis_label(canvas, "台面 Y (mm)", axis_font, x=50, center_y=(top + bottom) // 2)

    for item in samples:
        target = np.asarray(item.get("target_table_mm", [0.0, 0.0]), dtype=np.float64).reshape((2,))
        error = max(0.0, float(item.get("error_mm", 0.0)))
        x = _chart_x(target[0], x_max, left, right)
        y = _chart_y(target[1], y_max, top, bottom)
        radius = 8.0 + 1.25 * min(error, 16.0)
        color = residual_bucket_color_rgb(error)
        outline = (124, 51, 12) if error >= 5.0 else color
        draw.ellipse(
            (x - radius, y - radius, x + radius, y + radius),
            fill=color,
            outline=outline,
            width=2 if error >= 5.0 else 1,
        )

    legend = [
        ("2–5 mm", (49, 151, 244), 7),
        ("≥5 mm", (163, 65, 14), 7),
        ("<2 mm", (0, 108, 54), 7),
    ]
    total_width = sum(32 + draw.textlength(label, font=legend_font) for label, _, _ in legend) + 48
    cursor = (width - total_width) / 2
    legend_y = height - 52
    for label, color, radius in legend:
        draw.ellipse((cursor, legend_y - radius, cursor + 2 * radius, legend_y + radius), fill=color)
        cursor += 24
        draw.text((cursor, legend_y - 14), label, font=legend_font, fill=muted)
        cursor += draw.textlength(label, font=legend_font) + 42

    return cv2.cvtColor(np.asarray(canvas, dtype=np.uint8), cv2.COLOR_RGB2BGR)


def render_projector_ball_residual_view(
    calibration,
    training_report: Mapping[str, Any],
    *,
    projector_size: tuple[int, int],
    holdout_report: Mapping[str, Any] | None = None,
) -> np.ndarray:
    """Render all residual locations in projector coordinates."""

    width, height = (max(1, int(projector_size[0])), max(1, int(projector_size[1])))
    image = np.zeros((height, width, 3), dtype=np.uint8)
    _draw_projected_table_outline(image, calibration)
    _draw_report_on_projector(
        image,
        calibration,
        training_report,
        prefix="T",
        marker="circle",
    )
    if holdout_report:
        _draw_report_on_projector(
            image,
            calibration,
            holdout_report,
            prefix="H",
            marker="diamond",
        )
    _draw_legend(image, training_report, holdout_report)
    return image


def render_table_ball_residual_view(
    table_width_mm: float,
    table_height_mm: float,
    training_report: Mapping[str, Any],
    *,
    holdout_report: Mapping[str, Any] | None = None,
    output_size: tuple[int, int] = (1600, 900),
) -> np.ndarray:
    """Render a top-down residual map suitable for saving as a diagnostic PNG."""

    width, height = max(640, int(output_size[0])), max(360, int(output_size[1]))
    image = np.full((height, width, 3), (18, 18, 18), dtype=np.uint8)
    margin_x, margin_y = 90, 90
    table_left, table_top = margin_x, margin_y
    table_right, table_bottom = width - margin_x, height - margin_y
    cv2.rectangle(image, (table_left, table_top), (table_right, table_bottom), (55, 115, 70), -1)
    cv2.rectangle(image, (table_left, table_top), (table_right, table_bottom), (190, 210, 195), 2)

    def project(points: np.ndarray) -> np.ndarray:
        pts = np.asarray(points, dtype=np.float64).reshape((-1, 2))
        x = table_left + pts[:, 0] / max(1.0, float(table_width_mm)) * (table_right - table_left)
        y = table_top + pts[:, 1] / max(1.0, float(table_height_mm)) * (table_bottom - table_top)
        return np.column_stack((x, y)).astype(np.float32)

    _draw_report(image, training_report, project, prefix="T", marker="circle")
    if holdout_report:
        _draw_report(image, holdout_report, project, prefix="H", marker="diamond")
    _draw_legend(image, training_report, holdout_report)
    return image


def save_residual_view(path: str | Path, image: np.ndarray) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(target), np.asarray(image, dtype=np.uint8)):
        raise OSError(f"failed to save residual view: {target}")
    return target


def residual_color_bgr(error_mm: float) -> tuple[int, int, int]:
    error = float(error_mm)
    if error >= 5.0:
        return 45, 45, 245
    if error >= 2.0:
        return 30, 135, 255
    if error >= 1.0:
        return 35, 215, 240
    return 80, 220, 95


def residual_bucket_color_rgb(error_mm: float) -> tuple[int, int, int]:
    error = float(error_mm)
    if error >= 5.0:
        return 163, 65, 14
    if error >= 2.0:
        return 49, 151, 244
    return 0, 108, 54


def _chart_font(size: int, *, bold: bool = False):
    candidates = [
        Path("C:/Windows/Fonts/msyhbd.ttc" if bold else "C:/Windows/Fonts/msyh.ttc"),
        Path("C:/Windows/Fonts/simhei.ttf"),
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return ImageFont.truetype(str(candidate), int(size))
    return ImageFont.load_default()


def _chart_x(value: float, maximum: float, left: int, right: int) -> float:
    return float(left) + np.clip(float(value) / max(1.0, float(maximum)), 0.0, 1.0) * float(right - left)


def _chart_y(value: float, maximum: float, top: int, bottom: int) -> float:
    return float(bottom) - np.clip(float(value) / max(1.0, float(maximum)), 0.0, 1.0) * float(bottom - top)


def _draw_rotated_axis_label(canvas: Image.Image, text: str, font, *, x: int, center_y: int) -> None:
    bbox = font.getbbox(text)
    label = Image.new("RGBA", (bbox[2] - bbox[0] + 12, bbox[3] - bbox[1] + 12), (255, 255, 255, 0))
    ImageDraw.Draw(label).text((6 - bbox[0], 6 - bbox[1]), text, font=font, fill=(31, 35, 38, 255))
    rotated = label.rotate(90, expand=True, resample=Image.Resampling.BICUBIC)
    canvas.paste(rotated, (int(x - rotated.width / 2), int(center_y - rotated.height / 2)), rotated)


def _draw_projected_table_outline(image: np.ndarray, calibration) -> None:
    corners = np.asarray(
        [
            [0.0, 0.0],
            [float(calibration.table.width_mm), 0.0],
            [float(calibration.table.width_mm), float(calibration.table.height_mm)],
            [0.0, float(calibration.table.height_mm)],
        ],
        dtype=np.float32,
    )
    projected = calibration.table_mm_to_projector_px(corners)
    cv2.polylines(image, [np.rint(projected).astype(np.int32)], True, (120, 170, 130), 2, cv2.LINE_AA)


def _draw_report_on_projector(
    image: np.ndarray,
    calibration,
    report: Mapping[str, Any],
    *,
    prefix: str,
    marker: str,
) -> None:
    _draw_report(
        image,
        report,
        lambda points: calibration.table_mm_to_projector_px(np.asarray(points, dtype=np.float32)),
        prefix=prefix,
        marker=marker,
    )


def _draw_report(image, report, project, *, prefix: str, marker: str) -> None:
    samples = list(report.get("samples", []))
    if not samples:
        return
    targets = np.asarray([item["target_table_mm"] for item in samples], dtype=np.float64)
    vectors = np.asarray([item["error_vector_mm"] for item in samples], dtype=np.float64)
    starts = project(targets)
    ends = project(targets + vectors * RESIDUAL_VECTOR_SCALE)
    for display_index, (item, start, end) in enumerate(zip(samples, starts, ends), start=1):
        p0 = tuple(np.rint(start).astype(int))
        p1 = tuple(np.rint(end).astype(int))
        error = float(item.get("error_mm", 0.0))
        color = residual_color_bgr(error)
        cv2.arrowedLine(image, p0, p1, color, 2, cv2.LINE_AA, tipLength=0.25)
        if marker == "diamond":
            x, y = p0
            diamond = np.asarray([[x, y - 7], [x + 7, y], [x, y + 7], [x - 7, y]], dtype=np.int32)
            cv2.polylines(image, [diamond], True, color, 2, cv2.LINE_AA)
        else:
            cv2.circle(image, p0, 5, color, 2, cv2.LINE_AA)
        sample_index = (
            int(item.get("sample_index", display_index - 1)) + 1
            if prefix == "T"
            else display_index
        )
        label = f"{prefix}{sample_index}:{error:.1f}"
        label_x = max(3, min(image.shape[1] - 70, p0[0] + 7))
        label_y_offset = 17 if marker == "diamond" else -7
        label_y = max(13, min(image.shape[0] - 3, p0[1] + label_y_offset))
        cv2.putText(image, label, (label_x, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.38, color, 1, cv2.LINE_AA)


def _draw_legend(
    image: np.ndarray,
    training_report: Mapping[str, Any],
    holdout_report: Mapping[str, Any] | None,
) -> None:
    training_error = dict(training_report.get("error_mm", {}))
    title = (
        f"Training {int(training_report.get('sample_count', 0))}: "
        f"mean {float(training_error.get('mean', 0.0)):.2f}  "
        f"median {float(training_error.get('median', 0.0)):.2f}  "
        f"P95 {float(training_error.get('p95', 0.0)):.2f}  "
        f"max {float(training_error.get('max', 0.0)):.2f} mm"
    )
    cv2.putText(image, title, (24, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (235, 235, 235), 2, cv2.LINE_AA)
    if holdout_report:
        holdout_error = dict(holdout_report.get("error_mm", {}))
        holdout_title = (
            f"Reused Holdout {int(holdout_report.get('observation_count', holdout_report.get('sample_count', 0)))} "
            f"observations / {int(holdout_report.get('sample_count', 0))} locations: "
            f"median {float(holdout_error.get('median', 0.0)):.2f}  "
            f"P95 {float(holdout_error.get('p95', 0.0)):.2f} mm (diagnostic only)"
        )
        cv2.putText(image, holdout_title, (24, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (205, 205, 205), 1, cv2.LINE_AA)
    legend = [("<1", 0.5), ("1-2", 1.5), ("2-5", 3.0), (">=5 mm", 5.0)]
    x = max(24, image.shape[1] - 430)
    for label, value in legend:
        color = residual_color_bgr(value)
        cv2.circle(image, (x, 30), 6, color, -1, cv2.LINE_AA)
        cv2.putText(image, label, (x + 10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (230, 230, 230), 1, cv2.LINE_AA)
        x += 95
    cv2.putText(
        image,
        f"arrows x{RESIDUAL_VECTOR_SCALE:.0f}",
        (24, image.shape[0] - 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (210, 210, 210),
        1,
        cv2.LINE_AA,
    )
