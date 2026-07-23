from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from ..schemas import OverlayText


_FONT_CANDIDATES = (
    Path("C:/Windows/Fonts/msyh.ttc"),
    Path("C:/Windows/Fonts/msyhbd.ttc"),
    Path("C:/Windows/Fonts/simhei.ttf"),
    Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
    Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
)


def draw_overlay_texts(image_bgr: np.ndarray, texts: list[OverlayText]) -> None:
    """Draw Unicode text in final projector pixels, after all geometry calibration."""

    if not texts or image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
        return
    height, width = image_bgr.shape[:2]
    base = Image.fromarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)).convert("RGBA")
    layer = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    for item in texts:
        _draw_text_item(draw, width, height, item)
    composed = Image.alpha_composite(base, layer).convert("RGB")
    image_bgr[:] = cv2.cvtColor(np.asarray(composed), cv2.COLOR_RGB2BGR)


def _draw_text_item(
    draw: ImageDraw.ImageDraw,
    canvas_width: int,
    canvas_height: int,
    item: OverlayText,
) -> None:
    font_size = max(8, int(round(item.font_size_px)))
    font = _font(font_size)
    max_width = max(40, int(round(canvas_width * max(0.1, min(1.0, item.max_width_ratio)))))
    wrapped = _wrap_text(draw, str(item.text), font, max_width)
    spacing = max(2, int(round(font_size * 0.18)))
    stroke = max(0, int(round(item.outline_width_px)))
    bbox = draw.multiline_textbbox(
        (0, 0),
        wrapped,
        font=font,
        spacing=spacing,
        align="center",
        stroke_width=stroke,
    )
    text_width = max(1, int(bbox[2] - bbox[0]))
    text_height = max(1, int(bbox[3] - bbox[1]))
    center_x = int(round(item.position[0]))
    center_y = int(round(item.position[1]))
    left = max(4, min(canvas_width - text_width - 4, center_x - text_width // 2))
    top = max(4, min(canvas_height - text_height - 4, center_y - text_height // 2))

    padding_x = max(8, font_size // 3)
    padding_y = max(5, font_size // 5)
    if item.background_alpha > 0:
        draw.rounded_rectangle(
            (
                max(0, left - padding_x),
                max(0, top - padding_y),
                min(canvas_width - 1, left + text_width + padding_x),
                min(canvas_height - 1, top + text_height + padding_y),
            ),
            radius=max(4, font_size // 4),
            fill=(0, 0, 0, max(0, min(255, int(item.background_alpha)))),
        )

    blue, green, red = (int(value) for value in item.color)
    draw.multiline_text(
        (left - bbox[0], top - bbox[1]),
        wrapped,
        font=font,
        fill=(red, green, blue, 255),
        spacing=spacing,
        align="center",
        stroke_width=stroke,
        stroke_fill=(0, 0, 0, 235),
    )


def _wrap_text(
    draw: ImageDraw.ImageDraw,
    text: str,
    font: ImageFont.FreeTypeFont | ImageFont.ImageFont,
    max_width: int,
) -> str:
    wrapped_lines: list[str] = []
    for source_line in text.splitlines() or [""]:
        current = ""
        for char in source_line:
            candidate = current + char
            bbox = draw.textbbox((0, 0), candidate, font=font)
            if current and (bbox[2] - bbox[0]) > max_width:
                wrapped_lines.append(current)
                current = char
            else:
                current = candidate
        wrapped_lines.append(current)
    return "\n".join(wrapped_lines)


@lru_cache(maxsize=64)
def _font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for path in _FONT_CANDIDATES:
        if not path.is_file():
            continue
        try:
            return ImageFont.truetype(str(path), size=size)
        except OSError:
            continue
    return ImageFont.load_default(size=size)


__all__ = ["draw_overlay_texts"]
