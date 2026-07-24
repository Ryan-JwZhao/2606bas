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
    for item in texts:
        _draw_text_item(layer, width, height, item)
    composed = Image.alpha_composite(base, layer).convert("RGB")
    image_bgr[:] = cv2.cvtColor(np.asarray(composed), cv2.COLOR_RGB2BGR)


def _draw_text_item(
    layer: Image.Image,
    canvas_width: int,
    canvas_height: int,
    item: OverlayText,
) -> None:
    font_size = max(8, int(round(item.font_size_px)))
    font = _font(font_size)
    max_width = max(40, int(round(canvas_width * max(0.1, min(1.0, item.max_width_ratio)))))
    measure = ImageDraw.Draw(Image.new("L", (1, 1), 0))
    wrapped = _wrap_text(measure, str(item.text), font, max_width)
    spacing = max(2, int(round(font_size * 0.18)))
    stroke = max(0, int(round(item.outline_width_px)))
    bbox = measure.multiline_textbbox(
        (0, 0),
        wrapped,
        font=font,
        spacing=spacing,
        align="center",
        stroke_width=stroke,
    )
    text_width = max(1, int(bbox[2] - bbox[0]))
    text_height = max(1, int(bbox[3] - bbox[1]))
    padding_x = max(8, font_size // 3)
    padding_y = max(5, font_size // 5)
    patch = Image.new(
        "RGBA",
        (text_width + 2 * padding_x, text_height + 2 * padding_y),
        (0, 0, 0, 0),
    )
    draw = ImageDraw.Draw(patch)
    if item.background_alpha > 0:
        draw.rounded_rectangle(
            (0, 0, patch.width - 1, patch.height - 1),
            radius=max(4, font_size // 4),
            fill=(0, 0, 0, max(0, min(255, int(item.background_alpha)))),
        )

    blue, green, red = (int(value) for value in item.color)
    draw.multiline_text(
        (padding_x - bbox[0], padding_y - bbox[1]),
        wrapped,
        font=font,
        fill=(red, green, blue, 255),
        spacing=spacing,
        align="center",
        stroke_width=stroke,
        stroke_fill=(0, 0, 0, 235),
    )
    rotation_deg = float(item.rotation_deg)
    if abs(rotation_deg) > 1e-4:
        patch = patch.rotate(
            -rotation_deg,
            resample=Image.Resampling.BICUBIC,
            expand=True,
        )
    _alpha_composite_centered(
        layer,
        patch,
        center=(int(round(item.position[0])), int(round(item.position[1]))),
        canvas_size=(canvas_width, canvas_height),
    )


def _alpha_composite_centered(
    layer: Image.Image,
    patch: Image.Image,
    *,
    center: tuple[int, int],
    canvas_size: tuple[int, int],
) -> None:
    canvas_width, canvas_height = canvas_size
    left = int(center[0] - patch.width // 2)
    top = int(center[1] - patch.height // 2)
    source_left = max(0, -left)
    source_top = max(0, -top)
    destination_left = max(0, left)
    destination_top = max(0, top)
    source_right = min(patch.width, canvas_width - destination_left + source_left)
    source_bottom = min(patch.height, canvas_height - destination_top + source_top)
    if source_right <= source_left or source_bottom <= source_top:
        return
    visible = patch.crop((source_left, source_top, source_right, source_bottom))
    layer.alpha_composite(visible, (destination_left, destination_top))


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
