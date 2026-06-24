from __future__ import annotations

from typing import Tuple

import cv2
import numpy as np

from ..geometry import TableGeometry

OUTLINE_COLOR = (255, 255, 255)
INNER_COLOR = (0, 255, 128)
INLINE_COLOR = (0, 255, 255)
POCKET_COLOR = (0, 180, 255)


def draw_geometry_reference_lines(image: np.ndarray, geometry: TableGeometry, *, enabled: bool = True) -> int:
    if not enabled or image is None or image.size == 0 or geometry.is_empty:
        return 0
    height, width = image.shape[:2]
    outline, inline_lines, pockets = geometry.reference_scaled(width, height)
    inner, _ = geometry.boundary_scaled(width, height)
    thickness = max(2, int(round(min(width, height) / 540.0)))
    drawn = 0
    if _draw_polyline(image, outline, OUTLINE_COLOR, thickness + 1, close=outline.shape[0] >= 3):
        drawn += 1
    if _draw_polyline(image, inner, INNER_COLOR, thickness, close=inner.shape[0] >= 3):
        drawn += 1
    for line in inline_lines:
        if _draw_polyline(image, line, INLINE_COLOR, thickness, close=False):
            drawn += 1
    for pocket in pockets:
        if _draw_polyline(image, pocket, POCKET_COLOR, thickness, close=False):
            drawn += 1
    return drawn


def _draw_polyline(
    image: np.ndarray,
    points: np.ndarray,
    color: Tuple[int, int, int],
    thickness: int,
    *,
    close: bool,
) -> bool:
    pts = np.asarray(points, dtype=np.float32).reshape((-1, 2))
    if pts.shape[0] < 2:
        return False
    cv2.polylines(
        image,
        [np.round(pts).astype(np.int32).reshape((-1, 1, 2))],
        isClosed=bool(close),
        color=color,
        thickness=max(1, int(thickness)),
        lineType=cv2.LINE_AA,
    )
    return True
