from __future__ import annotations

from typing import Tuple

import numpy as np

from ..calibration.service import CalibrationService
from ..config import ProjectionConfig
from ..geometry import TableGeometry
from ..schemas import OverlayLine, ProjectionOverlay

OUTLINE_COLOR = (255, 255, 255)
INLINE_COLOR = (255, 255, 0)
POCKET_COLOR = (0, 180, 255)


def build_geometry_reference_overlay(
    config: ProjectionConfig,
    calibration: CalibrationService,
    geometry: TableGeometry,
    *,
    frame_size: Tuple[int, int],
    frame_id: int = 0,
) -> ProjectionOverlay:
    size = (int(config.projector_width), int(config.projector_height))
    overlay = ProjectionOverlay(overlay_id="geometry_reference", frame_id=int(frame_id), projector_size=size)
    if geometry.is_empty:
        return overlay

    frame_w = max(1, int(frame_size[0]))
    frame_h = max(1, int(frame_size[1]))
    outline, inline_lines, pockets = geometry.reference_scaled(frame_w, frame_h)

    if outline.shape[0] >= 2:
        _append_camera_polyline(
            overlay,
            calibration,
            outline,
            size,
            (frame_w, frame_h),
            color=OUTLINE_COLOR,
            width=3,
            label="outline",
            close=outline.shape[0] >= 3,
        )
    for idx, line in enumerate(inline_lines):
        _append_camera_polyline(
            overlay,
            calibration,
            line,
            size,
            (frame_w, frame_h),
            color=INLINE_COLOR,
            width=2,
            label=f"inline_{idx}",
            close=False,
        )
    for idx, pocket in enumerate(pockets):
        _append_camera_polyline(
            overlay,
            calibration,
            pocket,
            size,
            (frame_w, frame_h),
            color=POCKET_COLOR,
            width=2,
            label=f"pocket_{idx}",
            close=pocket.shape[0] >= 3,
        )
    return overlay


def _append_camera_polyline(
    overlay: ProjectionOverlay,
    calibration: CalibrationService,
    points_cam: np.ndarray,
    projector_size: Tuple[int, int],
    frame_size: Tuple[int, int],
    *,
    color: Tuple[int, int, int],
    width: int,
    label: str,
    close: bool,
) -> None:
    points = np.asarray(points_cam, dtype=np.float32).reshape((-1, 2))
    if points.shape[0] < 2:
        return
    if calibration.projection.is_valid:
        points_proj = calibration.camera_px_to_projector_px(points)
    else:
        points_proj = _fallback_camera_to_projector(points, projector_size, frame_size)
    if close and points_proj.shape[0] >= 3 and not np.allclose(points_proj[0], points_proj[-1]):
        points_proj = np.vstack([points_proj, points_proj[0]])
    overlay.lines.append(
        OverlayLine(
            points=[(float(x), float(y)) for x, y in points_proj],
            color=color,
            width=width,
            label=label,
        )
    )


def _fallback_camera_to_projector(points: np.ndarray, projector_size: Tuple[int, int], frame_size: Tuple[int, int]) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float32).reshape((-1, 2)).copy()
    if pts.shape[0] == 0:
        return pts
    scale_x = float(projector_size[0]) / max(1.0, float(frame_size[0]))
    scale_y = float(projector_size[1]) / max(1.0, float(frame_size[1]))
    pts[:, 0] *= scale_x
    pts[:, 1] *= scale_y
    return pts
