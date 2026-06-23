from __future__ import annotations

from typing import List, Optional, Tuple

import cv2
import numpy as np

from ..calibration.service import CalibrationService
from ..config import ProjectionConfig
from ..schemas import OverlayLine, ProjectionOverlay, ShotCandidate, ShotPlan
from ..utils import wall_time_id


class OverlayBuilder:
    def __init__(self, config: ProjectionConfig, calibration: CalibrationService):
        self.config = config
        self.calibration = calibration

    def from_plan(self, plan: ShotPlan) -> ProjectionOverlay:
        size = (int(self.config.projector_width), int(self.config.projector_height))
        overlay = ProjectionOverlay(
            overlay_id=f"overlay_{plan.frame_id}_{wall_time_id()}",
            frame_id=plan.frame_id,
            projector_size=size,
        )
        if plan.best is None:
            return overlay
        candidate = plan.best
        aim = self._table_line_to_projector(candidate.aim_line)
        obj = self._table_line_to_projector(candidate.object_line)
        if len(aim) >= 2:
            overlay.lines.append(OverlayLine(points=aim, color=(0, 240, 160), width=4, label="aim"))
        if len(obj) >= 2:
            overlay.lines.append(OverlayLine(points=obj, color=(80, 180, 255), width=4, label="object"))
        circles = self.calibration.table_mm_to_projector_px(
            np.asarray([candidate.ghost_ball, candidate.object_ball, candidate.pocket_point], dtype=np.float32)
        )
        overlay.circles.append(((float(circles[0, 0]), float(circles[0, 1])), 10.0, (0, 255, 180)))
        overlay.circles.append(((float(circles[1, 0]), float(circles[1, 1])), 8.0, (255, 220, 80)))
        overlay.circles.append(((float(circles[2, 0]), float(circles[2, 1])), 12.0, (255, 255, 255)))
        label_pos = (float(circles[0, 0] + 14), float(circles[0, 1] - 14))
        overlay.labels.append((label_pos, f"{candidate.score:.2f}", (220, 255, 220)))
        return overlay

    def _table_line_to_projector(self, points) -> List[Tuple[float, float]]:
        arr = self.calibration.table_mm_to_projector_px(np.asarray(points, dtype=np.float32))
        return [(float(x), float(y)) for x, y in arr]


def render_overlay_image(overlay: ProjectionOverlay, background: Optional[np.ndarray] = None) -> np.ndarray:
    width, height = overlay.projector_size
    if background is None:
        img = np.zeros((height, width, 3), dtype=np.uint8)
    else:
        img = cv2.resize(background, (width, height)).copy()
    table_color = (20, 80, 44)
    if background is None:
        img[:] = table_color
    for line in overlay.lines:
        pts = np.asarray(line.points, dtype=np.float32).reshape((-1, 2))
        if pts.shape[0] >= 2:
            cv2.polylines(
                img,
                [np.round(pts).astype(np.int32).reshape((-1, 1, 2))],
                isClosed=False,
                color=tuple(int(c) for c in line.color),
                thickness=max(1, int(line.width)),
                lineType=cv2.LINE_AA,
            )
            _draw_arrow_head(img, pts[-2], pts[-1], tuple(int(c) for c in line.color), max(1, int(line.width)))
    for center, radius, color in overlay.circles:
        cv2.circle(
            img,
            (int(round(center[0])), int(round(center[1]))),
            max(1, int(round(radius))),
            tuple(int(c) for c in color),
            2,
            cv2.LINE_AA,
        )
    for pos, text, color in overlay.labels:
        cv2.putText(
            img,
            text,
            (int(round(pos[0])), int(round(pos[1]))),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            tuple(int(c) for c in color),
            1,
            cv2.LINE_AA,
        )
    return img


def _draw_arrow_head(img: np.ndarray, a: np.ndarray, b: np.ndarray, color: Tuple[int, int, int], width: int) -> None:
    v = np.asarray(b, dtype=np.float32) - np.asarray(a, dtype=np.float32)
    n = float(np.linalg.norm(v))
    if n < 1e-6:
        return
    d = v / n
    perp = np.asarray([-d[1], d[0]], dtype=np.float32)
    size = max(12.0, 5.0 * width)
    p1 = np.asarray(b, dtype=np.float32)
    p2 = p1 - d * size + perp * size * 0.45
    p3 = p1 - d * size - perp * size * 0.45
    cv2.fillConvexPoly(img, np.round(np.vstack([p1, p2, p3])).astype(np.int32), color, cv2.LINE_AA)

