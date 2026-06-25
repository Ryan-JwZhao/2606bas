from __future__ import annotations

import math
from typing import List, Optional, Tuple

import cv2
import numpy as np

from ..calibration.service import CalibrationService
from ..config import ProjectionConfig
from ..route_geometry import cue_alignment_start, estimate_route_end, rule_cue_separation_end
from ..schemas import FreeRouteSuggestion, OverlayLine, ProjectionOverlay, ShotCandidate, ShotPlan
from ..utils import unit, wall_time_id
from .star_formula import StarFormulaConfig, draw_star_formula

ROUTE_COLOR = (255, 255, 255)
CUE_STICK_COLOR = ROUTE_COLOR


class OverlayBuilder:
    def __init__(self, config: ProjectionConfig, calibration: CalibrationService, star_formula: StarFormulaConfig | None = None):
        self.config = config
        self.calibration = calibration
        self.star_formula = star_formula or StarFormulaConfig()

    def from_plan(self, plan: ShotPlan) -> ProjectionOverlay:
        size = (int(self.config.projector_width), int(self.config.projector_height))
        overlay = ProjectionOverlay(
            overlay_id=f"overlay_{plan.frame_id}_{wall_time_id()}",
            frame_id=plan.frame_id,
            projector_size=size,
        )
        if plan.shot_mode == "free":
            if plan.free_route is not None:
                self._add_free_route(overlay, plan.free_route)
            return overlay
        if plan.best is None:
            return overlay
        self._add_rule_route(overlay, plan.best)
        return overlay

    def _add_rule_route(self, overlay: ProjectionOverlay, candidate: ShotCandidate) -> None:
        ball_radius = 0.5 * float(self.calibration.table.ball_diameter_mm)
        cue = np.asarray(candidate.cue_ball, dtype=np.float32)
        ghost = np.asarray(candidate.ghost_ball, dtype=np.float32)
        target = np.asarray(candidate.object_ball, dtype=np.float32)
        pocket = np.asarray(candidate.pocket_point, dtype=np.float32)
        visible_polygon = self.calibration.table.projection_visible_polygon_mm or self.calibration.table.inner_polygon_mm
        inner = np.asarray(visible_polygon, dtype=np.float32).reshape((-1, 2))
        table = self.calibration.table

        guide_start = cue_alignment_start(
            cue,
            ghost,
            inner,
            table_width_mm=float(table.width_mm),
            table_height_mm=float(table.height_mm),
        )
        if float(np.linalg.norm(guide_start - cue)) >= 1.0:
            self._append_line_mm(overlay, [guide_start, cue], width=3, trim_end_mm=ball_radius, label="cue_guide")
        self._append_line_mm(overlay, [cue, ghost], width=4, trim_start_mm=ball_radius, trim_end_mm=ball_radius, label="aim")
        self._append_line_mm(
            overlay,
            [target, pocket],
            width=3,
            style="dashed",
            trim_start_mm=ball_radius,
            label="object",
        )
        cue_sep_end = rule_cue_separation_end(
            cue,
            ghost,
            target,
            inner,
            fallback_length_mm=math.hypot(float(table.width_mm), float(table.height_mm)),
        )
        if cue_sep_end is not None:
            self._append_line_mm(
                overlay,
                [ghost, cue_sep_end],
                width=3,
                style="dashed",
                trim_start_mm=ball_radius,
                label="cue_separation",
            )
        circles = self.calibration.table_mm_to_projector_px(
            np.asarray([candidate.ghost_ball, candidate.object_ball, candidate.pocket_point], dtype=np.float32)
        )
        radius_px = self._projector_radius_px(candidate.ghost_ball, ball_radius)
        overlay.circles.append(((float(circles[0, 0]), float(circles[0, 1])), radius_px, ROUTE_COLOR))
        overlay.circles.append(((float(circles[1, 0]), float(circles[1, 1])), radius_px, ROUTE_COLOR))
        overlay.circles.append(((float(circles[2, 0]), float(circles[2, 1])), max(8.0, radius_px * 0.45), ROUTE_COLOR))
        label_pos = (float(circles[0, 0] + 14), float(circles[0, 1] - 14))
        overlay.labels.append((label_pos, f"{candidate.score:.2f}", ROUTE_COLOR))

    def _add_free_route(self, overlay: ProjectionOverlay, route: FreeRouteSuggestion) -> None:
        cue = np.asarray(route.cue_ball, dtype=np.float32)
        radius = float(max(1.0, route.cue_radius))
        tip = np.asarray(route.cue_stick_tip, dtype=np.float32)
        tail = np.asarray(route.cue_stick_tail, dtype=np.float32)
        aim_dir = unit(np.asarray(route.aim_direction, dtype=np.float32))

        self._append_line_mm(overlay, [tail, tip], color=CUE_STICK_COLOR, width=2, label="cue_stick")
        guide_back = cue - aim_dir * max(26.0, 2.2 * radius)
        self._append_line_mm(overlay, [guide_back, cue], width=3, trim_end_mm=radius, label="cue_guide")

        nodes = [np.asarray(p, dtype=np.float32) for p in route.path_points]
        collision_count = len(route.collision_points or [])
        for idx, node in enumerate(nodes[1 : 1 + collision_count]):
            overlay.circles.append((self._projector_point(node), self._projector_radius_px(node, radius), ROUTE_COLOR))

        for i in range(max(0, len(nodes) - 1)):
            start_radius = radius if i <= collision_count else 0.0
            end_radius = radius if (i + 1) <= collision_count else 0.0
            self._append_line_mm(
                overlay,
                [nodes[i], nodes[i + 1]],
                width=4,
                trim_start_mm=start_radius,
                trim_end_mm=end_radius,
                label="free_path",
            )

        normals = [np.asarray(n, dtype=np.float32) for n in route.collision_normals or []]
        collision_types = list(route.collision_types or [])
        collisions = [np.asarray(p, dtype=np.float32) for p in route.collision_points or []]
        for idx, collision in enumerate(collisions):
            if idx >= len(collision_types) or str(collision_types[idx]) != "ball":
                continue
            normal = unit(normals[idx] if idx < len(normals) else aim_dir)
            hit_center = collision + normal * (2.0 * radius)
            hit_end = estimate_route_end(
                hit_center,
                normal,
                np.asarray(self.calibration.table.projection_visible_polygon_mm or self.calibration.table.inner_polygon_mm, dtype=np.float32).reshape((-1, 2)),
                fallback_length_mm=math.hypot(float(self.calibration.table.width_mm), float(self.calibration.table.height_mm)),
            )
            self._append_line_mm(
                overlay,
                [hit_center, hit_end],
                width=3,
                style="dashed",
                trim_start_mm=radius,
                label="free_hit_ball",
            )

        if route.pocket_point is not None:
            pocket = np.asarray(route.pocket_point, dtype=np.float32)
            overlay.circles.append((self._projector_point(pocket), max(8.0, self._projector_radius_px(pocket, radius) * 0.45), ROUTE_COLOR))

    def _append_line_mm(
        self,
        overlay: ProjectionOverlay,
        points_mm,
        *,
        color: Tuple[int, int, int] = ROUTE_COLOR,
        width: int = 3,
        style: str = "solid",
        trim_start_mm: float = 0.0,
        trim_end_mm: float = 0.0,
        label: str | None = None,
    ) -> None:
        points = self._table_line_to_projector(points_mm)
        if len(points) == 2 and (trim_start_mm > 0.0 or trim_end_mm > 0.0):
            points = self._trim_projector_segment(
                points,
                self._projector_radius_px(points_mm[0], trim_start_mm) if trim_start_mm > 0.0 else 0.0,
                self._projector_radius_px(points_mm[-1], trim_end_mm) if trim_end_mm > 0.0 else 0.0,
            )
        if len(points) >= 2:
            overlay.lines.append(OverlayLine(points=points, color=color, width=width, label=label, style=style))

    def _table_line_to_projector(self, points) -> List[Tuple[float, float]]:
        arr = self.calibration.table_mm_to_projector_px(np.asarray(points, dtype=np.float32))
        return [(float(x), float(y)) for x, y in arr]

    def _projector_point(self, point_mm) -> Tuple[float, float]:
        arr = self.calibration.table_mm_to_projector_px(np.asarray([point_mm], dtype=np.float32))[0]
        return (float(arr[0]), float(arr[1]))

    def _projector_radius_px(self, point_mm, radius_mm: float) -> float:
        radius = float(max(0.0, radius_mm))
        if radius <= 0.0:
            return 0.0
        point = np.asarray(point_mm, dtype=np.float32).reshape((2,))
        arr = self.calibration.table_mm_to_projector_px(np.asarray([point, point + np.asarray([radius, 0.0], dtype=np.float32)], dtype=np.float32))
        return float(max(1.0, np.linalg.norm(arr[1] - arr[0])))

    @staticmethod
    def _trim_projector_segment(points: List[Tuple[float, float]], trim_start_px: float, trim_end_px: float) -> List[Tuple[float, float]]:
        a = np.asarray(points[0], dtype=np.float32)
        b = np.asarray(points[1], dtype=np.float32)
        v = b - a
        dist = float(np.linalg.norm(v))
        if dist < 1e-3:
            return points
        start = float(max(0.0, trim_start_px))
        end = float(max(0.0, trim_end_px))
        if start + end >= dist - 1.0:
            return []
        d = v / dist
        aa = a + d * start
        bb = b - d * end
        return [(float(aa[0]), float(aa[1])), (float(bb[0]), float(bb[1]))]

def render_overlay_image(overlay: ProjectionOverlay, background: Optional[np.ndarray] = None) -> np.ndarray:
    width, height = overlay.projector_size
    if background is None:
        img = np.zeros((height, width, 3), dtype=np.uint8)
    else:
        img = cv2.resize(background, (width, height)).copy()
    if background is None:
        img[:] = (0, 0, 0)
    for line in overlay.lines:
        pts = np.asarray(line.points, dtype=np.float32).reshape((-1, 2))
        if pts.shape[0] >= 2:
            color = tuple(int(c) for c in line.color)
            width = max(1, int(line.width))
            if str(line.style).lower() == "dashed":
                _draw_dashed_polyline(img, pts, color, width)
            else:
                cv2.polylines(
                    img,
                    [np.round(pts).astype(np.int32).reshape((-1, 1, 2))],
                    isClosed=False,
                    color=color,
                    thickness=width,
                    lineType=cv2.LINE_AA,
                )
            if line.arrow:
                _draw_arrow_head(img, pts[-2], pts[-1], color, width)
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


def render_overlay_with_star(overlay: ProjectionOverlay, star_formula: StarFormulaConfig) -> np.ndarray:
    img = render_overlay_image(overlay)
    draw_star_formula(img, star_formula)
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


def _draw_dashed_polyline(
    img: np.ndarray,
    pts: np.ndarray,
    color: Tuple[int, int, int],
    width: int,
    dash_px: float = 12.0,
    gap_px: float = 16.0,
) -> None:
    for idx in range(pts.shape[0] - 1):
        _draw_dashed_segment(img, pts[idx], pts[idx + 1], color, width, dash_px=dash_px, gap_px=gap_px)


def _draw_dashed_segment(
    img: np.ndarray,
    start: np.ndarray,
    end: np.ndarray,
    color: Tuple[int, int, int],
    width: int,
    *,
    dash_px: float,
    gap_px: float,
) -> None:
    v = np.asarray(end, dtype=np.float32) - np.asarray(start, dtype=np.float32)
    dist = float(np.linalg.norm(v))
    if dist < 1e-3:
        return
    d = v / dist
    step = float(max(2.0, dash_px + gap_px))
    drawn = 0.0
    while drawn < dist:
        seg_end = min(dist, drawn + float(max(2.0, dash_px)))
        p1 = np.asarray(start, dtype=np.float32) + d * drawn
        p2 = np.asarray(start, dtype=np.float32) + d * seg_end
        i1 = (int(round(float(p1[0]))), int(round(float(p1[1]))))
        i2 = (int(round(float(p2[0]))), int(round(float(p2[1]))))
        if i1 != i2:
            cv2.line(img, i1, i2, color, max(1, width - 1), cv2.LINE_AA)
        drawn += step
