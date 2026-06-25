from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

from ..calibration.service import CalibrationService
from ..config import PlannerConfig
from ..schemas import FreeRouteSuggestion, MatchStateFrame, TrackObservation
from ..utils import clamp, point_segment_distance, unit


@dataclass
class _Ball:
    track_id: int
    group: str
    center_px: np.ndarray
    radius_px: float
    quality: float


class FreeShotPlanner:
    version = "free_shot_collision_v1"

    def __init__(self, config: PlannerConfig, calibration: CalibrationService):
        self.config = config
        self.calibration = calibration
        self.min_line_length_px = 72.0
        self.cue_tip_near_factor = 5.2
        self.cue_line_near_factor = 3.2
        self.ray_eps_px = 1.5
        self.ball_collision_padding_px = 1.2
        self.last_status = "idle"

    def plan(self, state: MatchStateFrame, frame_bgr: Optional[np.ndarray] = None) -> Optional[FreeRouteSuggestion]:
        balls = self._extract_balls(state.layout)
        cue_ball = next((ball for ball in balls if ball.group == "cue"), None)
        if cue_ball is None:
            return self._miss("no_cue_ball")

        frame_shape = frame_bgr.shape if frame_bgr is not None else None
        inner_polygon = self._inner_polygon_px(frame_shape)
        pockets = self._pocket_points_px()
        cue_stick = self._detect_cue_stick(frame_bgr, state.layout, cue_ball, inner_polygon)
        if cue_stick is None:
            return self._miss("no_cue_stick")

        tip, tail, initial_dir = cue_stick
        if float(np.linalg.norm(initial_dir)) < 1e-6:
            return self._miss("invalid_direction")

        sim = self._simulate_collision_path(cue_ball, initial_dir, balls, inner_polygon, pockets)
        if sim is None:
            return self._miss("invalid_path")

        path_points, collisions, normals, collision_types, collision_track_ids, pocket_index, pocket_point = sim
        route = self._to_table_route(
            cue_ball=cue_ball,
            tip=tip,
            tail=tail,
            aim_direction=initial_dir,
            path_points=path_points,
            collisions=collisions,
            normals=normals,
            collision_types=collision_types,
            collision_track_ids=collision_track_ids,
            pocket_index=pocket_index,
            pocket_point=pocket_point,
        )
        self.last_status = "ok"
        return route

    def _miss(self, status: str) -> Optional[FreeRouteSuggestion]:
        self.last_status = str(status)
        return None

    def _extract_balls(self, tracks: Sequence[TrackObservation]) -> List[_Ball]:
        balls: List[_Ball] = []
        for tr in tracks:
            if tr.group not in {"cue", "solid", "stripe", "black"}:
                continue
            if tr.quality <= 0.15:
                continue
            balls.append(
                _Ball(
                    track_id=int(tr.track_id),
                    group=str(tr.group),
                    center_px=np.asarray(tr.center_px, dtype=np.float32).reshape((2,)),
                    radius_px=float(max(2.0, tr.radius_px)),
                    quality=float(tr.quality),
                )
            )
        return balls

    def _inner_polygon_px(self, frame_shape: Optional[Tuple[int, ...]]) -> np.ndarray:
        base_polygon = self.calibration.table.center_playable_polygon_mm or self.calibration.table.inner_polygon_mm
        poly_mm = np.asarray(base_polygon, dtype=np.float32).reshape((-1, 2))
        if poly_mm.shape[0] >= 3:
            try:
                poly_px = self.calibration.table_mm_to_camera_px(poly_mm).astype(np.float32)
                if np.all(np.isfinite(poly_px)):
                    return poly_px
            except Exception:
                pass
        if frame_shape is not None and len(frame_shape) >= 2:
            h, w = frame_shape[:2]
            return np.asarray([(0.0, 0.0), (float(w), 0.0), (float(w), float(h)), (0.0, float(h))], dtype=np.float32)
        return np.asarray([(0.0, 0.0), (1280.0, 0.0), (1280.0, 800.0), (0.0, 800.0)], dtype=np.float32)

    def _pocket_points_px(self) -> List[np.ndarray]:
        pockets_mm = np.asarray(self.calibration.table.pockets_mm, dtype=np.float32).reshape((-1, 2))
        if pockets_mm.shape[0] == 0:
            return []
        try:
            pockets_px = self.calibration.table_mm_to_camera_px(pockets_mm).astype(np.float32)
        except Exception:
            return []
        return [pockets_px[i].copy() for i in range(pockets_px.shape[0]) if np.all(np.isfinite(pockets_px[i]))]

    def _detect_cue_stick(
        self,
        frame_bgr: Optional[np.ndarray],
        tracks: Sequence[TrackObservation],
        cue_ball: _Ball,
        inner_polygon: np.ndarray,
    ) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        if frame_bgr is not None:
            found = self._detect_cue_stick_from_edges(frame_bgr, cue_ball.center_px, cue_ball.radius_px, inner_polygon)
            if found is not None:
                return found
        return self._detect_cue_stick_from_tracks(tracks, cue_ball)

    def _detect_cue_stick_from_tracks(
        self,
        tracks: Sequence[TrackObservation],
        cue_ball: _Ball,
    ) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        best: Optional[Tuple[float, np.ndarray, np.ndarray, np.ndarray]] = None
        cue_center = cue_ball.center_px
        cue_radius = max(2.0, cue_ball.radius_px)
        line_limit = max(18.0, 4.5 * cue_radius)
        for tr in tracks:
            if tr.group != "cue_stick":
                continue
            x1, y1, x2, y2 = [float(v) for v in tr.bbox]
            w = max(1.0, x2 - x1)
            h = max(1.0, y2 - y1)
            if w >= h:
                p1 = np.asarray([x1, (y1 + y2) * 0.5], dtype=np.float32)
                p2 = np.asarray([x2, (y1 + y2) * 0.5], dtype=np.float32)
            else:
                p1 = np.asarray([(x1 + x2) * 0.5, y1], dtype=np.float32)
                p2 = np.asarray([(x1 + x2) * 0.5, y2], dtype=np.float32)
            seg_len = float(np.linalg.norm(p2 - p1))
            if seg_len < 24.0:
                continue
            line_dist = point_segment_distance(cue_center, p1, p2)
            near, far = (p1, p2) if float(np.linalg.norm(p1 - cue_center)) <= float(np.linalg.norm(p2 - cue_center)) else (p2, p1)
            near_dist = float(np.linalg.norm(near - cue_center))
            if line_dist > line_limit and near_dist > max(line_limit, 8.0 * cue_radius):
                continue
            direction = unit(cue_center - near)
            score = float(seg_len + 60.0 * float(tr.confidence) - 1.5 * line_dist - 0.25 * near_dist)
            if best is None or score > best[0]:
                best = (score, near.astype(np.float32), far.astype(np.float32), direction.astype(np.float32))
        if best is None:
            return None
        _score, tip, tail, direction = best
        return tip, tail, direction

    def _detect_cue_stick_from_edges(
        self,
        frame_bgr: np.ndarray,
        cue_center: np.ndarray,
        cue_radius: float,
        cue_search_polygon: np.ndarray,
    ) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        h, w = frame_bgr.shape[:2]
        diag = float(math.hypot(w, h))
        if cue_search_polygon.size >= 6:
            x_min = float(np.min(cue_search_polygon[:, 0]))
            x_max = float(np.max(cue_search_polygon[:, 0]))
            y_min = float(np.min(cue_search_polygon[:, 1]))
            y_max = float(np.max(cue_search_polygon[:, 1]))
            diag = max(120.0, float(math.hypot(x_max - x_min, y_max - y_min)))

        roi_half = int(clamp(0.28 * diag, 220.0, 560.0))
        cx = int(round(float(cue_center[0])))
        cy = int(round(float(cue_center[1])))
        x1 = max(0, cx - roi_half)
        y1 = max(0, cy - roi_half)
        x2 = min(w, cx + roi_half)
        y2 = min(h, cy + roi_half)
        if (x2 - x1) < 24 or (y2 - y1) < 24:
            return None

        roi = frame_bgr[y1:y2, x1:x2]
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0.0)
        edges = cv2.Canny(gray, 55, 150)
        if int(np.count_nonzero(edges)) < 16:
            return None

        line_blocks = []
        for threshold, min_len_factor, max_gap in ((36, 0.60, 14), (26, 0.45, 20), (18, 0.35, 28)):
            lines = cv2.HoughLinesP(
                edges,
                rho=1.0,
                theta=float(np.pi / 180.0),
                threshold=int(threshold),
                minLineLength=int(max(24.0, self.min_line_length_px * float(min_len_factor))),
                maxLineGap=int(max_gap),
            )
            if lines is not None and lines.size > 0:
                line_blocks.append(lines.reshape((-1, 4)))
        if not line_blocks:
            return None

        tip_limit = max(26.0, float(self.cue_tip_near_factor) * float(cue_radius) + 10.0)
        line_limit = max(14.0, float(self.cue_line_near_factor) * float(cue_radius) + 8.0)
        best: Optional[Tuple[float, np.ndarray, np.ndarray, np.ndarray]] = None
        for line_block in line_blocks:
            for ln in line_block:
                p1 = np.asarray([float(ln[0] + x1), float(ln[1] + y1)], dtype=np.float32)
                p2 = np.asarray([float(ln[2] + x1), float(ln[3] + y1)], dtype=np.float32)
                seg = p2 - p1
                seg_len = float(np.linalg.norm(seg))
                if seg_len < self.min_line_length_px:
                    continue
                line_dist = point_segment_distance(cue_center, p1, p2)
                if line_dist > line_limit:
                    continue
                d1 = float(np.linalg.norm(p1 - cue_center))
                d2 = float(np.linalg.norm(p2 - cue_center))
                near = p1 if d1 <= d2 else p2
                far = p2 if d1 <= d2 else p1

                denom = float(np.dot(seg, seg))
                if denom <= 1e-9:
                    continue
                t = clamp(float(np.dot(cue_center - p1, seg) / denom), 0.0, 1.0)
                projected = p1 + seg * t
                tip = projected if float(np.linalg.norm(projected - cue_center)) >= max(3.0, 0.22 * cue_radius) else near
                tip_dist = float(np.linalg.norm(tip - cue_center))
                if tip_dist > tip_limit:
                    tip = near
                    tip_dist = float(np.linalg.norm(tip - cue_center))
                    if tip_dist > tip_limit:
                        continue

                axis = unit(tip - far)
                cue_to_tip = unit(cue_center - tip)
                align = abs(float(np.dot(axis, cue_to_tip)))
                if align < 0.42:
                    continue
                direction = axis if float(np.dot(axis, cue_to_tip)) >= 0.0 else -axis
                score = float(seg_len + 70.0 * align - 1.2 * line_dist - 0.9 * tip_dist + 0.1 * max(d1, d2))
                if best is None or score > best[0]:
                    best = (score, tip.astype(np.float32), far.astype(np.float32), direction.astype(np.float32))
        if best is None:
            return None
        _score, tip, tail, direction = best
        return tip, tail, direction

    def _simulate_collision_path(
        self,
        cue_ball: _Ball,
        initial_dir: np.ndarray,
        balls: List[_Ball],
        inner_polygon: np.ndarray,
        pockets: Sequence[np.ndarray],
    ) -> Optional[
        Tuple[
            List[np.ndarray],
            List[np.ndarray],
            List[np.ndarray],
            List[str],
            List[Optional[int]],
            Optional[int],
            Optional[np.ndarray],
        ]
    ]:
        path_points: List[np.ndarray] = [cue_ball.center_px.copy().astype(np.float32)]
        collisions: List[np.ndarray] = []
        normals: List[np.ndarray] = []
        collision_types: List[str] = []
        collision_track_ids: List[Optional[int]] = []

        ray_origin = cue_ball.center_px.copy().astype(np.float32)
        ray_dir = unit(initial_dir)
        cue_radius = float(max(2.0, cue_ball.radius_px))
        previous_ball_hit_id: Optional[int] = None
        pocket_index: Optional[int] = None
        pocket_point: Optional[np.ndarray] = None
        route_finished = False

        for _collision_idx in range(max(0, int(self.config.free_max_collisions))):
            boundary_hit = self._first_boundary_hit(ray_origin, ray_dir, inner_polygon, min_t=max(0.8, self.ray_eps_px))
            boundary_t = float(np.linalg.norm(boundary_hit[0] - ray_origin)) if boundary_hit is not None else None

            ignore_ids = {int(cue_ball.track_id)}
            if previous_ball_hit_id is not None:
                ignore_ids.add(int(previous_ball_hit_id))
            ball_hit = self._first_ball_hit(
                ray_origin,
                ray_dir,
                balls,
                moving_radius=cue_radius,
                ignore_ids=ignore_ids,
                min_t=max(2.0, 0.35 * cue_radius, self.ray_eps_px),
            )
            ball_t = ball_hit[0] if ball_hit is not None else None

            nearest_collision_t = min([v for v in (boundary_t, ball_t) if v is not None], default=None)
            max_pocket_t = float(nearest_collision_t + max(42.0, 2.5 * cue_radius)) if nearest_collision_t is not None else 900.0
            pocket_hit = self._first_pocket_hit(
                ray_origin,
                ray_dir,
                pockets,
                ball_radius=cue_radius,
                min_t=max(4.0, 0.35 * cue_radius),
                max_t=max_pocket_t,
            )
            if pocket_hit is not None:
                pocket_t, pk_center, pk_idx, capture = pocket_hit
                before_ball = ball_t is None or pocket_t <= ball_t - max(2.0, 0.25 * cue_radius)
                before_boundary = boundary_t is None or pocket_t <= boundary_t + capture + cue_radius
                if before_ball and before_boundary:
                    path_points.append(pk_center.copy().astype(np.float32))
                    pocket_index = int(pk_idx)
                    pocket_point = pk_center.copy().astype(np.float32)
                    route_finished = True
                    break

            if nearest_collision_t is None:
                end_point, pocket_index = self._estimate_route_end(ray_origin, ray_dir, inner_polygon, pockets, cue_radius)
                path_points.append(end_point.astype(np.float32))
                if pocket_index is not None:
                    pocket_point = end_point.astype(np.float32)
                route_finished = True
                break

            use_ball = ball_hit is not None and (boundary_t is None or float(ball_hit[0]) <= float(boundary_t))
            if use_ball and ball_hit is not None:
                _ball_t, collision_node, ball, hit_normal = ball_hit
                collisions.append(collision_node.copy().astype(np.float32))
                normals.append(hit_normal.copy().astype(np.float32))
                collision_types.append("ball")
                collision_track_ids.append(int(ball.track_id))
                path_points.append(collision_node.copy().astype(np.float32))

                incoming = unit(ray_dir)
                normal_speed = max(0.0, float(np.dot(incoming, hit_normal)))
                cue_after = incoming - hit_normal * normal_speed
                if float(np.linalg.norm(cue_after)) < 0.08:
                    route_finished = True
                    break
                ray_dir = unit(cue_after.astype(np.float32))
                ray_origin = (collision_node + ray_dir * float(self.ray_eps_px)).astype(np.float32)
                previous_ball_hit_id = int(ball.track_id)
            elif boundary_hit is not None:
                hit_point, inward_n = boundary_hit
                collision_node = (hit_point + inward_n * cue_radius).astype(np.float32)
                collisions.append(collision_node.copy())
                normals.append(inward_n.copy().astype(np.float32))
                collision_types.append("edge")
                collision_track_ids.append(None)
                path_points.append(collision_node.copy())

                ray_dir = self._reflect(ray_dir, inward_n)
                ray_origin = (collision_node + ray_dir * float(self.ray_eps_px)).astype(np.float32)
                previous_ball_hit_id = None
            else:
                end_point, pocket_index = self._estimate_route_end(ray_origin, ray_dir, inner_polygon, pockets, cue_radius)
                path_points.append(end_point.astype(np.float32))
                if pocket_index is not None:
                    pocket_point = end_point.astype(np.float32)
                route_finished = True
                break

        if not route_finished:
            end_point, pocket_index = self._estimate_route_end(ray_origin, ray_dir, inner_polygon, pockets, cue_radius)
            path_points.append(end_point.astype(np.float32))
            if pocket_index is not None:
                pocket_point = end_point.astype(np.float32)

        if len(path_points) < 2:
            return None
        return path_points, collisions, normals, collision_types, collision_track_ids, pocket_index, pocket_point

    def _to_table_route(
        self,
        *,
        cue_ball: _Ball,
        tip: np.ndarray,
        tail: np.ndarray,
        aim_direction: np.ndarray,
        path_points: Sequence[np.ndarray],
        collisions: Sequence[np.ndarray],
        normals: Sequence[np.ndarray],
        collision_types: Sequence[str],
        collision_track_ids: Sequence[Optional[int]],
        pocket_index: Optional[int],
        pocket_point: Optional[np.ndarray],
    ) -> FreeRouteSuggestion:
        def pt(point: np.ndarray) -> Tuple[float, float]:
            mm = self.calibration.ball_camera_px_to_table_mm(np.asarray([point], dtype=np.float32))[0]
            return (float(mm[0]), float(mm[1]))

        def pts(points: Sequence[np.ndarray]) -> List[Tuple[float, float]]:
            if not points:
                return []
            arr = np.asarray(points, dtype=np.float32).reshape((-1, 2))
            mm = self.calibration.ball_camera_px_to_table_mm(arr)
            return [(float(x), float(y)) for x, y in mm]

        def direction_at(point: np.ndarray, direction: np.ndarray) -> Tuple[float, float]:
            step = max(20.0, cue_ball.radius_px)
            arr = np.asarray([point, point + unit(direction) * step], dtype=np.float32)
            mm = self.calibration.ball_camera_px_to_table_mm(arr)
            d = unit(mm[1] - mm[0])
            return (float(d[0]), float(d[1]))

        cue_radius_mm = self.calibration.ball_pixel_radius_to_mm((float(cue_ball.center_px[0]), float(cue_ball.center_px[1])), cue_ball.radius_px)
        pocket_mm = pt(pocket_point) if pocket_point is not None else None
        collision_normals = [
            direction_at(collisions[idx], normals[idx])
            for idx in range(min(len(collisions), len(normals)))
        ]
        return FreeRouteSuggestion(
            cue_ball=pt(cue_ball.center_px),
            cue_radius=float(max(1.0, cue_radius_mm)),
            cue_stick_tip=pt(tip),
            cue_stick_tail=pt(tail),
            aim_direction=direction_at(cue_ball.center_px, aim_direction),
            path_points=pts(path_points),
            collision_points=pts(collisions),
            collision_normals=collision_normals,
            collision_types=[str(v) for v in collision_types],
            collision_track_ids=[int(v) if v is not None else None for v in collision_track_ids],
            pocket_index=int(pocket_index) if pocket_index is not None else None,
            pocket_point=pocket_mm,
        )

    @staticmethod
    def _ray_segment_intersection(
        origin: np.ndarray,
        direction: np.ndarray,
        a: np.ndarray,
        b: np.ndarray,
        min_t: float,
    ) -> Optional[Tuple[float, float, np.ndarray]]:
        s = b - a
        denom = _cross2d(direction, s)
        if abs(denom) < 1e-7:
            return None
        ao = a - origin
        t = _cross2d(ao, s) / denom
        u = _cross2d(ao, direction) / denom
        if t < float(min_t) or u < -1e-4 or u > 1.0001:
            return None
        p = origin + direction * float(t)
        return float(t), float(u), p.astype(np.float32)

    def _first_boundary_hit(
        self,
        origin: np.ndarray,
        direction: np.ndarray,
        inner_polygon: np.ndarray,
        min_t: float,
    ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        if inner_polygon.size < 6 or inner_polygon.shape[0] < 3:
            return None
        best_t: Optional[float] = None
        best_point: Optional[np.ndarray] = None
        best_normal: Optional[np.ndarray] = None
        for idx in range(inner_polygon.shape[0]):
            a = inner_polygon[idx].astype(np.float32)
            b = inner_polygon[(idx + 1) % inner_polygon.shape[0]].astype(np.float32)
            if float(np.linalg.norm(b - a)) < 1e-6:
                continue
            hit = self._ray_segment_intersection(origin, direction, a, b, min_t=min_t)
            if hit is None:
                continue
            t, _u, p = hit
            if best_t is None or t < best_t:
                best_t = t
                best_point = p
                best_normal = self._segment_inward_normal(inner_polygon, a, b)
        if best_point is None or best_normal is None:
            return None
        return best_point.astype(np.float32), unit(best_normal)

    @staticmethod
    def _segment_inward_normal(inner_polygon: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        v = b - a
        normal = unit(np.asarray([-float(v[1]), float(v[0])], dtype=np.float32))
        poly = inner_polygon.reshape((-1, 1, 2)).astype(np.float32)
        mid = (a + b) * 0.5
        p_pos = mid + normal * 4.0
        p_neg = mid - normal * 4.0
        inside_pos = cv2.pointPolygonTest(poly, (float(p_pos[0]), float(p_pos[1])), False) >= 0.0
        inside_neg = cv2.pointPolygonTest(poly, (float(p_neg[0]), float(p_neg[1])), False) >= 0.0
        if inside_pos and not inside_neg:
            return normal
        if inside_neg and not inside_pos:
            return -normal
        return normal

    @staticmethod
    def _reflect(direction: np.ndarray, normal_vec: np.ndarray) -> np.ndarray:
        n = unit(normal_vec)
        reflected = direction - (2.0 * float(np.dot(direction, n))) * n
        if float(np.linalg.norm(reflected)) < 1e-6:
            return -unit(direction)
        return unit(reflected.astype(np.float32))

    @staticmethod
    def _ray_circle_first_hit(origin: np.ndarray, direction: np.ndarray, center: np.ndarray, radius: float, min_t: float) -> Optional[float]:
        d = unit(direction)
        oc = origin - center
        b = 2.0 * float(np.dot(d, oc))
        c = float(np.dot(oc, oc) - float(radius) * float(radius))
        disc = b * b - 4.0 * c
        if disc < 0.0:
            return None
        root = math.sqrt(max(0.0, disc))
        valid = [float(t) for t in [(-b - root) * 0.5, (-b + root) * 0.5] if float(t) >= float(min_t)]
        return min(valid) if valid else None

    def _first_ball_hit(
        self,
        origin: np.ndarray,
        direction: np.ndarray,
        balls: Sequence[_Ball],
        moving_radius: float,
        ignore_ids: set[int],
        min_t: float,
    ) -> Optional[Tuple[float, np.ndarray, _Ball, np.ndarray]]:
        best: Optional[Tuple[float, np.ndarray, _Ball, np.ndarray]] = None
        d = unit(direction)
        for ball in balls:
            if int(ball.track_id) in ignore_ids:
                continue
            radius_sum = max(2.0, float(moving_radius) + float(ball.radius_px) + float(self.ball_collision_padding_px))
            t = self._ray_circle_first_hit(origin, d, ball.center_px, radius_sum, min_t)
            if t is None:
                continue
            collision_node = (origin + d * float(t)).astype(np.float32)
            normal = unit(ball.center_px - collision_node, fallback=(float(d[0]), float(d[1])))
            if float(np.dot(normal, d)) < 0.0:
                normal = -normal
            if best is None or t < best[0]:
                best = (float(t), collision_node, ball, normal.astype(np.float32))
        return best

    def _estimate_route_end(
        self,
        origin: np.ndarray,
        direction: np.ndarray,
        inner_polygon: np.ndarray,
        pockets: Sequence[np.ndarray],
        ball_radius: float,
    ) -> Tuple[np.ndarray, Optional[int]]:
        d = unit(direction)
        boundary_hit = self._first_boundary_hit(origin, d, inner_polygon, min_t=max(0.8, self.ray_eps_px))
        boundary_t = None
        boundary_end = None
        if boundary_hit is not None:
            hit_point, inward_n = boundary_hit
            boundary_t = float(np.linalg.norm(hit_point - origin))
            boundary_end = (hit_point + inward_n * float(max(0.0, ball_radius))).astype(np.float32)

        max_t = float(boundary_t + max(42.0, 2.5 * ball_radius)) if boundary_t is not None else 900.0
        pocket_hit = self._first_pocket_hit(origin, d, pockets, ball_radius=ball_radius, min_t=max(4.0, 0.35 * ball_radius), max_t=max_t)
        if pocket_hit is not None:
            pocket_t, pocket_center, pocket_idx, capture = pocket_hit
            if boundary_t is None or pocket_t <= boundary_t + capture + ball_radius:
                return pocket_center.astype(np.float32), int(pocket_idx)
        if boundary_end is not None:
            return boundary_end.astype(np.float32), None

        if inner_polygon.size >= 6:
            x_min = float(np.min(inner_polygon[:, 0]))
            x_max = float(np.max(inner_polygon[:, 0]))
            y_min = float(np.min(inner_polygon[:, 1]))
            y_max = float(np.max(inner_polygon[:, 1]))
            diag = max(120.0, float(math.hypot(x_max - x_min, y_max - y_min)))
        else:
            diag = 900.0
        return (origin + d * float(0.45 * diag)).astype(np.float32), None

    @staticmethod
    def _first_pocket_hit(
        origin: np.ndarray,
        direction: np.ndarray,
        pockets: Sequence[np.ndarray],
        ball_radius: float,
        min_t: float,
        max_t: float,
    ) -> Optional[Tuple[float, np.ndarray, int, float]]:
        d = unit(direction)
        best: Optional[Tuple[float, np.ndarray, int, float]] = None
        for idx, center in enumerate(pockets):
            rel = center - origin
            t = float(np.dot(rel, d))
            if t < float(min_t) or t > float(max_t):
                continue
            closest = origin + d * t
            capture = max(18.0, 1.65 * float(ball_radius))
            if float(np.linalg.norm(center - closest)) > capture:
                continue
            score_t = max(0.0, t - 0.35 * capture)
            if best is None or score_t < max(0.0, best[0] - 0.35 * best[3]):
                best = (float(t), center.astype(np.float32), int(idx), float(capture))
        return best


def _cross2d(a: np.ndarray, b: np.ndarray) -> float:
    return float(a[0] * b[1] - a[1] * b[0])
