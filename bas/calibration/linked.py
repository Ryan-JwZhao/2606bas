from __future__ import annotations

from dataclasses import dataclass, field
import time
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from ..geometry import TableGeometry
from ..utils import clamp, ensure_numpy_points
from .charuco import CharucoBoardSpec, detect_charuco_corners, render_charuco_board
from .projector import (
    MAX_PROJECTION_INLIER_P95_PX,
    MIN_PROJECTION_INLIER_RATIO,
    ProjectionCalibration,
    polygon_quad,
    table_bbox_from_polygon,
)


PointArray = np.ndarray
MAX_LINKED_PATTERN_CV_P95_PX = 5.0


@dataclass
class LinkedCalibrationPattern:
    pattern_id: str
    title: str
    image: np.ndarray
    emphasis_zone: str
    collect_for_solver: bool = True
    board_spec: Optional[CharucoBoardSpec] = None
    projector_points: PointArray = field(default_factory=lambda: np.zeros((0, 2), dtype=np.float32))
    projector_ids: np.ndarray = field(default_factory=lambda: np.zeros((0,), dtype=np.int32))
    roi_proj: Tuple[int, int, int, int] = (0, 0, 0, 0)


@dataclass
class LinkedCalibrationObservation:
    pattern_id: str
    title: str
    emphasis_zone: str
    camera_points: PointArray
    projector_points: PointArray
    ids: np.ndarray
    detected_count: int
    matched_count: int


@dataclass
class LinkedCalibrationResult:
    projection: ProjectionCalibration
    observations: List[LinkedCalibrationObservation]
    summary: Dict[str, Any]


@dataclass
class LinkedPatternCaptureResult:
    observation: Optional[LinkedCalibrationObservation]
    transition_frames_read: int
    detection_frames_read: int
    matched_frames: int


def collect_linked_pattern_observation(
    pattern: LinkedCalibrationPattern,
    read_frame: Callable[[], Optional[np.ndarray]],
    *,
    undistort_points: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    transition_frames: int = 8,
    max_detection_frames: int = 18,
    inter_frame_delay_seconds: float = 0.0,
    on_frame: Optional[Callable[[np.ndarray], None]] = None,
) -> LinkedPatternCaptureResult:
    """Flush transition frames, then retain the strongest pattern observation.

    Camera backends may return buffered frames after the projector switches to a
    new image. Those frames must not be associated with the new projector
    coordinates, especially because adjacent focus patterns reuse ChArUco IDs.
    """
    transition_read = 0
    delay = max(0.0, float(inter_frame_delay_seconds))
    for _ in range(max(0, int(transition_frames))):
        frame = read_frame()
        if frame is None or frame.size == 0:
            if delay > 0.0:
                time.sleep(delay)
            continue
        transition_read += 1
        if on_frame is not None:
            on_frame(frame)
        if delay > 0.0:
            time.sleep(delay)

    best: Optional[LinkedCalibrationObservation] = None
    detection_read = 0
    matched_frames = 0
    for _ in range(max(1, int(max_detection_frames))):
        frame = read_frame()
        if frame is None or frame.size == 0:
            if delay > 0.0:
                time.sleep(delay)
            continue
        detection_read += 1
        if on_frame is not None:
            on_frame(frame)
        observation = match_linked_pattern_observation(
            pattern,
            frame,
            undistort_points=undistort_points,
        )
        if delay > 0.0:
            time.sleep(delay)
        if observation is None:
            continue
        matched_frames += 1
        if best is None or observation.matched_count > best.matched_count:
            best = observation

    return LinkedPatternCaptureResult(
        observation=best,
        transition_frames_read=transition_read,
        detection_frames_read=detection_read,
        matched_frames=matched_frames,
    )


def build_linked_patterns(
    geometry: TableGeometry,
    projector_size: Tuple[int, int],
    *,
    prior_projection: Optional[ProjectionCalibration] = None,
) -> List[LinkedCalibrationPattern]:
    width, height = [max(1, int(v)) for v in projector_size]
    projector_bbox = _projector_focus_bbox(projector_size, prior_projection)
    source_bbox = _source_geometry_bbox(geometry)
    outline_proj = _map_norm_points(geometry.outer_norm, source_bbox, projector_bbox)
    inner_proj = _map_norm_points(geometry.inner_norm, source_bbox, projector_bbox)
    pockets_proj = [_map_norm_points(pocket, source_bbox, projector_bbox) for pocket in geometry.pockets_norm]
    pocket_centers = [_polygon_center(pocket) for pocket in pockets_proj if pocket.shape[0] >= 2]
    if not pocket_centers:
        pocket_centers = _default_pocket_centers(projector_bbox)
    center = _polygon_center(inner_proj) if inner_proj.shape[0] >= 3 else np.asarray(
        [0.5 * (projector_bbox[0] + projector_bbox[2]), 0.5 * (projector_bbox[1] + projector_bbox[3])],
        dtype=np.float32,
    )

    coarse_spec = CharucoBoardSpec(squares_x=9, squares_y=6, square_length_m=0.035, marker_length_m=0.027)
    dense_spec = CharucoBoardSpec(squares_x=8, squares_y=6, square_length_m=0.030, marker_length_m=0.022)

    patterns: List[LinkedCalibrationPattern] = []
    patterns.append(
        LinkedCalibrationPattern(
            pattern_id="guide_overview",
            title="联动校正总览",
            image=_render_guide_image(width, height, projector_bbox, outline_proj, inner_proj, pockets_proj, center, None),
            emphasis_zone="overview",
            collect_for_solver=False,
        )
    )

    center_roi = _roi_around_point(center, projector_bbox, width, height, width_ratio=0.58, height_ratio=0.50)
    patterns.append(_make_charuco_pattern("center_focus", "台面中心预检与细化", dense_spec, width, height, center_roi, "center", projector_bbox, outline_proj, inner_proj, pockets_proj, center))

    # Keep the large board on the cloth instead of extending across reflective
    # rails and the extreme projector boundary. Pocket boards provide the edge
    # constraints separately.
    full_roi = _expand_roi(projector_bbox, -0.07, width, height)
    patterns.append(_make_charuco_pattern("full_cover", "台布全域 ChArUco", coarse_spec, width, height, full_roi, "full", projector_bbox, outline_proj, inner_proj, pockets_proj, center))

    names = ["pocket_lt", "pocket_mt", "pocket_rt", "pocket_rb", "pocket_mb", "pocket_lb"]
    for idx, point in enumerate(pocket_centers[:6]):
        roi = _roi_around_point(point, projector_bbox, width, height, width_ratio=0.30, height_ratio=0.28)
        patterns.append(
            _make_charuco_pattern(
                f"focus_{idx}",
                f"袋口重点 {idx + 1}",
                dense_spec,
                width,
                height,
                roi,
                names[idx] if idx < len(names) else f"pocket_{idx}",
                projector_bbox,
                outline_proj,
                inner_proj,
                pockets_proj,
                point,
            )
        )

    patterns.append(
        LinkedCalibrationPattern(
            pattern_id="guide_verify",
            title="覆盖校验网格",
            image=_render_guide_image(width, height, projector_bbox, outline_proj, inner_proj, pockets_proj, center, "verify"),
            emphasis_zone="verify",
            collect_for_solver=False,
        )
    )
    return patterns


def match_linked_pattern_observation(
    pattern: LinkedCalibrationPattern,
    frame_bgr: np.ndarray,
    *,
    undistort_points: Optional[Callable[[np.ndarray], np.ndarray]] = None,
) -> Optional[LinkedCalibrationObservation]:
    if not pattern.collect_for_solver or pattern.board_spec is None:
        return None
    if frame_bgr is None or frame_bgr.size == 0:
        return None
    camera_points, camera_ids = detect_charuco_corners(frame_bgr, pattern.board_spec)
    if camera_ids.size == 0 or pattern.projector_ids.size == 0:
        return None
    pairs = _match_ids(pattern.projector_points, pattern.projector_ids, camera_points, camera_ids)
    if pairs is None:
        return None
    matched_camera, matched_projector, ids = pairs
    if undistort_points is not None and matched_camera.shape[0] > 0:
        matched_camera = ensure_numpy_points(undistort_points(matched_camera))
    if matched_camera.shape[0] < 4 or matched_projector.shape[0] < 4:
        return None
    return LinkedCalibrationObservation(
        pattern_id=pattern.pattern_id,
        title=pattern.title,
        emphasis_zone=pattern.emphasis_zone,
        camera_points=matched_camera.astype(np.float32),
        projector_points=matched_projector.astype(np.float32),
        ids=ids.astype(np.int32),
        detected_count=int(camera_ids.size),
        matched_count=int(ids.size),
    )


def solve_linked_projection_calibration(
    observations: Sequence[LinkedCalibrationObservation],
    projector_size: Tuple[int, int],
    *,
    table_polygon_cam: Optional[np.ndarray] = None,
    minimum_pocket_zones: int = 4,
) -> LinkedCalibrationResult:
    cam_poly = polygon_quad(ensure_numpy_points(table_polygon_cam))
    if cam_poly.shape[0] != 4:
        raise ValueError(
            "Linked calibration requires a valid four-corner camera table polygon anchor."
        )
    usable = [obs for obs in observations if obs.matched_count >= 6]
    if len(usable) < 2:
        raise ValueError("联动校正至少需要两个有效采样图样。")
    camera_points = np.vstack([obs.camera_points for obs in usable]).astype(np.float64)
    projector_points = np.vstack([obs.projector_points for obs in usable]).astype(np.float64)
    if camera_points.shape[0] < 12 or projector_points.shape[0] < 12:
        raise ValueError("联动校正有效对应点不足，至少需要 12 个匹配角点。")
    projection = ProjectionCalibration.fit_from_correspondences(
        camera_points,
        projector_points,
        mode="linked_hybrid_charuco",
        projector_size=projector_size,
    )
    inlier_ratio = float(projection.quality_report.get("ransac_inlier_ratio", 0.0))
    if inlier_ratio < MIN_PROJECTION_INLIER_RATIO:
        raise RuntimeError(
            "Linked calibration rejected: RANSAC inlier ratio "
            f"{inlier_ratio:.1%} is below the required {MIN_PROJECTION_INLIER_RATIO:.0%}."
        )
    inlier_p95 = float(projection.quality_report.get("ransac_inlier_p95_px", float("inf")))
    if not np.isfinite(inlier_p95) or inlier_p95 > MAX_PROJECTION_INLIER_P95_PX:
        raise RuntimeError(
            "Linked calibration rejected: inlier reprojection P95 "
            f"{inlier_p95:.2f}px exceeds {MAX_PROJECTION_INLIER_P95_PX:.2f}px."
        )
    zones = {str(obs.emphasis_zone).strip().lower() for obs in usable}
    pocket_zones = {zone for zone in zones if zone.startswith("pocket_")}
    missing_core = [zone for zone in ("full", "center") if zone not in zones]
    if missing_core or len(pocket_zones) < max(0, int(minimum_pocket_zones)):
        raise RuntimeError(
            "Linked calibration rejected: insufficient spatial coverage; "
            f"missing core zones={missing_core}, pocket zones={len(pocket_zones)}/"
            f"{max(0, int(minimum_pocket_zones))}."
        )
    pattern_cv_errors = _leave_one_pattern_out_errors(usable)
    pattern_cv_p95 = float(np.percentile(pattern_cv_errors, 95)) if pattern_cv_errors.size else float("inf")
    if not np.isfinite(pattern_cv_p95) or pattern_cv_p95 > MAX_LINKED_PATTERN_CV_P95_PX:
        raise RuntimeError(
            "Linked calibration rejected: leave-one-pattern-out P95 "
            f"{pattern_cv_p95:.2f}px exceeds {MAX_LINKED_PATTERN_CV_P95_PX:.2f}px."
        )
    projection.table_polygon_cam = cam_poly.astype(np.float64)
    projection.table_polygon_proj = projection.camera_to_projector_points(cam_poly).astype(np.float64)
    if projection.residual_field.control_points_cam.shape[0] >= 4:
        normalized_rect = np.asarray(
            [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
            dtype=np.float32,
        )
        camera_to_normalized = cv2.getPerspectiveTransform(
            cam_poly.astype(np.float32),
            normalized_rect,
        )
        controls_camera = projection.residual_field.control_points_cam.astype(np.float64)
        projection.table_control_points_norm = cv2.perspectiveTransform(
            controls_camera.reshape((-1, 1, 2)),
            camera_to_normalized,
        ).reshape((-1, 2)).astype(np.float64)
        projection.table_control_points_proj = projection.camera_to_projector_points(
            controls_camera,
        ).astype(np.float64)
    summary = {
        "patterns_total": len(observations),
        "patterns_used": len(usable),
        "matched_points_total": int(camera_points.shape[0]),
        "matched_points_by_pattern": {obs.pattern_id: int(obs.matched_count) for obs in usable},
        "zones": sorted(zones),
        "pocket_zones_used": len(pocket_zones),
        "geometry_model": "independent_2d",
        "camera_extrinsics_used": False,
    }
    projection.quality_report.update({
        "workflow": "linked_hybrid_charuco",
        "patterns_total": summary["patterns_total"],
        "patterns_used": summary["patterns_used"],
        "matched_points_total": summary["matched_points_total"],
        "matched_points_by_pattern": summary["matched_points_by_pattern"],
        "zones": summary["zones"],
        "quality_gate_passed": True,
        "minimum_inlier_ratio": float(MIN_PROJECTION_INLIER_RATIO),
        "minimum_pocket_zones": max(0, int(minimum_pocket_zones)),
        "maximum_inlier_p95_px": float(MAX_PROJECTION_INLIER_P95_PX),
        "pattern_cv_p95_px": pattern_cv_p95,
        "maximum_pattern_cv_p95_px": float(MAX_LINKED_PATTERN_CV_P95_PX),
    })
    projection.quality_report.update(projection.calibration_error_stats())
    return LinkedCalibrationResult(projection=projection, observations=usable, summary=summary)


def _leave_one_pattern_out_errors(observations: Sequence[LinkedCalibrationObservation]) -> np.ndarray:
    errors: list[float] = []
    for validation_index, validation in enumerate(observations):
        training = [obs for index, obs in enumerate(observations) if index != validation_index]
        if not training:
            continue
        source = np.vstack([obs.camera_points for obs in training]).astype(np.float64)
        target = np.vstack([obs.projector_points for obs in training]).astype(np.float64)
        if source.shape[0] < 12:
            continue
        homography, _ = cv2.findHomography(source, target, cv2.RANSAC, 3.0)
        if homography is None:
            continue
        predicted = cv2.perspectiveTransform(
            validation.camera_points.astype(np.float64).reshape((-1, 1, 2)),
            homography,
        ).reshape((-1, 2))
        errors.extend(np.linalg.norm(predicted - validation.projector_points, axis=1).tolist())
    return np.asarray(errors, dtype=np.float64)


def projection_output_summary(result: LinkedCalibrationResult) -> str:
    report = dict(result.projection.quality_report)
    stats = result.projection.calibration_error_stats()
    p95 = stats.get("p95_px", stats.get("max_px", 0.0))
    return (
        f"图样 {report.get('patterns_used', 0)}/{report.get('patterns_total', 0)} | "
        f"匹配角点 {report.get('matched_points_total', 0)} | "
        f"mean={stats.get('mean_px', 0.0):.2f}px p95={p95:.2f}px max={stats.get('max_px', 0.0):.2f}px"
    )


def _make_charuco_pattern(
    pattern_id: str,
    title: str,
    board_spec: CharucoBoardSpec,
    width: int,
    height: int,
    roi: Tuple[int, int, int, int],
    emphasis_zone: str,
    projector_bbox: Tuple[float, float, float, float],
    outline_proj: np.ndarray,
    inner_proj: np.ndarray,
    pockets_proj: Sequence[np.ndarray],
    focus_point: Optional[np.ndarray],
) -> LinkedCalibrationPattern:
    image = _render_guide_image(width, height, projector_bbox, outline_proj, inner_proj, pockets_proj, focus_point, title)
    x1, y1, x2, y2 = roi
    board_w = max(120, x2 - x1)
    board_h = max(120, y2 - y1)
    board = render_charuco_board(board_spec, board_w, board_h)
    image[y1:y2, x1:x2] = board[: y2 - y1, : x2 - x1]
    projector_points, projector_ids = detect_charuco_corners(board, board_spec)
    projector_points = projector_points + np.asarray([x1, y1], dtype=np.float32)
    return LinkedCalibrationPattern(
        pattern_id=pattern_id,
        title=title,
        image=image,
        emphasis_zone=emphasis_zone,
        collect_for_solver=True,
        board_spec=board_spec,
        projector_points=projector_points,
        projector_ids=projector_ids,
        roi_proj=roi,
    )


def _render_guide_image(
    width: int,
    height: int,
    projector_bbox: Tuple[float, float, float, float],
    outline_proj: np.ndarray,
    inner_proj: np.ndarray,
    pockets_proj: Sequence[np.ndarray],
    focus_point: Optional[np.ndarray],
    label: Optional[str],
) -> np.ndarray:
    image = np.zeros((height, width, 3), dtype=np.uint8)
    image[:] = (10, 10, 10)
    x1, y1, x2, y2 = [int(round(v)) for v in projector_bbox]
    step_x = max(60, int(round((x2 - x1) / 10.0)))
    step_y = max(60, int(round((y2 - y1) / 6.0)))
    for x in range(x1, x2 + 1, step_x):
        cv2.line(image, (x, y1), (x, y2), (55, 55, 55), 1, cv2.LINE_AA)
    for y in range(y1, y2 + 1, step_y):
        cv2.line(image, (x1, y), (x2, y), (55, 55, 55), 1, cv2.LINE_AA)
    cv2.rectangle(image, (x1, y1), (x2, y2), (180, 180, 180), 2, cv2.LINE_AA)
    _draw_polyline(image, outline_proj, (220, 220, 220), 2, close=outline_proj.shape[0] >= 3)
    _draw_polyline(image, inner_proj, (0, 255, 128), 2, close=inner_proj.shape[0] >= 3)
    for pocket in pockets_proj:
        _draw_polyline(image, pocket, (0, 180, 255), 2, close=False)
    if focus_point is not None and focus_point.size >= 2:
        cx, cy = [int(round(v)) for v in focus_point[:2]]
        cv2.circle(image, (cx, cy), max(18, min(width, height) // 40), (0, 255, 255), 2, cv2.LINE_AA)
    if label:
        cv2.putText(image, str(label), (x1 + 16, max(34, y1 + 28)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
    return image


def _projector_focus_bbox(
    projector_size: Tuple[int, int],
    prior_projection: Optional[ProjectionCalibration],
) -> Tuple[float, float, float, float]:
    width, height = [max(1, int(v)) for v in projector_size]
    if prior_projection is not None:
        prior_size = prior_projection.projector_size
        bbox = table_bbox_from_polygon(
            prior_projection.table_polygon_proj,
            prior_size if prior_size[0] > 0 and prior_size[1] > 0 else projector_size,
        )
        x1, y1, x2, y2 = bbox
        x1 = clamp(x1, 0.0, float(width - 1))
        y1 = clamp(y1, 0.0, float(height - 1))
        x2 = clamp(x2, x1 + 1.0, float(width))
        y2 = clamp(y2, y1 + 1.0, float(height))
        if (x2 - x1) > width * 0.2 and (y2 - y1) > height * 0.2:
            return (x1, y1, x2, y2)
    margin_x = width * 0.05
    margin_y = height * 0.08
    return (margin_x, margin_y, float(width) - margin_x, float(height) - margin_y)


def _source_geometry_bbox(geometry: TableGeometry) -> Tuple[float, float, float, float]:
    for candidate in [geometry.outer_norm, geometry.inner_norm, *_flatten_segments(geometry.inline_norm), *_flatten_segments(geometry.pockets_norm)]:
        pts = ensure_numpy_points(candidate)
        if pts.shape[0] >= 2:
            return (
                float(np.min(pts[:, 0])),
                float(np.min(pts[:, 1])),
                float(np.max(pts[:, 0])),
                float(np.max(pts[:, 1])),
            )
    return (0.0, 0.0, 1.0, 1.0)


def _flatten_segments(segments: Sequence[np.ndarray]) -> List[np.ndarray]:
    return [ensure_numpy_points(seg) for seg in segments if ensure_numpy_points(seg).shape[0] >= 2]


def _map_norm_points(
    points: np.ndarray,
    src_bbox: Tuple[float, float, float, float],
    dst_bbox: Tuple[float, float, float, float],
) -> np.ndarray:
    pts = ensure_numpy_points(points)
    if pts.shape[0] == 0:
        return pts
    sx1, sy1, sx2, sy2 = src_bbox
    dx1, dy1, dx2, dy2 = dst_bbox
    src_w = max(1e-6, sx2 - sx1)
    src_h = max(1e-6, sy2 - sy1)
    dst_w = max(1e-6, dx2 - dx1)
    dst_h = max(1e-6, dy2 - dy1)
    out = pts.copy().astype(np.float32)
    out[:, 0] = dx1 + (out[:, 0] - sx1) / src_w * dst_w
    out[:, 1] = dy1 + (out[:, 1] - sy1) / src_h * dst_h
    return out


def _default_pocket_centers(bbox: Tuple[float, float, float, float]) -> List[np.ndarray]:
    x1, y1, x2, y2 = bbox
    cx = 0.5 * (x1 + x2)
    cy = 0.5 * (y1 + y2)
    return [
        np.asarray([x1, y1], dtype=np.float32),
        np.asarray([cx, y1], dtype=np.float32),
        np.asarray([x2, y1], dtype=np.float32),
        np.asarray([x2, y2], dtype=np.float32),
        np.asarray([cx, y2], dtype=np.float32),
        np.asarray([x1, y2], dtype=np.float32),
    ]


def _polygon_center(points: np.ndarray) -> np.ndarray:
    pts = ensure_numpy_points(points)
    if pts.shape[0] == 0:
        return np.zeros((2,), dtype=np.float32)
    return np.mean(pts, axis=0).astype(np.float32)


def _roi_around_point(
    point: np.ndarray,
    bbox: Tuple[float, float, float, float],
    width: int,
    height: int,
    *,
    width_ratio: float,
    height_ratio: float,
) -> Tuple[int, int, int, int]:
    x1, y1, x2, y2 = bbox
    available_w = max(2.0, min(float(width), x2) - max(0.0, x1))
    available_h = max(2.0, min(float(height), y2) - max(0.0, y1))
    bw = min(available_w, max(140.0, (x2 - x1) * float(width_ratio)))
    bh = min(available_h, max(140.0, (y2 - y1) * float(height_ratio)))
    cx, cy = [float(v) for v in point[:2]]
    min_x = clamp(x1, 0.0, float(width - 2))
    min_y = clamp(y1, 0.0, float(height - 2))
    max_x = clamp(x2, min_x + 1.0, float(width))
    max_y = clamp(y2, min_y + 1.0, float(height))
    rx1 = clamp(cx - bw * 0.5, min_x, max(min_x, max_x - bw))
    ry1 = clamp(cy - bh * 0.5, min_y, max(min_y, max_y - bh))
    rx2 = min(max_x, rx1 + bw)
    ry2 = min(max_y, ry1 + bh)
    return (int(round(rx1)), int(round(ry1)), int(round(rx2)), int(round(ry2)))


def _expand_roi(
    roi: Tuple[float, float, float, float],
    margin_ratio: float,
    width: int,
    height: int,
) -> Tuple[int, int, int, int]:
    x1, y1, x2, y2 = roi
    dw = (x2 - x1) * margin_ratio
    dh = (y2 - y1) * margin_ratio
    return (
        int(round(clamp(x1 - dw, 0.0, float(width - 2)))),
        int(round(clamp(y1 - dh, 0.0, float(height - 2)))),
        int(round(clamp(x2 + dw, 1.0, float(width)))),
        int(round(clamp(y2 + dh, 1.0, float(height)))),
    )


def _match_ids(
    projector_points: np.ndarray,
    projector_ids: np.ndarray,
    camera_points: np.ndarray,
    camera_ids: np.ndarray,
) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    proj_map = {int(idx): projector_points[i] for i, idx in enumerate(projector_ids.tolist())}
    cam_map = {int(idx): camera_points[i] for i, idx in enumerate(camera_ids.tolist())}
    common = sorted(set(proj_map).intersection(cam_map))
    if len(common) < 4:
        return None
    proj = np.asarray([proj_map[idx] for idx in common], dtype=np.float32).reshape((-1, 2))
    cam = np.asarray([cam_map[idx] for idx in common], dtype=np.float32).reshape((-1, 2))
    ids = np.asarray(common, dtype=np.int32)
    return cam, proj, ids


def _draw_polyline(image: np.ndarray, points: np.ndarray, color: Tuple[int, int, int], thickness: int, *, close: bool) -> None:
    pts = ensure_numpy_points(points)
    if pts.shape[0] < 2:
        return
    cv2.polylines(
        image,
        [np.round(pts).astype(np.int32).reshape((-1, 1, 2))],
        isClosed=bool(close),
        color=color,
        thickness=max(1, int(thickness)),
        lineType=cv2.LINE_AA,
    )
