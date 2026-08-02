from __future__ import annotations

from dataclasses import dataclass, field
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from ..geometry import TableGeometry
from ..projection.text import draw_overlay_texts
from ..schemas import OverlayText
from ..utils import clamp, ensure_numpy_points
from .charuco import CharucoBoardSpec, detect_charuco_corners, render_charuco_board
from .projector import (
    MAX_PROJECTION_INLIER_P95_PX,
    MIN_PROJECTION_INLIER_RATIO,
    ProjectionCalibration,
    polygon_quad,
    table_bbox_from_polygon,
)
from .linked_coverage import (
    MAX_LINKED_HOLE_DISTANCE_RATIO,
    MIN_LINKED_CORE_GRID_OCCUPIED_RATIO,
    MIN_LINKED_COVERAGE_HEIGHT_RATIO,
    MIN_LINKED_COVERAGE_HULL_AREA_RATIO,
    MIN_LINKED_COVERAGE_WIDTH_RATIO,
    MIN_LINKED_EDGE_COVERAGE_RATIO,
    MIN_LINKED_TOTAL_MATCHED_POINTS,
    dense_linked_pattern_placements,
    evaluate_linked_point_coverage,
    linked_coverage_errors,
)
from .linked_detection import detect_linked_charuco_corners


PointArray = np.ndarray
MAX_LINKED_PATTERN_CV_P95_PX = 5.0
MIN_LINKED_PATTERN_MATCHED_POINTS = 4
MAX_LINKED_RESIDUAL_CONTROL_POINTS = 120
LINKED_CALIBRATION_ALGORITHM_VERSION = "linked-geometry-v11"


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
    attempts: int = 1
    diagnostic_frame: Optional[np.ndarray] = field(default=None, repr=False)


def linked_calibration_runtime_summary() -> str:
    return (
        f"{LINKED_CALIBRATION_ALGORITHM_VERSION} | "
        f"coverage={MIN_LINKED_COVERAGE_WIDTH_RATIO:.0%}/"
        f"{MIN_LINKED_COVERAGE_HEIGHT_RATIO:.0%}/"
        f"{MIN_LINKED_COVERAGE_HULL_AREA_RATIO:.0%} | "
        "core_grid=5x4:100% | edges=100% | "
        f"max_hole={MAX_LINKED_HOLE_DISTANCE_RATIO:.2f} | "
        "dense_tiles=5x4 | middle_pockets=edge_v2 | "
        "corner_pockets=edge_v2 | capture_retry=2 | "
        f"pattern_min={MIN_LINKED_PATTERN_MATCHED_POINTS} total_min={MIN_LINKED_TOTAL_MATCHED_POINTS} | "
        "edge_contrast=raw_plus_clahe_v2 | "
        f"source={Path(__file__).resolve()}"
    )


def collect_linked_pattern_observation(
    pattern: LinkedCalibrationPattern,
    read_frame: Callable[[], Optional[np.ndarray]],
    *,
    undistort_points: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    transition_frames: int = 8,
    max_detection_frames: int = 18,
    attempts: int = 1,
    minimum_matched_points: int = MIN_LINKED_PATTERN_MATCHED_POINTS,
    inter_frame_delay_seconds: float = 0.0,
    on_frame: Optional[Callable[[np.ndarray], None]] = None,
) -> LinkedPatternCaptureResult:
    """Flush transition frames, then retain the strongest pattern observation.

    Camera backends may return buffered frames after the projector switches to a
    new image. Those frames must not be associated with the new projector
    coordinates, especially because adjacent focus patterns reuse ChArUco IDs.
    """
    transition_read = 0
    detection_read = 0
    matched_frames = 0
    attempts_completed = 0
    diagnostic_frame: Optional[np.ndarray] = None
    delay = max(0.0, float(inter_frame_delay_seconds))
    best: Optional[LinkedCalibrationObservation] = None
    for _attempt_index in range(max(1, int(attempts))):
        attempts_completed += 1
        for _ in range(max(0, int(transition_frames))):
            frame = read_frame()
            if frame is None or frame.size == 0:
                if delay > 0.0:
                    time.sleep(delay)
                continue
            transition_read += 1
            diagnostic_frame = frame
            if on_frame is not None:
                on_frame(frame)
            if delay > 0.0:
                time.sleep(delay)
        for _ in range(max(1, int(max_detection_frames))):
            frame = read_frame()
            if frame is None or frame.size == 0:
                if delay > 0.0:
                    time.sleep(delay)
                continue
            detection_read += 1
            diagnostic_frame = frame
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
        if best is not None and best.matched_count >= max(4, int(minimum_matched_points)):
            break

    return LinkedPatternCaptureResult(
        observation=best,
        transition_frames_read=transition_read,
        detection_frames_read=detection_read,
        matched_frames=matched_frames,
        attempts=attempts_completed,
        diagnostic_frame=diagnostic_frame,
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
    tile_spec = CharucoBoardSpec(squares_x=6, squares_y=5, square_length_m=0.030, marker_length_m=0.022)

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
    edge_focus_positions = {
        0: (0.10, 0.10),
        1: (0.50, 0.06),
        2: (0.90, 0.10),
        3: (0.90, 0.90),
        4: (0.50, 0.94),
        5: (0.10, 0.90),
    }
    for idx, point in enumerate(pocket_centers[:6]):
        x_ratio, y_ratio = edge_focus_positions.get(idx, (0.5, 0.5))
        focus_point = np.asarray(
            [
                projector_bbox[0] + x_ratio * (projector_bbox[2] - projector_bbox[0]),
                projector_bbox[1] + y_ratio * (projector_bbox[3] - projector_bbox[1]),
            ],
            dtype=np.float32,
        )
        roi = _roi_around_point(focus_point, projector_bbox, width, height, width_ratio=0.24, height_ratio=0.30)
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
                focus_point,
            )
        )

    for placement in dense_linked_pattern_placements():
        focus_point = np.asarray(
            [
                projector_bbox[0] + placement.center_x_ratio * (projector_bbox[2] - projector_bbox[0]),
                projector_bbox[1] + placement.center_y_ratio * (projector_bbox[3] - projector_bbox[1]),
            ],
            dtype=np.float32,
        )
        roi = _roi_around_point(
            focus_point,
            projector_bbox,
            width,
            height,
            width_ratio=placement.width_ratio,
            height_ratio=placement.height_ratio,
        )
        patterns.append(
            _make_charuco_pattern(
                placement.pattern_id,
                placement.title,
                tile_spec,
                width,
                height,
                roi,
                placement.emphasis_zone,
                projector_bbox,
                outline_proj,
                inner_proj,
                pockets_proj,
                focus_point,
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


def linked_table_surface_polygon(
    geometry: TableGeometry,
    frame_size: Tuple[int, int],
    *,
    undistort_points: Optional[Callable[[np.ndarray], np.ndarray]] = None,
) -> np.ndarray:
    """Return the coplanar cloth quad used by linked calibration coverage.

    ``outer`` can include raised rails or even the whole camera frame in legacy
    geometry files.  ChArUco calibration is a planar cloth measurement, so the
    pocket-aware inner boundary is preferred and ``outer`` is only a fallback.
    """

    frame_w, frame_h = [max(1, int(value)) for value in frame_size]
    outer_px, inner_px, _ = geometry.scaled(frame_w, frame_h)
    minimum_area = float(frame_w * frame_h) * 0.10
    for candidate in (inner_px, outer_px):
        points = ensure_numpy_points(candidate).astype(np.float32)
        if points.shape[0] < 4:
            continue
        if undistort_points is not None:
            points = ensure_numpy_points(undistort_points(points)).astype(np.float32)
        quad = polygon_quad(points)
        if quad.shape[0] == 4 and abs(float(cv2.contourArea(quad.reshape((-1, 1, 2))))) >= minimum_area:
            return quad.astype(np.float32)
    return np.zeros((0, 2), dtype=np.float32)


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
    camera_points, camera_ids = detect_linked_charuco_corners(
        frame_bgr,
        pattern.board_spec,
        pattern.emphasis_zone,
    )
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
    del minimum_pocket_zones  # Kept in the interface for older callers; names no longer gate coverage.
    cam_poly = polygon_quad(ensure_numpy_points(table_polygon_cam))
    if cam_poly.shape[0] != 4:
        raise ValueError(
            "Linked calibration requires a valid four-corner camera table polygon anchor."
        )
    usable = [obs for obs in observations if obs.matched_count >= MIN_LINKED_PATTERN_MATCHED_POINTS]
    if len(usable) < 2:
        raise ValueError("联动校正至少需要两个有效采样图样。")
    camera_points = np.vstack([obs.camera_points for obs in usable]).astype(np.float64)
    projector_points = np.vstack([obs.projector_points for obs in usable]).astype(np.float64)
    if camera_points.shape[0] < MIN_LINKED_TOTAL_MATCHED_POINTS or projector_points.shape[0] < MIN_LINKED_TOTAL_MATCHED_POINTS:
        raise ValueError(
            f"联动校正有效对应点不足，至少需要 {MIN_LINKED_TOTAL_MATCHED_POINTS} 个累计匹配角点。"
        )
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
    base_projection = projection.camera_to_projector_points(camera_points, refined=False)
    base_errors = np.linalg.norm(base_projection - projector_points, axis=1)
    coverage_inliers = base_errors <= float(projection.quality_report.get("ransac_threshold_px", 3.0))
    coverage_camera_points = camera_points[coverage_inliers]
    spatial_coverage = evaluate_linked_point_coverage(coverage_camera_points, cam_poly)
    coverage_errors = linked_coverage_errors(
        spatial_coverage,
        matched_point_count=int(spatial_coverage.get("evaluated_point_count", 0)),
    )
    if coverage_errors:
        raise RuntimeError(
            "Linked calibration rejected: incomplete full-table coverage; "
            + "; ".join(coverage_errors)
        )
    coverage_gate = "uniform_grid_and_segmented_edges"
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
        normalized_controls = cv2.perspectiveTransform(
            controls_camera.reshape((-1, 1, 2)),
            camera_to_normalized,
        ).reshape((-1, 2)).astype(np.float64)
        projector_controls = projection.camera_to_projector_points(
            controls_camera,
        ).astype(np.float64)
        selected = _spread_control_indices(normalized_controls, MAX_LINKED_RESIDUAL_CONTROL_POINTS)
        projection.table_control_points_norm = normalized_controls[selected]
        projection.table_control_points_proj = projector_controls[selected]
    summary = {
        "patterns_total": len(observations),
        "patterns_used": len(usable),
        "matched_points_total": int(camera_points.shape[0]),
        "coverage_inlier_points_total": int(spatial_coverage.get("evaluated_point_count", 0)),
        "matched_points_by_pattern": {obs.pattern_id: int(obs.matched_count) for obs in usable},
        "zones": sorted(zones),
        "pocket_zones_used": len(pocket_zones),
        "coverage_gate": coverage_gate,
        "spatial_coverage": spatial_coverage,
        "geometry_model": "independent_2d",
        "camera_extrinsics_used": False,
    }
    projection.quality_report.update({
        "workflow": "linked_hybrid_charuco",
        "patterns_total": summary["patterns_total"],
        "patterns_used": summary["patterns_used"],
        "matched_points_total": summary["matched_points_total"],
        "coverage_inlier_points_total": summary["coverage_inlier_points_total"],
        "matched_points_by_pattern": summary["matched_points_by_pattern"],
        "zones": summary["zones"],
        "pocket_zones_used": summary["pocket_zones_used"],
        "quality_gate_passed": True,
        "minimum_inlier_ratio": float(MIN_PROJECTION_INLIER_RATIO),
        "coverage_gate": coverage_gate,
        "spatial_coverage": spatial_coverage,
        "minimum_coverage_width_ratio": float(MIN_LINKED_COVERAGE_WIDTH_RATIO),
        "minimum_coverage_height_ratio": float(MIN_LINKED_COVERAGE_HEIGHT_RATIO),
        "minimum_coverage_hull_area_ratio": float(MIN_LINKED_COVERAGE_HULL_AREA_RATIO),
        "minimum_core_grid_occupied_ratio": float(MIN_LINKED_CORE_GRID_OCCUPIED_RATIO),
        "minimum_edge_coverage_ratio": float(MIN_LINKED_EDGE_COVERAGE_RATIO),
        "maximum_hole_distance_ratio": float(MAX_LINKED_HOLE_DISTANCE_RATIO),
        "minimum_total_matched_points": int(MIN_LINKED_TOTAL_MATCHED_POINTS),
        "minimum_pattern_matched_points": int(MIN_LINKED_PATTERN_MATCHED_POINTS),
        "maximum_inlier_p95_px": float(MAX_PROJECTION_INLIER_P95_PX),
        "pattern_cv_p95_px": pattern_cv_p95,
        "maximum_pattern_cv_p95_px": float(MAX_LINKED_PATTERN_CV_P95_PX),
    })
    projection.quality_report.update(projection.calibration_error_stats())
    return LinkedCalibrationResult(projection=projection, observations=usable, summary=summary)


def _linked_spatial_coverage(camera_points: np.ndarray, table_polygon_cam: np.ndarray) -> Dict[str, float]:
    return evaluate_linked_point_coverage(camera_points, table_polygon_cam)  # type: ignore[return-value]


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


def _spread_control_indices(points: np.ndarray, maximum_count: int) -> np.ndarray:
    """Keep a bounded, spatially spread residual support set for runtime fitting."""

    pts = np.asarray(points, dtype=np.float64).reshape((-1, 2))
    limit = max(4, int(maximum_count))
    if pts.shape[0] <= limit:
        return np.arange(pts.shape[0], dtype=np.int32)
    center = np.mean(pts, axis=0)
    first = int(np.argmax(np.linalg.norm(pts - center.reshape((1, 2)), axis=1)))
    selected = [first]
    minimum_distance = np.linalg.norm(pts - pts[first].reshape((1, 2)), axis=1)
    minimum_distance[first] = -1.0
    while len(selected) < limit:
        index = int(np.argmax(minimum_distance))
        selected.append(index)
        distance = np.linalg.norm(pts - pts[index].reshape((1, 2)), axis=1)
        minimum_distance = np.minimum(minimum_distance, distance)
        minimum_distance[np.asarray(selected, dtype=np.int32)] = -1.0
    return np.asarray(selected, dtype=np.int32)


def projection_output_summary(
    result: LinkedCalibrationResult,
    *,
    geometry_report: Optional[Dict[str, object]] = None,
) -> str:
    report = dict(result.projection.quality_report)
    stats = result.projection.calibration_error_stats()
    p95 = stats.get("p95_px", stats.get("max_px", 0.0))
    coverage = report.get("spatial_coverage", {})
    if not isinstance(coverage, dict):
        coverage = {}
    lines = [
        f"图样 {report.get('patterns_used', 0)}/{report.get('patterns_total', 0)} | "
        f"匹配角点 {report.get('matched_points_total', 0)} | "
        f"mean={stats.get('mean_px', 0.0):.2f}px p95={p95:.2f}px max={stats.get('max_px', 0.0):.2f}px"
    ]
    if coverage:
        lines.append(
            "覆盖率 "
            f"宽={float(coverage.get('width_ratio', 0.0)):.1%} "
            f"高={float(coverage.get('height_ratio', 0.0)):.1%} "
            f"凸包={float(coverage.get('hull_area_ratio', 0.0)):.1%} | "
            f"袋口区域={int(report.get('pocket_zones_used', 0))} | "
            f"跨图样CV P95={float(report.get('pattern_cv_p95_px', 0.0)):.2f}px"
        )
        edge = coverage.get("edge_coverage", {})
        if not isinstance(edge, dict):
            edge = {}
        lines.append(
            "均匀覆盖 "
            f"5×4网格={float(coverage.get('core_grid_occupied_ratio', 0.0)):.1%} | "
            f"四边=上{float(edge.get('top', 0.0)):.0%}/"
            f"右{float(edge.get('right', 0.0)):.0%}/"
            f"下{float(edge.get('bottom', 0.0)):.0%}/"
            f"左{float(edge.get('left', 0.0)):.0%} | "
            f"最大空洞={float(coverage.get('maximum_hole_distance_ratio', 0.0)):.1%}"
        )
    runtime = dict(geometry_report or {})
    if "projector_residual_support_grid_ratio" in runtime:
        lines.append(
            "投影残差支撑 "
            f"全域={float(runtime.get('projector_residual_support_grid_ratio', 0.0)):.1%} "
            f"边缘={float(runtime.get('projector_residual_support_edge_ratio', 0.0)):.1%}"
        )
    return "\n".join(lines)


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
        label_text = str(label)
        label_center_x = x1 + min(max(130, int(round((x2 - x1) * 0.18))), 260)
        draw_overlay_texts(
            image,
            [
                OverlayText(
                    position=(float(label_center_x), float(max(36, y1 + 30))),
                    text=label_text,
                    color=(255, 255, 255),
                    font_size_px=28.0,
                    max_width_ratio=0.38,
                    outline_width_px=1.0,
                    background_alpha=120,
                )
            ],
        )
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
    # With no prior projection, keep only a tiny safety inset. The dense edge
    # sweep needs actual control points close to all four table boundaries.
    margin_x = width * 0.01
    margin_y = height * 0.01
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
