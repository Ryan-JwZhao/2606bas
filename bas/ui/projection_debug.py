from __future__ import annotations

from typing import Iterable

import numpy as np

from ..projection.overlay import ROUTE_COLOR
from ..calibration.service import CalibrationService
from ..schemas import Detection, OverlayCircle, OverlayLine, ProjectionOverlay, TrackObservation
from ..tracking.confirmation import is_track_confirmed
from ..utils import group_from_class


BALL_GROUPS = {"cue", "solid", "stripe", "black"}
BOUNDARY_OVERLAY_SPECS = (
    ("projection_visible_polygon_mm", "debug_visible_boundary", "visible", (255, 255, 0), 2),
    ("inner_polygon_mm", "debug_physical_boundary", "physical", (255, 0, 255), 2),
    ("center_playable_polygon_mm", "debug_center_boundary", "center", (80, 80, 255), 2),
)


def append_projected_boundary_overlays(
    overlay: ProjectionOverlay,
    calibration: CalibrationService,
) -> int:
    appended = 0
    table = calibration.table
    for attr_name, line_label, text_label, color, width in BOUNDARY_OVERLAY_SPECS:
        polygon = list(getattr(table, attr_name, []) or [])
        if attr_name != "inner_polygon_mm" and not polygon:
            polygon = list(table.inner_polygon_mm or [])
        if _append_projected_boundary_line(
            overlay,
            calibration,
            polygon_mm=polygon,
            line_label=line_label,
            text_label=text_label,
            color=color,
            width=width,
        ):
            appended += 1
    return appended


def append_projected_ball_overlays(
    overlay: ProjectionOverlay,
    calibration: CalibrationService,
    *,
    tracks: Iterable[TrackObservation] = (),
    detections: Iterable[Detection] = (),
) -> int:
    appended = 0
    track_list = list(tracks)
    visible_tracks = [track for track in track_list if _track_is_projectable_ball(track)]
    if track_list:
        for track in visible_tracks:
            if _append_projected_ball_marker(
                overlay,
                calibration,
                center_px=track.center_px,
                radius_px=track.radius_px,
                geometry_quality=float(getattr(track, "geometry_quality", track.quality)),
                geometry_method=str(getattr(track, "geometry_method", "unknown")),
            ):
                appended += 1
        return appended

    for det in detections:
        if group_from_class(getattr(det, "cls_name", "")) not in BALL_GROUPS:
            continue
        if _append_projected_ball_marker(
            overlay,
            calibration,
            center_px=det.center,
            radius_px=det.radius_px,
            geometry_quality=float(det.geometry_quality),
            geometry_method=str(det.geometry_method),
        ):
            appended += 1
    return appended


def _append_projected_ball_marker(
    overlay: ProjectionOverlay,
    calibration: CalibrationService,
    *,
    center_px,
    radius_px: float,
    geometry_quality: float,
    geometry_method: str,
) -> bool:
    cx, cy = [float(v) for v in center_px]
    try:
        ellipse = calibration.ball_geometry.locate(
            (cx, cy),
            radius_px=float(radius_px),
            geometry_quality=float(geometry_quality),
            geometry_method=str(geometry_method),
        ).projector_ellipse
    except Exception:
        return False
    overlay.circles.append(
        OverlayCircle(
            center=ellipse.center_px,
            radius=max(4.0, float(ellipse.radius_x_px)),
            radius_y=max(4.0, float(ellipse.radius_y_px)),
            rotation_deg=float(ellipse.rotation_deg),
            color=ROUTE_COLOR,
        )
    )
    return True


def _append_projected_boundary_line(
    overlay: ProjectionOverlay,
    calibration: CalibrationService,
    *,
    polygon_mm,
    line_label: str,
    text_label: str,
    color: tuple[int, int, int],
    width: int,
) -> bool:
    pts = np.asarray(polygon_mm, dtype=np.float32).reshape((-1, 2))
    if pts.shape[0] < 3:
        return False
    try:
        proj = calibration.table_mm_to_projector_px(pts).astype(np.float32)
    except Exception:
        return False
    closed = np.vstack([proj, proj[0]])
    points = [(float(x), float(y)) for x, y in closed]
    overlay.lines.append(OverlayLine(points=points, color=color, width=width, label=line_label))
    anchor = proj[0]
    overlay.labels.append(((float(anchor[0] + 12.0), float(anchor[1] - 8.0)), text_label, color))
    return True


def _track_is_projectable_ball(track: TrackObservation) -> bool:
    if not is_track_confirmed(track):
        return False
    if int(getattr(track, "lost_frames", 0)) > 0:
        return False
    if str(getattr(track, "visibility", "visible")) != "visible":
        return False
    return str(getattr(track, "group", "")) in BALL_GROUPS
