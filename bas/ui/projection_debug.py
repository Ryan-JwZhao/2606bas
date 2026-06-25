from __future__ import annotations

from typing import Iterable

import numpy as np

from ..projection.overlay import ROUTE_COLOR
from ..calibration.service import CalibrationService
from ..schemas import Detection, ProjectionOverlay, TrackObservation
from ..utils import group_from_class


BALL_GROUPS = {"cue", "solid", "stripe", "black"}


def append_projected_ball_overlays(
    overlay: ProjectionOverlay,
    calibration: CalibrationService,
    *,
    tracks: Iterable[TrackObservation] = (),
    detections: Iterable[Detection] = (),
) -> int:
    appended = 0
    visible_tracks = [track for track in tracks if _track_is_projectable_ball(track)]
    if visible_tracks:
        for track in visible_tracks:
            if _append_projected_ball_marker(
                overlay,
                calibration,
                center_px=track.center_px,
                radius_px=track.radius_px,
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
        ):
            appended += 1
    return appended


def _append_projected_ball_marker(
    overlay: ProjectionOverlay,
    calibration: CalibrationService,
    *,
    center_px,
    radius_px: float,
) -> bool:
    cx, cy = [float(v) for v in center_px]
    radius = float(max(1.0, radius_px))
    refs = np.asarray(
        [
            [cx, cy],
            [cx + radius, cy],
            [cx, cy + radius],
        ],
        dtype=np.float32,
    )
    try:
        proj = calibration.camera_px_to_projector_px(refs).astype(np.float32)
    except Exception:
        return False
    rx = float(np.linalg.norm(proj[1] - proj[0]))
    ry = float(np.linalg.norm(proj[2] - proj[0]))
    radius_proj = max(4.0, 0.5 * (rx + ry))
    center = (float(proj[0, 0]), float(proj[0, 1]))
    overlay.circles.append((center, radius_proj, ROUTE_COLOR))
    return True


def _track_is_projectable_ball(track: TrackObservation) -> bool:
    if int(getattr(track, "lost_frames", 0)) > 0:
        return False
    if str(getattr(track, "visibility", "visible")) != "visible":
        return False
    return str(getattr(track, "group", "")) in BALL_GROUPS
