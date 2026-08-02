from __future__ import annotations

import re
from typing import Tuple

import cv2
import numpy as np

from .charuco import CharucoBoardSpec, detect_charuco_corners


PointArray = np.ndarray
EDGE_RECOVERY_TARGET_CORNERS = 12
_COVERAGE_ZONE = re.compile(r"^coverage_r(?P<row>\d+)_c(?P<col>\d+)$")


def detect_linked_charuco_corners(
    frame_bgr: np.ndarray,
    spec: CharucoBoardSpec,
    emphasis_zone: str,
) -> Tuple[PointArray, np.ndarray]:
    """Detect linked-calibration corners, recovering contrast at physical edges.

    The installed camera/projector pair is substantially darker along the left
    and bottom edges.  The normal image is retained whenever it supplies enough
    corners; an edge-only CLAHE pass is selected only when it finds more IDs.
    """

    points, ids = detect_charuco_corners(frame_bgr, spec)
    if not is_edge_recovery_zone(emphasis_zone) or ids.size >= EDGE_RECOVERY_TARGET_CORNERS:
        return points, ids

    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    enhanced_gray = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    enhanced_bgr = cv2.cvtColor(enhanced_gray, cv2.COLOR_GRAY2BGR)
    recovered_points, recovered_ids = detect_charuco_corners(enhanced_bgr, spec)
    if recovered_ids.size > ids.size:
        return recovered_points, recovered_ids
    return points, ids


def is_edge_recovery_zone(emphasis_zone: str) -> bool:
    zone = str(emphasis_zone).strip().lower()
    if zone.startswith("pocket_"):
        return True
    match = _COVERAGE_ZONE.fullmatch(zone)
    if match is None:
        return False
    row = int(match.group("row"))
    col = int(match.group("col"))
    return row in {1, 4} or col in {1, 5}
