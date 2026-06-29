from __future__ import annotations

import math

import cv2
import numpy as np

from ..planning.cue_sector import CueSectorDebugView
from ..utils import unit

FRAME_COLOR = (80, 220, 255)
CANDIDATE_COLOR = (80, 220, 255)
TEXT_COLOR = (240, 240, 240)


def draw_cue_sector_candidate_box(
    image: np.ndarray,
    debug_view: CueSectorDebugView | None,
    *,
    enabled: bool = True,
    ui_scale: float = 1.0,
) -> int:
    if not enabled or debug_view is None:
        return 0

    cue = np.asarray(debug_view.cue_center_px, dtype=np.float32).reshape((2,))
    direction = unit(np.asarray(debug_view.direction_px, dtype=np.float32).reshape((2,)))
    if float(np.linalg.norm(direction)) < 1e-6:
        return 0

    half_width = max(2.0, float(debug_view.half_width_px))
    normal = np.asarray([-float(direction[1]), float(direction[0])], dtype=np.float32)
    corridor_length = float(math.hypot(image.shape[1], image.shape[0]))
    end = cue + direction * corridor_length
    corners = np.asarray(
        [
            cue - normal * half_width,
            end - normal * half_width,
            end + normal * half_width,
            cue + normal * half_width,
        ],
        dtype=np.float32,
    )
    corners_i = np.round(corners).astype(np.int32).reshape((-1, 1, 2))
    thickness = max(1, int(round(2.0 * float(ui_scale))))
    cv2.polylines(image, [corners_i], True, FRAME_COLOR, thickness, cv2.LINE_AA)

    arrow_end = cue + direction * min(corridor_length, 120.0)
    cv2.arrowedLine(
        image,
        tuple(int(round(v)) for v in cue),
        tuple(int(round(v)) for v in arrow_end),
        FRAME_COLOR,
        thickness,
        cv2.LINE_AA,
        tipLength=0.14,
    )

    box_radius = max(6, int(round(7.0 * float(ui_scale))))
    for idx, center in enumerate(debug_view.candidate_centers_px):
        point = np.asarray(center, dtype=np.float32).reshape((2,))
        x = int(round(float(point[0])))
        y = int(round(float(point[1])))
        cv2.rectangle(
            image,
            (x - box_radius, y - box_radius),
            (x + box_radius, y + box_radius),
            CANDIDATE_COLOR,
            thickness,
            cv2.LINE_AA,
        )
        track_id = debug_view.candidate_track_ids[idx] if idx < len(debug_view.candidate_track_ids) else None
        if track_id is not None:
            cv2.putText(
                image,
                f"sector #{track_id}",
                (x + box_radius + 4, y - 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                max(0.35, 0.45 * float(ui_scale)),
                TEXT_COLOR,
                1,
                cv2.LINE_AA,
            )

    status_anchor = cue + normal * (half_width + 18.0)
    cv2.putText(
        image,
        f"corridor {debug_view.status}",
        (int(round(float(status_anchor[0]))), int(round(float(status_anchor[1])))),
        cv2.FONT_HERSHEY_SIMPLEX,
        max(0.35, 0.48 * float(ui_scale)),
        TEXT_COLOR,
        1,
        cv2.LINE_AA,
    )
    return 1 + len(debug_view.candidate_centers_px)
