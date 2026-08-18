from __future__ import annotations

from collections import Counter, OrderedDict
from dataclasses import replace
from typing import Hashable, Sequence

import cv2
import numpy as np

from .schemas import OverlayCircle, OverlayLine, ProjectionOverlay


DISPLAY_GEOMETRY_DEADBAND_PX = 0.35
DISPLAY_SHAPE_DEADBAND_PX = 0.75
SUBPIXEL_BITS = 8
SUBPIXEL_SCALE = 1 << SUBPIXEL_BITS


class DisplayGeometryStabilizer:
    """Suppress sub-pixel display noise without modifying source geometry.

    The stored state belongs exclusively to a render target. Values farther than
    ``deadband_px`` from the last displayed value pass through on the same frame;
    values inside the deadband reuse the previous displayed value. Callers retain
    their original tracking, calibration, and planning values.
    """

    def __init__(
        self,
        deadband_px: float = DISPLAY_GEOMETRY_DEADBAND_PX,
        *,
        max_entries: int = 512,
    ) -> None:
        self.deadband_px = max(0.0, float(deadband_px))
        self.max_entries = max(1, int(max_entries))
        self._values: OrderedDict[Hashable, np.ndarray] = OrderedDict()

    def reset(self) -> None:
        self._values.clear()

    def stabilize(self, key: Hashable, values) -> np.ndarray:
        source = np.asarray(values, dtype=np.float32)
        previous = self._values.pop(key, None)
        if previous is None or previous.shape != source.shape:
            shown = source.copy()
        elif source.ndim >= 1 and source.shape[-1] == 2:
            shown = previous.copy()
            moving = np.linalg.norm(source - previous, axis=-1) > self.deadband_px
            shown[moving] = source[moving]
        else:
            shown = np.where(np.abs(source - previous) > self.deadband_px, source, previous).astype(
                np.float32,
                copy=False,
            )
        self._values[key] = shown.copy()
        while len(self._values) > self.max_entries:
            self._values.popitem(last=False)
        return shown.copy()


def stabilize_projection_overlay(
    overlay: ProjectionOverlay,
    stabilizer: DisplayGeometryStabilizer,
) -> ProjectionOverlay:
    """Return a presentation-only stabilized copy of a projection overlay."""

    line_occurrences: Counter[str] = Counter()
    lines: list[OverlayLine] = []
    for line in overlay.lines:
        label = str(line.label or "unlabeled")
        occurrence = line_occurrences[label]
        line_occurrences[label] += 1
        points = np.asarray(line.points, dtype=np.float32).reshape((-1, 2))
        shown = stabilizer.stabilize(
            ("projection_line", label, occurrence, points.shape[0]),
            points,
        )
        lines.append(
            replace(
                line,
                points=[(float(point[0]), float(point[1])) for point in shown],
            )
        )

    circles: list[OverlayCircle] = []
    for circle in overlay.circles:
        stability_key = getattr(circle, "stability_key", None)
        if not stability_key:
            circles.append(circle)
            continue
        center = stabilizer.stabilize(
            ("projection_circle_center", str(stability_key)),
            np.asarray(circle.center, dtype=np.float32),
        )
        radii = stabilizer.stabilize(
            ("projection_circle_radius", str(stability_key)),
            np.asarray(
                [
                    float(circle.radius),
                    float(circle.radius if circle.radius_y is None else circle.radius_y),
                ],
                dtype=np.float32,
            ),
        )
        circles.append(
            replace(
                circle,
                center=(float(center[0]), float(center[1])),
                radius=float(radii[0]),
                radius_y=float(radii[1]) if circle.radius_y is not None else None,
            )
        )

    labels = []
    for occurrence, (position, text, color) in enumerate(overlay.labels):
        shown = stabilizer.stabilize(
            ("projection_label", occurrence),
            np.asarray(position, dtype=np.float32),
        )
        labels.append(((float(shown[0]), float(shown[1])), text, color))

    return replace(overlay, lines=lines, circles=circles, labels=labels)


def draw_subpixel_polyline(
    image: np.ndarray,
    points: Sequence[Sequence[float]] | np.ndarray,
    color,
    thickness: int,
    *,
    closed: bool = False,
) -> None:
    values = np.asarray(points, dtype=np.float64).reshape((-1, 2))
    if values.shape[0] < 2:
        return
    fixed = np.round(values * SUBPIXEL_SCALE).astype(np.int32).reshape((-1, 1, 2))
    cv2.polylines(
        image,
        [fixed],
        bool(closed),
        color,
        max(1, int(thickness)),
        cv2.LINE_AA,
        SUBPIXEL_BITS,
    )


def draw_subpixel_line(
    image: np.ndarray,
    start: Sequence[float] | np.ndarray,
    end: Sequence[float] | np.ndarray,
    color,
    thickness: int,
) -> None:
    first = _fixed_point(start)
    second = _fixed_point(end)
    cv2.line(
        image,
        first,
        second,
        color,
        max(1, int(thickness)),
        cv2.LINE_AA,
        SUBPIXEL_BITS,
    )


def draw_subpixel_rectangle(
    image: np.ndarray,
    top_left: Sequence[float] | np.ndarray,
    bottom_right: Sequence[float] | np.ndarray,
    color,
    thickness: int,
) -> None:
    cv2.rectangle(
        image,
        _fixed_point(top_left),
        _fixed_point(bottom_right),
        color,
        max(1, int(thickness)),
        cv2.LINE_AA,
        SUBPIXEL_BITS,
    )


def draw_subpixel_circle(
    image: np.ndarray,
    center: Sequence[float] | np.ndarray,
    radius: float,
    color,
    thickness: int,
) -> None:
    cv2.circle(
        image,
        _fixed_point(center),
        max(1, int(round(float(radius) * SUBPIXEL_SCALE))),
        color,
        max(1, int(thickness)),
        cv2.LINE_AA,
        SUBPIXEL_BITS,
    )


def draw_subpixel_ellipse(
    image: np.ndarray,
    center: Sequence[float] | np.ndarray,
    radii: Sequence[float] | np.ndarray,
    rotation_deg: float,
    color,
    thickness: int,
) -> None:
    axes = np.asarray(radii, dtype=np.float64).reshape((2,))
    cv2.ellipse(
        image,
        _fixed_point(center),
        (
            max(1, int(round(float(axes[0]) * SUBPIXEL_SCALE))),
            max(1, int(round(float(axes[1]) * SUBPIXEL_SCALE))),
        ),
        float(rotation_deg),
        0.0,
        360.0,
        color,
        max(1, int(thickness)),
        cv2.LINE_AA,
        SUBPIXEL_BITS,
    )


def _fixed_point(point: Sequence[float] | np.ndarray) -> tuple[int, int]:
    values = np.asarray(point, dtype=np.float64).reshape((2,))
    return (
        int(round(float(values[0]) * SUBPIXEL_SCALE)),
        int(round(float(values[1]) * SUBPIXEL_SCALE)),
    )


__all__ = [
    "DISPLAY_GEOMETRY_DEADBAND_PX",
    "DISPLAY_SHAPE_DEADBAND_PX",
    "DisplayGeometryStabilizer",
    "draw_subpixel_circle",
    "draw_subpixel_ellipse",
    "draw_subpixel_line",
    "draw_subpixel_polyline",
    "draw_subpixel_rectangle",
    "stabilize_projection_overlay",
]
