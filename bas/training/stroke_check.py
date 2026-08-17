from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from ..config import ProjectionConfig
from ..schemas import OverlayLine, ProjectionOverlay


STROKE_CHECK_SCENARIO_ID = "OTHER_STROKE_CHECK"
STROKE_CHECK_RENDERER_ID = "stroke_check"
INCH_MM = 25.4


class TableProjection(Protocol):
    table: object

    def table_mm_to_projector_px(self, points: np.ndarray) -> np.ndarray:
        ...


@dataclass(frozen=True)
class StrokeCheckDimensions:
    """Field-sized stroke guide derived from the printable reference."""

    ball_diameter_mm: float = 2.25 * INCH_MM
    ball_outline_mm: float = 1.68
    # The printable body is 250 mm long. On the calibrated 1280x800 field
    # projector it occupied only about 122 px, so the working body is 1.5x.
    # The cue-ball circle intentionally remains regulation size.
    ball_center_from_rear_mm: float = 375.0
    cue_band_width_mm: float = 12.17
    cue_band_end_mm: float = 375.0 - (2.25 * INCH_MM * 0.5)
    checkpoint_first_mm: float = 3.0 * INCH_MM
    checkpoint_spacing_mm: float = 4.5 * INCH_MM
    checkpoint_length_mm: float = 139.14
    checkpoint_width_mm: float = 1.32
    guide_half_angle_deg: float = 3.5
    rear_guide_half_width_mm: float | None = None
    guide_width_mm: float = 0.53
    minimum_mark_width_px: int = 2

    @property
    def checkpoint_centers_mm(self) -> tuple[float, float, float]:
        return tuple(
            self.checkpoint_first_mm + index * self.checkpoint_spacing_mm
            for index in range(3)
        )

    @property
    def resolved_rear_guide_half_width_mm(self) -> float:
        if self.rear_guide_half_width_mm is not None:
            return float(self.rear_guide_half_width_mm)
        band_half = 0.5 * float(self.cue_band_width_mm)
        guide_run = max(0.0, float(self.cue_band_end_mm))
        return band_half + math.tan(math.radians(float(self.guide_half_angle_deg))) * guide_run


class StrokeCheckOverlayBuilder:
    """Render the print template in calibrated table millimetres.

    The cue-ball circle is anchored at one quarter of the table length and on
    the table centreline. The shot points toward the top long rail, therefore
    the cue band is perpendicular to that rail.
    """

    COLOR = (255, 255, 255)

    def __init__(
        self,
        config: ProjectionConfig,
        calibration: TableProjection,
        dimensions: StrokeCheckDimensions | None = None,
    ):
        self.config = config
        self.calibration = calibration
        self.dimensions = dimensions or StrokeCheckDimensions()

    def build(self, *, frame_id: int = 0) -> ProjectionOverlay:
        overlay = ProjectionOverlay(
            overlay_id=f"training_{STROKE_CHECK_SCENARIO_ID}_{int(frame_id)}",
            frame_id=int(frame_id),
            projector_size=(
                int(self.config.projector_width),
                int(self.config.projector_height),
            ),
            suppress_star_formula=True,
        )
        table = self.calibration.table
        ball_center = np.asarray(
            [
                0.25 * float(getattr(table, "width_mm")),
                0.50 * float(getattr(table, "height_mm")),
            ],
            dtype=np.float32,
        )
        shot_direction = np.asarray([0.0, -1.0], dtype=np.float32)
        transverse = np.asarray([1.0, 0.0], dtype=np.float32)
        rear = ball_center - shot_direction * float(
            self.dimensions.ball_center_from_rear_mm
        )

        for index, center_u in enumerate(self.dimensions.checkpoint_centers_mm):
            center = rear + shot_direction * float(center_u)
            half = 0.5 * float(self.dimensions.checkpoint_length_mm)
            self._append_physical_line(
                overlay,
                [center - transverse * half, center + transverse * half],
                physical_width_mm=self.dimensions.checkpoint_width_mm,
                width_axis=shot_direction,
                minimum_width_px=self.dimensions.minimum_mark_width_px,
                label=f"stroke_check_checkpoint_{index + 1}",
            )

        band_half = 0.5 * float(self.dimensions.cue_band_width_mm)
        guide_end = rear + shot_direction * float(
            self.dimensions.cue_band_end_mm
        )
        for sign in (-1.0, 1.0):
            self._append_physical_line(
                overlay,
                [
                    rear
                    + transverse
                    * (
                        sign
                        * float(
                            self.dimensions.resolved_rear_guide_half_width_mm
                        )
                    ),
                    guide_end + transverse * (sign * band_half),
                ],
                physical_width_mm=self.dimensions.guide_width_mm,
                width_axis=transverse,
                minimum_width_px=self.dimensions.minimum_mark_width_px,
                label=(
                    "stroke_check_guide_left"
                    if sign < 0.0
                    else "stroke_check_guide_right"
                ),
            )

        self._append_physical_line(
            overlay,
            [rear, guide_end],
            physical_width_mm=self.dimensions.cue_band_width_mm,
            width_axis=transverse,
            label="stroke_check_cue_direction",
        )

        radius = 0.5 * float(self.dimensions.ball_diameter_mm)
        circle_points = []
        for angle in np.linspace(0.0, math.tau, 97):
            circle_points.append(
                ball_center
                + transverse * (math.cos(float(angle)) * radius)
                + shot_direction * (math.sin(float(angle)) * radius)
            )
        self._append_physical_line(
            overlay,
            circle_points,
            physical_width_mm=self.dimensions.ball_outline_mm,
            width_axis=transverse,
            minimum_width_px=self.dimensions.minimum_mark_width_px,
            label="stroke_check_shot_direction_circle",
        )
        return overlay

    def _append_physical_line(
        self,
        overlay: ProjectionOverlay,
        points_mm,
        *,
        physical_width_mm: float,
        width_axis: np.ndarray,
        minimum_width_px: int = 1,
        label: str,
    ) -> None:
        points = np.asarray(points_mm, dtype=np.float32).reshape((-1, 2))
        projected = self.calibration.table_mm_to_projector_px(points)
        midpoint = np.mean(points, axis=0)
        half_width = 0.5 * max(0.1, float(physical_width_mm))
        width_samples = self.calibration.table_mm_to_projector_px(
            np.asarray(
                [
                    midpoint - width_axis * half_width,
                    midpoint + width_axis * half_width,
                ],
                dtype=np.float32,
            )
        )
        width_px = max(
            max(1, int(minimum_width_px)),
            int(round(float(np.linalg.norm(width_samples[1] - width_samples[0])))),
        )
        overlay.lines.append(
            OverlayLine(
                points=[(float(x), float(y)) for x, y in projected],
                color=self.COLOR,
                width=width_px,
                label=label,
            )
        )


def build_stroke_check_overlay(
    config: ProjectionConfig,
    calibration: TableProjection,
    *,
    frame_id: int = 0,
) -> ProjectionOverlay:
    return StrokeCheckOverlayBuilder(config, calibration).build(frame_id=frame_id)
