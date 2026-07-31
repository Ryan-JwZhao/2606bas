from __future__ import annotations

import math

import numpy as np

from ..calibration import CalibrationService
from ..config import ProjectionConfig
from ..schemas import OverlayCircle, OverlayLine, OverlayText, ProjectionOverlay, TracksFrame
from .models import SetupTargetZoneGuide, TrainingStateFrame
from .numbered_tracker import ball_number_from_track
from .projection_drills import build_projection_drill_overlay
from .prompts import projection_prompt_for_state
from .scenarios import get_training_scenario


class TrainingOverlayBuilder:
    def __init__(self, config: ProjectionConfig, calibration: CalibrationService):
        self.config = config
        self.calibration = calibration

    def build(
        self,
        tracks: TracksFrame,
        state: TrainingStateFrame,
        *,
        route_overlay: ProjectionOverlay | None = None,
    ) -> ProjectionOverlay:
        scenario = get_training_scenario(state.scenario_id)
        if scenario.projection_only:
            return build_projection_drill_overlay(
                scenario,
                self.config,
                self.calibration,
                frame_id=state.frame_id,
            )
        overlay = ProjectionOverlay(
            overlay_id=f"training_{state.scenario_id}_{state.frame_id}",
            frame_id=state.frame_id,
            projector_size=(int(self.config.projector_width), int(self.config.projector_height)),
            lines=list(route_overlay.lines) if route_overlay is not None else [],
            circles=list(route_overlay.circles) if route_overlay is not None else [],
            labels=list(route_overlay.labels) if route_overlay is not None else [],
            texts=list(route_overlay.texts) if route_overlay is not None else [],
        )
        if state.phase in {"setup", "ready"}:
            self._append_setup_target_zone_guides(overlay, state.setup_target_zones)
        if state.cue_ball_goal_polygon_mm:
            try:
                goal_points = self.calibration.table_mm_to_projector_px(
                    np.asarray(state.cue_ball_goal_polygon_mm, dtype=np.float32)
                )
            except Exception:
                goal_points = np.empty((0, 2), dtype=np.float32)
            if goal_points.shape[0] >= 3:
                goal_color = (
                    (0, 220, 80)
                    if state.cue_ball_goal_result == "passed"
                    else (0, 80, 255)
                    if state.cue_ball_goal_result == "failed"
                    else (0, 220, 255)
                )
                overlay.lines.append(
                    OverlayLine(
                        points=[(float(x), float(y)) for x, y in goal_points],
                        color=goal_color,
                        width=4,
                        label="cue_ball_goal",
                        style="dashed",
                    )
                )
        expected = set(state.expected_numbers)
        for track in tracks.tracks:
            if track.visibility != "visible":
                continue
            number = ball_number_from_track(track)
            if number is None:
                continue
            try:
                point = self.calibration.ball_camera_px_to_projector_px(
                    np.asarray([track.center_px], dtype=np.float32)
                )[0]
                ellipse = self.calibration.ball_projector_ellipse(track.center_px)
                radius = 0.5 * (float(ellipse.radius_x_px) + float(ellipse.radius_y_px))
            except Exception:
                continue
            center = ellipse.center_px
            color = (255, 255, 255) if number == 0 else ((0, 220, 255) if number in expected else (160, 160, 160))
            overlay.circles.append(
                OverlayCircle(
                    center=center,
                    radius=max(8.0, float(ellipse.radius_x_px) * 1.25),
                    radius_y=max(8.0, float(ellipse.radius_y_px) * 1.25),
                    rotation_deg=float(ellipse.rotation_deg),
                    color=color,
                    width=3,
                )
            )
            overlay.labels.append(((center[0] + radius, center[1] - radius), str(number), color))

        phase_text = str(state.phase or "setup").upper()
        status_color = (0, 220, 120) if state.phase == "passed" else ((0, 80, 255) if state.phase == "failed" else (255, 255, 255))
        overlay.labels.append(((36.0, 48.0), f"TRAINING · {state.scenario_id}", status_color))
        target_text = " / ".join(str(number) for number in state.expected_numbers) or "-"
        overlay.labels.append(((36.0, 82.0), f"{phase_text} · TARGET {target_text}", status_color))
        overlay.labels.append(
            ((36.0, 116.0), f"PROGRESS {state.progress_current}/{state.progress_total}  TIME {state.elapsed_s:.1f}s", status_color)
        )
        if state.mode_hint == "自由选择模式":
            remaining = " / ".join(str(number) for number in state.remaining_numbers) or "-"
            overlay.labels.append(((36.0, 150.0), f"FREE ORDER · REMAINING {remaining} · ERRORS {state.error_count}", status_color))
        if self.config.training_prompt_enabled:
            prompt = projection_prompt_for_state(state)
            width, height = overlay.projector_size
            prompt_position = (
                float(width) * float(self.config.training_prompt_x_pct) / 100.0,
                float(height) * float(self.config.training_prompt_y_pct) / 100.0,
            )
            overlay.texts.append(
                OverlayText(
                    position=prompt_position,
                    text=prompt.text,
                    color=prompt.color,
                    font_size_px=float(self.config.training_prompt_font_size_px),
                    max_width_ratio=0.9,
                    outline_width_px=max(1.0, float(self.config.training_prompt_font_size_px) / 18.0),
                    background_alpha=105,
                    rotation_deg=self._table_axis_rotation_deg(projector_anchor=prompt_position),
                )
            )
        return overlay

    def _append_setup_target_zone_guides(
        self,
        overlay: ProjectionOverlay,
        guides: list[SetupTargetZoneGuide],
    ) -> None:
        palette = (
            (0, 220, 255),
            (255, 170, 0),
            (180, 80, 255),
            (0, 220, 120),
            (255, 120, 160),
        )
        for guide in guides:
            try:
                points = self.calibration.table_mm_to_projector_px(
                    np.asarray(guide.polygon_mm, dtype=np.float32)
                )
            except Exception:
                continue
            if points.shape[0] < 4:
                continue
            projected = [(float(x), float(y)) for x, y in points]
            color = (255, 255, 255) if guide.ball == 0 else palette[(guide.ball - 1) % len(palette)]
            overlay.lines.append(
                OverlayLine(
                    points=projected,
                    color=color,
                    width=4,
                    label=f"setup_target_zone_{guide.ball}",
                    style="dashed",
                )
            )
            center_mm = np.mean(np.asarray(guide.polygon_mm[:-1], dtype=np.float32), axis=0)
            center = np.mean(points[:-1], axis=0)
            font_size = max(
                18.0,
                min(32.0, float(self.config.training_prompt_font_size_px) * 0.65),
            )
            overlay.texts.append(
                OverlayText(
                    position=(float(center[0]), float(center[1])),
                    text=f"摆放 {guide.ball} 号球",
                    color=color,
                    font_size_px=font_size,
                    max_width_ratio=0.32,
                    outline_width_px=max(1.0, font_size / 18.0),
                    background_alpha=75,
                    rotation_deg=self._table_axis_rotation_deg(table_anchor=center_mm),
                )
            )

    def _table_axis_rotation_deg(
        self,
        *,
        projector_anchor: tuple[float, float] | None = None,
        table_anchor: np.ndarray | tuple[float, float] | None = None,
    ) -> float:
        try:
            if table_anchor is None:
                if projector_anchor is None:
                    return 0.0
                table_anchor = self.calibration.projector_px_to_table_mm(
                    np.asarray([projector_anchor], dtype=np.float32)
                )[0]
            anchor = np.asarray(table_anchor, dtype=np.float32).reshape((2,))
            table = getattr(self.calibration, "table", None)
            reference_mm = max(
                100.0,
                2.0 * float(getattr(table, "ball_diameter_mm", 57.15)),
            )
            projected = self.calibration.table_mm_to_projector_px(
                np.asarray(
                    [
                        anchor,
                        anchor + np.asarray([reference_mm, 0.0], dtype=np.float32),
                    ],
                    dtype=np.float32,
                )
            )
            direction = np.asarray(projected[1], dtype=np.float32) - np.asarray(
                projected[0],
                dtype=np.float32,
            )
            if not np.all(np.isfinite(direction)) or float(np.linalg.norm(direction)) < 1e-4:
                return 0.0
            angle = math.degrees(math.atan2(float(direction[1]), float(direction[0])))
            while angle >= 90.0:
                angle -= 180.0
            while angle < -90.0:
                angle += 180.0
            return float(angle)
        except Exception:
            return 0.0
