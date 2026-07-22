from __future__ import annotations

import numpy as np

from ..calibration import CalibrationService
from ..config import ProjectionConfig
from ..schemas import OverlayCircle, ProjectionOverlay, TracksFrame
from .models import TrainingStateFrame
from .numbered_tracker import ball_number_from_track


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
        overlay = ProjectionOverlay(
            overlay_id=f"training_{state.scenario_id}_{state.frame_id}",
            frame_id=state.frame_id,
            projector_size=(int(self.config.projector_width), int(self.config.projector_height)),
            lines=list(route_overlay.lines) if route_overlay is not None else [],
            circles=list(route_overlay.circles) if route_overlay is not None else [],
            labels=list(route_overlay.labels) if route_overlay is not None else [],
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
                radius = self.calibration.ball_projector_radius_px(track.center_px)
            except Exception:
                continue
            center = (float(point[0]), float(point[1]))
            color = (255, 255, 255) if number == 0 else ((0, 220, 255) if number in expected else (160, 160, 160))
            overlay.circles.append(OverlayCircle(center=center, radius=max(8.0, float(radius) * 1.25), color=color, width=3))
            overlay.labels.append(((center[0] + radius, center[1] - radius), str(number), color))

        phase_text = str(state.phase or "setup").upper()
        status_color = (0, 220, 120) if state.phase == "passed" else ((0, 80, 255) if state.phase == "failed" else (255, 255, 255))
        overlay.labels.append(((36.0, 48.0), f"TRAINING · {state.scenario_id}", status_color))
        target_text = " / ".join(str(number) for number in state.expected_numbers) or "-"
        overlay.labels.append(((36.0, 82.0), f"{phase_text} · TARGET {target_text}", status_color))
        overlay.labels.append(
            ((36.0, 116.0), f"PROGRESS {state.progress_current}/{state.progress_total}  TIME {state.elapsed_s:.1f}s", status_color)
        )
        return overlay
