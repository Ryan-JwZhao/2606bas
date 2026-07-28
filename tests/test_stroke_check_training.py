from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from bas.config import ProjectionConfig, TrainingConfig
from bas.projection.overlay import render_overlay_image, render_overlay_with_star
from bas.projection.star_formula import StarFormulaConfig
from bas.schemas import TracksFrame
from bas.training import (
    STROKE_CHECK_SCENARIO_ID,
    StrokeCheckDimensions,
    StrokeCheckOverlayBuilder,
    TrainingSession,
    get_training_scenario,
)


class _IdentityTableProjection:
    table = SimpleNamespace(width_mm=2540.0, height_mm=1270.0)

    @staticmethod
    def table_mm_to_projector_px(points):
        return np.asarray(points, dtype=np.float32)


def _line_by_label(overlay, label: str):
    return next(line for line in overlay.lines if line.label == label)


def test_stroke_check_is_first_other_projection_only_training() -> None:
    scenario = get_training_scenario(STROKE_CHECK_SCENARIO_ID)

    assert scenario.group == "other"
    assert scenario.display_number == 1
    assert scenario.display_title == "其它训练1：出杆检测"
    assert scenario.projection_only is True
    assert scenario.require_cue_ball is False
    assert scenario.required_balls == ()


def test_stroke_check_keeps_printed_three_inch_spacing_and_initial_pose() -> None:
    dimensions = StrokeCheckDimensions()
    overlay = StrokeCheckOverlayBuilder(
        ProjectionConfig(projector_width=2540, projector_height=1270),
        _IdentityTableProjection(),
        dimensions,
    ).build()

    checkpoints = [
        _line_by_label(overlay, f"stroke_check_checkpoint_{index}")
        for index in (1, 2, 3)
    ]
    centers = [
        np.mean(np.asarray(line.points, dtype=np.float32), axis=0)
        for line in checkpoints
    ]
    assert float(np.linalg.norm(centers[1] - centers[0])) == pytest.approx(76.2)
    assert float(np.linalg.norm(centers[2] - centers[1])) == pytest.approx(76.2)

    cue_direction = _line_by_label(overlay, "stroke_check_cue_direction")
    cue_points = np.asarray(cue_direction.points, dtype=np.float32)
    assert np.allclose(cue_points[:, 0], 2540.0 * 0.25)
    assert cue_points[1, 1] < cue_points[0, 1]

    circle = _line_by_label(overlay, "stroke_check_shot_direction_circle")
    circle_points = np.asarray(circle.points[:-1], dtype=np.float32)
    assert np.allclose(np.mean(circle_points, axis=0), (635.0, 635.0), atol=0.05)
    diameter_x = float(np.max(circle_points[:, 0]) - np.min(circle_points[:, 0]))
    diameter_y = float(np.max(circle_points[:, 1]) - np.min(circle_points[:, 1]))
    assert abs(diameter_x - 57.15) < 0.02
    assert abs(diameter_y - 57.15) < 0.02
    assert overlay.suppress_star_formula is True
    assert np.array_equal(
        render_overlay_with_star(overlay, StarFormulaConfig(enabled=True)),
        render_overlay_image(overlay),
    )


def test_projection_only_training_starts_and_runs_without_camera_tracks() -> None:
    session = TrainingSession(
        TrainingConfig(scenario_id=STROKE_CHECK_SCENARIO_ID)
    )

    started, state = session.start()
    assert started is True
    assert state.phase == "running"
    assert state.setup_ready is True
    assert state.mode_hint == "纯投影训练"

    updated = session.update(
        TracksFrame(frame_id=7, ts_cam_ns=700_000_000, tracks=[])
    )
    assert updated.phase == "running"
    assert updated.visible_numbers == []
    assert updated.progress_total == 0
