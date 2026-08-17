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


class _FieldScaleTableProjection:
    """Approximate the current 1280x800 field calibration near the drill."""

    table = SimpleNamespace(width_mm=2540.0, height_mm=1270.0)

    @staticmethod
    def table_mm_to_projector_px(points):
        return np.asarray(points, dtype=np.float32) * 0.44


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


def test_stroke_check_uses_enlarged_field_profile_and_symmetric_guide_angle() -> None:
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
    assert float(np.linalg.norm(centers[1] - centers[0])) == pytest.approx(114.3)
    assert float(np.linalg.norm(centers[2] - centers[1])) == pytest.approx(114.3)
    checkpoint_length = float(
        np.linalg.norm(
            np.asarray(checkpoints[0].points[1])
            - np.asarray(checkpoints[0].points[0])
        )
    )
    assert checkpoint_length == pytest.approx(139.14)

    cue_direction = _line_by_label(overlay, "stroke_check_cue_direction")
    cue_points = np.asarray(cue_direction.points, dtype=np.float32)
    assert np.allclose(cue_points[:, 0], 2540.0 * 0.25)
    assert cue_points[1, 1] < cue_points[0, 1]
    assert float(np.linalg.norm(cue_points[1] - cue_points[0])) == pytest.approx(
        375.0 - 0.5 * dimensions.ball_diameter_mm
    )

    axis = cue_points[1] - cue_points[0]
    axis /= np.linalg.norm(axis)
    guide_angles = []
    for label in ("stroke_check_guide_left", "stroke_check_guide_right"):
        guide = np.asarray(_line_by_label(overlay, label).points, dtype=np.float32)
        vector = guide[1] - guide[0]
        signed_angle = np.degrees(
            np.arctan2(
                axis[0] * vector[1] - axis[1] * vector[0],
                np.dot(axis, vector),
            )
        )
        guide_angles.append(float(signed_angle))
    assert abs(guide_angles[0]) == pytest.approx(3.5, abs=0.01)
    assert abs(guide_angles[1]) == pytest.approx(3.5, abs=0.01)

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


def test_stroke_check_thin_marks_remain_two_pixels_at_field_scale() -> None:
    overlay = StrokeCheckOverlayBuilder(
        ProjectionConfig(projector_width=1280, projector_height=800),
        _FieldScaleTableProjection(),
    ).build()

    for label in (
        "stroke_check_guide_left",
        "stroke_check_guide_right",
        "stroke_check_checkpoint_1",
        "stroke_check_checkpoint_2",
        "stroke_check_checkpoint_3",
        "stroke_check_shot_direction_circle",
    ):
        assert _line_by_label(overlay, label).width >= 2


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
