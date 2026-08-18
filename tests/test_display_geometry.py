from __future__ import annotations

from dataclasses import replace

import cv2
import numpy as np

from bas.display_geometry import DisplayGeometryStabilizer, stabilize_projection_overlay
from bas.projection.interaction import ProjectionInteractionController
from bas.projection.overlay import render_overlay_image
from bas.projection.star_formula import StarFormulaConfig
from bas.schemas import OverlayCircle, OverlayLine, ProjectionOverlay


def _overlay(frame_id: int, *, aim_end_y: float, cue_x: float = 1121.51) -> ProjectionOverlay:
    return ProjectionOverlay(
        overlay_id=f"route_{frame_id}",
        frame_id=frame_id,
        projector_size=(1280, 800),
        lines=[
            OverlayLine(
                points=[(931.62, 231.17), (153.34, aim_end_y)],
                width=4,
                label="aim",
            ),
            OverlayLine(
                points=[(cue_x, 125.19), (954.00, 218.68)],
                width=4,
                label="cue_guide",
            ),
        ],
        circles=[
            OverlayCircle(
                center=(931.62, 231.17),
                radius=14.2,
                radius_y=14.8,
                stability_key="route_cue",
            )
        ],
    )


def test_display_stabilizer_does_not_modify_geometric_source_and_bounds_display_error() -> None:
    stabilizer = DisplayGeometryStabilizer(deadband_px=0.35)
    first = np.asarray([[1121.49, 125.18], [954.00, 218.68]], dtype=np.float32)
    jittered = np.asarray([[1121.55, 125.22], [954.08, 218.61]], dtype=np.float32)
    source_copy = jittered.copy()

    shown_first = stabilizer.stabilize("cue_guide", first)
    shown_jittered = stabilizer.stabilize("cue_guide", jittered)

    assert np.array_equal(jittered, source_copy)
    assert np.array_equal(shown_jittered, shown_first)
    assert float(np.max(np.linalg.norm(jittered - shown_jittered, axis=1))) <= 0.35


def test_display_stabilizer_follows_real_movement_immediately() -> None:
    stabilizer = DisplayGeometryStabilizer(deadband_px=0.35)
    first = np.asarray([[100.0, 100.0], [500.0, 300.0]], dtype=np.float32)
    moved = first + np.asarray([10.0, -4.0], dtype=np.float32)

    stabilizer.stabilize("aim", first)
    shown = stabilizer.stabilize("aim", moved)

    assert np.array_equal(shown, moved)


def test_projection_overlay_stability_is_presentation_only() -> None:
    stabilizer = DisplayGeometryStabilizer(deadband_px=0.35)
    first = _overlay(1, aim_end_y=665.49)
    jittered = _overlay(2, aim_end_y=665.53, cue_x=1121.55)
    source_aim = list(jittered.lines[0].points)
    source_circle = jittered.circles[0].center

    shown_first = stabilize_projection_overlay(first, stabilizer)
    shown_jittered = stabilize_projection_overlay(jittered, stabilizer)

    assert jittered.lines[0].points == source_aim
    assert jittered.circles[0].center == source_circle
    assert shown_jittered.lines[0].points == shown_first.lines[0].points
    assert shown_jittered.lines[1].points == shown_first.lines[1].points
    assert shown_jittered.circles[0].center == shown_first.circles[0].center


def test_projection_controller_removes_stationary_route_flash_without_route_freeze(tmp_path) -> None:
    controller = ProjectionInteractionController(asset_root=tmp_path, time_source=lambda: 0.0)
    first = _overlay(1, aim_end_y=665.49, cue_x=1121.49)
    jittered = _overlay(2, aim_end_y=665.53, cue_x=1121.55)
    moved = replace(
        _overlay(3, aim_end_y=665.53, cue_x=1121.55),
        lines=[
            replace(first.lines[0], points=[(941.62, 231.17), (163.34, 665.53)]),
            replace(first.lines[1], points=[(1131.55, 125.19), (964.00, 218.68)]),
        ],
    )
    star = StarFormulaConfig(enabled=False)

    first_image = controller.compose_frame(first, star_formula=star)
    jittered_image = controller.compose_frame(jittered, star_formula=star)
    moved_image = controller.compose_frame(moved, star_formula=star)

    assert np.array_equal(jittered_image, first_image)
    assert not np.array_equal(moved_image, first_image)


def test_projection_renderer_uses_subpixel_rasterization_instead_of_integer_flip() -> None:
    first = _overlay(1, aim_end_y=665.49)
    jittered = _overlay(2, aim_end_y=665.51)

    first_image = render_overlay_image(first)
    jittered_image = render_overlay_image(jittered)
    diff = cv2.absdiff(first_image, jittered_image)

    assert int(np.count_nonzero(diff >= 32)) < 300

