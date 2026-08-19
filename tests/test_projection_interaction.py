from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from bas.projection.interaction import ProjectionInteractionController
from bas.projection.star_formula import StarFormulaConfig
from bas.projection.overlay import render_overlay_image
from bas.schemas import Event, MatchStateFrame, OverlayCircle, OverlayText, ProjectionOverlay, ShotPlan


def _write_bgra_png(path: Path, frame: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(path), frame)
    assert ok


def test_unicode_overlay_text_is_rendered_in_final_projector_pixels() -> None:
    text = OverlayText(
        position=(160.0, 90.0),
        text="当前目标：1 号球\n请准备击球",
        font_size_px=28.0,
    )
    first = render_overlay_image(
        ProjectionOverlay(
            overlay_id="calibration_a",
            frame_id=1,
            projector_size=(320, 180),
            texts=[text],
        )
    )
    second = render_overlay_image(
        ProjectionOverlay(
            overlay_id="calibration_b",
            frame_id=2,
            projector_size=(320, 180),
            texts=[text],
        )
    )

    assert int(np.count_nonzero(first)) > 0
    assert np.array_equal(first, second)


def test_projector_overlay_renders_anisotropic_circle_as_ellipse() -> None:
    image = render_overlay_image(
        ProjectionOverlay(
            overlay_id="ellipse",
            frame_id=1,
            projector_size=(120, 120),
            circles=[OverlayCircle(center=(60.0, 60.0), radius=10.0, radius_y=25.0, width=2)],
        )
    )
    ys, xs = np.nonzero(np.any(image != 0, axis=2))

    assert int(np.max(ys) - np.min(ys)) > int(np.max(xs) - np.min(xs)) * 1.8


def test_unicode_overlay_text_supports_table_aligned_rotation() -> None:
    straight = render_overlay_image(
        ProjectionOverlay(
            overlay_id="straight_text",
            frame_id=1,
            projector_size=(420, 180),
            texts=[
                OverlayText(
                    position=(210.0, 90.0),
                    text="摆放 1 号球",
                    font_size_px=34.0,
                    rotation_deg=0.0,
                )
            ],
        )
    )
    rotated = render_overlay_image(
        ProjectionOverlay(
            overlay_id="rotated_text",
            frame_id=2,
            projector_size=(420, 180),
            texts=[
                OverlayText(
                    position=(210.0, 90.0),
                    text="摆放 1 号球",
                    font_size_px=34.0,
                    rotation_deg=-8.0,
                )
            ],
        )
    )

    assert int(np.count_nonzero(rotated)) > 0
    assert not np.array_equal(straight, rotated)


def test_projection_interaction_blends_rgba_sequence_without_geometry_warp(tmp_path: Path) -> None:
    pocket_dir = tmp_path / "Goal" / "pocket0"
    frame = np.zeros((2, 2, 4), dtype=np.uint8)
    frame[0, 0] = (0, 0, 255, 255)
    frame[0, 1] = (0, 255, 0, 128)
    _write_bgra_png(pocket_dir / "frame_000.png", frame)

    now = [0.0]
    controller = ProjectionInteractionController(asset_root=tmp_path, time_source=lambda: now[0])
    assert controller.trigger_pocket_animation(0, fps_hint=30.0) is True

    overlay = ProjectionOverlay(overlay_id="blank", frame_id=1, projector_size=(2, 2))
    image = controller.compose_frame(overlay, star_formula=StarFormulaConfig(enabled=False))

    assert tuple(int(v) for v in image[0, 0]) == (0, 0, 255)
    assert 120 <= int(image[0, 1, 1]) <= 128
    assert int(image[0, 1, 0]) == 0
    assert int(image[0, 1, 2]) == 0


def test_projection_calibration_cannot_change_star_formula_pixels(tmp_path: Path) -> None:
    controller = ProjectionInteractionController(asset_root=tmp_path, time_source=lambda: 0.0)
    overlay = ProjectionOverlay(overlay_id="star-only", frame_id=1, projector_size=(320, 200))
    star_formula = StarFormulaConfig(enabled=True)

    without_calibration = controller.compose_frame(
        overlay,
        star_formula=star_formula,
        calibration=None,
    )
    with_unrelated_calibration_object = controller.compose_frame(
        overlay,
        star_formula=star_formula,
        calibration=object(),
    )

    assert np.array_equal(without_calibration, with_unrelated_calibration_object)


def test_projection_interaction_deduplicates_pocket_events_and_notices_target_activation(tmp_path: Path) -> None:
    pocket_dir = tmp_path / "Goal" / "pocket0"
    victory_dir = tmp_path / "Win"
    _write_bgra_png(pocket_dir / "frame_000.png", np.zeros((8, 8, 4), dtype=np.uint8))
    _write_bgra_png(victory_dir / "frame_000.png", np.zeros((8, 8, 4), dtype=np.uint8))

    now = [0.0]
    controller = ProjectionInteractionController(asset_root=tmp_path, time_source=lambda: now[0])
    state = MatchStateFrame(
        frame_id=12,
        ts_cam_ns=12,
        phase="TURN_RESOLVE",
        events=[
            Event(
                name="POCKET_CONFIRMED",
                ts_cam_ns=12,
                frame_id=12,
                payload={"track_id": 8, "pocket_index": 0, "shot_id": 3, "decision_id": "pocket:1"},
            )
        ],
    )
    plan = ShotPlan(
        plan_id="p1",
        frame_id=12,
        ts_cam_ns=12,
        shot_mode="target",
        locked_target_id=5,
        target_shot_status="active_new:no_theoretical_route",
    )

    assert controller.observe_output(state=state, plan=plan, fps_hint=30.0) is True
    assert controller.observe_output(state=state, plan=plan, fps_hint=30.0) is False

    image = controller.compose_frame(
        ProjectionOverlay(overlay_id="blank", frame_id=12, projector_size=(220, 120)),
        star_formula=StarFormulaConfig(enabled=False),
    )
    assert int(np.count_nonzero(image)) > 0


def test_projection_interaction_ignores_nonfinal_pocket_events_for_auto_animation(tmp_path: Path) -> None:
    pocket_dir = tmp_path / "Goal" / "pocket0"
    frame = np.zeros((8, 8, 4), dtype=np.uint8)
    frame[:, :, 1] = 255
    frame[:, :, 3] = 255
    _write_bgra_png(pocket_dir / "frame_000.png", frame)

    now = [0.0]
    controller = ProjectionInteractionController(asset_root=tmp_path, time_source=lambda: now[0])
    state = MatchStateFrame(
        frame_id=13,
        ts_cam_ns=13,
        phase="SHOT_ACTIVE",
        events=[
            Event(
                name="POCKET_DETECTED",
                ts_cam_ns=13,
                frame_id=13,
                payload={"track_id": 9, "pocket_index": 0, "shot_id": 3, "decision_id": "pocket:detected"},
            ),
            Event(
                name="POT_PROBABLE",
                ts_cam_ns=13,
                frame_id=13,
                payload={"track_id": 10, "pocket_index": 0, "shot_id": 3, "decision_id": "pocket:legacy"},
            ),
        ],
    )
    plan = ShotPlan(plan_id="p-pot", frame_id=13, ts_cam_ns=13)

    assert controller.observe_output(state=state, plan=plan, fps_hint=30.0) is False
    image = controller.compose_frame(
        ProjectionOverlay(overlay_id="blank", frame_id=13, projector_size=(220, 120)),
        star_formula=StarFormulaConfig(enabled=False),
    )
    assert int(np.count_nonzero(image)) == 0


def test_projection_interaction_triggers_victory_animation_once(tmp_path: Path) -> None:
    victory_dir = tmp_path / "Win"
    frame = np.zeros((4, 4, 4), dtype=np.uint8)
    frame[:, :, 2] = 255
    frame[:, :, 3] = 255
    _write_bgra_png(victory_dir / "frame_000.png", frame)

    now = [0.0]
    controller = ProjectionInteractionController(asset_root=tmp_path, time_source=lambda: now[0])
    state = MatchStateFrame(
        frame_id=99,
        ts_cam_ns=99,
        phase="TURN_RESOLVE",
        events=[
            Event(
                name="GAME_STATUS_CHANGED",
                ts_cam_ns=99,
                frame_id=99,
                payload={"shot_id": 7, "decision_id": "game-status:7:in_progress->ended", "from_status": "in_progress", "to_status": "ended"},
            )
        ],
    )
    plan = ShotPlan(plan_id="p2", frame_id=99, ts_cam_ns=99)

    assert controller.observe_output(state=state, plan=plan, fps_hint=30.0) is True
    image = controller.compose_frame(
        ProjectionOverlay(overlay_id="blank", frame_id=99, projector_size=(4, 4)),
        star_formula=StarFormulaConfig(enabled=False),
    )
    assert tuple(int(v) for v in image[1, 1]) == (0, 0, 255)


def test_projection_interaction_ignores_referee_intent_for_auto_victory(tmp_path: Path) -> None:
    victory_dir = tmp_path / "Win"
    frame = np.zeros((4, 4, 4), dtype=np.uint8)
    frame[:, :, 2] = 255
    frame[:, :, 3] = 255
    _write_bgra_png(victory_dir / "frame_000.png", frame)

    now = [0.0]
    controller = ProjectionInteractionController(asset_root=tmp_path, time_source=lambda: now[0])
    state = MatchStateFrame(
        frame_id=100,
        ts_cam_ns=100,
        phase="TURN_RESOLVE",
        events=[Event(name="REFEREE_INTENT", ts_cam_ns=100, frame_id=100, payload={"game_status": "won"})],
    )
    plan = ShotPlan(plan_id="p3", frame_id=100, ts_cam_ns=100)

    assert controller.observe_output(state=state, plan=plan, fps_hint=30.0) is False
    image = controller.compose_frame(
        ProjectionOverlay(overlay_id="blank", frame_id=100, projector_size=(4, 4)),
        star_formula=StarFormulaConfig(enabled=False),
    )
    assert int(np.count_nonzero(image)) == 0


def test_projection_interaction_can_disable_auto_pocket_animation_without_affecting_manual_trigger(tmp_path: Path) -> None:
    pocket_dir = tmp_path / "Goal" / "pocket0"
    frame = np.zeros((8, 8, 4), dtype=np.uint8)
    frame[:, :, 1] = 255
    frame[:, :, 3] = 255
    _write_bgra_png(pocket_dir / "frame_000.png", frame)

    controller = ProjectionInteractionController(
        asset_root=tmp_path,
        time_source=lambda: 0.0,
        auto_pocket_animation_enabled=False,
    )
    state = MatchStateFrame(
        frame_id=14,
        ts_cam_ns=14,
        phase="SHOT_ACTIVE",
        events=[Event(name="POCKET_CONFIRMED", ts_cam_ns=14, frame_id=14, payload={"track_id": 10, "pocket_index": 0, "shot_id": 4, "decision_id": "pocket:4"})],
    )
    plan = ShotPlan(plan_id="p-pocket-off", frame_id=14, ts_cam_ns=14)
    overlay = ProjectionOverlay(overlay_id="blank", frame_id=14, projector_size=(220, 120))

    assert controller.observe_output(state=state, plan=plan, fps_hint=30.0) is False
    assert int(np.count_nonzero(controller.compose_frame(overlay, star_formula=StarFormulaConfig(enabled=False)))) == 0

    assert controller.trigger_pocket_animation(0, fps_hint=30.0) is True
    assert int(np.count_nonzero(controller.compose_frame(overlay, star_formula=StarFormulaConfig(enabled=False)))) > 0


def test_projection_interaction_can_disable_auto_victory_animation_without_affecting_manual_trigger(tmp_path: Path) -> None:
    victory_dir = tmp_path / "Win"
    frame = np.zeros((4, 4, 4), dtype=np.uint8)
    frame[:, :, 2] = 255
    frame[:, :, 3] = 255
    _write_bgra_png(victory_dir / "frame_000.png", frame)

    controller = ProjectionInteractionController(
        asset_root=tmp_path,
        time_source=lambda: 0.0,
        auto_victory_animation_enabled=False,
    )
    state = MatchStateFrame(
        frame_id=101,
        ts_cam_ns=101,
        phase="TURN_RESOLVE",
        events=[
            Event(
                name="GAME_STATUS_CHANGED",
                ts_cam_ns=101,
                frame_id=101,
                payload={"shot_id": 8, "decision_id": "game-status:8:in_progress->ended", "from_status": "in_progress", "to_status": "ended"},
            )
        ],
    )
    plan = ShotPlan(plan_id="p-victory-off", frame_id=101, ts_cam_ns=101)
    overlay = ProjectionOverlay(overlay_id="blank", frame_id=101, projector_size=(4, 4))

    assert controller.observe_output(state=state, plan=plan, fps_hint=30.0) is False
    assert int(np.count_nonzero(controller.compose_frame(overlay, star_formula=StarFormulaConfig(enabled=False)))) == 0

    assert controller.trigger_victory_animation(fps_hint=30.0) is True
    assert tuple(int(v) for v in controller.compose_frame(overlay, star_formula=StarFormulaConfig(enabled=False))[1, 1]) == (0, 0, 255)
