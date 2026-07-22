from __future__ import annotations

import cv2
from types import SimpleNamespace

import numpy as np

from bas.calibration.camera import CameraCalibration
from bas.calibration.projector import ProjectionCalibration
from bas.calibration.service import CalibrationService
from bas.config import PlannerConfig
from bas.planning.cue_aim import CueStickAimPx
from bas.projection.star_formula import StarFormulaConfig
from bas.planning import GeometryPhysicsPlanner
from bas.projection.overlay import OverlayBuilder, projection_route_stroke_style
from bas.config import ProjectionConfig
from bas.schemas import Event, MatchStateFrame, ShotCandidate, ShotPlan, TableModel, TrackObservation


def _service() -> CalibrationService:
    return _service_for_table(1000.0, 500.0)


def _service_for_table(width_mm: float, height_mm: float) -> CalibrationService:
    projection = ProjectionCalibration.fit_from_correspondences(
        np.array([[0, 0], [width_mm, 0], [width_mm, height_mm], [0, height_mm]], dtype=np.float64),
        np.array([[0, 0], [width_mm, 0], [width_mm, height_mm], [0, height_mm]], dtype=np.float64),
        projector_size=(int(round(width_mm)), int(round(height_mm))),
    )
    projection.table_polygon_proj = np.array([[0, 0], [width_mm, 0], [width_mm, height_mm], [0, height_mm]], dtype=np.float64)
    return CalibrationService(
        camera=CameraCalibration(metadata={}),
        projection=projection,
        table=TableModel(
            width_mm=width_mm,
            height_mm=height_mm,
            ball_diameter_mm=57.15,
            inner_polygon_mm=[(0, 0), (width_mm, 0), (width_mm, height_mm), (0, height_mm)],
            pockets_mm=[(0, 0), (width_mm * 0.5, 0), (width_mm, 0), (width_mm, height_mm), (width_mm * 0.5, height_mm), (0, height_mm)],
        ),
    )


def _obs(track_id: int, group: str, x: float, y: float) -> TrackObservation:
    return TrackObservation(
        track_id=track_id,
        bbox=(x - 15, y - 15, x + 15, y + 15),
        center_px=(x, y),
        radius_px=15,
        cls_name=group,
        group=group,
        confidence=0.9,
        quality=0.9,
    )


def _stick(track_id: int, x1: float, y1: float, x2: float, y2: float) -> TrackObservation:
    return TrackObservation(
        track_id=track_id,
        bbox=(x1, y1, x2, y2),
        center_px=((x1 + x2) * 0.5, (y1 + y2) * 0.5),
        radius_px=0.25 * ((x2 - x1) + (y2 - y1)),
        cls_name="cue_stick",
        group="cue_stick",
        confidence=0.9,
        quality=0.9,
    )


def _ts_ms(value: int) -> int:
    return int(value) * 1_000_000


def test_planner_generates_candidate() -> None:
    planner = GeometryPhysicsPlanner(PlannerConfig(top_k=3), _service())
    state = MatchStateFrame(
        frame_id=1,
        ts_cam_ns=1,
        phase="STABLE_IDLE",
        layout=[_obs(1, "cue", 120, 250), _obs(2, "solid", 620, 250)],
    )
    plan = planner.plan(state)
    assert plan.best is not None
    assert len(plan.candidates) >= 1
    assert plan.best.score > -5


def test_planner_manual_web_target_limits_candidates_until_cleared() -> None:
    planner = GeometryPhysicsPlanner(PlannerConfig(top_k=20), _service())
    state = MatchStateFrame(
        frame_id=1,
        ts_cam_ns=1,
        phase="STABLE_IDLE",
        layout=[
            _obs(1, "cue", 120, 250),
            _obs(2, "solid", 620, 250),
            _obs(3, "stripe", 620, 350),
        ],
    )

    planner.set_manual_target(3)
    plan = planner.plan(state)

    assert plan.best is not None
    assert plan.locked_target_id == 3
    assert plan.target_lock_status == "manual"
    assert all(candidate.target_track_id == 3 for candidate in plan.candidates)

    planner.clear_manual_target()
    assert planner.manual_target_id is None


def test_planner_manual_web_target_overrides_hook_mode() -> None:
    planner = GeometryPhysicsPlanner(PlannerConfig(shot_mode="hook", top_k=20), _service())
    state = MatchStateFrame(
        frame_id=1,
        ts_cam_ns=1,
        phase="STABLE_IDLE",
        layout=[
            _obs(1, "cue", 120, 250),
            _obs(2, "solid", 620, 250),
            _obs(3, "stripe", 620, 350),
        ],
    )

    planner.set_manual_target(3)
    plan = planner.plan(state)

    assert plan.shot_mode == "hook"
    assert plan.best is not None
    assert plan.best.target_track_id == 3
    assert plan.locked_target_id == 3
    assert plan.target_lock_status == "manual"
    assert plan.best.explanation["hook_selection_source"] == "manual"


def test_hook_mode_traverses_only_the_current_turn_group() -> None:
    planner = GeometryPhysicsPlanner(
        PlannerConfig(shot_mode="hook", top_k=20, target_shot_enabled=False),
        _service(),
    )
    state = MatchStateFrame(
        frame_id=1,
        ts_cam_ns=1,
        phase="STABLE_IDLE",
        turn_target_group="stripe",
        layout=[
            _obs(1, "cue", 120, 250),
            _obs(2, "solid", 620, 250),
            _obs(3, "stripe", 620, 350),
        ],
    )

    plan = planner.plan(state)

    assert plan.shot_mode == "hook"
    assert plan.best is not None
    assert {candidate.target_track_id for candidate in plan.candidates} == {3}
    assert plan.best.explanation["hook_selection_source"] == "automatic"
    assert plan.target_lock_status == "hook_global"


def test_hook_mode_global_selection_does_not_use_rule_cue_sector(monkeypatch) -> None:
    planner = GeometryPhysicsPlanner(
        PlannerConfig(shot_mode="hook", top_k=20, target_shot_enabled=False),
        _service(),
    )
    monkeypatch.setattr(
        planner.cue_sector,
        "detect_aim",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("hook mode must not use cue-sector filtering")),
    )
    state = MatchStateFrame(
        frame_id=1,
        ts_cam_ns=1,
        phase="STABLE_IDLE",
        turn_target_group="solid",
        layout=[_obs(1, "cue", 120, 250), _obs(2, "solid", 620, 250)],
    )

    plan = planner.plan(state)

    assert plan.shot_mode == "hook"
    assert plan.best is not None
    assert plan.best.target_track_id == 2


def test_training_explicit_target_reuses_hook_planner_for_only_that_number() -> None:
    planner = GeometryPhysicsPlanner(
        PlannerConfig(top_k=20, target_shot_enabled=False),
        _service(),
    )
    state = MatchStateFrame(
        frame_id=1,
        ts_cam_ns=1,
        phase="TRAINING_RUNNING",
        layout=[
            _obs(0, "cue", 120, 250),
            _obs(2, "solid", 620, 250),
            _obs(3, "solid", 620, 350),
            _obs(8, "black", 820, 250),
        ],
    )

    plan = planner.plan(
        state,
        forced_shot_mode="hook",
        forced_target_track_ids=[3],
    )

    assert plan.shot_mode == "hook"
    assert plan.best is not None
    assert {candidate.target_track_id for candidate in plan.candidates} == {3}
    assert plan.best.explanation["hook_selection_source"] == "training_target"
    assert plan.locked_target_id == 3


def test_training_open_stage_reuses_rule_planner_and_excludes_early_black() -> None:
    planner = GeometryPhysicsPlanner(
        PlannerConfig(
            top_k=20,
            target_shot_enabled=False,
            cue_sector_correction_enabled=False,
        ),
        _service(),
    )
    state = MatchStateFrame(
        frame_id=1,
        ts_cam_ns=1,
        phase="TRAINING_RUNNING",
        layout=[
            _obs(0, "cue", 120, 250),
            _obs(2, "solid", 620, 250),
            _obs(3, "solid", 620, 350),
            _obs(8, "black", 820, 250),
        ],
    )

    plan = planner.plan(
        state,
        forced_shot_mode="rule",
        forced_turn_target_group="solid",
        forced_target_track_ids=[2, 3],
    )

    assert plan.shot_mode == "rule"
    assert plan.best is not None
    assert {candidate.target_track_id for candidate in plan.candidates} <= {2, 3}
    assert all(candidate.target_group == "solid" for candidate in plan.candidates)


def test_planner_manual_web_target_is_released_on_shot_started() -> None:
    planner = GeometryPhysicsPlanner(PlannerConfig(top_k=20), _service())
    layout = [
        _obs(1, "cue", 120, 250),
        _obs(2, "solid", 620, 250),
        _obs(3, "stripe", 620, 350),
    ]
    before_shot = MatchStateFrame(
        frame_id=1,
        ts_cam_ns=1,
        phase="PRE_SHOT_ARMED",
        layout=layout,
    )
    shot_started = MatchStateFrame(
        frame_id=2,
        ts_cam_ns=2,
        phase="SHOT_ACTIVE",
        events=[Event(name="SHOT_STARTED", frame_id=2, ts_cam_ns=2)],
        layout=layout,
    )

    planner.set_manual_target(3)
    held = planner.plan(before_shot)
    released = planner.plan(shot_started)

    assert held.best is not None
    assert held.best.target_track_id == 3
    assert planner.manual_target_id is None
    assert released.target_lock_status != "manual"
    assert all(candidate.target_track_id != 3 for candidate in released.candidates) or released.locked_target_id is None


def test_planner_manual_web_target_does_not_fallback_when_temporarily_missing() -> None:
    planner = GeometryPhysicsPlanner(PlannerConfig(top_k=20), _service())
    selected_layout = [
        _obs(1, "cue", 120, 250),
        _obs(2, "solid", 620, 250),
        _obs(3, "stripe", 620, 350),
    ]
    missing_layout = [selected_layout[0], selected_layout[1]]

    planner.set_manual_target(3)
    selected = planner.plan(MatchStateFrame(frame_id=1, ts_cam_ns=1, phase="PRE_SHOT_ARMED", layout=selected_layout))
    missing = planner.plan(MatchStateFrame(frame_id=2, ts_cam_ns=2, phase="PRE_SHOT_ARMED", layout=missing_layout))

    assert selected.best is not None
    assert selected.best.target_track_id == 3
    assert missing.best is None
    assert missing.locked_target_id == 3
    assert missing.target_lock_status == "manual_missing"


def test_planner_excludes_black_on_open_table() -> None:
    planner = GeometryPhysicsPlanner(PlannerConfig(top_k=20), _service())
    state = MatchStateFrame(
        frame_id=1,
        ts_cam_ns=1,
        phase="STABLE_IDLE",
        layout=[
            _obs(1, "cue", 120, 250),
            _obs(2, "solid", 620, 250),
            _obs(3, "stripe", 620, 380),
            _obs(4, "black", 320, 250),
        ],
    )

    plan = planner.plan(state)

    assert plan.best is not None
    assert all(candidate.target_group != "black" for candidate in plan.candidates)


def test_planner_uses_black_only_for_cleared_solid_turn() -> None:
    planner = GeometryPhysicsPlanner(PlannerConfig(top_k=20), _service())
    state = MatchStateFrame(
        frame_id=1,
        ts_cam_ns=1,
        phase="STABLE_IDLE",
        turn_target_group="solid",
        layout=[
            _obs(1, "cue", 120, 250),
            _obs(2, "stripe", 760, 250),
            _obs(3, "black", 320, 250),
        ],
    )

    plan = planner.plan(state)

    assert plan.best is not None
    assert all(candidate.target_group == "black" for candidate in plan.candidates)


def test_planner_keeps_recommending_stripes_when_solids_are_cleared_on_stripe_turn() -> None:
    planner = GeometryPhysicsPlanner(PlannerConfig(top_k=20), _service())
    state = MatchStateFrame(
        frame_id=1,
        ts_cam_ns=1,
        phase="STABLE_IDLE",
        turn_target_group="stripe",
        layout=[
            _obs(1, "cue", 120, 250),
            _obs(2, "stripe", 620, 250),
            _obs(3, "black", 320, 100),
        ],
    )

    plan = planner.plan(state)

    assert plan.best is not None
    assert all(candidate.target_group == "stripe" for candidate in plan.candidates)


def test_cue_sector_correction_keeps_turn_group_when_available_in_sector() -> None:
    planner = GeometryPhysicsPlanner(PlannerConfig(top_k=20, cue_sector_switch_confirm_frames=1), _service())
    state = MatchStateFrame(
        frame_id=1,
        ts_cam_ns=1,
        phase="PRE_SHOT_ARMED",
        turn_target_group="solid",
        layout=[
            _obs(1, "cue", 120, 250),
            _stick(9, 20, 240, 90, 260),
            _obs(2, "solid", 620, 250),
            _obs(3, "stripe", 620, 300),
        ],
    )

    plan = planner.plan(state)

    assert plan.best is not None
    assert plan.best.target_group == "solid"
    assert all(candidate.target_group == "solid" for candidate in plan.candidates)
    assert plan.best.explanation["cue_sector_policy"] == "own_group"
    assert plan.best.explanation["cue_sector_geometry"] == "corridor"
    assert plan.best.explanation["cue_sector_corridor_width_px"] == 140.0


def test_cue_sector_correction_recommends_opponent_with_secondary_confirmation() -> None:
    planner = GeometryPhysicsPlanner(PlannerConfig(top_k=20, cue_sector_switch_confirm_frames=1), _service())
    state = MatchStateFrame(
        frame_id=1,
        ts_cam_ns=1,
        phase="PRE_SHOT_ARMED",
        turn_target_group="solid",
        layout=[
            _obs(1, "cue", 120, 250),
            _stick(9, 20, 240, 90, 260),
            _obs(2, "solid", 620, 380),
            _obs(3, "stripe", 620, 250),
        ],
    )

    plan = planner.plan(state)

    assert plan.best is not None
    assert plan.best.target_group == "stripe"
    assert all(candidate.target_group == "stripe" for candidate in plan.candidates)
    assert plan.best.explanation["cue_sector_policy"] == "opponent_confirmation"
    assert plan.best.explanation["cue_sector_geometry"] == "corridor"
    assert plan.best.explanation["cue_sector_target_ids"] == [3]
    assert plan.best.explanation["cue_sector_requires_confirmation"] is True
    assert plan.best.explanation["cue_sector_confirmation_target_id"] == 3


def test_cue_sector_correction_prefers_black_over_opponent_when_no_turn_group_in_sector() -> None:
    planner = GeometryPhysicsPlanner(PlannerConfig(top_k=20, cue_sector_switch_confirm_frames=1), _service())
    state = MatchStateFrame(
        frame_id=1,
        ts_cam_ns=1,
        phase="PRE_SHOT_ARMED",
        turn_target_group="solid",
        layout=[
            _obs(1, "cue", 120, 250),
            _stick(9, 20, 240, 90, 260),
            _obs(2, "solid", 620, 380),
            _obs(3, "stripe", 620, 300),
            _obs(4, "black", 620, 250),
        ],
    )

    plan = planner.plan(state)

    assert plan.best is not None
    assert plan.best.target_group == "black"
    assert all(candidate.target_group == "black" for candidate in plan.candidates)
    assert plan.best.explanation["cue_sector_policy"] == "black_fallback"
    assert plan.best.explanation["cue_sector_requires_confirmation"] is False


def test_cue_sector_correction_leaves_existing_algorithm_when_stick_is_not_pointing_cue() -> None:
    planner = GeometryPhysicsPlanner(PlannerConfig(top_k=20, cue_sector_switch_confirm_frames=1), _service())
    state = MatchStateFrame(
        frame_id=1,
        ts_cam_ns=1,
        phase="PRE_SHOT_ARMED",
        turn_target_group="solid",
        layout=[
            _obs(1, "cue", 120, 250),
            _stick(9, 40, 20, 70, 110),
            _obs(2, "solid", 620, 380),
            _obs(3, "stripe", 620, 250),
        ],
    )

    plan = planner.plan(state)

    assert plan.best is not None
    assert plan.best.target_group == "solid"
    assert "cue_sector_policy" not in plan.best.explanation


def test_cue_sector_correction_waits_until_balls_are_stationary() -> None:
    planner = GeometryPhysicsPlanner(PlannerConfig(top_k=20, cue_sector_switch_confirm_frames=1), _service())
    moving_stripe = _obs(3, "stripe", 620, 250)
    moving_stripe.velocity_mm_s = (12.0, 0.0)
    state = MatchStateFrame(
        frame_id=1,
        ts_cam_ns=1,
        phase="PRE_SHOT_ARMED",
        turn_target_group="solid",
        layout=[
            _obs(1, "cue", 120, 250),
            _stick(9, 20, 240, 90, 260),
            _obs(2, "solid", 620, 380),
            moving_stripe,
        ],
    )

    plan = planner.plan(state)

    assert plan.best is not None
    assert plan.best.target_group == "solid"
    assert "cue_sector_policy" not in plan.best.explanation


def test_cue_sector_correction_does_not_require_stationary_cue_stick() -> None:
    planner = GeometryPhysicsPlanner(PlannerConfig(top_k=20, cue_sector_switch_confirm_frames=1), _service())
    moving_stick = _stick(9, 20, 240, 90, 260)
    moving_stick.velocity_mm_s = (800.0, 0.0)
    state = MatchStateFrame(
        frame_id=1,
        ts_cam_ns=1,
        phase="PRE_SHOT_ARMED",
        turn_target_group="solid",
        layout=[
            _obs(1, "cue", 120, 250),
            moving_stick,
            _obs(2, "solid", 620, 380),
            _obs(3, "stripe", 620, 250),
        ],
    )

    plan = planner.plan(state)

    assert plan.best is not None
    assert plan.best.target_group == "stripe"
    assert plan.best.explanation["cue_sector_policy"] == "opponent_confirmation"


def test_cue_sector_correction_prefers_frame_line_over_stick_bbox_axis() -> None:
    planner = GeometryPhysicsPlanner(
        PlannerConfig(top_k=20, cue_sector_switch_confirm_frames=1, cue_sector_corridor_width_px=140.0),
        _service(),
    )
    frame = np.zeros((500, 1000, 3), dtype=np.uint8)
    cv2.line(frame, (20, 250), (114, 250), (255, 255, 255), 4)
    state = MatchStateFrame(
        frame_id=1,
        ts_cam_ns=1,
        phase="PRE_SHOT_ARMED",
        turn_target_group="solid",
        layout=[
            _obs(1, "cue", 120, 250),
            _stick(9, 80, 200, 100, 240),
            _obs(2, "solid", 620, 380),
            _obs(3, "stripe", 620, 250),
        ],
    )

    plan = planner.plan(state, frame_bgr=frame)

    assert plan.best is not None
    assert plan.best.target_group == "stripe"
    assert plan.best.explanation["cue_sector_policy"] == "opponent_confirmation"


def test_target_lock_keeps_selected_ball_when_cue_stick_disappears(monkeypatch) -> None:
    planner = GeometryPhysicsPlanner(
        PlannerConfig(top_k=20, target_lock_confirm_frames=1),
        _service(),
    )
    raw_aims = iter(
        [
            CueStickAimPx(
                tip_px=np.asarray([140.0, 250.0], dtype=np.float32),
                tail_px=np.asarray([80.0, 250.0], dtype=np.float32),
                direction_px=np.asarray([1.0, 0.0], dtype=np.float32),
                source="test_lock",
                score=1.0,
            ),
            None,
        ]
    )
    monkeypatch.setattr(planner.cue_sector.aim_detector, "detect", lambda **_kwargs: next(raw_aims))
    layout = [
        _obs(1, "cue", 120, 250),
        _obs(2, "solid", 620, 250),
        _obs(3, "stripe", 620, 380),
    ]

    seeded = planner.plan(MatchStateFrame(frame_id=1, ts_cam_ns=1, phase="PRE_SHOT_ARMED", layout=layout))
    held = planner.plan(
        MatchStateFrame(
            frame_id=2,
            ts_cam_ns=2,
            phase="PRE_SHOT_ARMED",
            turn_target_group="stripe",
            layout=layout,
        )
    )

    assert seeded.best is not None
    assert seeded.best.target_track_id == 2
    assert seeded.locked_target_id == 2
    assert held.best is not None
    assert held.best.target_track_id == 2
    assert held.locked_target_id == 2
    assert held.target_lock_status == "locked_hold_no_aim"
    assert held.best.explanation["target_lock"] is True


def test_target_lock_switches_only_after_user_points_elsewhere_for_confirmation_frames(monkeypatch) -> None:
    planner = GeometryPhysicsPlanner(
        PlannerConfig(
            top_k=20,
            target_lock_confirm_frames=1,
            target_lock_switch_confirm_frames=3,
        ),
        _service(),
    )
    raw_aims = iter(
        [
            CueStickAimPx(
                tip_px=np.asarray([140.0, 250.0], dtype=np.float32),
                tail_px=np.asarray([80.0, 250.0], dtype=np.float32),
                direction_px=np.asarray([1.0, 0.0], dtype=np.float32),
                source="test_lock_solid",
                score=1.0,
            ),
            CueStickAimPx(
                tip_px=np.asarray([138.0, 254.5], dtype=np.float32),
                tail_px=np.asarray([80.0, 239.5], dtype=np.float32),
                direction_px=np.asarray([0.968, 0.252], dtype=np.float32),
                source="test_switch_pending_1",
                score=1.0,
            ),
            CueStickAimPx(
                tip_px=np.asarray([138.0, 254.5], dtype=np.float32),
                tail_px=np.asarray([80.0, 239.5], dtype=np.float32),
                direction_px=np.asarray([0.968, 0.252], dtype=np.float32),
                source="test_switch_pending_2",
                score=1.0,
            ),
            CueStickAimPx(
                tip_px=np.asarray([138.0, 254.5], dtype=np.float32),
                tail_px=np.asarray([80.0, 239.5], dtype=np.float32),
                direction_px=np.asarray([0.968, 0.252], dtype=np.float32),
                source="test_switch_commit",
                score=1.0,
            ),
        ]
    )
    monkeypatch.setattr(planner.cue_sector.aim_detector, "detect", lambda **_kwargs: next(raw_aims))
    layout = [
        _obs(1, "cue", 120, 250),
        _obs(2, "solid", 620, 250),
        _obs(3, "stripe", 620, 380),
    ]

    planner.plan(MatchStateFrame(frame_id=1, ts_cam_ns=1, phase="PRE_SHOT_ARMED", layout=layout))
    pending_1 = planner.plan(MatchStateFrame(frame_id=2, ts_cam_ns=2, phase="PRE_SHOT_ARMED", layout=layout))
    pending_2 = planner.plan(MatchStateFrame(frame_id=3, ts_cam_ns=3, phase="PRE_SHOT_ARMED", layout=layout))
    committed = planner.plan(MatchStateFrame(frame_id=4, ts_cam_ns=4, phase="PRE_SHOT_ARMED", layout=layout))

    assert pending_1.best is not None
    assert pending_1.best.target_track_id == 2
    assert pending_1.target_lock_status == "switch_pending 1/3"
    assert pending_2.best is not None
    assert pending_2.best.target_track_id == 2
    assert pending_2.target_lock_status == "switch_pending 2/3"
    assert committed.best is not None
    assert committed.best.target_track_id == 3
    assert committed.target_lock_status == "switch_commit"


def test_target_lock_ignores_ball_that_only_overlaps_corridor_edge(monkeypatch) -> None:
    planner = GeometryPhysicsPlanner(
        PlannerConfig(
            top_k=20,
            target_lock_confirm_frames=1,
            target_lock_switch_confirm_frames=2,
        ),
        _service(),
    )
    raw_aims = iter(
        [
            CueStickAimPx(
                tip_px=np.asarray([140.0, 250.0], dtype=np.float32),
                tail_px=np.asarray([80.0, 250.0], dtype=np.float32),
                direction_px=np.asarray([1.0, 0.0], dtype=np.float32),
                source="test_lock_seed",
                score=1.0,
            ),
            CueStickAimPx(
                tip_px=np.asarray([140.0, 250.0], dtype=np.float32),
                tail_px=np.asarray([80.0, 250.0], dtype=np.float32),
                direction_px=np.asarray([1.0, 0.0], dtype=np.float32),
                source="test_lock_hold_1",
                score=1.0,
            ),
            CueStickAimPx(
                tip_px=np.asarray([140.0, 250.0], dtype=np.float32),
                tail_px=np.asarray([80.0, 250.0], dtype=np.float32),
                direction_px=np.asarray([1.0, 0.0], dtype=np.float32),
                source="test_lock_hold_2",
                score=1.0,
            ),
        ]
    )
    monkeypatch.setattr(planner.cue_sector.aim_detector, "detect", lambda **_kwargs: next(raw_aims))
    seed_layout = [
        _obs(1, "cue", 120, 250),
        _obs(2, "stripe", 720, 298),
    ]
    edge_layout = [
        _obs(1, "cue", 120, 250),
        _obs(2, "stripe", 720, 298),
        _obs(3, "stripe", 350, 325),
    ]

    planner.plan(
        MatchStateFrame(
            frame_id=1,
            ts_cam_ns=1,
            phase="PRE_SHOT_ARMED",
            turn_target_group="stripe",
            layout=seed_layout,
        )
    )
    held_1 = planner.plan(
        MatchStateFrame(
            frame_id=2,
            ts_cam_ns=2,
            phase="PRE_SHOT_ARMED",
            turn_target_group="stripe",
            layout=edge_layout,
        )
    )
    held_2 = planner.plan(
        MatchStateFrame(
            frame_id=3,
            ts_cam_ns=3,
            phase="PRE_SHOT_ARMED",
            turn_target_group="stripe",
            layout=edge_layout,
        )
    )

    assert held_1.best is not None
    assert held_1.best.target_track_id == 2
    assert held_1.locked_target_id == 2
    assert held_2.best is not None
    assert held_2.best.target_track_id == 2
    assert held_2.locked_target_id == 2
    assert held_2.target_lock_status == "locked_hold_same"


def test_target_lock_ignores_new_aim_during_shot_motion(monkeypatch) -> None:
    planner = GeometryPhysicsPlanner(
        PlannerConfig(
            top_k=20,
            target_lock_confirm_frames=1,
            target_lock_switch_confirm_frames=2,
        ),
        _service(),
    )
    raw_aims = iter(
        [
            CueStickAimPx(
                tip_px=np.asarray([140.0, 250.0], dtype=np.float32),
                tail_px=np.asarray([80.0, 250.0], dtype=np.float32),
                direction_px=np.asarray([1.0, 0.0], dtype=np.float32),
                source="test_lock_solid",
                score=1.0,
            ),
            CueStickAimPx(
                tip_px=np.asarray([138.0, 254.5], dtype=np.float32),
                tail_px=np.asarray([80.0, 239.5], dtype=np.float32),
                direction_px=np.asarray([0.968, 0.252], dtype=np.float32),
                source="test_motion_1",
                score=1.0,
            ),
            CueStickAimPx(
                tip_px=np.asarray([138.0, 254.5], dtype=np.float32),
                tail_px=np.asarray([80.0, 239.5], dtype=np.float32),
                direction_px=np.asarray([0.968, 0.252], dtype=np.float32),
                source="test_motion_2",
                score=1.0,
            ),
        ]
    )
    monkeypatch.setattr(planner.cue_sector.aim_detector, "detect", lambda **_kwargs: next(raw_aims))
    layout = [
        _obs(1, "cue", 120, 250),
        _obs(2, "solid", 620, 250),
        _obs(3, "stripe", 620, 380),
    ]

    planner.plan(MatchStateFrame(frame_id=1, ts_cam_ns=1, phase="PRE_SHOT_ARMED", layout=layout))
    moving_1 = planner.plan(MatchStateFrame(frame_id=2, ts_cam_ns=2, phase="SHOT_ACTIVE", layout=layout))
    moving_2 = planner.plan(MatchStateFrame(frame_id=3, ts_cam_ns=3, phase="SHOT_ACTIVE", layout=layout))

    assert moving_1.best is not None
    assert moving_1.best.target_track_id == 2
    assert moving_1.target_lock_status == "locked_hold_motion"
    assert moving_2.best is not None
    assert moving_2.best.target_track_id == 2
    assert moving_2.target_lock_status == "locked_hold_motion"


def test_target_shot_mode_activates_after_pointing_object_for_threshold(monkeypatch) -> None:
    service = _service()
    service.table.pockets_mm = [(500, 0)]
    planner = GeometryPhysicsPlanner(
        PlannerConfig(
            top_k=20,
            target_lock_enabled=False,
            target_shot_activate_hold_ms=1000,
        ),
        service,
    )
    layout = [
        _obs(1, "cue", 100, 400),
        _obs(2, "solid", 500, 250),
        _stick(9, 490, 320, 510, 390),
    ]
    pointed = SimpleNamespace(track_id=2, group="solid", score=1.0)
    monkeypatch.setattr(planner.target_shot_mode, "_pointed_ball", lambda *args, **kwargs: pointed)

    pending = planner.plan(MatchStateFrame(frame_id=1, ts_cam_ns=_ts_ms(0), phase="PRE_SHOT_ARMED", layout=layout))
    almost = planner.plan(MatchStateFrame(frame_id=2, ts_cam_ns=_ts_ms(600), phase="PRE_SHOT_ARMED", layout=layout))
    active = planner.plan(MatchStateFrame(frame_id=3, ts_cam_ns=_ts_ms(1000), phase="PRE_SHOT_ARMED", layout=layout))

    assert pending.shot_mode == "rule"
    assert pending.target_shot_status == "activate_pending 0/1000ms"
    assert almost.shot_mode == "rule"
    assert almost.target_shot_status == "activate_pending 600/1000ms"
    assert active.shot_mode == "target"
    assert active.locked_target_id == 2
    assert active.best is not None
    assert active.best.target_track_id == 2
    assert active.best.explanation["target_shot"] is True
    assert active.best.explanation["target_shot_independent_of_cue_stick"] is True


def test_target_shot_mode_keeps_edge_detection_available_when_scanning_candidates(monkeypatch) -> None:
    service = _service()
    service.table.pockets_mm = [(500, 0)]
    planner = GeometryPhysicsPlanner(
        PlannerConfig(
            top_k=20,
            target_lock_enabled=False,
            target_shot_activate_hold_ms=1000,
        ),
        service,
    )
    original_detect = planner.target_shot_mode.aim_detector.detect
    detect_kwargs_seen: list[tuple[object, object]] = []

    def _wrapped_detect(**kwargs):
        detect_kwargs_seen.append((kwargs.get("prefer_tracks", False), kwargs.get("allow_edge_detection", True)))
        return original_detect(**kwargs)

    monkeypatch.setattr(planner.target_shot_mode.aim_detector, "detect", _wrapped_detect)
    layout = [
        _obs(1, "cue", 100, 400),
        _obs(2, "solid", 500, 250),
        _obs(3, "stripe", 500, 320),
        _stick(9, 490, 320, 510, 390),
    ]

    plan = planner.plan(MatchStateFrame(frame_id=1, ts_cam_ns=_ts_ms(0), phase="PRE_SHOT_ARMED", layout=layout))

    assert detect_kwargs_seen == [(False, True)]
    assert plan.shot_mode == "rule"
    assert plan.best is not None


def test_planner_calls_shared_aim_detector_at_most_once_per_frame(monkeypatch) -> None:
    service = _service()
    planner = GeometryPhysicsPlanner(
        PlannerConfig(
            top_k=20,
            target_lock_enabled=False,
            target_shot_activate_hold_ms=1000,
        ),
        service,
    )
    detect_calls = 0
    original_detect = planner.cue_sector.aim_detector.detect

    def _wrapped_detect(**kwargs):
        nonlocal detect_calls
        detect_calls += 1
        return original_detect(**kwargs)

    monkeypatch.setattr(planner.cue_sector.aim_detector, "detect", _wrapped_detect)
    state = MatchStateFrame(
        frame_id=1,
        ts_cam_ns=_ts_ms(0),
        phase="PRE_SHOT_ARMED",
        turn_target_group="solid",
        layout=[
            _obs(1, "cue", 120, 250),
            _stick(9, 20, 240, 90, 260),
            _obs(2, "solid", 620, 250),
            _obs(3, "stripe", 620, 320),
        ],
    )

    plan = planner.plan(state)

    assert detect_calls == 1
    assert plan.best is not None


def test_planner_reuses_same_shared_aim_for_target_shot_and_cue_sector(monkeypatch) -> None:
    planner = GeometryPhysicsPlanner(
        PlannerConfig(
            top_k=20,
            target_lock_enabled=False,
            target_shot_activate_hold_ms=1000,
        ),
        _service(),
    )
    shared_ids: dict[str, int | None] = {}
    detect_calls = 0
    original_detect = planner.cue_sector.aim_detector.detect
    original_pointed_ball = planner.target_shot_mode._pointed_ball
    original_detect_aim = planner.cue_sector.detect_aim

    def _wrapped_detect(**kwargs):
        nonlocal detect_calls
        detect_calls += 1
        return original_detect(**kwargs)

    def _wrapped_pointed_ball(*args, **kwargs):
        frame_context = kwargs.get("frame_context")
        aim = frame_context.shared_aim() if frame_context is not None else None
        shared_ids["target_shot"] = id(aim) if aim is not None else None
        return original_pointed_ball(*args, **kwargs)

    def _wrapped_detect_aim(*args, **kwargs):
        frame_context = kwargs.get("frame_context")
        aim = frame_context.shared_aim() if frame_context is not None else None
        shared_ids["cue_sector"] = id(aim) if aim is not None else None
        return original_detect_aim(*args, **kwargs)

    monkeypatch.setattr(planner.cue_sector.aim_detector, "detect", _wrapped_detect)
    monkeypatch.setattr(planner.target_shot_mode, "_pointed_ball", _wrapped_pointed_ball)
    monkeypatch.setattr(planner.cue_sector, "detect_aim", _wrapped_detect_aim)
    state = MatchStateFrame(
        frame_id=1,
        ts_cam_ns=_ts_ms(0),
        phase="PRE_SHOT_ARMED",
        turn_target_group="solid",
        layout=[
            _obs(1, "cue", 120, 250),
            _stick(9, 20, 240, 90, 260),
            _obs(2, "solid", 620, 250),
            _obs(3, "stripe", 620, 320),
        ],
    )

    plan = planner.plan(state)

    assert detect_calls == 1
    assert plan.best is not None
    assert shared_ids["target_shot"] is not None
    assert shared_ids["target_shot"] == shared_ids["cue_sector"]


def test_target_shot_mode_holds_without_cue_stick_and_keeps_route_independent_of_direction(monkeypatch) -> None:
    service = _service()
    service.table.pockets_mm = [(500, 0)]
    planner = GeometryPhysicsPlanner(
        PlannerConfig(
            top_k=20,
            target_lock_enabled=False,
            target_shot_activate_hold_ms=100,
            target_shot_miss_grace_ms=200,
        ),
        service,
    )
    locked_layout = [
        _obs(1, "cue", 100, 400),
        _obs(2, "solid", 500, 250),
        _stick(9, 490, 320, 510, 390),
    ]
    no_stick_layout = [
        _obs(1, "cue", 100, 400),
        _obs(2, "solid", 500, 250),
    ]
    pointed = iter(
        [
            SimpleNamespace(track_id=2, group="solid", score=1.0),
            SimpleNamespace(track_id=2, group="solid", score=1.0),
            None,
        ]
    )
    monkeypatch.setattr(planner.target_shot_mode, "_pointed_ball", lambda *args, **kwargs: next(pointed))

    planner.plan(MatchStateFrame(frame_id=1, ts_cam_ns=_ts_ms(0), phase="PRE_SHOT_ARMED", layout=locked_layout))
    locked = planner.plan(MatchStateFrame(frame_id=2, ts_cam_ns=_ts_ms(100), phase="PRE_SHOT_ARMED", layout=locked_layout))
    held = planner.plan(MatchStateFrame(frame_id=3, ts_cam_ns=_ts_ms(180), phase="PRE_SHOT_ARMED", layout=no_stick_layout))

    assert locked.shot_mode == "target"
    assert held.shot_mode == "target"
    assert held.locked_target_id == 2
    assert held.target_shot_status.startswith("active_hold_no_aim")
    assert locked.best is not None
    assert held.best is not None
    assert held.best.object_line == locked.best.object_line


def test_target_shot_mode_releases_after_release_confirmation_window(monkeypatch) -> None:
    service = _service()
    service.table.pockets_mm = [(500, 0)]
    planner = GeometryPhysicsPlanner(
        PlannerConfig(
            top_k=20,
            target_lock_enabled=False,
            target_shot_activate_hold_ms=100,
            target_shot_release_confirm_ms=300,
        ),
        service,
    )
    layout = [
        _obs(1, "cue", 100, 400),
        _obs(2, "solid", 500, 250),
        _stick(9, 490, 320, 510, 390),
    ]
    pointed = SimpleNamespace(track_id=2, group="solid", score=1.0)
    monkeypatch.setattr(planner.target_shot_mode, "_pointed_ball", lambda *args, **kwargs: pointed)

    planner.plan(MatchStateFrame(frame_id=1, ts_cam_ns=_ts_ms(0), phase="PRE_SHOT_ARMED", layout=layout))
    locked = planner.plan(MatchStateFrame(frame_id=2, ts_cam_ns=_ts_ms(100), phase="PRE_SHOT_ARMED", layout=layout))
    still_locked = planner.plan(MatchStateFrame(frame_id=3, ts_cam_ns=_ts_ms(250), phase="SHOT_ACTIVE", layout=layout))
    moving = planner.plan(MatchStateFrame(frame_id=4, ts_cam_ns=_ts_ms(450), phase="SHOT_ACTIVE", layout=layout))

    assert locked.shot_mode == "target"
    assert still_locked.shot_mode == "target"
    assert still_locked.target_shot_status.startswith("release_pending 150/300ms")
    assert moving.shot_mode != "target"
    assert moving.locked_target_id is None


def test_target_shot_mode_keeps_activation_progress_during_short_no_aim_gap(monkeypatch) -> None:
    service = _service()
    service.table.pockets_mm = [(500, 0)]
    planner = GeometryPhysicsPlanner(
        PlannerConfig(
            top_k=20,
            target_lock_enabled=False,
            target_shot_activate_hold_ms=1000,
            target_shot_miss_grace_ms=250,
        ),
        service,
    )
    layout = [
        _obs(1, "cue", 100, 400),
        _obs(2, "solid", 500, 250),
    ]
    pointed = iter(
        [
            SimpleNamespace(track_id=2, group="solid", score=1.0),
            None,
            SimpleNamespace(track_id=2, group="solid", score=1.0),
            SimpleNamespace(track_id=2, group="solid", score=1.0),
        ]
    )
    monkeypatch.setattr(planner.target_shot_mode, "_pointed_ball", lambda *args, **kwargs: next(pointed))

    planner.plan(MatchStateFrame(frame_id=1, ts_cam_ns=_ts_ms(0), phase="PRE_SHOT_ARMED", layout=layout))
    grace = planner.plan(MatchStateFrame(frame_id=2, ts_cam_ns=_ts_ms(100), phase="PRE_SHOT_ARMED", layout=layout))
    regain = planner.plan(MatchStateFrame(frame_id=3, ts_cam_ns=_ts_ms(700), phase="PRE_SHOT_ARMED", layout=layout))
    active = planner.plan(MatchStateFrame(frame_id=4, ts_cam_ns=_ts_ms(1100), phase="PRE_SHOT_ARMED", layout=layout))

    assert grace.shot_mode == "rule"
    assert grace.target_shot_status == "activate_pending 0/1000ms grace 100/250ms"
    assert regain.shot_mode == "rule"
    assert regain.target_shot_status == "activate_pending 600/1000ms"
    assert active.shot_mode == "target"
    assert active.locked_target_id == 2


def test_target_shot_mode_resets_activation_after_miss_exceeds_grace(monkeypatch) -> None:
    service = _service()
    service.table.pockets_mm = [(500, 0)]
    planner = GeometryPhysicsPlanner(
        PlannerConfig(
            top_k=20,
            target_lock_enabled=False,
            target_shot_activate_hold_ms=1000,
            target_shot_miss_grace_ms=200,
        ),
        service,
    )
    layout = [
        _obs(1, "cue", 100, 400),
        _obs(2, "solid", 500, 250),
    ]
    pointed = iter(
        [
            SimpleNamespace(track_id=2, group="solid", score=1.0),
            None,
            SimpleNamespace(track_id=2, group="solid", score=1.0),
            SimpleNamespace(track_id=2, group="solid", score=1.0),
        ]
    )
    monkeypatch.setattr(planner.target_shot_mode, "_pointed_ball", lambda *args, **kwargs: next(pointed))

    planner.plan(MatchStateFrame(frame_id=1, ts_cam_ns=_ts_ms(0), phase="PRE_SHOT_ARMED", layout=layout))
    expired = planner.plan(MatchStateFrame(frame_id=2, ts_cam_ns=_ts_ms(300), phase="PRE_SHOT_ARMED", layout=layout))
    restarted = planner.plan(MatchStateFrame(frame_id=3, ts_cam_ns=_ts_ms(700), phase="PRE_SHOT_ARMED", layout=layout))
    progress = planner.plan(MatchStateFrame(frame_id=4, ts_cam_ns=_ts_ms(1000), phase="PRE_SHOT_ARMED", layout=layout))

    assert expired.target_shot_status == "inactive_no_aim"
    assert restarted.target_shot_status == "activate_pending 0/1000ms"
    assert progress.target_shot_status == "activate_pending 300/1000ms"


def test_target_shot_mode_switches_only_after_longer_confirmation_window(monkeypatch) -> None:
    service = _service()
    planner = GeometryPhysicsPlanner(
        PlannerConfig(
            top_k=20,
            target_lock_enabled=False,
            target_shot_activate_hold_ms=200,
            target_shot_switch_hold_ms=1500,
        ),
        service,
    )
    layout = [
        _obs(1, "cue", 100, 400),
        _obs(2, "solid", 500, 250),
        _obs(3, "stripe", 650, 250),
    ]
    pointed = iter(
        [
            SimpleNamespace(track_id=2, group="solid", score=1.0),
            SimpleNamespace(track_id=2, group="solid", score=1.0),
            SimpleNamespace(track_id=3, group="stripe", score=1.0),
            SimpleNamespace(track_id=3, group="stripe", score=1.0),
            SimpleNamespace(track_id=3, group="stripe", score=1.0),
        ]
    )
    monkeypatch.setattr(planner.target_shot_mode, "_pointed_ball", lambda *args, **kwargs: next(pointed))

    planner.plan(MatchStateFrame(frame_id=1, ts_cam_ns=_ts_ms(0), phase="PRE_SHOT_ARMED", layout=layout))
    locked = planner.plan(MatchStateFrame(frame_id=2, ts_cam_ns=_ts_ms(200), phase="PRE_SHOT_ARMED", layout=layout))
    switch_start = planner.plan(MatchStateFrame(frame_id=3, ts_cam_ns=_ts_ms(300), phase="PRE_SHOT_ARMED", layout=layout))
    switch_mid = planner.plan(MatchStateFrame(frame_id=4, ts_cam_ns=_ts_ms(1000), phase="PRE_SHOT_ARMED", layout=layout))
    switched = planner.plan(MatchStateFrame(frame_id=5, ts_cam_ns=_ts_ms(1800), phase="PRE_SHOT_ARMED", layout=layout))

    assert locked.shot_mode == "target"
    assert locked.locked_target_id == 2
    assert locked.best is not None and locked.best.target_track_id == 2
    assert switch_start.shot_mode == "target"
    assert switch_start.locked_target_id == 2
    assert switch_start.target_shot_status.startswith("switch_pending 0/1500ms")
    assert switch_mid.shot_mode == "target"
    assert switch_mid.locked_target_id == 2
    assert switch_mid.target_shot_status.startswith("switch_pending 700/1500ms")
    assert switched.shot_mode == "target"
    assert switched.locked_target_id == 3
    assert switched.best is not None and switched.best.target_track_id == 3


def test_target_shot_mode_can_choose_bank_route_when_direct_route_is_not_possible(monkeypatch) -> None:
    service = _service()
    service.table.pockets_mm = [(500, 0)]
    planner = GeometryPhysicsPlanner(
        PlannerConfig(
            top_k=20,
            target_lock_enabled=False,
            target_shot_activate_hold_ms=100,
            target_shot_max_rebounds=1,
        ),
        service,
    )
    layout = [
        _obs(1, "cue", 700, 300),
        _obs(2, "solid", 500, 250),
        _obs(3, "stripe", 500, 130),
        _stick(9, 490, 320, 510, 390),
    ]
    pointed = SimpleNamespace(track_id=2, group="solid", score=1.0)
    monkeypatch.setattr(planner.target_shot_mode, "_pointed_ball", lambda *args, **kwargs: pointed)

    planner.plan(MatchStateFrame(frame_id=1, ts_cam_ns=_ts_ms(0), phase="PRE_SHOT_ARMED", layout=layout))
    plan = planner.plan(MatchStateFrame(frame_id=2, ts_cam_ns=_ts_ms(100), phase="PRE_SHOT_ARMED", layout=layout))

    assert plan.shot_mode == "target"
    assert plan.best is not None
    assert plan.best.explanation["target_shot_rebounds"] == 1
    assert len(plan.best.object_line) == 3


def test_target_shot_direct_route_matches_rule_route_pocket_relief(monkeypatch) -> None:
    service = _service_for_table(2540.0, 1270.0)
    service.table.inner_polygon_mm = [(172.16, 143.93), (2373.49, 143.93), (2373.49, 1117.6), (172.16, 1117.6)]
    service.table.center_playable_polygon_mm = list(service.table.inner_polygon_mm)
    service.table.pockets_mm = [
        (163.7, 148.2),
        (1259.41, 118.99),
        (2375.26, 137.54),
        (2383.8, 1118.35),
        (1261.18, 1142.0),
        (160.53, 1113.65),
    ]
    planner = GeometryPhysicsPlanner(
        PlannerConfig(
            top_k=20,
            target_lock_enabled=False,
            target_shot_activate_hold_ms=100,
        ),
        service,
    )
    layout = [
        _obs(1, "cue", 1158.54, 225.0),
        _obs(2, "stripe", 378.95, 206.16),
    ]
    pointed = SimpleNamespace(track_id=2, group="stripe", score=1.0)
    monkeypatch.setattr(planner.target_shot_mode, "_pointed_ball", lambda *args, **kwargs: pointed)

    pending = planner.plan(
        MatchStateFrame(
            frame_id=1,
            ts_cam_ns=_ts_ms(0),
            phase="PRE_SHOT_ARMED",
            turn_target_group="stripe",
            layout=layout,
        )
    )
    active = planner.plan(
        MatchStateFrame(
            frame_id=2,
            ts_cam_ns=_ts_ms(100),
            phase="PRE_SHOT_ARMED",
            turn_target_group="stripe",
            layout=layout,
        )
    )

    assert pending.shot_mode == "rule"
    assert pending.best is not None
    assert pending.best.target_track_id == 2
    assert active.shot_mode == "target"
    assert active.best is not None
    assert active.best.target_track_id == 2
    assert "no_theoretical_route" not in active.target_shot_status


def test_target_shot_activation_keeps_original_target_when_edge_grazer_appears(monkeypatch) -> None:
    planner = GeometryPhysicsPlanner(
        PlannerConfig(
            top_k=20,
            target_lock_confirm_frames=1,
            target_lock_switch_confirm_frames=2,
            target_shot_activate_hold_ms=200,
        ),
        _service(),
    )
    raw_aims = iter(
        [
            CueStickAimPx(
                tip_px=np.asarray([140.0, 250.0], dtype=np.float32),
                tail_px=np.asarray([80.0, 250.0], dtype=np.float32),
                direction_px=np.asarray([1.0, 0.0], dtype=np.float32),
                source="test_lock_seed",
                score=1.0,
            ),
            CueStickAimPx(
                tip_px=np.asarray([140.0, 250.0], dtype=np.float32),
                tail_px=np.asarray([80.0, 250.0], dtype=np.float32),
                direction_px=np.asarray([1.0, 0.0], dtype=np.float32),
                source="test_lock_hold_1",
                score=1.0,
            ),
            CueStickAimPx(
                tip_px=np.asarray([140.0, 250.0], dtype=np.float32),
                tail_px=np.asarray([80.0, 250.0], dtype=np.float32),
                direction_px=np.asarray([1.0, 0.0], dtype=np.float32),
                source="test_lock_hold_2",
                score=1.0,
            ),
        ]
    )
    monkeypatch.setattr(planner.cue_sector.aim_detector, "detect", lambda **_kwargs: next(raw_aims))
    seed_layout = [
        _obs(1, "cue", 120, 250),
        _obs(2, "stripe", 720, 298),
    ]
    edge_layout = [
        _obs(1, "cue", 120, 250),
        _obs(2, "stripe", 720, 298),
        _obs(3, "stripe", 350, 325),
    ]

    pending = planner.plan(
        MatchStateFrame(
            frame_id=1,
            ts_cam_ns=_ts_ms(0),
            phase="PRE_SHOT_ARMED",
            turn_target_group="stripe",
            layout=seed_layout,
        )
    )
    progress = planner.plan(
        MatchStateFrame(
            frame_id=2,
            ts_cam_ns=_ts_ms(100),
            phase="PRE_SHOT_ARMED",
            turn_target_group="stripe",
            layout=edge_layout,
        )
    )
    active = planner.plan(
        MatchStateFrame(
            frame_id=3,
            ts_cam_ns=_ts_ms(200),
            phase="PRE_SHOT_ARMED",
            turn_target_group="stripe",
            layout=edge_layout,
        )
    )

    assert pending.shot_mode == "rule"
    assert progress.shot_mode == "rule"
    assert progress.locked_target_id == 2
    assert active.shot_mode == "target"
    assert active.locked_target_id == 2
    assert active.best is not None
    assert active.best.target_track_id == 2


def test_cue_sector_correction_holds_locked_target_when_strict_corridor_temporarily_loses_it(monkeypatch) -> None:
    planner = GeometryPhysicsPlanner(
        PlannerConfig(
            top_k=20,
            cue_sector_corridor_width_px=140.0,
            cue_sector_switch_confirm_frames=1,
            cue_sector_lock_margin_px=18.0,
            cue_sector_lock_release_frames=3,
        ),
        _service(),
    )
    raw_aims = iter(
        [
            CueStickAimPx(
                tip_px=np.asarray([140.0, 258.0], dtype=np.float32),
                tail_px=np.asarray([80.0, 248.0], dtype=np.float32),
                direction_px=np.asarray([0.9878, 0.1560], dtype=np.float32),
                source="test_lock_seed",
                score=1.0,
            ),
            CueStickAimPx(
                tip_px=np.asarray([140.0, 250.0], dtype=np.float32),
                tail_px=np.asarray([80.0, 250.0], dtype=np.float32),
                direction_px=np.asarray([1.0, 0.0], dtype=np.float32),
                source="test_lock_hold",
                score=1.0,
            ),
        ]
    )
    monkeypatch.setattr(planner.cue_sector.aim_detector, "detect", lambda **_kwargs: next(raw_aims))
    layout = [
        _obs(1, "cue", 120, 250),
        _obs(2, "solid", 620, 329),
    ]
    locked = MatchStateFrame(
        frame_id=1,
        ts_cam_ns=1,
        phase="PRE_SHOT_ARMED",
        turn_target_group="solid",
        layout=layout,
    )
    jitter = MatchStateFrame(
        frame_id=2,
        ts_cam_ns=2,
        phase="PRE_SHOT_ARMED",
        turn_target_group="solid",
        layout=layout,
    )

    seeded = planner.plan(locked)
    held = planner.plan(jitter)

    assert seeded.best is not None
    assert seeded.best.target_track_id == 2
    assert held.best is not None
    assert held.best.target_track_id == 2
    assert held.best.explanation["cue_sector_policy"] == "locked_hold"
    assert held.best.explanation["cue_sector_target_ids"] == [2]


def test_cue_sector_lock_releases_after_consecutive_misses(monkeypatch) -> None:
    planner = GeometryPhysicsPlanner(
        PlannerConfig(
            top_k=20,
            cue_sector_corridor_width_px=140.0,
            cue_sector_switch_confirm_frames=1,
            cue_sector_lock_margin_px=18.0,
            cue_sector_lock_release_frames=2,
        ),
        _service(),
    )
    raw_aims = iter(
        [
            CueStickAimPx(
                tip_px=np.asarray([140.0, 260.0], dtype=np.float32),
                tail_px=np.asarray([80.0, 248.0], dtype=np.float32),
                direction_px=np.asarray([0.9824, 0.1867], dtype=np.float32),
                source="test_lock_seed",
                score=1.0,
            ),
            CueStickAimPx(
                tip_px=np.asarray([140.0, 250.0], dtype=np.float32),
                tail_px=np.asarray([80.0, 250.0], dtype=np.float32),
                direction_px=np.asarray([1.0, 0.0], dtype=np.float32),
                source="test_lock_miss_1",
                score=1.0,
            ),
            CueStickAimPx(
                tip_px=np.asarray([140.0, 250.0], dtype=np.float32),
                tail_px=np.asarray([80.0, 250.0], dtype=np.float32),
                direction_px=np.asarray([1.0, 0.0], dtype=np.float32),
                source="test_lock_miss_2",
                score=1.0,
            ),
        ]
    )
    monkeypatch.setattr(planner.cue_sector.aim_detector, "detect", lambda **_kwargs: next(raw_aims))
    layout = [
        _obs(1, "cue", 120, 250),
        _obs(2, "solid", 620, 345),
    ]
    seeded = MatchStateFrame(
        frame_id=1,
        ts_cam_ns=1,
        phase="PRE_SHOT_ARMED",
        turn_target_group="solid",
        layout=layout,
    )
    miss_1 = MatchStateFrame(
        frame_id=2,
        ts_cam_ns=2,
        phase="PRE_SHOT_ARMED",
        turn_target_group="solid",
        layout=layout,
    )
    miss_2 = MatchStateFrame(
        frame_id=3,
        ts_cam_ns=3,
        phase="PRE_SHOT_ARMED",
        turn_target_group="solid",
        layout=layout,
    )

    planner.plan(seeded)
    first_miss = planner.plan(miss_1)
    assert first_miss.best is None
    assert planner.cue_sector._held_target_id == 2

    second_miss = planner.plan(miss_2)
    assert second_miss.best is None
    assert planner.cue_sector._held_target_id is None


def test_cue_sector_correction_holds_previous_target_when_jitter_points_at_opponent(monkeypatch) -> None:
    planner = GeometryPhysicsPlanner(
        PlannerConfig(
            top_k=20,
            cue_sector_switch_confirm_frames=3,
            cue_sector_corridor_width_px=240.0,
            target_lock_enabled=False,
        ),
        _service(),
    )
    base_layout = [
        _obs(1, "cue", 120, 250),
        _obs(2, "solid", 620, 250),
        _obs(3, "stripe", 620, 380),
    ]
    solid_aim = MatchStateFrame(
        frame_id=1,
        ts_cam_ns=1,
        phase="PRE_SHOT_ARMED",
        turn_target_group="solid",
        layout=[base_layout[0], _stick(9, 20, 240, 90, 260), base_layout[1], base_layout[2]],
    )
    stripe_jitter = MatchStateFrame(
        frame_id=2,
        ts_cam_ns=2,
        phase="PRE_SHOT_ARMED",
        turn_target_group="solid",
        layout=[base_layout[0], _stick(9, 50, 230, 100, 250), base_layout[1], base_layout[2]],
    )
    raw_aims = iter(
        [
            CueStickAimPx(
                tip_px=np.asarray([140.0, 250.0], dtype=np.float32),
                tail_px=np.asarray([80.0, 250.0], dtype=np.float32),
                direction_px=np.asarray([1.0, 0.0], dtype=np.float32),
                source="test",
                score=1.0,
            ),
            CueStickAimPx(
                tip_px=np.asarray([138.0, 254.5], dtype=np.float32),
                tail_px=np.asarray([80.0, 239.5], dtype=np.float32),
                direction_px=np.asarray([0.968, 0.252], dtype=np.float32),
                source="test_jitter",
                score=1.0,
            ),
        ]
    )
    monkeypatch.setattr(planner.cue_sector.aim_detector, "detect", lambda **_kwargs: next(raw_aims))

    first = planner.plan(solid_aim)
    second = planner.plan(stripe_jitter)

    assert first.best is not None
    assert first.best.target_group == "solid"
    assert second.best is not None
    assert second.best.target_group == "solid"
    assert second.best.explanation["cue_sector_policy"] == "stable_hold"


def test_cue_sector_correction_switches_after_confirmation_frames(monkeypatch) -> None:
    planner = GeometryPhysicsPlanner(
        PlannerConfig(
            top_k=20,
            cue_sector_switch_confirm_frames=3,
            cue_sector_corridor_width_px=240.0,
            target_lock_enabled=False,
        ),
        _service(),
    )
    layout = [
        _obs(1, "cue", 120, 250),
        _obs(2, "solid", 620, 250),
        _obs(3, "stripe", 620, 380),
    ]
    solid_aim = MatchStateFrame(
        frame_id=1,
        ts_cam_ns=1,
        phase="PRE_SHOT_ARMED",
        turn_target_group="solid",
        layout=[layout[0], _stick(9, 20, 240, 90, 260), layout[1], layout[2]],
    )
    stripe_aim = [
        MatchStateFrame(
            frame_id=frame_id,
            ts_cam_ns=frame_id,
            phase="PRE_SHOT_ARMED",
            turn_target_group="solid",
            layout=[layout[0], _stick(9, 50, 230, 100, 250), layout[1], layout[2]],
        )
        for frame_id in (2, 3, 4)
    ]
    raw_aims = iter(
        [
            CueStickAimPx(
                tip_px=np.asarray([140.0, 250.0], dtype=np.float32),
                tail_px=np.asarray([80.0, 250.0], dtype=np.float32),
                direction_px=np.asarray([1.0, 0.0], dtype=np.float32),
                source="test",
                score=1.0,
            ),
            CueStickAimPx(
                tip_px=np.asarray([138.0, 254.5], dtype=np.float32),
                tail_px=np.asarray([80.0, 239.5], dtype=np.float32),
                direction_px=np.asarray([0.968, 0.252], dtype=np.float32),
                source="test_pending_1",
                score=1.0,
            ),
            CueStickAimPx(
                tip_px=np.asarray([138.0, 254.5], dtype=np.float32),
                tail_px=np.asarray([80.0, 239.5], dtype=np.float32),
                direction_px=np.asarray([0.968, 0.252], dtype=np.float32),
                source="test_pending_2",
                score=1.0,
            ),
            CueStickAimPx(
                tip_px=np.asarray([138.0, 254.5], dtype=np.float32),
                tail_px=np.asarray([80.0, 239.5], dtype=np.float32),
                direction_px=np.asarray([0.968, 0.252], dtype=np.float32),
                source="test_commit",
                score=1.0,
            ),
        ]
    )
    monkeypatch.setattr(planner.cue_sector.aim_detector, "detect", lambda **_kwargs: next(raw_aims))

    planner.plan(solid_aim)
    first_pending = planner.plan(stripe_aim[0])
    second_pending = planner.plan(stripe_aim[1])
    confirmed = planner.plan(stripe_aim[2])

    assert first_pending.best is not None
    assert first_pending.best.target_group == "solid"
    assert second_pending.best is not None
    assert second_pending.best.target_group == "solid"
    assert confirmed.best is not None
    assert confirmed.best.target_group == "stripe"
    assert confirmed.best.explanation["cue_sector_policy"] == "opponent_confirmation"


def test_cue_sector_correction_keeps_forward_target_when_detector_direction_flips(monkeypatch) -> None:
    planner = GeometryPhysicsPlanner(
        PlannerConfig(top_k=20, cue_sector_switch_confirm_frames=1, cue_sector_corridor_width_px=220.0),
        _service(),
    )
    raw_aims = iter(
        [
            CueStickAimPx(
                tip_px=np.asarray([430.0, 250.0], dtype=np.float32),
                tail_px=np.asarray([360.0, 250.0], dtype=np.float32),
                direction_px=np.asarray([1.0, 0.0], dtype=np.float32),
                source="test",
                score=1.0,
            ),
            CueStickAimPx(
                tip_px=np.asarray([560.0, 250.0], dtype=np.float32),
                tail_px=np.asarray([630.0, 250.0], dtype=np.float32),
                direction_px=np.asarray([-1.0, 0.0], dtype=np.float32),
                source="test_backstroke",
                score=1.0,
            ),
        ]
    )
    monkeypatch.setattr(planner.cue_sector.aim_detector, "detect", lambda **_kwargs: next(raw_aims))
    layout = [
        _obs(1, "cue", 500, 250),
        _obs(2, "solid", 820, 250),
        _obs(3, "stripe", 260, 250),
    ]
    first = MatchStateFrame(
        frame_id=1,
        ts_cam_ns=1,
        phase="PRE_SHOT_ARMED",
        turn_target_group="solid",
        layout=layout,
    )
    backstroke = MatchStateFrame(
        frame_id=2,
        ts_cam_ns=2,
        phase="PRE_SHOT_ARMED",
        turn_target_group="solid",
        layout=layout,
    )

    stable = planner.plan(first)
    during_backstroke = planner.plan(backstroke)

    assert stable.best is not None
    assert stable.best.target_track_id == 2
    assert stable.best.target_group == "solid"
    assert during_backstroke.best is not None
    assert during_backstroke.best.target_track_id == 2
    assert during_backstroke.best.target_group == "solid"
    assert during_backstroke.best.explanation["cue_sector_policy"] == "own_group"


def test_planner_uses_center_playable_boundary_for_edge_rejection() -> None:
    service = _service()
    service.table.inner_polygon_mm = [(0, 0), (1000, 0), (1000, 500), (0, 500)]
    service.table.center_playable_polygon_mm = [(120, 120), (880, 120), (880, 380), (120, 380)]
    planner = GeometryPhysicsPlanner(PlannerConfig(top_k=3), service)
    state = MatchStateFrame(
        frame_id=1,
        ts_cam_ns=1,
        phase="STABLE_IDLE",
        layout=[_obs(1, "cue", 140, 240), _obs(2, "solid", 600, 72)],
    )

    plan = planner.plan(state)

    assert plan.best is None
    assert plan.candidates == []


def test_rule_overlay_includes_dashed_object_and_cue_separation_lines() -> None:
    service = _service()
    config = ProjectionConfig(projector_width=1000, projector_height=500)
    candidate = ShotCandidate(
        candidate_id="rule",
        cue_track_id=1,
        target_track_id=2,
        target_group="solid",
        pocket_index=2,
        cue_ball=(120.0, 250.0),
        object_ball=(520.0, 220.0),
        ghost_ball=(465.0, 236.0),
        pocket_point=(1000.0, 0.0),
        aim_line=[(120.0, 250.0), (465.0, 236.0)],
        object_line=[(520.0, 220.0), (1000.0, 0.0)],
        cut_angle_deg=25.0,
        cue_distance_mm=345.0,
        object_distance_mm=528.0,
        score=1.0,
        risk=0.2,
    )
    plan = ShotPlan(plan_id="p", frame_id=1, ts_cam_ns=1, best=candidate, candidates=[candidate], shot_mode="rule")
    overlay = OverlayBuilder(config, service).from_plan(plan)
    dashed_labels = {line.label for line in overlay.lines if line.style == "dashed"}
    style = projection_route_stroke_style((config.projector_width, config.projector_height), StarFormulaConfig())
    assert "object" in dashed_labels
    assert "cue_separation" in dashed_labels
    assert overlay.lines
    assert all(line.width == style.solid_line_width for line in overlay.lines if line.style != "dashed")
    assert all(line.width == style.dashed_line_width for line in overlay.lines if line.style == "dashed")
    assert overlay.circles
    assert all(circle.width == style.circle_width for circle in overlay.circles)


def test_legacy_free_mode_migrates_to_hook_without_loading_free_planner() -> None:
    service = _service()
    planner = GeometryPhysicsPlanner(PlannerConfig(shot_mode="free", free_max_collisions=2), service)
    state = MatchStateFrame(
        frame_id=1,
        ts_cam_ns=1,
        phase="STABLE_IDLE",
        layout=[
            _obs(1, "cue", 100, 250),
            _obs(2, "solid", 620, 250),
        ],
    )
    plan = planner.plan(state, frame_bgr=None)
    assert plan.shot_mode == "hook"
    assert plan.free_route is None
    assert plan.free_status == "archived"
    assert not hasattr(planner, "free_planner")
    assert plan.best is not None
    assert plan.best.explanation["hook_shot"] is True


def test_planner_can_force_black_target_for_current_turn() -> None:
    planner = GeometryPhysicsPlanner(PlannerConfig(top_k=20), _service())
    state = MatchStateFrame(
        frame_id=1,
        ts_cam_ns=1,
        phase="STABLE_IDLE",
        layout=[
            _obs(1, "cue", 120, 250),
            _obs(2, "solid", 620, 250),
            _obs(3, "stripe", 620, 380),
            _obs(4, "black", 320, 250),
        ],
    )

    plan = planner.plan(state, forced_turn_target_group="black")

    assert plan.best is not None
    assert all(candidate.target_group == "black" for candidate in plan.candidates)
