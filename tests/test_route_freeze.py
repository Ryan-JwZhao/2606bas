from __future__ import annotations

from bas.config import PlannerConfig
from bas.route_freeze import MotionRouteFreezeController
from bas.schemas import MatchStateFrame, ProjectionOverlay, ShotCandidate, ShotPlan


def _state(frame_id: int, phase: str) -> MatchStateFrame:
    return MatchStateFrame(frame_id=frame_id, ts_cam_ns=frame_id, phase=phase)


def _plan(
    frame_id: int,
    *,
    target_track_id: int | None,
    pocket_index: int | None,
    shot_mode: str = "rule",
    locked_target_id: int | None = None,
    target_shot_status: str = "off",
    cue_ball: tuple[float, float] = (100.0, 100.0),
    object_ball: tuple[float, float] = (200.0, 100.0),
    ghost_ball: tuple[float, float] = (180.0, 100.0),
    pocket_point: tuple[float, float] = (300.0, 0.0),
    score: float = 1.0,
) -> ShotPlan:
    best = None
    candidates = []
    if target_track_id is not None and pocket_index is not None:
        best = ShotCandidate(
            candidate_id=f"t{target_track_id}_p{pocket_index}",
            cue_track_id=1,
            target_track_id=target_track_id,
            target_group="solid",
            pocket_index=pocket_index,
            cue_ball=cue_ball,
            object_ball=object_ball,
            ghost_ball=ghost_ball,
            pocket_point=pocket_point,
            aim_line=[cue_ball, ghost_ball],
            object_line=[object_ball, pocket_point],
            cut_angle_deg=20.0,
            cue_distance_mm=80.0,
            object_distance_mm=100.0,
            score=score,
            risk=0.2,
        )
        candidates = [best]
    return ShotPlan(
        plan_id=f"plan_{frame_id}",
        frame_id=frame_id,
        ts_cam_ns=frame_id,
        best=best,
        candidates=candidates,
        shot_mode=shot_mode,
        locked_target_id=locked_target_id,
        target_shot_status=target_shot_status,
    )


def _overlay(frame_id: int) -> ProjectionOverlay:
    return ProjectionOverlay(overlay_id=f"overlay_{frame_id}", frame_id=frame_id, projector_size=(1280, 800))


def test_motion_route_freeze_holds_last_stable_plan_until_release() -> None:
    config = PlannerConfig(
        route_freeze_enabled=True,
        route_freeze_enter_frames=1,
        route_freeze_release_frames=2,
        route_freeze_switch_confirm_frames=1,
    )
    controller = MotionRouteFreezeController(config)

    first = controller.update(_state(1, "STABLE_IDLE"), _plan(1, target_track_id=2, pocket_index=1), _overlay(1))
    assert first.plan.best is not None
    assert first.plan.best.target_track_id == 2

    moving = controller.update(
        _state(2, "SHOT_ACTIVE"),
        _plan(2, target_track_id=8, pocket_index=5, cue_ball=(140.0, 120.0), object_ball=(280.0, 180.0), ghost_ball=(250.0, 165.0)),
        _overlay(2),
    )
    assert moving.frozen is True
    assert moving.plan.best is not None
    assert moving.plan.best.target_track_id == 2
    assert moving.plan.frame_id == 2

    release_pending = controller.update(
        _state(3, "STABLE_IDLE"),
        _plan(3, target_track_id=8, pocket_index=5, cue_ball=(140.0, 120.0), object_ball=(280.0, 180.0), ghost_ball=(250.0, 165.0)),
        _overlay(3),
    )
    assert release_pending.frozen is True
    assert release_pending.plan.best is not None
    assert release_pending.plan.best.target_track_id == 2

    released = controller.update(
        _state(4, "STABLE_IDLE"),
        _plan(4, target_track_id=8, pocket_index=5, cue_ball=(140.0, 120.0), object_ball=(280.0, 180.0), ghost_ball=(250.0, 165.0)),
        _overlay(4),
    )
    assert released.frozen is False
    assert released.plan.best is not None
    assert released.plan.best.target_track_id == 8


def test_same_route_small_jitter_keeps_previous_displayed_plan() -> None:
    config = PlannerConfig(
        route_freeze_enabled=True,
        route_freeze_same_route_refresh_mm=12.0,
        route_freeze_same_route_refresh_score_delta=0.08,
    )
    controller = MotionRouteFreezeController(config)

    first = controller.update(_state(1, "STABLE_IDLE"), _plan(1, target_track_id=2, pocket_index=1), _overlay(1))
    jittered = controller.update(
        _state(2, "STABLE_IDLE"),
        _plan(
            2,
            target_track_id=2,
            pocket_index=1,
            cue_ball=(104.0, 103.0),
            object_ball=(206.0, 105.0),
            ghost_ball=(184.0, 103.0),
            score=1.03,
        ),
        _overlay(2),
    )

    assert jittered.frozen is False
    assert jittered.plan.best is not None
    assert first.plan.best is not None
    assert jittered.plan.best.cue_ball == first.plan.best.cue_ball
    assert jittered.plan.frame_id == 2

    refreshed = controller.update(
        _state(3, "STABLE_IDLE"),
        _plan(
            3,
            target_track_id=2,
            pocket_index=1,
            cue_ball=(118.0, 112.0),
            object_ball=(220.0, 110.0),
            ghost_ball=(198.0, 106.0),
            score=1.20,
        ),
        _overlay(3),
    )
    assert refreshed.plan.best is not None
    assert refreshed.plan.best.cue_ball == (118.0, 112.0)


def test_route_switch_requires_confirmation_and_significant_change() -> None:
    config = PlannerConfig(
        route_freeze_enabled=True,
        route_freeze_switch_confirm_frames=2,
        route_freeze_switch_min_distance_mm=25.0,
        route_freeze_switch_min_score_delta=0.2,
    )
    controller = MotionRouteFreezeController(config)

    original = controller.update(_state(1, "STABLE_IDLE"), _plan(1, target_track_id=2, pocket_index=1, score=1.0), _overlay(1))

    pending = controller.update(
        _state(2, "STABLE_IDLE"),
        _plan(2, target_track_id=3, pocket_index=4, cue_ball=(140.0, 130.0), object_ball=(260.0, 180.0), ghost_ball=(232.0, 170.0), score=1.25),
        _overlay(2),
    )
    assert pending.plan.best is not None
    assert original.plan.best is not None
    assert pending.plan.best.target_track_id == original.plan.best.target_track_id

    committed = controller.update(
        _state(3, "STABLE_IDLE"),
        _plan(3, target_track_id=3, pocket_index=4, cue_ball=(140.0, 130.0), object_ball=(260.0, 180.0), ghost_ball=(232.0, 170.0), score=1.25),
        _overlay(3),
    )
    assert committed.plan.best is not None
    assert committed.plan.best.target_track_id == 3


def test_target_no_route_clears_previous_route_without_switch_delay() -> None:
    config = PlannerConfig(route_freeze_enabled=True, route_freeze_switch_confirm_frames=8)
    controller = MotionRouteFreezeController(config)

    first = controller.update(
        _state(1, "STABLE_IDLE"),
        _plan(
            1,
            target_track_id=2,
            pocket_index=1,
            shot_mode="target",
            locked_target_id=2,
            target_shot_status="active_new:ok:0_rebounds",
        ),
        _overlay(1),
    )
    cleared = controller.update(
        _state(2, "STABLE_IDLE"),
        _plan(
            2,
            target_track_id=None,
            pocket_index=None,
            shot_mode="target",
            locked_target_id=2,
            target_shot_status="active_hold_same:no_theoretical_route",
        ),
        _overlay(2),
    )

    assert first.plan.best is not None
    assert cleared.plan.best is None
    assert cleared.frozen is False
    assert cleared.status_text == "target_no_route_clear"
