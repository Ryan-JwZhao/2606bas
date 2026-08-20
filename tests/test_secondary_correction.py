from __future__ import annotations

from bas.config import PlannerConfig
from bas.schemas import Event, MatchStateFrame, ShotCandidate, ShotPlan, TrackObservation
from bas.secondary_correction import SecondaryCorrectionController


class _StateMachine:
    def __init__(self):
        self.calls: list[tuple[str, int, int, str]] = []

    def set_turn_target_group(self, group, *, frame_id=0, ts_cam_ns=0, reason="operator"):
        self.calls.append((group, frame_id, ts_cam_ns, reason))


def _track(track_id: int, group: str) -> TrackObservation:
    return TrackObservation(
        track_id=track_id,
        bbox=(0.0, 0.0, 10.0, 10.0),
        center_px=(5.0, 5.0),
        radius_px=5.0,
        cls_name=group,
        group=group,
        confidence=0.9,
        quality=0.9,
    )


def _candidate(*, target_id: int = 3, target_group: str = "stripe") -> ShotCandidate:
    return ShotCandidate(
        candidate_id="sector",
        cue_track_id=1,
        target_track_id=target_id,
        target_group=target_group,
        pocket_index=0,
        cue_ball=(100.0, 100.0),
        object_ball=(300.0, 100.0),
        ghost_ball=(250.0, 100.0),
        pocket_point=(500.0, 0.0),
        aim_line=[(100.0, 100.0), (250.0, 100.0)],
        object_line=[(300.0, 100.0), (500.0, 0.0)],
        cut_angle_deg=10.0,
        cue_distance_mm=150.0,
        object_distance_mm=220.0,
        score=1.0,
        risk=0.1,
        explanation={
            "cue_sector_requires_confirmation": True,
            "cue_sector_confirmation_target_id": target_id,
            "cue_sector_confirmation_target_group": target_group,
        },
    )


def test_secondary_correction_updates_turn_group_on_confirmed_first_hit() -> None:
    controller = SecondaryCorrectionController(PlannerConfig())
    state_machine = _StateMachine()
    arm_state = MatchStateFrame(frame_id=1, ts_cam_ns=1, phase="PRE_SHOT_ARMED")
    plan = ShotPlan(plan_id="p", frame_id=1, ts_cam_ns=1, best=_candidate(), candidates=[_candidate()])

    controller.arm_from_plan(arm_state, plan)
    updated = controller.advance_from_state(
        MatchStateFrame(
            frame_id=2,
            ts_cam_ns=2,
            phase="SHOT_ACTIVE",
            events=[Event("BALL_COLLISION_CANDIDATE", 2, 2, payload={"track_a": 1, "track_b": 3})],
            layout=[_track(1, "cue"), _track(3, "stripe")],
            turn_target_group="solid",
        ),
        state_machine,
    )

    assert updated == "stripe"
    assert state_machine.calls == [("stripe", 2, 2, "cue_sector_first_contact")]
    assert controller.pending is None


def test_secondary_correction_does_not_update_after_wrong_first_hit() -> None:
    controller = SecondaryCorrectionController(PlannerConfig())
    state_machine = _StateMachine()
    arm_state = MatchStateFrame(frame_id=1, ts_cam_ns=1, phase="PRE_SHOT_ARMED")
    plan = ShotPlan(plan_id="p", frame_id=1, ts_cam_ns=1, best=_candidate(), candidates=[_candidate()])
    controller.arm_from_plan(arm_state, plan)

    first = controller.advance_from_state(
        MatchStateFrame(
            frame_id=2,
            ts_cam_ns=2,
            phase="SHOT_ACTIVE",
            events=[Event("BALL_COLLISION_CANDIDATE", 2, 2, payload={"track_a": 1, "track_b": 2})],
            layout=[_track(1, "cue"), _track(2, "solid"), _track(3, "stripe")],
            turn_target_group="solid",
        ),
        state_machine,
    )
    second = controller.advance_from_state(
        MatchStateFrame(
            frame_id=3,
            ts_cam_ns=3,
            phase="SHOT_ACTIVE",
            events=[Event("BALL_COLLISION_CANDIDATE", 3, 3, payload={"track_a": 1, "track_b": 3})],
            layout=[_track(1, "cue"), _track(2, "solid"), _track(3, "stripe")],
            turn_target_group="solid",
        ),
        state_machine,
    )

    assert first is None
    assert second is None
    assert state_machine.calls == []
    assert controller.pending is None


def test_secondary_correction_does_not_arm_while_break_is_pending() -> None:
    controller = SecondaryCorrectionController(PlannerConfig())
    plan = ShotPlan(plan_id="p", frame_id=1, ts_cam_ns=1, best=_candidate(), candidates=[_candidate()])
    state = MatchStateFrame(
        frame_id=1,
        ts_cam_ns=1,
        phase="PRE_SHOT_ARMED",
        break_shot_pending=True,
    )

    controller.arm_from_plan(state, plan)

    assert controller.pending is None
    assert controller.last_status == "break_pending"
