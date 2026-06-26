from __future__ import annotations

import numpy as np

from bas.calibration.camera import CameraCalibration
from bas.calibration.projector import ProjectionCalibration
from bas.calibration.service import CalibrationService
from bas.config import PlannerConfig
from bas.planning import GeometryPhysicsPlanner
from bas.projection.overlay import OverlayBuilder
from bas.config import ProjectionConfig
from bas.schemas import MatchStateFrame, ShotCandidate, ShotPlan, TableModel, TrackObservation


def _service() -> CalibrationService:
    projection = ProjectionCalibration.fit_from_correspondences(
        np.array([[0, 0], [1000, 0], [1000, 500], [0, 500]], dtype=np.float64),
        np.array([[0, 0], [1000, 0], [1000, 500], [0, 500]], dtype=np.float64),
        projector_size=(1000, 500),
    )
    projection.table_polygon_proj = np.array([[0, 0], [1000, 0], [1000, 500], [0, 500]], dtype=np.float64)
    return CalibrationService(
        camera=CameraCalibration(metadata={}),
        projection=projection,
        table=TableModel(
            width_mm=1000,
            height_mm=500,
            ball_diameter_mm=57.15,
            inner_polygon_mm=[(0, 0), (1000, 0), (1000, 500), (0, 500)],
            pockets_mm=[(0, 0), (500, 0), (1000, 0), (1000, 500), (500, 500), (0, 500)],
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
    overlay = OverlayBuilder(ProjectionConfig(projector_width=1000, projector_height=500), service).from_plan(plan)
    dashed_labels = {line.label for line in overlay.lines if line.style == "dashed"}
    assert "object" in dashed_labels
    assert "cue_separation" in dashed_labels


def test_free_mode_predicts_two_cushion_collisions() -> None:
    service = _service()
    planner = GeometryPhysicsPlanner(PlannerConfig(shot_mode="free", free_max_collisions=2), service)
    state = MatchStateFrame(
        frame_id=1,
        ts_cam_ns=1,
        phase="STABLE_IDLE",
        layout=[
            _obs(1, "cue", 100, 250),
            _stick(2, 20, 240, 90, 260),
        ],
    )
    plan = planner.plan(state, frame_bgr=None)
    assert plan.shot_mode == "free"
    assert plan.free_route is not None
    assert len(plan.free_route.collision_points) == 2
    assert plan.free_route.collision_types == ["edge", "edge"]
    overlay = OverlayBuilder(ProjectionConfig(projector_width=1000, projector_height=500), service).from_plan(plan)
    assert overlay.lines
    assert all(line.color == (255, 255, 255) for line in overlay.lines)
    assert all(color == (255, 255, 255) for _center, _radius, color in overlay.circles)
    assert all(color == (255, 255, 255) for _pos, _text, color in overlay.labels)
