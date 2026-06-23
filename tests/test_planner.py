from __future__ import annotations

import numpy as np

from bas.calibration.camera import CameraCalibration
from bas.calibration.projector import ProjectionCalibration
from bas.calibration.service import CalibrationService
from bas.config import PlannerConfig
from bas.planning import GeometryPhysicsPlanner
from bas.schemas import MatchStateFrame, TableModel, TrackObservation


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

