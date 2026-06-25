from __future__ import annotations

import numpy as np

from bas.calibration.camera import CameraCalibration
from bas.calibration.projector import ProjectionCalibration
from bas.calibration.service import CalibrationService
from bas.schemas import Detection, ProjectionOverlay, TableModel, TrackObservation
from bas.ui.projection_debug import append_projected_ball_overlays


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
            pockets_mm=[],
        ),
    )


def test_projection_debug_appends_visible_track_markers() -> None:
    overlay = ProjectionOverlay(overlay_id="debug", frame_id=1, projector_size=(1000, 500))
    track = TrackObservation(
        track_id=7,
        bbox=(85.0, 185.0, 115.0, 215.0),
        center_px=(100.0, 200.0),
        radius_px=15.0,
        cls_name="cue",
        group="cue",
        confidence=0.9,
        quality=0.9,
    )

    count = append_projected_ball_overlays(overlay, _service(), tracks=[track])

    assert count == 1
    assert len(overlay.circles) == 1
    assert overlay.circles[0][2] == (255, 255, 255)
    assert not overlay.labels


def test_projection_debug_falls_back_to_detections_when_tracks_absent() -> None:
    overlay = ProjectionOverlay(overlay_id="debug", frame_id=1, projector_size=(1000, 500))
    detection = Detection(bbox=(185.0, 235.0, 215.0, 265.0), conf=0.9, cls_id=0, cls_name="cue")

    count = append_projected_ball_overlays(overlay, _service(), detections=[detection])

    assert count == 1
    assert len(overlay.circles) == 1
    assert overlay.circles[0][2] == (255, 255, 255)
    assert not overlay.labels
