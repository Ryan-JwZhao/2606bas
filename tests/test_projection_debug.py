from __future__ import annotations

import numpy as np
import pytest

from bas.calibration.camera import CameraCalibration
from bas.calibration.ball_compensation import BallCompensationModel
from bas.calibration.projector import ProjectionCalibration
from bas.calibration.service import CalibrationService
from bas.schemas import Detection, ProjectionOverlay, TableModel, TrackObservation
from bas.ui.projection_debug import append_projected_ball_overlays, append_projected_boundary_overlays


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
            projection_visible_polygon_mm=[(20, 10), (980, 10), (980, 490), (20, 490)],
            center_playable_polygon_mm=[(30, 20), (970, 20), (970, 480), (30, 480)],
        ),
    )


def _ball_compensated_service() -> CalibrationService:
    projection = ProjectionCalibration.fit_from_correspondences(
        np.array([[0, 0], [1000, 0], [1000, 500], [0, 500]], dtype=np.float64),
        np.array([[0, 0], [1000, 0], [1000, 500], [0, 500]], dtype=np.float64),
        projector_size=(1000, 500),
    )
    projection.table_polygon_proj = np.array([[0, 0], [1000, 0], [1000, 500], [0, 500]], dtype=np.float64)
    return CalibrationService(
        camera=CameraCalibration(metadata={}),
        projection=projection,
        ball_compensation_model=BallCompensationModel(
            mode="engineered_ball_comp_v2",
            control_points_camera_px=np.array([[100.0, 200.0]], dtype=np.float64),
            delta_table_mm=np.array([[0.0, 0.0]], dtype=np.float64),
        ),
        table=TableModel(
            width_mm=1000,
            height_mm=500,
            ball_diameter_mm=57.15,
            inner_polygon_mm=[(0, 0), (1000, 0), (1000, 500), (0, 500)],
            pockets_mm=[],
        ),
    )


def _perspective_service_without_ball_model() -> CalibrationService:
    camera_quad = np.array([[0, 0], [1000, 0], [1000, 500], [0, 500]], dtype=np.float64)
    projector_quad = np.array([[0, 0], [1000, 40], [900, 500], [100, 470]], dtype=np.float64)
    projection = ProjectionCalibration.fit_from_correspondences(
        camera_quad,
        projector_quad,
        projector_size=(1000, 500),
    )
    projection.table_polygon_cam = camera_quad.copy()
    projection.table_polygon_proj = projector_quad.copy()
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


def test_projection_debug_appends_three_boundary_layers() -> None:
    overlay = ProjectionOverlay(overlay_id="debug", frame_id=1, projector_size=(1000, 500))

    count = append_projected_boundary_overlays(overlay, _service())

    assert count == 3
    assert [line.label for line in overlay.lines] == [
        "debug_visible_boundary",
        "debug_physical_boundary",
        "debug_center_boundary",
    ]
    assert [text for _, text, _ in overlay.labels] == ["visible", "physical", "center"]
    assert all(len(line.points) == 5 for line in overlay.lines)
    assert overlay.lines[0].points[0] == overlay.lines[0].points[-1]


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
    assert overlay.circles[0].color == (255, 255, 255)
    assert not overlay.labels


def test_projection_debug_falls_back_to_detections_when_tracks_absent() -> None:
    overlay = ProjectionOverlay(overlay_id="debug", frame_id=1, projector_size=(1000, 500))
    detection = Detection(bbox=(185.0, 235.0, 215.0, 265.0), conf=0.9, cls_id=0, cls_name="cue")

    count = append_projected_ball_overlays(overlay, _service(), detections=[detection])

    assert count == 1
    assert len(overlay.circles) == 1
    assert overlay.circles[0].color == (255, 255, 255)
    assert not overlay.labels


def test_projection_debug_without_ball_model_uses_shared_perspective_ellipse() -> None:
    service = _perspective_service_without_ball_model()
    overlay = ProjectionOverlay(overlay_id="debug", frame_id=1, projector_size=(1000, 500))
    track = TrackObservation(
        track_id=7,
        bbox=(485.0, 235.0, 515.0, 265.0),
        center_px=(500.0, 250.0),
        radius_px=15.0,
        cls_name="cue",
        group="cue",
        confidence=0.9,
        quality=0.9,
        geometry_quality=0.9,
        geometry_method="appearance_ellipse",
    )
    expected = service.ball_geometry.locate(
        track.center_px,
        radius_px=track.radius_px,
        geometry_quality=track.geometry_quality,
        geometry_method=track.geometry_method,
    ).projector_ellipse

    append_projected_ball_overlays(overlay, service, tracks=[track])

    circle = overlay.circles[0]
    assert circle.center == pytest.approx(expected.center_px, abs=1e-5)
    assert circle.radius == pytest.approx(expected.radius_x_px, abs=1e-5)
    assert circle.radius_y == pytest.approx(expected.radius_y_px, abs=1e-5)
    assert circle.rotation_deg == pytest.approx(expected.rotation_deg, abs=1e-5)


def test_ball_compensated_projection_debug_uses_physical_ball_radius() -> None:
    service = _ball_compensated_service()
    overlay_small = ProjectionOverlay(overlay_id="debug_small", frame_id=1, projector_size=(1000, 500))
    overlay_large = ProjectionOverlay(overlay_id="debug_large", frame_id=1, projector_size=(1000, 500))
    track_small = TrackObservation(
        track_id=1,
        bbox=(90.0, 190.0, 110.0, 210.0),
        center_px=(100.0, 200.0),
        radius_px=10.0,
        cls_name="cue",
        group="cue",
        confidence=0.9,
    )
    track_large = TrackObservation(
        track_id=2,
        bbox=(80.0, 180.0, 120.0, 220.0),
        center_px=(100.0, 200.0),
        radius_px=20.0,
        cls_name="cue",
        group="cue",
        confidence=0.9,
    )

    append_projected_ball_overlays(overlay_small, service, tracks=[track_small])
    append_projected_ball_overlays(overlay_large, service, tracks=[track_large])

    assert overlay_small.circles[0].radius == pytest.approx(overlay_large.circles[0].radius, abs=1e-6)
    assert overlay_small.circles[0].radius == pytest.approx(service.ball_projector_radius_px((100.0, 200.0)), abs=1e-6)
