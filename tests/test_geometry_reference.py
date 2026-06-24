from __future__ import annotations

import json

import numpy as np

from bas.calibration.camera import CameraCalibration
from bas.calibration.projector import ProjectionCalibration
from bas.calibration.service import CalibrationService
from bas.config import AppConfig, ProjectionConfig
from bas.geometry import TableGeometry, TableGeometryLoader
from bas.projection.geometry_reference import build_geometry_reference_overlay
from bas.schemas import TableModel
from bas.user_settings import UserSettings


def _service() -> CalibrationService:
    projection = ProjectionCalibration.fit_from_correspondences(
        np.array([[0, 0], [1000, 0], [1000, 500], [0, 500]], dtype=np.float64),
        np.array([[0, 0], [1000, 0], [1000, 500], [0, 500]], dtype=np.float64),
        projector_size=(1000, 500),
    )
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


def test_geometry_loader_preserves_inline_segments(tmp_path) -> None:
    inline_path = tmp_path / "inline.json"
    inline_path.write_text(
        json.dumps(
            {
                "imageWidth": 100,
                "imageHeight": 50,
                "shapes": [
                    {"label": "inline", "points": [[10, 20], [90, 20]]},
                    {"label": "other", "points": [[0, 0], [1, 1]]},
                ],
            }
        ),
        encoding="utf-8",
    )

    geometry = TableGeometryLoader.load(None, str(inline_path), None)

    assert len(geometry.inline_norm) == 1
    np.testing.assert_allclose(geometry.inline_norm[0], np.array([[0.1, 0.4], [0.9, 0.4]], dtype=np.float32))


def test_geometry_reference_overlay_includes_outline_inline_and_pocket_lines() -> None:
    geometry = TableGeometry(
        outer_norm=np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float32),
        inner_norm=np.array([[0.1, 0.1], [0.9, 0.1], [0.9, 0.9], [0.1, 0.9]], dtype=np.float32),
        inline_norm=[np.array([[0.1, 0.2], [0.9, 0.2]], dtype=np.float32)],
        pockets_norm=[np.array([[0.0, 0.0], [0.08, 0.0], [0.08, 0.08]], dtype=np.float32)],
    )

    overlay = build_geometry_reference_overlay(
        ProjectionConfig(projector_width=1000, projector_height=500),
        _service(),
        geometry,
        frame_size=(1000, 500),
    )

    labels = {line.label for line in overlay.lines}
    assert {"outline", "inline_0", "pocket_0"} <= labels
    outline = next(line for line in overlay.lines if line.label == "outline")
    assert outline.points[0] == outline.points[-1]
    inline = next(line for line in overlay.lines if line.label == "inline_0")
    assert inline.points == [(100.0, 100.0), (900.0, 100.0)]


def test_user_settings_persists_geometry_reference_toggle(tmp_path) -> None:
    path = tmp_path / "user_settings.json"
    path.write_text(json.dumps({"projection_geometry_reference_enabled": False}), encoding="utf-8")
    cfg = AppConfig()
    cfg.projection.geometry_reference_enabled = True

    UserSettings.load(path).apply_to_config(cfg)

    assert cfg.projection.geometry_reference_enabled is False
    saved = UserSettings.from_config(cfg)
    assert saved.projection_geometry_reference_enabled is False
