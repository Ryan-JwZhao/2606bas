from __future__ import annotations

import json

import numpy as np

from bas.config import AppConfig
from bas.geometry import TableGeometry, TableGeometryLoader
from bas.ui.geometry_reference import draw_geometry_reference_lines
from bas.user_settings import UserSettings


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


def test_geometry_reference_lines_draw_on_frontend_frame_only() -> None:
    geometry = TableGeometry(
        outer_norm=np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float32),
        inner_norm=np.array([[0.1, 0.1], [0.9, 0.1], [0.9, 0.9], [0.1, 0.9]], dtype=np.float32),
        inline_norm=[np.array([[0.1, 0.2], [0.9, 0.2]], dtype=np.float32)],
        pockets_norm=[np.array([[0.0, 0.0], [0.08, 0.0], [0.08, 0.08]], dtype=np.float32)],
    )
    image = np.zeros((500, 1000, 3), dtype=np.uint8)

    drawn = draw_geometry_reference_lines(image, geometry)

    assert drawn == 3
    assert np.count_nonzero(image) > 0
    assert np.any(np.all(image == np.array([0, 255, 255], dtype=np.uint8), axis=2))


def test_geometry_reference_draw_respects_toggle() -> None:
    geometry = TableGeometry(outer_norm=np.array([[0, 0], [1, 0], [1, 1]], dtype=np.float32))
    image = np.zeros((100, 100, 3), dtype=np.uint8)

    drawn = draw_geometry_reference_lines(image, geometry, enabled=False)

    assert drawn == 0
    assert np.count_nonzero(image) == 0


def test_user_settings_persists_geometry_reference_toggle(tmp_path) -> None:
    path = tmp_path / "user_settings.json"
    path.write_text(json.dumps({"ui_geometry_reference_enabled": False}), encoding="utf-8")
    cfg = AppConfig()
    cfg.projection.geometry_reference_enabled = True

    UserSettings.load(path).apply_to_config(cfg)

    assert cfg.projection.geometry_reference_enabled is False
    saved = UserSettings.from_config(cfg)
    assert saved.ui_geometry_reference_enabled is False


def test_user_settings_accepts_deprecated_projection_geometry_reference_key(tmp_path) -> None:
    path = tmp_path / "user_settings.json"
    path.write_text(json.dumps({"projection_geometry_reference_enabled": False}), encoding="utf-8")
    cfg = AppConfig()
    cfg.projection.geometry_reference_enabled = True

    UserSettings.load(path).apply_to_config(cfg)

    assert cfg.projection.geometry_reference_enabled is False
