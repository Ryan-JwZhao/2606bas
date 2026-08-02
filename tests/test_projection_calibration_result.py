from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from bas.ui.projection_calibration_result import build_projection_calibration_result_overlay


def test_result_overlay_displays_every_correspondence_with_error_colors_and_metrics() -> None:
    point_count = 96
    xs = np.linspace(80.0, 1120.0, 12)
    ys = np.linspace(90.0, 650.0, 8)
    projector_points = np.asarray([[x, y] for y in ys for x in xs], dtype=np.float64)
    camera_points = projector_points.copy()
    error_px = np.linspace(0.2, 4.0, point_count)

    class Projection:
        table_polygon_proj = np.asarray(
            [[40.0, 40.0], [1240.0, 40.0], [1240.0, 680.0], [40.0, 680.0]],
            dtype=np.float64,
        )
        proj_points = projector_points
        cam_points = camera_points
        quality_report = {
            "quality_gate_passed": True,
            "patterns_used": 8,
            "patterns_total": 8,
            "ransac_inlier_mean_px": 1.6,
            "ransac_inlier_p95_px": 3.2,
            "ransac_inlier_max_px": 4.0,
            "spatial_coverage": {"width_ratio": 0.91, "height_ratio": 0.78, "hull_area_ratio": 0.70},
            "pocket_zones_used": 6,
            "pattern_cv_p95_px": 1.4,
            "maximum_pattern_cv_p95_px": 4.0,
        }

        @staticmethod
        def camera_to_projector_points(points, *, refined=True):
            assert refined is False
            vectors = np.column_stack([error_px, np.zeros(point_count)])
            return np.asarray(points, dtype=np.float64) + vectors

        @staticmethod
        def calibration_error_stats():
            return {"mean_px": 1.6, "p95_px": 3.2, "max_px": 4.0}

    overlay = build_projection_calibration_result_overlay(
        SimpleNamespace(projection=Projection()),
        (1280, 720),
    )

    assert len(overlay.circles) == point_count
    assert len(overlay.labels) == point_count
    assert overlay.circles[0].color == (80, 255, 120)
    assert overlay.circles[-1].color == (60, 60, 255)
    assert len(overlay.texts) == 1
    summary = overlay.texts[0].text
    assert "PASS" in summary
    assert "patterns 8/8" in summary
    assert "96 points" in summary
    assert "P95 3.20px" in summary
    assert "coverage W91% H78% hull70%" in summary
    assert "pockets 6" in summary
    assert "CV P95 1.40px" in summary
