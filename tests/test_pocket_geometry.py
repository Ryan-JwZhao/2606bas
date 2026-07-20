from __future__ import annotations

import numpy as np
import pytest

from bas.config import StateConfig
from bas.state.pocket_geometry import PocketGeometryContext, PocketGeometryModel


TABLE_EDGE = [(0.0, 0.0), (1000.0, 0.0), (1000.0, 500.0), (0.0, 500.0)]
REACHABLE = [(30.0, 30.0), (970.0, 30.0), (970.0, 470.0), (30.0, 470.0)]
POCKET_CURVES = [
    [(0.0, 80.0), (80.0, 0.0)],
    [(450.0, 0.0), (550.0, 0.0)],
    [(920.0, 0.0), (1000.0, 80.0)],
    [(1000.0, 420.0), (920.0, 500.0)],
    [(550.0, 500.0), (450.0, 500.0)],
    [(80.0, 500.0), (0.0, 420.0)],
]
PHYSICALLY_INSET_POCKETS = [
    (55.0, 55.0),
    (500.0, 30.0),
    (945.0, 55.0),
    (945.0, 445.0),
    (500.0, 470.0),
    (55.0, 445.0),
]


def _model(*, pockets=PHYSICALLY_INSET_POCKETS, curves=POCKET_CURVES) -> PocketGeometryModel:
    return PocketGeometryModel.build(
        PocketGeometryContext(
            table_edge_polygon_mm=TABLE_EDGE,
            ball_center_reachable_polygon_mm=REACHABLE,
            pockets_mm=pockets,
            pocket_curves_mm=curves,
            ball_diameter_mm=56.0,
        ),
        StateConfig(engine="modern"),
        log_diagnostics=False,
    )


def _axis_point(geometry, *, depth: float, lateral: float = 0.0) -> tuple[float, float]:
    center = np.asarray(geometry.center_mm, dtype=np.float64)
    outward = np.asarray(geometry.outward_normal, dtype=np.float64)
    tangent = np.asarray(geometry.tangent_unit, dtype=np.float64)
    point = center + outward * depth + tangent * lateral
    return (float(point[0]), float(point[1]))


def test_real_curves_determine_all_six_outward_normals() -> None:
    model = _model()
    expected = [
        (-1.0, -1.0),
        (0.0, -1.0),
        (1.0, -1.0),
        (1.0, 1.0),
        (0.0, 1.0),
        (-1.0, 1.0),
    ]

    assert model.valid is True
    assert len(model.geometries) == 6
    for geometry, raw_expected in zip(model.geometries, expected):
        normalized = np.asarray(raw_expected, dtype=np.float64)
        normalized /= np.linalg.norm(normalized)
        assert geometry.outward_normal == pytest.approx(tuple(normalized), abs=1e-5)
        assert geometry.inward_normal == pytest.approx(tuple(-normalized), abs=1e-5)
        assert geometry.outward_probe_distance_mm < geometry.inward_probe_distance_mm


def test_table_center_is_not_in_any_pocket_zone() -> None:
    model = _model()

    sample = model.sample((500.0, 250.0))

    assert sample.zone is None
    assert sample.pocket_index is None


def test_reachable_polygon_random_audit_has_less_than_two_percent_interior() -> None:
    diagnostics = _model().diagnostics()

    assert diagnostics["sampled_points"] == 500
    assert diagnostics["sampled_interior_ratio"] < 0.02


def test_outward_axis_transitions_none_mouth_throat_interior() -> None:
    model = _model()
    radius = model.context.ball_diameter_mm * 0.5

    for geometry in model.geometries:
        samples = [
            model.sample(_axis_point(geometry, depth=-radius)),
            model.sample(_axis_point(geometry, depth=0.0)),
            model.sample(_axis_point(geometry, depth=geometry.throat_depth_mm + 1.0)),
            model.sample(_axis_point(geometry, depth=geometry.interior_depth_mm + 1.0)),
        ]
        assert [sample.zone for sample in samples] == [None, "mouth", "throat", "interior"]
        assert [sample.pocket_index for sample in samples[1:]] == [geometry.index] * 3


def test_every_depth_band_requires_lateral_membership() -> None:
    model = _model()
    radius = model.context.ball_diameter_mm * 0.5

    for geometry in model.geometries:
        outside_all_zones = geometry.mouth_width_mm * 0.5 + radius + 1.0
        for depth in (0.0, geometry.throat_depth_mm + 1.0, geometry.interior_depth_mm + 1.0):
            sample = model.sample(_axis_point(geometry, depth=depth, lateral=outside_all_zones))
            assert sample.zone is None
            assert sample.pocket_index is None


def test_degenerate_curve_fails_closed_with_diagnostics() -> None:
    model = _model(curves=[[(100.0, 100.0), (100.0, 100.0)]], pockets=[])

    diagnostics = model.diagnostics()
    sample = model.sample((100.0, 100.0))
    assert model.valid is False
    assert "no_valid_pocket_geometries" in diagnostics["reasons"]
    assert "curve_endpoints_are_degenerate" in diagnostics["pockets"][0]["validation_reasons"]
    assert sample.zone is None
    assert sample.geometry_valid is False


def test_physical_inset_centers_do_not_change_curve_outward_direction() -> None:
    inset_model = _model(pockets=PHYSICALLY_INSET_POCKETS)
    raw_model = _model(
        pockets=[(40.0, 40.0), (500.0, 0.0), (960.0, 40.0), (960.0, 460.0), (500.0, 500.0), (40.0, 460.0)]
    )

    assert [item.outward_normal for item in inset_model.geometries] == pytest.approx(
        [item.outward_normal for item in raw_model.geometries],
        abs=1e-6,
    )
