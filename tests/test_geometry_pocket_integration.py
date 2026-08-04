from __future__ import annotations

import json
from dataclasses import replace
from types import SimpleNamespace

import cv2
import numpy as np

from bas.app import RuntimePipeline
from bas.config import CalibrationConfig, StateConfig
from bas.geometry import TableGeometryLoader
from bas.geometry_runtime import RuntimeGeometryReloader
from bas.perception.pocket_observer import PocketObserver
from bas.schemas import FramePacket, MatchPhase, TrackObservation, TracksFrame
from bas.state import ModernMatchStateMachine
from bas.ui.geometry_reference import draw_geometry_reference_lines


def _write_labelme(path, shapes: list[dict[str, object]]) -> None:
    path.write_text(
        json.dumps(
            {
                "imageWidth": 100,
                "imageHeight": 100,
                "shapes": shapes,
            }
        ),
        encoding="utf-8",
    )


def _write_geometry_set(root, *, pocket_x: int) -> tuple[str, str, str]:
    root.mkdir()
    outline = root / "outline.json"
    inline = root / "inline.json"
    pocket = root / "pocket.json"
    gap_left = pocket_x - 10
    gap_right = pocket_x + 10
    _write_labelme(
        outline,
        [
            {
                "label": "outline",
                "points": [[5, 5], [95, 5], [95, 95], [5, 95]],
            }
        ],
    )
    _write_labelme(
        inline,
        [
            {"label": "inline", "points": [[15, 10], [gap_left, 10]]},
            {"label": "inline", "points": [[gap_right, 10], [85, 10]]},
            {"label": "inline", "points": [[90, 15], [90, 85]]},
            {"label": "inline", "points": [[85, 90], [55, 90]]},
            {"label": "inline", "points": [[45, 90], [15, 90]]},
            {"label": "inline", "points": [[10, 85], [10, 15]]},
        ],
    )
    _write_labelme(
        pocket,
        [
            {"label": "pocket0", "points": [[10, 15], [8, 8], [15, 10]]},
            {
                "label": "pocket1",
                "points": [[gap_left, 10], [pocket_x, 2], [gap_right, 10]],
            },
            {"label": "pocket2", "points": [[85, 10], [92, 8], [90, 15]]},
            {"label": "pocket3", "points": [[90, 85], [92, 92], [85, 90]]},
            {"label": "pocket4", "points": [[55, 90], [50, 98], [45, 90]]},
            {"label": "pocket5", "points": [[15, 90], [8, 92], [10, 85]]},
        ],
    )
    return str(outline), str(inline), str(pocket)


def _identity_calibration() -> SimpleNamespace:
    square = [(0.0, 0.0), (100.0, 0.0), (100.0, 100.0), (0.0, 100.0)]
    identity = lambda points: np.asarray(points, dtype=np.float32).reshape((-1, 2))
    return SimpleNamespace(
        table=SimpleNamespace(
            width_mm=100.0,
            height_mm=100.0,
            ball_diameter_mm=12.0,
            center_playable_polygon_mm=square,
        ),
        projection=SimpleNamespace(
            is_valid=True,
            table_polygon_cam=np.asarray(square, dtype=np.float32),
        ),
        camera_px_to_table_mm=identity,
        ball_camera_px_to_table_mm=identity,
        table_mm_to_camera_px=identity,
    )


def _track(track_id: int, x: float, y: float, *, visible: bool = True) -> TrackObservation:
    return TrackObservation(
        track_id=track_id,
        bbox=(x - 6, y - 6, x + 6, y + 6),
        center_px=(x, y),
        radius_px=6,
        cls_name="solid",
        group="solid",
        confidence=0.95,
        velocity_px_s=(0.0, -240.0),
        center_mm=(x, y),
        velocity_mm_s=(0.0, -240.0),
        radius_mm=6.0,
        quality=0.95,
        visibility="visible" if visible else "occluded",
        lost_frames=0 if visible else 1,
    )


def _ball_image(x: int | None, y: int | None) -> np.ndarray:
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    if x is not None and y is not None:
        cv2.circle(image, (x, y), 6, (240, 240, 240), -1)
    return image


def test_cold_invalid_geometry_disables_runtime_detection_policy(tmp_path) -> None:
    pipeline = RuntimePipeline.__new__(RuntimePipeline)
    pipeline.geometry_reloader = RuntimeGeometryReloader()
    pipeline.geometry, _ = pipeline.geometry_reloader.refresh(
        None,
        str(tmp_path / "missing-inline.json"),
        None,
    )
    pipeline.calibration = _identity_calibration()

    policy = pipeline._camera_detection_regions(
        FramePacket(1, 1_000_000_000, "cam", image=_ball_image(None, None))
    )

    assert policy is not None
    assert policy.detection_enabled is False


def test_geometry_loader_canonicalizes_shuffled_six_pocket_shape_order(tmp_path) -> None:
    pocket_path = tmp_path / "pocket.json"
    source_centers = [
        (10, 90),
        (90, 10),
        (50, 90),
        (10, 10),
        (90, 90),
        (50, 10),
    ]
    _write_labelme(
        pocket_path,
        [
            {
                "label": "pocket",
                "points": [[x - 2, y], [x, y - 2], [x + 2, y]],
            }
            for x, y in source_centers
        ],
    )

    geometry = TableGeometryLoader.load(None, None, str(pocket_path))
    centers = [tuple(np.mean(curve, axis=0)) for curve in geometry.pockets_norm]

    np.testing.assert_allclose(
        centers,
        [
            (0.10, 0.09333333),
            (0.50, 0.09333333),
            (0.90, 0.09333333),
            (0.90, 0.89333333),
            (0.50, 0.89333333),
            (0.10, 0.89333333),
        ],
        atol=1e-6,
    )


def test_geometry_loader_splices_pocket_curve_into_continuous_inline_rail(tmp_path) -> None:
    outline_path = tmp_path / "outline.json"
    inline_path = tmp_path / "inline.json"
    pocket_path = tmp_path / "pocket.json"
    _write_labelme(
        outline_path,
        [{"label": "outline", "points": [[5, 5], [95, 5], [95, 95], [5, 95]]}],
    )
    _write_labelme(
        inline_path,
        [
            {"label": "inline", "points": [[10, 10], [10, 88]]},
            {"label": "inline", "points": [[10, 88], [90, 92]]},
            {"label": "inline", "points": [[90, 92], [90, 10]]},
            {"label": "inline", "points": [[90, 10], [10, 10]]},
        ],
    )
    _write_labelme(
        pocket_path,
        [{"label": "pocket", "points": [[40, 89.5], [50, 98], [60, 90.5]]}],
    )

    geometry = TableGeometryLoader.load(str(outline_path), str(inline_path), str(pocket_path))

    assert len(geometry.inline_norm) == 5
    assert geometry.boundary_complete is True
    assert geometry.boundary_source_count == 6
    assert len(geometry.boundary_segments_norm) == geometry.boundary_source_count
    for pocket_point in geometry.pockets_norm[0]:
        assert any(np.allclose(point, pocket_point) for point in geometry.inner_norm)
    assert any(np.allclose(line[-1], (0.4, 0.895)) for line in geometry.inline_norm)
    assert any(np.allclose(line[0], (0.6, 0.905)) for line in geometry.inline_norm)

    reloader = RuntimeGeometryReloader()
    retained, changed = reloader.refresh(str(outline_path), str(inline_path), str(pocket_path))
    assert changed is False
    assert retained.is_empty is True
    assert reloader.last_error is not None
    assert "exactly six" in reloader.last_error


def test_runtime_geometry_reloader_rejects_disconnected_boundary_segments(tmp_path) -> None:
    outline_path = tmp_path / "outline.json"
    inline_path = tmp_path / "inline.json"
    pocket_path = tmp_path / "pocket.json"
    _write_labelme(
        outline_path,
        [{"label": "outline", "points": [[5, 5], [95, 5], [95, 95], [5, 95]]}],
    )
    _write_labelme(
        inline_path,
        [
            {"label": "inline", "points": [[10, 10], [10, 90]]},
            {"label": "inline", "points": [[10, 90], [90, 90]]},
            {"label": "inline", "points": [[90, 90], [90, 10]]},
            {"label": "inline", "points": [[90, 10], [10, 10]]},
        ],
    )
    _write_labelme(
        pocket_path,
        [{"label": "pocket", "points": [[45, 45], [50, 40], [55, 45]]}],
    )
    reloader = RuntimeGeometryReloader()

    geometry, changed = reloader.refresh(str(outline_path), str(inline_path), str(pocket_path))

    assert changed is False
    assert geometry.is_empty is True
    assert reloader.is_ready is False
    assert reloader.last_error is not None
    assert "incomplete boundary" in reloader.last_error

def test_geometry_reload_recovers_inline_and_pocket_files_selected_in_swapped_rows(tmp_path) -> None:
    outline_path, inline_path, pocket_path = _write_geometry_set(tmp_path / "geometry", pocket_x=50)
    reloader = RuntimeGeometryReloader()

    geometry, changed = reloader.refresh(outline_path, pocket_path, inline_path)
    preview = np.zeros((100, 100, 3), dtype=np.uint8)

    assert changed is True
    assert reloader.is_ready is True
    assert reloader.last_error is None
    assert len(geometry.inline_norm) == 6
    assert len(geometry.pockets_norm) == 6
    assert geometry.inner_norm.shape[0] >= 3
    assert draw_geometry_reference_lines(preview, geometry) == 14
    assert np.count_nonzero(preview) > 0


def test_switching_inline_and_pocket_keeps_new_pocket_observer_and_state_effective(tmp_path) -> None:
    outline_a, inline_a, pocket_a = _write_geometry_set(tmp_path / "a", pocket_x=30)
    outline_b, inline_b, pocket_b = _write_geometry_set(tmp_path / "b", pocket_x=50)
    pipeline = RuntimePipeline.__new__(RuntimePipeline)
    pipeline.geometry_reloader = RuntimeGeometryReloader()
    pipeline.geometry, changed = pipeline.geometry_reloader.refresh(outline_a, inline_a, pocket_a)
    assert changed is True
    pipeline.config = SimpleNamespace(
        geometry=SimpleNamespace(
            outline_path=outline_a,
            inline_path=inline_a,
            pocket_path=pocket_a,
        ),
        calibration=CalibrationConfig(),
    )
    reset_calls: list[bool] = []
    pipeline._reset_temporal_processing_state = lambda: reset_calls.append(True)
    pipeline.calibration = _identity_calibration()
    frame = FramePacket(1, 1_000_000_000, "cam", image=_ball_image(None, None))
    pipeline._update_table_geometry_for_frame(frame)
    old_center_polygon = np.asarray(pipeline.calibration.table.center_playable_polygon_mm, dtype=np.float32)
    old_pockets = list(pipeline.calibration.table.pockets_mm)

    # Match the field swap found in the live settings: content labels, rather
    # than picker-row order, must determine each geometry role.
    pipeline.config.geometry.outline_path = outline_b
    pipeline.config.geometry.inline_path = pocket_b
    pipeline.config.geometry.pocket_path = inline_b

    pipeline._refresh_geometry_if_needed()
    pipeline._update_table_geometry_for_frame(frame)
    policy = pipeline._camera_detection_regions(frame)
    new_center_polygon = np.asarray(pipeline.calibration.table.center_playable_polygon_mm, dtype=np.float32)
    new_pockets = list(pipeline.calibration.table.pockets_mm)

    assert reset_calls == [True]
    assert old_pockets != new_pockets
    assert abs(new_pockets[1][0] - 50.0) < 0.1
    assert old_center_polygon.shape[0] >= 3
    assert new_center_polygon.shape[0] >= 3
    assert not np.array_equal(old_center_polygon, new_center_polygon)
    assert policy is not None
    assert policy.detection_enabled is True
    assert len(policy.ball_guard_regions) == 6
    guard = policy.ball_guard_regions[1]
    assert abs(guard.center_px[0] - 50.0) < 0.1
    selected_policy = replace(
        policy,
        ball_guard_regions=(replace(guard, pocket_index=0),),
    )

    observer = PocketObserver(history_ms=1500)
    state = ModernMatchStateMachine(
        StateConfig(engine="modern", pocket_visual_confirmation_ms=1300)
    )
    inner_px = np.asarray(policy.ball_polygon, dtype=np.float32)
    pocket_curve_px = pipeline.geometry.scaled(100, 100)[2][1]
    state.set_table_context(
        inner_polygon_mm=[tuple(map(float, point)) for point in inner_px],
        pockets_mm=[guard.center_px],
        ball_diameter_mm=12.0,
        table_edge_polygon_mm=[tuple(map(float, point)) for point in inner_px],
        pocket_curves_mm=[
            [tuple(map(float, point)) for point in pocket_curve_px]
        ],
    )
    state.phase = MatchPhase.SHOT_ACTIVE

    approach_track = _track(7, 50.0, 32.0)
    first_frame = FramePacket(1, 1_000_000_000, "cam", image=_ball_image(50, 32))
    first_tracks = TracksFrame(1, 1_000_000_000, [approach_track])
    first_visual = observer.update(first_frame, first_tracks, selected_policy)
    state.update(first_tracks, first_visual)

    crossing_track = _track(7, 50.0, 8.0)
    crossing_frame = FramePacket(2, 1_100_000_000, "cam", image=_ball_image(50, 8))
    crossing_tracks = TracksFrame(2, 1_100_000_000, [crossing_track])
    crossing_visual = observer.update(crossing_frame, crossing_tracks, selected_policy)
    candidate = state.update(crossing_tracks, crossing_visual)

    empty_tracks = TracksFrame(3, 1_200_000_000, [])
    empty_visual = observer.update(
        FramePacket(3, 1_200_000_000, "cam", image=_ball_image(None, None)),
        empty_tracks,
        selected_policy,
    )
    state.update(empty_tracks, empty_visual)
    confirm_tracks = TracksFrame(4, 2_500_000_000, [])
    confirm_visual = observer.update(
        FramePacket(4, 2_500_000_000, "cam", image=_ball_image(None, None)),
        confirm_tracks,
        selected_policy,
    )
    detected = state.update(confirm_tracks, confirm_visual)

    assert crossing_visual.observations[0].inward_crossing is True
    assert any(event.name == "POCKET_CANDIDATE" for event in candidate.events)
    assert any(event.name == "POCKET_DETECTED" for event in detected.events), (
        [event.name for event in detected.events],
        confirm_visual.observations,
        state.debug_snapshot(),
    )
