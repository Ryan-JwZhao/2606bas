from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from bas.calibration.ball_compensation_checkpoint import (
    HoldoutCheckpointObservation,
    ball_compensation_checkpoint_compatibility_errors,
    checkpoint_from_audit_events,
    load_ball_compensation_checkpoint,
    make_ball_compensation_checkpoint,
    save_ball_compensation_checkpoint,
)
from bas.calibration.ball_compensation_sampling import BallCompensationSample


def _sample(index: int, point: tuple[float, float]) -> BallCompensationSample:
    return BallCompensationSample(
        sample_index=index,
        target_table_mm=point,
        detected_camera_px=point,
        projected_target_px=point,
        expected_camera_px=point,
        observed_table_mm=point,
        delta_table_mm=(0.0, 0.0),
        detected_radius_px=20.0,
        detection_confidence=0.92,
        stability_spread_px=0.2,
        geometry_quality=0.88,
        geometry_method="appearance_ellipse",
        detector_version="test",
    )


def _context() -> dict:
    return {
        "projection_file": "C:/calibration/projection.json",
        "detector_model_file": "C:/models/ball.pt",
        "camera_coordinate_domain": "undistorted",
        "frame_rotation_degrees": 180,
        "frame_width": 1920,
        "frame_height": 1080,
        "ball_diameter_mm": 57.15,
        "sampling_detection": "test-v1",
    }


def test_checkpoint_round_trip_and_single_slot_overwrite(tmp_path: Path) -> None:
    path = tmp_path / "checkpoint.json"
    first = make_ball_compensation_checkpoint(
        context=_context(),
        sampling_grid_table_mm=[(10.0, 20.0), (30.0, 40.0)],
        training_cursor=1,
        training_samples=[_sample(0, (10.0, 20.0))],
    )
    save_ball_compensation_checkpoint(path, first)
    second = make_ball_compensation_checkpoint(
        context=_context(),
        sampling_grid_table_mm=[(10.0, 20.0), (30.0, 40.0)],
        training_cursor=2,
        training_samples=[_sample(0, (10.0, 20.0)), _sample(1, (30.0, 40.0))],
        holdout_observations=[HoldoutCheckpointObservation(0, 0, _sample(2, (10.0, 20.0)))],
    )
    save_ball_compensation_checkpoint(path, second)

    loaded = load_ball_compensation_checkpoint(path)

    assert loaded is not None
    assert loaded.training_cursor == 2
    assert len(loaded.training_samples) == 2
    assert loaded.holdout_completed_count == 1
    assert loaded.holdout_observations[0].sample.geometry_method == "appearance_ellipse"
    assert not path.with_suffix(".json.tmp").exists()


def test_checkpoint_compatibility_rejects_changed_calibration_or_grid() -> None:
    checkpoint = make_ball_compensation_checkpoint(
        context=_context(),
        sampling_grid_table_mm=[(10.0, 20.0), (30.0, 40.0)],
        training_cursor=2,
        training_samples=[_sample(0, (10.0, 20.0)), _sample(1, (30.0, 40.0))],
    )
    changed = {**_context(), "projection_file": "C:/calibration/other.json"}

    errors = ball_compensation_checkpoint_compatibility_errors(
        checkpoint,
        context=changed,
        sampling_grid_table_mm=[(10.0, 20.0), (31.0, 40.0)],
    )

    assert any("projection_file" in error for error in errors)
    assert "sampling grid changed" in errors


def test_checkpoint_can_be_recovered_from_complete_training_audit(tmp_path: Path) -> None:
    path = tmp_path / "events.jsonl"
    events = [
        {
            "action": "session_started",
            "details": {"context": {"projection_file": "p.json", "detector_model": "ball.pt"}},
        },
        {
            "action": "calibration_loaded",
            "metrics": {"frame_width": 1920, "frame_height": 1080},
            "details": {"projection_source": "p.json"},
        },
    ]
    for index, point in enumerate(((10.0, 20.0), (30.0, 40.0)), start=1):
        events.append(
            {
                "action": "sample_accepted",
                "metrics": {
                    "sample_index": index,
                    "detection_confidence": 0.92,
                    "stability_spread_px": 0.2,
                    "geometry_quality": 0.88,
                },
                "details": {
                    "target_table_mm": list(point),
                    "detected_camera_px": list(point),
                    "delta_table_mm": [1.0, 2.0],
                    "geometry_method": "appearance_ellipse",
                },
            }
        )
    path.write_text("\n".join(json.dumps(event) for event in events), encoding="utf-8")

    checkpoint = checkpoint_from_audit_events(
        path,
        frame_rotation_degrees=180,
        camera_coordinate_domain="undistorted",
        ball_diameter_mm=57.15,
        sampling_detection="v3",
    )

    assert checkpoint.training_cursor == 2
    assert len(checkpoint.training_samples) == 2
    assert np.allclose(checkpoint.training_samples[1].observed_table_mm, (29.0, 38.0))
    assert checkpoint.context["frame_width"] == 1920
