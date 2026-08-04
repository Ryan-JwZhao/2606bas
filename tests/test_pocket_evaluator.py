from __future__ import annotations

import json
from pathlib import Path

import pytest

from bas.user_settings import UserSettings
from scripts.evaluate_pocket_video import (
    _create_evaluation_calibration,
    _evaluate,
    _load_evaluation_geometry,
    _load_effective_config,
    _validate_replay_context,
)


def test_evaluator_applies_runtime_user_geometry_settings(tmp_path) -> None:
    paths = {
        "outline_path": str(tmp_path / "0804-outline.json"),
        "inline_path": str(tmp_path / "0804-inline.json"),
        "pocket_path": str(tmp_path / "0804-pocket.json"),
        "frame_rotation_degrees": 180,
    }
    settings_path = tmp_path / "user_settings.json"
    settings_path.write_text(json.dumps(paths), encoding="utf-8")

    config = _load_effective_config(
        config_path=Path("configs/default.yaml"),
        user_settings=UserSettings.load(settings_path),
    )

    assert config.geometry.outline_path == paths["outline_path"]
    assert config.geometry.inline_path == paths["inline_path"]
    assert config.geometry.pocket_path == paths["pocket_path"]
    assert config.camera.frame_rotation_degrees == 180


def test_evaluator_can_use_explicit_config_without_runtime_user_settings(tmp_path) -> None:
    settings_path = tmp_path / "user_settings.json"
    settings_path.write_text(
        json.dumps({"outline_path": str(tmp_path / "local-outline.json")}),
        encoding="utf-8",
    )

    config = _load_effective_config(
        config_path=Path("configs/default.yaml"),
        user_settings=UserSettings.load(settings_path),
        apply_user_settings=False,
    )

    assert config.geometry.outline_path.endswith("0804_outline.json")


def test_evaluator_builds_setting_aware_calibration_for_video_dimensions() -> None:
    config = _load_effective_config(
        config_path=Path("configs/default.yaml"),
        apply_user_settings=False,
    )
    config.camera.distortion_correction_enabled = False
    config.calibration.camera_file = None
    config.calibration.projection_file = None
    config.calibration.engineered_ball_compensation_file = None

    calibration = _create_evaluation_calibration(config, 640, 360)

    assert calibration.frame_undistorted is False
    assert calibration.distortion_correction_enabled is False
    assert calibration.projection.is_valid is False


def test_evaluator_uses_runtime_geometry_validation(tmp_path) -> None:
    config = _load_effective_config(
        config_path=Path("configs/default.yaml"),
        apply_user_settings=False,
    )
    outline = tmp_path / "outline.json"
    inline = tmp_path / "inline.json"
    pocket = tmp_path / "pocket.json"
    base = {"imageWidth": 100, "imageHeight": 100, "shapes": []}
    outline.write_text(
        json.dumps(
            base | {"shapes": [{"label": "outline", "points": [[5, 5], [95, 5], [95, 95], [5, 95]]}]}
        ),
        encoding="utf-8",
    )
    inline.write_text(
        json.dumps(
            base
            | {
                "shapes": [
                    {"label": "inline", "points": [[10, 10], [10, 90]]},
                    {"label": "inline", "points": [[10, 90], [90, 90]]},
                    {"label": "inline", "points": [[90, 90], [90, 10]]},
                    {"label": "inline", "points": [[90, 10], [10, 10]]},
                ]
            }
        ),
        encoding="utf-8",
    )
    pocket.write_text(
        json.dumps(
            base | {"shapes": [{"label": "pocket", "points": [[40, 90], [50, 98], [60, 90]]}]}
        ),
        encoding="utf-8",
    )
    config.geometry.outline_path = str(outline)
    config.geometry.inline_path = str(inline)
    config.geometry.pocket_path = str(pocket)

    with pytest.raises(RuntimeError, match="exactly six"):
        _load_evaluation_geometry(config)


def test_evaluator_rejects_replay_recorded_with_other_geometry() -> None:
    with pytest.raises(RuntimeError, match="geometry replay=.*previous.*current=current"):
        _validate_replay_context(
            replay_calibrations={"calib-current"},
            replay_geometries={"previous"},
            calibration_version="calib-current",
            geometry_version="current",
            allow_mismatch=False,
        )


def test_evaluator_requires_explicit_override_for_legacy_replay_without_geometry_version() -> None:
    with pytest.raises(RuntimeError, match=r"geometry replay=\['missing'\]"):
        _validate_replay_context(
            replay_calibrations={"calib-current"},
            replay_geometries=set(),
            calibration_version="calib-current",
            geometry_version="current",
            allow_mismatch=False,
        )

    _validate_replay_context(
        replay_calibrations={"calib-old"},
        replay_geometries=set(),
        calibration_version="calib-current",
        geometry_version="current",
        allow_mismatch=True,
    )


def test_goal_matching_uses_contact_frame_when_shot_phase_has_false_starts() -> None:
    labels = {
        "max_notice_delay_ms": 1500,
        "max_match_frame_gap": 90,
        "goals": [
            {
                "shot": 4,
                "contact_frame": 750,
                "pocket_index": 2,
                "group": "stripe",
                "ball_code": "stb",
            }
        ],
        "no_goals": [{"shot": 3, "reason": "black ball remained on the lip"}],
    }
    detections = [
        {
            "shot": 6,
            "frame_id": 774,
            "group": "stripe",
            "ball_code": "stb",
            "pocket_index": 2,
            "decision_id": "pocket:9",
            "decision_latency_ms": 1328,
        }
    ]

    result = _evaluate(labels, detections, [1.0], [80.0])

    assert result["passed"] is True
    assert result["matched_goals"] == 1
    assert result["false_positives"] == []


def test_no_goal_shot_still_rejects_unmatched_detection() -> None:
    labels = {"goals": [], "no_goals": [{"shot": 3, "reason": "lip stop"}]}
    detection = {
        "shot": 3,
        "frame_id": 430,
        "group": "black",
        "pocket_index": 0,
        "decision_latency_ms": 1300,
    }

    result = _evaluate(labels, [detection], [1.0], [80.0])

    assert result["passed"] is False
    assert result["false_positives"] == [detection]
