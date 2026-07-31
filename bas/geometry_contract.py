from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Optional


CALIBRATION_ARTIFACT_SCHEMA_VERSION = 3
BALL_CENTER_REFINER_VERSION = "ball_center_refiner_v2"


def file_fingerprint(path: str | Path | None) -> Optional[str]:
    if path is None:
        return None
    candidate = Path(path)
    if not candidate.exists() or not candidate.is_file():
        return None
    digest = hashlib.sha256()
    with candidate.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return f"sha256:{digest.hexdigest()}"


def calibration_context(
    *,
    frame_width: int,
    frame_height: int,
    frame_rotation_degrees: int,
    camera_coordinate_domain: str,
    distortion_file: str | Path | None,
    projection_file: str | Path | None,
    detector_model_file: str | Path | None,
    ball_diameter_mm: float,
) -> dict[str, Any]:
    return {
        "artifact_schema_version": CALIBRATION_ARTIFACT_SCHEMA_VERSION,
        "frame_width": int(frame_width),
        "frame_height": int(frame_height),
        "frame_rotation_degrees": int(frame_rotation_degrees) % 360,
        "camera_coordinate_domain": str(camera_coordinate_domain),
        "distortion_fingerprint": file_fingerprint(distortion_file),
        "projection_fingerprint": file_fingerprint(projection_file),
        "detector_model_fingerprint": file_fingerprint(detector_model_file),
        "ball_center_refiner_version": BALL_CENTER_REFINER_VERSION,
        "ball_diameter_mm": float(ball_diameter_mm),
    }


def projection_calibration_context(
    *,
    frame_width: int,
    frame_height: int,
    frame_rotation_degrees: int,
    camera_coordinate_domain: str,
    distortion_file: str | Path | None,
    projector_width: int,
    projector_height: int,
) -> dict[str, Any]:
    """Describe the 2-D coordinate domains joined by a projection calibration."""

    return {
        "artifact_schema_version": CALIBRATION_ARTIFACT_SCHEMA_VERSION,
        "frame_width": int(frame_width),
        "frame_height": int(frame_height),
        "frame_rotation_degrees": int(frame_rotation_degrees) % 360,
        "camera_coordinate_domain": str(camera_coordinate_domain),
        "distortion_fingerprint": file_fingerprint(distortion_file),
        "projector_width": int(projector_width),
        "projector_height": int(projector_height),
    }


def context_compatibility_errors(
    stored: dict[str, Any],
    expected: Optional[dict[str, Any]],
) -> tuple[str, ...]:
    """Return context keys that are missing or incompatible."""

    if expected is None:
        return ()
    errors: list[str] = []
    for key, expected_value in expected.items():
        if expected_value is None:
            continue
        if key not in stored:
            errors.append(key)
            continue
        actual = stored[key]
        if isinstance(expected_value, float):
            try:
                matches = abs(float(actual) - expected_value) <= max(1e-6, abs(expected_value) * 1e-6)
            except (TypeError, ValueError):
                matches = False
        else:
            matches = actual == expected_value
        if not matches:
            errors.append(key)
    return tuple(errors)
