from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


VALID_PROJECTION_ROTATION_DEGREES = (0, 90, 180, 270)


def normalize_projection_rotation_degrees(value: object) -> int:
    """Return a supported clockwise rotation for the final projector frame."""

    try:
        degrees = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"projection rotation must be one of {VALID_PROJECTION_ROTATION_DEGREES}"
        ) from exc
    if degrees not in VALID_PROJECTION_ROTATION_DEGREES:
        raise ValueError(
            f"projection rotation must be one of {VALID_PROJECTION_ROTATION_DEGREES}"
        )
    return degrees


@dataclass(frozen=True)
class ProjectionFrameTransform:
    """Rotate only the final projector frame, without changing camera geometry."""

    calibration_rotation_degrees: int = 0
    output_rotation_degrees: int = 180

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "calibration_rotation_degrees",
            normalize_projection_rotation_degrees(self.calibration_rotation_degrees),
        )
        object.__setattr__(
            self,
            "output_rotation_degrees",
            normalize_projection_rotation_degrees(self.output_rotation_degrees),
        )

    def apply(self, image: np.ndarray, *, calibration_mode: bool) -> np.ndarray:
        if image.ndim not in {2, 3}:
            raise ValueError("projector frame must be a 2-D or 3-D image")
        degrees = (
            self.calibration_rotation_degrees
            if calibration_mode
            else self.output_rotation_degrees
        )
        if degrees == 0:
            return image
        rotate_code = {
            90: cv2.ROTATE_90_CLOCKWISE,
            180: cv2.ROTATE_180,
            270: cv2.ROTATE_90_COUNTERCLOCKWISE,
        }[degrees]
        return cv2.rotate(image, rotate_code)
