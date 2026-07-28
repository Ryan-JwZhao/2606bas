from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Optional, Protocol

import cv2

LOGGER = logging.getLogger(__name__)

# OpenCV's DirectShow backend maps these values to
# CameraControl_Flags_Manual and CameraControl_Flags_Auto.
OPENCV_EXPOSURE_MANUAL = 0.25
OPENCV_EXPOSURE_AUTO = 0.75


class OpenCVCameraControl(Protocol):
    def get(self, property_id: int) -> float: ...

    def set(self, property_id: int, value: float) -> bool: ...


@dataclass(frozen=True)
class UvcExposureState:
    requested_auto: Optional[bool]
    requested_level: Optional[int]
    auto_set: Optional[bool]
    level_set: Optional[bool]
    auto_readback: Optional[float]
    level_readback: Optional[float]

    @property
    def manual_level_verified(self) -> Optional[bool]:
        if self.requested_level is None or bool(self.requested_auto):
            return None
        if self.level_readback is None:
            return False
        return abs(float(self.level_readback) - float(self.requested_level)) <= 0.5

    def as_metadata(self) -> dict[str, object]:
        return {
            "exposure_auto_requested": self.requested_auto,
            "exposure_level_requested": self.requested_level,
            "exposure_auto_set": self.auto_set,
            "exposure_level_set": self.level_set,
            "exposure_auto_readback": self.auto_readback,
            "exposure_level_readback": self.level_readback,
            "exposure_manual_verified": self.manual_level_verified,
        }


def apply_uvc_exposure(
    capture: OpenCVCameraControl,
    *,
    exposure_auto: Optional[bool],
    exposure_level: Optional[int],
) -> UvcExposureState:
    """Apply standard UVC exposure controls through an OpenCV capture backend."""

    auto_set: Optional[bool] = None
    level_set: Optional[bool] = None

    if exposure_auto is not None:
        auto_value = OPENCV_EXPOSURE_AUTO if exposure_auto else OPENCV_EXPOSURE_MANUAL
        auto_set = bool(capture.set(cv2.CAP_PROP_AUTO_EXPOSURE, auto_value))

    if exposure_level is not None and not bool(exposure_auto):
        level_set = bool(capture.set(cv2.CAP_PROP_EXPOSURE, float(exposure_level)))

    auto_readback = _read_optional(capture, cv2.CAP_PROP_AUTO_EXPOSURE)
    level_readback = _read_optional(capture, cv2.CAP_PROP_EXPOSURE)
    state = UvcExposureState(
        requested_auto=exposure_auto,
        requested_level=exposure_level,
        auto_set=auto_set,
        level_set=level_set,
        auto_readback=auto_readback,
        level_readback=level_readback,
    )

    if auto_set is False:
        LOGGER.warning("UVC camera rejected the automatic-exposure mode request.")
    if level_set is False:
        LOGGER.warning("UVC camera rejected manual exposure level %s.", exposure_level)
    elif state.manual_level_verified is False:
        LOGGER.warning(
            "UVC exposure write was not confirmed: requested=%s, readback=%s.",
            exposure_level,
            level_readback,
        )
    return state


def _read_optional(capture: OpenCVCameraControl, property_id: int) -> Optional[float]:
    try:
        return float(capture.get(property_id))
    except Exception:
        return None
