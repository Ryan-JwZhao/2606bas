from __future__ import annotations

import cv2

from bas.capture.uvc_controls import (
    OPENCV_EXPOSURE_AUTO,
    OPENCV_EXPOSURE_MANUAL,
    apply_uvc_exposure,
)


class _StubCapture:
    def __init__(self, *, accept_exposure: bool = True) -> None:
        self.accept_exposure = accept_exposure
        self.values = {
            cv2.CAP_PROP_AUTO_EXPOSURE: -1.0,
            cv2.CAP_PROP_EXPOSURE: 0.0,
        }
        self.calls: list[tuple[int, float]] = []

    def set(self, property_id: int, value: float) -> bool:
        self.calls.append((property_id, value))
        if property_id == cv2.CAP_PROP_EXPOSURE and not self.accept_exposure:
            return True
        self.values[property_id] = value
        return True

    def get(self, property_id: int) -> float:
        return self.values[property_id]


def test_apply_uvc_manual_exposure_switches_mode_then_sets_and_verifies_level() -> None:
    capture = _StubCapture()

    state = apply_uvc_exposure(capture, exposure_auto=False, exposure_level=-8)

    assert capture.calls == [
        (cv2.CAP_PROP_AUTO_EXPOSURE, OPENCV_EXPOSURE_MANUAL),
        (cv2.CAP_PROP_EXPOSURE, -8.0),
    ]
    assert state.level_readback == -8.0
    assert state.manual_level_verified is True


def test_apply_uvc_auto_exposure_does_not_write_manual_level() -> None:
    capture = _StubCapture()

    state = apply_uvc_exposure(capture, exposure_auto=True, exposure_level=-8)

    assert capture.calls == [(cv2.CAP_PROP_AUTO_EXPOSURE, OPENCV_EXPOSURE_AUTO)]
    assert state.manual_level_verified is None


def test_apply_uvc_exposure_detects_driver_that_accepts_but_ignores_write() -> None:
    capture = _StubCapture(accept_exposure=False)

    state = apply_uvc_exposure(capture, exposure_auto=False, exposure_level=-8)

    assert state.level_set is True
    assert state.level_readback == 0.0
    assert state.manual_level_verified is False
