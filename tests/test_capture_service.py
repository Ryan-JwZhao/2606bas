from __future__ import annotations

from bas.capture import service as capture_service
from bas.config import CameraConfig
from bas.capture.base import CaptureInfo


class _StubController:
    def __init__(self) -> None:
        self.calls: list[tuple[str, int, int | bool]] = []

    def set_auto_exposure(self, device_id: int, enable: bool) -> None:
        self.calls.append(("exposure_auto", device_id, enable))

    def set_manual_exposure_level(self, device_id: int, level: int) -> None:
        self.calls.append(("exposure_level", device_id, level))

    def set_auto_white_balance(self, device_id: int, enable: bool) -> None:
        self.calls.append(("white_balance_auto", device_id, enable))

    def set_manual_white_balance_value(self, device_id: int, value: int) -> None:
        self.calls.append(("white_balance_value", device_id, value))


class _StubNoriCapture:
    def __init__(self, controller: _StubController, device_id: int = 3) -> None:
        self._controller = controller
        self._device_id = device_id

    def is_opened(self) -> bool:
        return True

    def read(self):
        return False, None, {}

    def release(self) -> None:
        return None

    def info(self) -> CaptureInfo:
        return CaptureInfo(backend="nori", camera_id="stub", width=1920, height=1080, fps=30.0, metadata={})


def test_create_capture_service_applies_nori_white_balance_controls(monkeypatch) -> None:
    controller = _StubController()
    nori = _StubNoriCapture(controller)
    monkeypatch.setattr(capture_service, "open_nori_capture", lambda **kwargs: nori)

    config = CameraConfig(
        backend="nori",
        distortion_correction_enabled=False,
        exposure_auto=False,
        exposure_level=-4,
        white_balance_auto=False,
        white_balance_value=5200,
    )

    service = capture_service.create_capture_service(config)

    assert service.source is nori
    assert controller.calls == [
        ("exposure_auto", 3, False),
        ("exposure_level", 3, -4),
        ("white_balance_auto", 3, False),
        ("white_balance_value", 3, 5200),
    ]
