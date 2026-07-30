from __future__ import annotations

from bas.capture import service as capture_service
from bas.capture.base import CaptureInfo, VideoTimelineState
from bas.config import CameraConfig


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


class _StubVideoCapture:
    def is_opened(self) -> bool:
        return True

    def read(self):
        return False, None, {}

    def release(self) -> None:
        return None

    def info(self) -> CaptureInfo:
        return CaptureInfo(backend="video", camera_id="video_stub", width=1920, height=1080, fps=30.0, metadata={})


class _SeekableStubVideoCapture(_StubVideoCapture):
    def __init__(self) -> None:
        self.current_frame = 0

    def timeline_state(self) -> VideoTimelineState:
        return VideoTimelineState(current_frame=self.current_frame, total_frames=3000, fps=25.0)

    def seek(self, frame_index: int) -> VideoTimelineState:
        self.current_frame = int(frame_index)
        return self.timeline_state()


class _StubOpenCVCapture(_StubVideoCapture):
    def __init__(self, *args, **kwargs) -> None:
        self.args = args
        self.kwargs = kwargs


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


def test_create_capture_service_applies_exposure_to_opencv_fallback(monkeypatch) -> None:
    source = _StubOpenCVCapture()

    def open_stub(*args, **kwargs):
        source.args = args
        source.kwargs = kwargs
        return source

    monkeypatch.setattr(capture_service, "OpenCVCapture", open_stub)
    config = CameraConfig(
        backend="opencv",
        device_index=2,
        exposure_auto=False,
        exposure_level=-8,
        distortion_correction_enabled=False,
    )

    service = capture_service.create_capture_service(config)

    assert service.source is source
    assert source.kwargs["exposure_auto"] is False
    assert source.kwargs["exposure_level"] == -8


def test_uvc_exposure_control_forces_opencv_when_capture_backend_is_auto(monkeypatch) -> None:
    source = _StubOpenCVCapture()
    nori_attempted = False

    def unexpected_nori(**_kwargs):
        nonlocal nori_attempted
        nori_attempted = True
        return _StubNoriCapture(_StubController())

    monkeypatch.setattr(capture_service, "open_nori_capture", unexpected_nori)
    monkeypatch.setattr(capture_service, "OpenCVCapture", lambda *args, **kwargs: source)
    config = CameraConfig(
        backend="auto",
        exposure_control="uvc",
        distortion_correction_enabled=False,
    )

    service = capture_service.create_capture_service(config)

    assert service.source is source
    assert nori_attempted is False


def test_manual_exposure_control_overrides_conflicting_live_capture_backend(monkeypatch) -> None:
    uvc_source = _StubOpenCVCapture()
    nori_source = _StubNoriCapture(_StubController())
    nori_attempts = 0

    def open_nori(**_kwargs):
        nonlocal nori_attempts
        nori_attempts += 1
        return nori_source

    monkeypatch.setattr(capture_service, "open_nori_capture", open_nori)
    monkeypatch.setattr(capture_service, "OpenCVCapture", lambda *args, **kwargs: uvc_source)

    uvc_service = capture_service.create_capture_service(
        CameraConfig(backend="nori", exposure_control="uvc", distortion_correction_enabled=False)
    )
    decxin_service = capture_service.create_capture_service(
        CameraConfig(backend="opencv", exposure_control="decxin", distortion_correction_enabled=False)
    )

    assert uvc_service.source is uvc_source
    assert decxin_service.source is nori_source
    assert nori_attempts == 1


def test_decxin_exposure_control_does_not_silently_fall_back_to_uvc(monkeypatch) -> None:
    opencv_attempted = False

    def unexpected_opencv(*_args, **_kwargs):
        nonlocal opencv_attempted
        opencv_attempted = True
        return _StubOpenCVCapture()

    monkeypatch.setattr(capture_service, "open_nori_capture", lambda **kwargs: None)
    monkeypatch.setattr(capture_service, "OpenCVCapture", unexpected_opencv)
    config = CameraConfig(
        backend="auto",
        exposure_control="decxin",
        distortion_correction_enabled=False,
    )

    try:
        capture_service.create_capture_service(config)
    except RuntimeError as exc:
        assert "Decxin" in str(exc)
    else:
        raise AssertionError("manual Decxin mode must fail instead of changing control systems")
    assert opencv_attempted is False


def test_capture_frames_are_distortion_corrected_treats_video_backend_as_precorrected(monkeypatch) -> None:
    def _unexpected_load(_path):
        raise AssertionError("video backend should not load OpenCV distortion calibration")

    monkeypatch.setattr(capture_service.CameraCalibration, "load_opencv_yaml", _unexpected_load)

    config = CameraConfig(
        backend="video",
        video_path="already_corrected.mp4",
        distortion_correction_enabled=True,
        distortion_correction_file="missing.yaml",
    )

    assert capture_service.capture_frames_are_distortion_corrected(config) is True


def test_create_capture_service_skips_runtime_undistortion_for_video_backend(monkeypatch) -> None:
    source = _StubVideoCapture()

    def _unexpected_load(_path):
        raise AssertionError("video backend should not load OpenCV distortion calibration")

    monkeypatch.setattr(capture_service, "VideoFileCapture", lambda *args, **kwargs: source)
    monkeypatch.setattr(capture_service.CameraCalibration, "load_opencv_yaml", _unexpected_load)

    config = CameraConfig(
        backend="video",
        video_path="already_corrected.mp4",
        camera_id="video_source",
        distortion_correction_enabled=True,
        distortion_correction_file="missing.yaml",
    )

    service = capture_service.create_capture_service(config)

    assert service.source is source
    assert service.frame_distortion_corrected is True
    assert service.info().backend == "video"


def test_capture_frames_are_distortion_corrected_requires_valid_calibration_for_live_capture() -> None:
    config = CameraConfig(
        backend="opencv",
        distortion_correction_enabled=True,
        distortion_correction_file="missing.yaml",
    )

    assert capture_service.capture_frames_are_distortion_corrected(config) is False


def test_capture_service_exposes_and_seeks_video_timeline() -> None:
    source = _SeekableStubVideoCapture()
    service = capture_service.CaptureService(source, camera_id="video_source")

    initial = service.video_timeline_state()
    sought = service.seek_video(1250)

    assert initial == VideoTimelineState(current_frame=0, total_frames=3000, fps=25.0)
    assert sought.current_frame == 1250
    assert sought.current_seconds == 50.0
    assert sought.duration_seconds == 120.0
    assert service.frame_id == 1250


def test_capture_service_rejects_seek_for_non_video_source() -> None:
    service = capture_service.CaptureService(_StubNoriCapture(_StubController()), camera_id="live")

    assert service.video_timeline_state() is None

    try:
        service.seek_video(10)
    except RuntimeError as exc:
        assert "does not support" in str(exc)
    else:
        raise AssertionError("live capture must not support timeline seeking")
