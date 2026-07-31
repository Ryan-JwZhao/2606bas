from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import List, Optional, Tuple

import cv2
import numpy as np

from ..calibration.camera import CameraCalibration
from ..config import CameraConfig, normalize_exposure_control
from ..schemas import FramePacket
from ..utils import monotonic_ns
from .base import CaptureInfo, CaptureSource, VideoTimelineState
from .nori_sdk import NoriProtocolController, open_nori_capture
from .opencv_capture import OpenCVCapture, SyntheticCapture, VideoFileCapture, probe_cameras as _probe_opencv
from .orientation import FrameOrientedCapture, normalize_frame_rotation_degrees

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class _DistortionCorrectionState:
    frames_are_corrected: bool
    should_wrap_source: bool
    calibration: Optional[CameraCalibration] = None


class CaptureService:
    def __init__(self, source: CaptureSource, camera_id: str, frame_distortion_corrected: bool = False):
        self.source = source
        self.camera_id = camera_id
        self.frame_id = 0
        self.frame_distortion_corrected = bool(frame_distortion_corrected)

    def read(self) -> Optional[FramePacket]:
        ok, frame, meta = self.source.read()
        if not ok or frame is None:
            return None
        pkt = FramePacket(
            frame_id=self.frame_id,
            ts_cam_ns=monotonic_ns(),
            camera_id=self.camera_id,
            image=frame,
            exposure_meta=meta,
        )
        self.frame_id += 1
        return pkt

    def info(self) -> CaptureInfo:
        return self.source.info()

    def video_timeline_state(self) -> Optional[VideoTimelineState]:
        timeline_state = getattr(self.source, "timeline_state", None)
        if not callable(timeline_state):
            return None
        return timeline_state()

    def seek_video(self, frame_index: int) -> VideoTimelineState:
        seek = getattr(self.source, "seek", None)
        if not callable(seek):
            raise RuntimeError("Current capture source does not support video seeking.")
        state = seek(frame_index)
        self.frame_id = int(state.current_frame)
        return state

    def release(self) -> None:
        self.source.release()


class DistortionCorrectedCapture:
    def __init__(self, source: CaptureSource, calibration: CameraCalibration):
        self._source = source
        self._calibration = calibration
        self._map_size: Tuple[int, int] = (0, 0)
        self._map1: Optional[np.ndarray] = None
        self._map2: Optional[np.ndarray] = None

    def is_opened(self) -> bool:
        return self._source.is_opened()

    def read(self) -> Tuple[bool, Optional[np.ndarray], dict[str, object]]:
        ok, frame, meta = self._source.read()
        if not ok or frame is None:
            return ok, frame, meta
        corrected = self._undistort(frame)
        out_meta = dict(meta)
        out_meta["distortion_correction"] = True
        out_meta["distortion_file"] = self._calibration.source_path
        return True, corrected, out_meta

    def release(self) -> None:
        self._source.release()

    def info(self):
        info = self._source.info()
        meta = dict(info.metadata)
        meta["distortion_correction"] = True
        meta["distortion_file"] = self._calibration.source_path
        info.metadata = meta
        return info

    def _undistort(self, frame: np.ndarray) -> np.ndarray:
        h, w = frame.shape[:2]
        size = (int(w), int(h))
        if self._map1 is None or self._map2 is None or self._map_size != size:
            k_scaled = self._calibration.scaled_camera_matrix(w, h)
            if k_scaled is None or self._calibration.distortion_coefficients is None:
                return frame
            self._map1, self._map2 = cv2.initUndistortRectifyMap(
                k_scaled,
                np.asarray(self._calibration.distortion_coefficients, dtype=np.float64),
                None,
                k_scaled,
                size,
                cv2.CV_16SC2,
            )
            self._map_size = size
        return cv2.remap(
            frame,
            self._map1,
            self._map2,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
        )


def _distortion_correction_state(config: CameraConfig) -> _DistortionCorrectionState:
    if not bool(config.distortion_correction_enabled):
        return _DistortionCorrectionState(frames_are_corrected=False, should_wrap_source=False)
    calibration = CameraCalibration.load_opencv_yaml(config.distortion_correction_file)
    valid = bool(calibration.is_valid)
    return _DistortionCorrectionState(
        frames_are_corrected=valid,
        should_wrap_source=valid,
        calibration=calibration,
    )


def capture_frames_are_distortion_corrected(config: CameraConfig) -> bool:
    return _distortion_correction_state(config).frames_are_corrected


def create_capture_service(config: CameraConfig) -> CaptureService:
    backend = str(config.backend or "auto").lower()
    exposure_control = normalize_exposure_control(config.exposure_control)
    effective_backend = backend
    if backend in {"auto", "nori", "opencv"}:
        if exposure_control == "decxin":
            effective_backend = "nori"
        elif exposure_control == "uvc":
            effective_backend = "opencv"
    source: CaptureSource

    def finish(src: CaptureSource) -> CaptureService:
        rotation_degrees = normalize_frame_rotation_degrees(config.frame_rotation_degrees)
        if rotation_degrees:
            # Mount orientation is normalized before undistortion so that camera
            # intrinsics remain expressed in the canonical image coordinates.
            src = FrameOrientedCapture(src, rotation_degrees)
        state = _distortion_correction_state(config)
        if state.should_wrap_source and state.calibration is not None:
            src = DistortionCorrectedCapture(src, state.calibration)
        elif bool(config.distortion_correction_enabled):
            LOGGER.warning(
                "Camera distortion correction requested but calibration is invalid or missing: %s",
                config.distortion_correction_file,
            )
        return CaptureService(src, camera_id=config.camera_id, frame_distortion_corrected=state.frames_are_corrected)

    if backend == "synthetic":
        source = SyntheticCapture(config.width, config.height, config.fps, camera_id=config.camera_id)
        return finish(source)
    if backend == "video":
        if not config.video_path:
            raise ValueError("camera.video_path is required when camera.backend=video")
        source = VideoFileCapture(config.video_path, camera_id=config.camera_id)
        return finish(source)
    if effective_backend in {"auto", "nori"}:
        nori = open_nori_capture(
            device_index=config.device_index,
            width=config.width,
            height=config.height,
            fps=config.fps,
            sdk_root=config.nori_sdk_root,
            nori_device_id=config.nori_device_id,
            camera_id=config.camera_id,
        )
        if nori is not None and nori.is_opened():
            if config.exposure_auto is not None or config.exposure_level is not None:
                try:
                    controller = nori._controller  # SDK-only runtime control.
                    if config.exposure_auto is not None:
                        controller.set_auto_exposure(nori._device_id, bool(config.exposure_auto))
                    if config.exposure_level is not None and not bool(config.exposure_auto):
                        controller.set_manual_exposure_level(nori._device_id, int(config.exposure_level))
                except Exception as exc:
                    LOGGER.warning("Failed to apply Nori exposure settings: %s", exc)
            if config.white_balance_auto is not None or config.white_balance_value is not None:
                try:
                    controller = nori._controller  # SDK-only runtime control.
                    if config.white_balance_auto is not None:
                        controller.set_auto_white_balance(nori._device_id, bool(config.white_balance_auto))
                    if config.white_balance_value is not None and not bool(config.white_balance_auto):
                        controller.set_manual_white_balance_value(nori._device_id, int(config.white_balance_value))
                except Exception as exc:
                    LOGGER.warning("Failed to apply Nori white balance settings: %s", exc)
            return finish(nori)
        if exposure_control == "decxin":
            raise RuntimeError(
                "Decxin exposure control was selected, but no Decxin/Nori SDK camera could be opened."
            )
        if effective_backend == "nori":
            raise RuntimeError("Nori camera requested but no MJPG SDK stream could be opened.")
        LOGGER.info("Nori SDK stream unavailable; falling back to OpenCV capture.")
    source = OpenCVCapture(
        config.device_index,
        config.width,
        config.height,
        config.fps,
        camera_id=config.camera_id,
        exposure_auto=config.exposure_auto if exposure_control in {"auto", "uvc"} else None,
        exposure_level=config.exposure_level if exposure_control in {"auto", "uvc"} else None,
    )
    return finish(source)


def probe_cameras(
    max_index: int = 12,
    nori_sdk_root: str | None = None,
    width: int = 1920,
    height: int = 1080,
    fps: int = 30,
) -> List[Tuple[str, int, int, int, float]]:
    rows: List[Tuple[str, int, int, int, float]] = []
    for idx, actual_width, actual_height, actual_fps in _probe_opencv(
        max_index=max_index,
        width=width,
        height=height,
        fps=fps,
    ):
        rows.append(("opencv_mjpg", idx, actual_width, actual_height, actual_fps))
    try:
        controller = NoriProtocolController(sdk_root=nori_sdk_root)
        for dev in controller.list_devices(max_scan=max_index):
            infos = controller.list_video_infos(dev.device_id)
            if infos:
                best = sorted(infos, key=lambda x: (x.width * x.height, x.fps), reverse=True)[0]
                rows.append(("nori_mjpg", dev.device_id, best.width, best.height, best.fps))
        controller.shutdown()
    except Exception:
        pass
    return rows
