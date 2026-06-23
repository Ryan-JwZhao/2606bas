from __future__ import annotations

import logging
from typing import List, Optional, Tuple

from ..config import CameraConfig
from ..schemas import FramePacket
from ..utils import monotonic_ns
from .base import CaptureInfo, CaptureSource
from .nori_sdk import NoriProtocolController, open_nori_capture
from .opencv_capture import OpenCVCapture, SyntheticCapture, VideoFileCapture, probe_cameras as _probe_opencv

LOGGER = logging.getLogger(__name__)


class CaptureService:
    def __init__(self, source: CaptureSource, camera_id: str):
        self.source = source
        self.camera_id = camera_id
        self.frame_id = 0

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

    def release(self) -> None:
        self.source.release()


def create_capture_service(config: CameraConfig) -> CaptureService:
    backend = str(config.backend or "auto").lower()
    source: CaptureSource
    if backend == "synthetic":
        source = SyntheticCapture(config.width, config.height, config.fps, camera_id=config.camera_id)
        return CaptureService(source, camera_id=config.camera_id)
    if backend == "video":
        if not config.video_path:
            raise ValueError("camera.video_path is required when camera.backend=video")
        source = VideoFileCapture(config.video_path, camera_id=config.camera_id)
        return CaptureService(source, camera_id=config.camera_id)
    if backend in {"auto", "nori"}:
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
                    if config.exposure_level is not None:
                        controller.set_manual_exposure_level(nori._device_id, int(config.exposure_level))
                except Exception as exc:
                    LOGGER.warning("Failed to apply Nori exposure settings: %s", exc)
            return CaptureService(nori, camera_id=config.camera_id)
        if backend == "nori":
            raise RuntimeError("Nori camera requested but no MJPG SDK stream could be opened.")
        LOGGER.info("Nori SDK stream unavailable; falling back to OpenCV capture.")
    source = OpenCVCapture(config.device_index, config.width, config.height, config.fps, camera_id=config.camera_id)
    return CaptureService(source, camera_id=config.camera_id)


def probe_cameras(max_index: int = 12, nori_sdk_root: str | None = None) -> List[Tuple[str, int, int, int, float]]:
    rows: List[Tuple[str, int, int, int, float]] = []
    for idx, width, height, fps in _probe_opencv(max_index=max_index):
        rows.append(("opencv", idx, width, height, fps))
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

