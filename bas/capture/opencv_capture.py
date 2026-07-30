from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from ..utils import fourcc_to_str
from .base import CaptureInfo, VideoTimelineState
from .uvc_controls import UvcExposureState, apply_uvc_exposure

LOGGER = logging.getLogger(__name__)
MJPG_FOURCC = cv2.VideoWriter_fourcc(*"MJPG")


def camera_api_candidates() -> List[int]:
    if sys.platform == "win32":
        return [cv2.CAP_DSHOW, getattr(cv2, "CAP_MSMF", cv2.CAP_ANY), cv2.CAP_ANY]
    if sys.platform.startswith("linux"):
        return [cv2.CAP_V4L2, cv2.CAP_ANY]
    if sys.platform == "darwin":
        return [cv2.CAP_AVFOUNDATION, cv2.CAP_ANY]
    return [cv2.CAP_ANY]


def mjpg_open_params(width: int, height: int, fps: int) -> List[int]:
    """Build one atomic OpenCV camera negotiation request.

    Some Windows UVC drivers lock the media subtype when the device is opened.
    Setting FOURCC afterwards can return success while the stream remains YUY2.
    """
    return [
        cv2.CAP_PROP_FOURCC,
        MJPG_FOURCC,
        cv2.CAP_PROP_FRAME_WIDTH,
        int(width),
        cv2.CAP_PROP_FRAME_HEIGHT,
        int(height),
        cv2.CAP_PROP_FPS,
        int(fps),
    ]


def _is_mjpg_capture(cap: cv2.VideoCapture) -> bool:
    return fourcc_to_str(cap.get(cv2.CAP_PROP_FOURCC)).upper() == "MJPG"


def _open_device(index: int, width: int = 1920, height: int = 1080, fps: int = 30) -> Optional[cv2.VideoCapture]:
    params = mjpg_open_params(width, height, fps)
    for api in camera_api_candidates():
        try:
            cap = cv2.VideoCapture(int(index), api, params)
        except Exception as exc:
            LOGGER.warning(
                "Camera index %s via API %s could not negotiate MJPG: %s",
                index,
                api,
                exc,
            )
            continue
        if cap.isOpened() and _is_mjpg_capture(cap):
            return cap
        if cap.isOpened():
            LOGGER.warning(
                "Rejected camera index %s via API %s: negotiated FOURCC=%s instead of MJPG.",
                index,
                api,
                fourcc_to_str(cap.get(cv2.CAP_PROP_FOURCC)),
            )
        cap.release()
    return None


def probe_cameras(
    max_index: int = 12,
    width: int = 1920,
    height: int = 1080,
    fps: int = 30,
) -> List[Tuple[int, int, int, float]]:
    found: List[Tuple[int, int, int, float]] = []
    for idx in range(max_index):
        cap = _open_device(idx, width=width, height=height, fps=fps)
        if cap is None:
            continue
        actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or width)
        actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or height)
        actual_fps = float(cap.get(cv2.CAP_PROP_FPS) or fps)
        if actual_width <= 0 or actual_height <= 0:
            ok, frame = cap.read()
            if ok and frame is not None:
                actual_height, actual_width = frame.shape[:2]
        found.append((idx, actual_width, actual_height, actual_fps))
        cap.release()
    return found


class OpenCVCapture:
    def __init__(
        self,
        device_index: int,
        width: int,
        height: int,
        fps: int,
        camera_id: str = "opencv",
        exposure_auto: Optional[bool] = None,
        exposure_level: Optional[int] = None,
    ):
        cap = _open_device(int(device_index), width=int(width), height=int(height), fps=int(fps))
        if cap is None or not cap.isOpened():
            raise RuntimeError(
                f"Cannot open OpenCV camera index {device_index} as MJPG "
                f"at {int(width)}x{int(height)}@{int(fps)}"
            )
        self._cap = cap
        self._camera_id = camera_id
        self._device_index = int(device_index)
        self._requested = (int(width), int(height), int(fps))
        try:
            self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass
        self._exposure_state: UvcExposureState = apply_uvc_exposure(
            self._cap,
            exposure_auto=exposure_auto,
            exposure_level=exposure_level,
        )
        if not _is_mjpg_capture(self._cap):
            negotiated = fourcc_to_str(self._cap.get(cv2.CAP_PROP_FOURCC))
            self._cap.release()
            raise RuntimeError(
                f"OpenCV camera index {device_index} changed to non-MJPG FOURCC={negotiated}"
            )

    def is_opened(self) -> bool:
        return bool(self._cap is not None and self._cap.isOpened())

    def read(self) -> Tuple[bool, Optional[np.ndarray], Dict[str, object]]:
        ok, frame = self._cap.read()
        meta = {
            "backend": "opencv",
            "device_index": self._device_index,
            "fourcc": int(self._cap.get(cv2.CAP_PROP_FOURCC) or 0),
            "media_type": "MJPG",
        }
        meta.update(self._exposure_state.as_metadata())
        return bool(ok), frame if ok else None, meta

    def release(self) -> None:
        if self._cap is not None:
            self._cap.release()

    def info(self) -> CaptureInfo:
        return CaptureInfo(
            backend="opencv",
            camera_id=self._camera_id,
            width=int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH) or self._requested[0]),
            height=int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or self._requested[1]),
            fps=float(self._cap.get(cv2.CAP_PROP_FPS) or self._requested[2]),
            metadata={
                "device_index": self._device_index,
                "media_type": "MJPG",
                "fourcc": int(self._cap.get(cv2.CAP_PROP_FOURCC) or 0),
                **self._exposure_state.as_metadata(),
            },
        )


class VideoFileCapture:
    def __init__(self, video_path: str | Path, camera_id: str = "video_file"):
        self._path = Path(video_path)
        self._cap = cv2.VideoCapture(str(self._path))
        if not self._cap.isOpened():
            raise RuntimeError(f"Cannot open video file: {self._path}")
        self._camera_id = camera_id

    def is_opened(self) -> bool:
        return bool(self._cap.isOpened())

    def read(self) -> Tuple[bool, Optional[np.ndarray], Dict[str, object]]:
        ok, frame = self._cap.read()
        meta = {"backend": "video", "path": str(self._path)}
        return bool(ok), frame if ok else None, meta

    def timeline_state(self) -> VideoTimelineState:
        total_frames = max(0, int(round(self._cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0)))
        current_frame = max(0, int(round(self._cap.get(cv2.CAP_PROP_POS_FRAMES) or 0.0)))
        if total_frames > 0:
            current_frame = min(current_frame, total_frames - 1)
        fps = max(0.0, float(self._cap.get(cv2.CAP_PROP_FPS) or 0.0))
        return VideoTimelineState(current_frame=current_frame, total_frames=total_frames, fps=fps)

    def seek(self, frame_index: int) -> VideoTimelineState:
        state = self.timeline_state()
        maximum = max(0, state.total_frames - 1)
        target = min(max(0, int(frame_index)), maximum)
        if not self._cap.set(cv2.CAP_PROP_POS_FRAMES, float(target)):
            raise RuntimeError(f"Cannot seek video file to frame {target}: {self._path}")
        return self.timeline_state()

    def release(self) -> None:
        self._cap.release()

    def info(self) -> CaptureInfo:
        timeline = self.timeline_state()
        return CaptureInfo(
            backend="video",
            camera_id=self._camera_id,
            width=int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0),
            height=int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0),
            fps=timeline.fps,
            metadata={
                "path": str(self._path),
                "total_frames": timeline.total_frames,
                "duration_seconds": timeline.duration_seconds,
            },
        )


class SyntheticCapture:
    def __init__(self, width: int, height: int, fps: int, camera_id: str = "synthetic"):
        self._width = int(width)
        self._height = int(height)
        self._fps = float(fps)
        self._camera_id = camera_id
        self._frame_id = 0
        self._opened = True

    def is_opened(self) -> bool:
        return self._opened

    def read(self) -> Tuple[bool, Optional[np.ndarray], Dict[str, object]]:
        img = np.zeros((self._height, self._width, 3), dtype=np.uint8)
        img[:] = (24, 64, 32)
        t = self._frame_id
        balls = [
            (0.25 + 0.10 * np.sin(t * 0.05), 0.42, (245, 245, 245)),
            (0.48, 0.45 + 0.08 * np.cos(t * 0.04), (30, 80, 240)),
            (0.68, 0.56, (240, 230, 40)),
            (0.76, 0.38, (20, 20, 20)),
        ]
        for x, y, color in balls:
            cx = int(x * self._width)
            cy = int(y * self._height)
            r = max(8, int(min(self._width, self._height) * 0.022))
            cv2.circle(img, (cx, cy), r, color, -1, cv2.LINE_AA)
            cv2.circle(img, (cx, cy), r, (12, 36, 24), 1, cv2.LINE_AA)
        self._frame_id += 1
        return True, img, {"backend": "synthetic"}

    def release(self) -> None:
        self._opened = False

    def info(self) -> CaptureInfo:
        return CaptureInfo(
            backend="synthetic",
            camera_id=self._camera_id,
            width=self._width,
            height=self._height,
            fps=self._fps,
            metadata={},
        )
