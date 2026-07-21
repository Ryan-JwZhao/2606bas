from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from .base import CaptureInfo, VideoTimelineState

LOGGER = logging.getLogger(__name__)


def camera_api_candidates() -> List[int]:
    if sys.platform == "win32":
        return [cv2.CAP_DSHOW, getattr(cv2, "CAP_MSMF", cv2.CAP_ANY), cv2.CAP_ANY]
    if sys.platform.startswith("linux"):
        return [cv2.CAP_V4L2, cv2.CAP_ANY]
    if sys.platform == "darwin":
        return [cv2.CAP_AVFOUNDATION, cv2.CAP_ANY]
    return [cv2.CAP_ANY]


def _open_device(index: int) -> Optional[cv2.VideoCapture]:
    for api in camera_api_candidates():
        cap = cv2.VideoCapture(index, api) if api != cv2.CAP_ANY else cv2.VideoCapture(index)
        if cap.isOpened():
            return cap
        cap.release()
    return None


def probe_cameras(max_index: int = 12) -> List[Tuple[int, int, int, float]]:
    found: List[Tuple[int, int, int, float]] = []
    for idx in range(max_index):
        cap = _open_device(idx)
        if cap is None:
            continue
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        if width <= 0 or height <= 0:
            ok, frame = cap.read()
            if ok and frame is not None:
                height, width = frame.shape[:2]
        found.append((idx, width, height, fps))
        cap.release()
    return found


class OpenCVCapture:
    def __init__(self, device_index: int, width: int, height: int, fps: int, camera_id: str = "opencv"):
        cap = _open_device(int(device_index))
        if cap is None or not cap.isOpened():
            raise RuntimeError(f"Cannot open OpenCV camera index {device_index}")
        self._cap = cap
        self._camera_id = camera_id
        self._device_index = int(device_index)
        self._requested = (int(width), int(height), int(fps))
        self._cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(width))
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(height))
        self._cap.set(cv2.CAP_PROP_FPS, int(fps))
        try:
            self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass

    def is_opened(self) -> bool:
        return bool(self._cap is not None and self._cap.isOpened())

    def read(self) -> Tuple[bool, Optional[np.ndarray], Dict[str, object]]:
        ok, frame = self._cap.read()
        meta = {
            "backend": "opencv",
            "device_index": self._device_index,
            "fourcc": int(self._cap.get(cv2.CAP_PROP_FOURCC) or 0),
        }
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
            metadata={"device_index": self._device_index},
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
