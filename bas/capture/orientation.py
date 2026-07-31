from __future__ import annotations

from dataclasses import replace
from typing import Optional, Tuple

import cv2
import numpy as np

from .base import CaptureInfo, CaptureSource, VideoTimelineState


VALID_FRAME_ROTATION_DEGREES = (0, 90, 180, 270)


def normalize_frame_rotation_degrees(value: object) -> int:
    """Return a canonical clockwise, right-angle frame rotation."""
    try:
        degrees = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"frame rotation must be one of {VALID_FRAME_ROTATION_DEGREES}") from exc
    degrees %= 360
    if degrees not in VALID_FRAME_ROTATION_DEGREES:
        raise ValueError(f"frame rotation must be one of {VALID_FRAME_ROTATION_DEGREES}, got {value!r}")
    return degrees


def rotate_frame_to_canonical(image: np.ndarray, rotation_degrees: int) -> np.ndarray:
    """Rotate a captured image clockwise into the system's canonical camera view."""
    degrees = normalize_frame_rotation_degrees(rotation_degrees)
    if degrees == 0:
        return image
    rotate_code = {
        90: cv2.ROTATE_90_CLOCKWISE,
        180: cv2.ROTATE_180,
        270: cv2.ROTATE_90_COUNTERCLOCKWISE,
    }[degrees]
    return cv2.rotate(image, rotate_code)


class FrameOrientedCapture:
    """Capture-source decorator that normalizes physical camera mounting direction."""

    def __init__(self, source: CaptureSource, rotation_degrees: int):
        self._source = source
        self.rotation_degrees = normalize_frame_rotation_degrees(rotation_degrees)

    def is_opened(self) -> bool:
        return self._source.is_opened()

    def read(self) -> Tuple[bool, Optional[np.ndarray], dict[str, object]]:
        ok, frame, meta = self._source.read()
        if not ok or frame is None:
            return ok, frame, meta
        normalized = rotate_frame_to_canonical(frame, self.rotation_degrees)
        out_meta = dict(meta)
        out_meta["frame_rotation_degrees"] = self.rotation_degrees
        return True, normalized, out_meta

    def release(self) -> None:
        self._source.release()

    def info(self) -> CaptureInfo:
        info = self._source.info()
        width, height = int(info.width), int(info.height)
        if self.rotation_degrees in {90, 270}:
            width, height = height, width
        metadata = dict(info.metadata)
        metadata["frame_rotation_degrees"] = self.rotation_degrees
        return replace(info, width=width, height=height, metadata=metadata)

    def timeline_state(self) -> Optional[VideoTimelineState]:
        getter = getattr(self._source, "timeline_state", None)
        return getter() if callable(getter) else None

    def seek(self, frame_index: int) -> VideoTimelineState:
        seek = getattr(self._source, "seek", None)
        if not callable(seek):
            raise RuntimeError("Current capture source does not support video seeking.")
        return seek(frame_index)

