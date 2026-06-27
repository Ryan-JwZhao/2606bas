from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

import numpy as np

from .calibration.camera import CameraCalibration


class RecordingFrameCorrector:
    def __init__(self, calibration_paths: Iterable[str | None] = ()) -> None:
        self._paths: tuple[str, ...] = ()
        self._loaded = False
        self._calibration: Optional[CameraCalibration] = None
        self.update_calibration_paths(calibration_paths)

    def update_calibration_paths(self, calibration_paths: Iterable[str | None]) -> None:
        normalized: list[str] = []
        for raw in calibration_paths:
            value = str(raw or "").strip()
            if value and value not in normalized:
                normalized.append(value)
        paths = tuple(normalized)
        if paths == self._paths:
            return
        self._paths = paths
        self.invalidate()

    def invalidate(self) -> None:
        self._loaded = False
        self._calibration = None

    def corrected_frame(self, frame_bgr: Optional[np.ndarray], *, already_corrected: bool) -> Optional[np.ndarray]:
        if frame_bgr is None:
            return None
        if already_corrected:
            return frame_bgr
        calibration = self._ensure_calibration_loaded()
        if calibration is None or not calibration.is_valid:
            return None
        return calibration.undistort(frame_bgr)

    def has_usable_calibration(self) -> bool:
        calibration = self._ensure_calibration_loaded()
        return calibration is not None and calibration.is_valid

    def _ensure_calibration_loaded(self) -> Optional[CameraCalibration]:
        if self._loaded:
            return self._calibration
        self._loaded = True
        self._calibration = None
        for raw_path in self._paths:
            path = Path(raw_path)
            if not path.exists():
                continue
            calibration = CameraCalibration.load_opencv_yaml(path)
            if calibration.is_valid:
                self._calibration = calibration
                break
        return self._calibration
