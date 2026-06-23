from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import cv2
import numpy as np


@dataclass
class CameraCalibration:
    image_size: Tuple[int, int] = (0, 0)
    camera_matrix: Optional[np.ndarray] = None
    distortion_coefficients: Optional[np.ndarray] = None
    rotation_matrix: Optional[np.ndarray] = None
    rvec: Optional[np.ndarray] = None
    tvec: Optional[np.ndarray] = None
    source_path: Optional[str] = None
    metadata: Dict[str, object] = None  # type: ignore[assignment]

    @classmethod
    def load_opencv_yaml(cls, path: str | Path | None) -> "CameraCalibration":
        if path is None:
            return cls(metadata={})
        p = Path(path)
        if not p.exists():
            return cls(source_path=str(p), metadata={"missing": True})
        fs = cv2.FileStorage(str(p), cv2.FILE_STORAGE_READ)
        if not fs.isOpened():
            raise RuntimeError(f"Cannot open OpenCV calibration file: {p}")
        try:
            width = int(fs.getNode("image_width").real() or 0)
            height = int(fs.getNode("image_height").real() or 0)
            K = _read_matrix(fs, "camera_matrix")
            D = _read_matrix(fs, "distortion_coefficients")
            R = _read_matrix(fs, "rotation_matrix")
            rvec = _read_matrix(fs, "rvec")
            tvec = _read_matrix(fs, "tvec")
            return cls(
                image_size=(width, height),
                camera_matrix=K,
                distortion_coefficients=D,
                rotation_matrix=R,
                rvec=rvec,
                tvec=tvec,
                source_path=str(p),
                metadata={
                    "halcon_to_opencv_fit_rms_px": _read_real(fs, "halcon_to_opencv_fit_rms_px"),
                    "halcon_to_opencv_fit_max_px": _read_real(fs, "halcon_to_opencv_fit_max_px"),
                },
            )
        finally:
            fs.release()

    @property
    def is_valid(self) -> bool:
        return self.camera_matrix is not None and self.distortion_coefficients is not None

    @property
    def version(self) -> str:
        if self.source_path:
            return Path(self.source_path).stem
        return "no_camera_calibration"

    def undistort(self, image: np.ndarray) -> np.ndarray:
        if not self.is_valid:
            return image
        return cv2.undistort(image, self.camera_matrix, self.distortion_coefficients)

    def undistort_points(self, points: np.ndarray) -> np.ndarray:
        pts = np.asarray(points, dtype=np.float32).reshape((-1, 1, 2))
        if not self.is_valid or pts.size == 0:
            return pts.reshape((-1, 2)).astype(np.float32)
        out = cv2.undistortPoints(pts, self.camera_matrix, self.distortion_coefficients, P=self.camera_matrix)
        return out.reshape((-1, 2)).astype(np.float32)


def _read_matrix(fs: cv2.FileStorage, name: str) -> Optional[np.ndarray]:
    node = fs.getNode(name)
    if node.empty():
        return None
    mat = node.mat()
    if mat is None or mat.size == 0:
        return None
    return np.asarray(mat, dtype=np.float64)


def _read_real(fs: cv2.FileStorage, name: str) -> Optional[float]:
    node = fs.getNode(name)
    if node.empty():
        return None
    try:
        return float(node.real())
    except Exception:
        return None

