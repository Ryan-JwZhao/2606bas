from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


@dataclass
class CharucoBoardSpec:
    squares_x: int = 8
    squares_y: int = 5
    square_length_m: float = 0.04
    marker_length_m: float = 0.030
    dictionary_id: int = 0


def _aruco_module():
    aruco = getattr(cv2, "aruco", None)
    if aruco is None:
        raise RuntimeError("OpenCV ArUco/ChArUco support is unavailable. Install opencv-contrib-python or use the encoded grid fallback.")
    return aruco


def create_charuco_board(spec: CharucoBoardSpec):
    aruco = _aruco_module()
    dictionary = aruco.getPredefinedDictionary(spec.dictionary_id)
    try:
        return aruco.CharucoBoard(
            (spec.squares_x, spec.squares_y),
            spec.square_length_m,
            spec.marker_length_m,
            dictionary,
        )
    except AttributeError:
        return aruco.CharucoBoard_create(
            spec.squares_x,
            spec.squares_y,
            spec.square_length_m,
            spec.marker_length_m,
            dictionary,
        )


def render_charuco_board(spec: CharucoBoardSpec, width_px: int, height_px: int) -> np.ndarray:
    board = create_charuco_board(spec)
    try:
        img = board.generateImage((int(width_px), int(height_px)))
    except AttributeError:
        img = board.draw((int(width_px), int(height_px)))
    return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)


def detect_charuco_corners(frame_bgr: np.ndarray, spec: CharucoBoardSpec) -> Tuple[np.ndarray, np.ndarray]:
    board = create_charuco_board(spec)
    aruco = _aruco_module()
    dictionary = aruco.getPredefinedDictionary(spec.dictionary_id)
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    params = aruco.DetectorParameters()
    if hasattr(aruco, "CharucoDetector"):
        detector = aruco.CharucoDetector(board)
        charuco_corners, charuco_ids, _, _ = detector.detectBoard(gray)
        if charuco_corners is None or charuco_ids is None:
            return np.zeros((0, 2), dtype=np.float32), np.zeros((0,), dtype=np.int32)
        return charuco_corners.reshape((-1, 2)).astype(np.float32), charuco_ids.flatten().astype(np.int32)
    try:
        detector = aruco.ArucoDetector(dictionary, params)
        marker_corners, marker_ids, _ = detector.detectMarkers(gray)
    except AttributeError:
        marker_corners, marker_ids, _ = aruco.detectMarkers(gray, dictionary, parameters=params)
    if marker_ids is None or len(marker_ids) == 0:
        return np.zeros((0, 2), dtype=np.float32), np.zeros((0,), dtype=np.int32)
    if not hasattr(aruco, "interpolateCornersCharuco"):
        raise RuntimeError("OpenCV ChArUco corner interpolation is unavailable. Install opencv-contrib-python or use a newer OpenCV with CharucoDetector.")
    _, charuco_corners, charuco_ids = aruco.interpolateCornersCharuco(
        marker_corners,
        marker_ids,
        gray,
        board,
    )
    if charuco_corners is None or charuco_ids is None:
        return np.zeros((0, 2), dtype=np.float32), np.zeros((0,), dtype=np.int32)
    return charuco_corners.reshape((-1, 2)).astype(np.float32), charuco_ids.flatten().astype(np.int32)
