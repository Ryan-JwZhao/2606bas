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
    dictionary_id: int = cv2.aruco.DICT_4X4_50


def create_charuco_board(spec: CharucoBoardSpec):
    dictionary = cv2.aruco.getPredefinedDictionary(spec.dictionary_id)
    try:
        return cv2.aruco.CharucoBoard(
            (spec.squares_x, spec.squares_y),
            spec.square_length_m,
            spec.marker_length_m,
            dictionary,
        )
    except AttributeError:
        return cv2.aruco.CharucoBoard_create(
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
    dictionary = cv2.aruco.getPredefinedDictionary(spec.dictionary_id)
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    params = cv2.aruco.DetectorParameters()
    try:
        detector = cv2.aruco.ArucoDetector(dictionary, params)
        marker_corners, marker_ids, _ = detector.detectMarkers(gray)
    except AttributeError:
        marker_corners, marker_ids, _ = cv2.aruco.detectMarkers(gray, dictionary, parameters=params)
    if marker_ids is None or len(marker_ids) == 0:
        return np.zeros((0, 2), dtype=np.float32), np.zeros((0,), dtype=np.int32)
    _, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
        marker_corners,
        marker_ids,
        gray,
        board,
    )
    if charuco_corners is None or charuco_ids is None:
        return np.zeros((0, 2), dtype=np.float32), np.zeros((0,), dtype=np.int32)
    return charuco_corners.reshape((-1, 2)).astype(np.float32), charuco_ids.flatten().astype(np.int32)

