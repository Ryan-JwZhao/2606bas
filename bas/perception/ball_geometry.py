from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import cv2
import numpy as np

from ..schemas import BBox, Point


@dataclass(frozen=True)
class BallGeometryEstimate:
    center_px: Point
    radius_px: float
    quality: float
    method: str


class BallCenterRefiner:
    """Refine a detector box into a sub-pixel ball center and radius."""

    def refine(
        self,
        frame_bgr: np.ndarray,
        bbox: BBox,
        *,
        mask_polygon: Optional[Sequence[Sequence[float]] | np.ndarray] = None,
    ) -> BallGeometryEstimate:
        fallback = _bbox_estimate(bbox)
        if frame_bgr is None or frame_bgr.size == 0:
            return fallback

        polygon = _normalize_polygon(mask_polygon)
        if polygon is not None:
            estimate = _estimate_from_contour(polygon.reshape((-1, 1, 2)), bbox, method="segmentation_ellipse")
            if estimate is not None and estimate.quality >= 0.35:
                return estimate

        contour = _foreground_contour(frame_bgr, bbox)
        if contour is not None:
            estimate = _estimate_from_contour(contour, bbox, method="appearance_ellipse")
            if estimate is not None and estimate.quality >= 0.28:
                return estimate
        return fallback


def _bbox_estimate(bbox: BBox) -> BallGeometryEstimate:
    x1, y1, x2, y2 = [float(value) for value in bbox]
    width = max(1.0, x2 - x1)
    height = max(1.0, y2 - y1)
    return BallGeometryEstimate(
        center_px=(0.5 * (x1 + x2), 0.5 * (y1 + y2)),
        radius_px=max(2.0, 0.25 * (width + height)),
        quality=0.45,
        method="bbox",
    )


def _normalize_polygon(
    polygon: Optional[Sequence[Sequence[float]] | np.ndarray],
) -> Optional[np.ndarray]:
    if polygon is None:
        return None
    points = np.asarray(polygon, dtype=np.float32).reshape((-1, 2))
    if points.shape[0] < 5 or not np.all(np.isfinite(points)):
        return None
    return points


def _estimate_from_contour(
    contour: np.ndarray,
    bbox: BBox,
    *,
    method: str,
) -> Optional[BallGeometryEstimate]:
    points = np.asarray(contour, dtype=np.float32).reshape((-1, 2))
    if points.shape[0] < 5:
        return None
    contour_cv = points.reshape((-1, 1, 2))
    area = float(abs(cv2.contourArea(contour_cv)))
    perimeter = float(cv2.arcLength(contour_cv, True))
    if area < 12.0 or perimeter < 8.0:
        return None

    try:
        (cx, cy), (diameter_a, diameter_b), _ = cv2.fitEllipse(contour_cv)
    except cv2.error:
        return None
    if diameter_a <= 1.0 or diameter_b <= 1.0:
        return None

    radius = 0.25 * float(diameter_a + diameter_b)
    axis_ratio = float(min(diameter_a, diameter_b) / max(diameter_a, diameter_b))
    circularity = float(np.clip(4.0 * np.pi * area / max(1e-6, perimeter * perimeter), 0.0, 1.0))
    expected_area = np.pi * max(1.0, 0.25 * diameter_a * diameter_b)
    fill = float(np.clip(area / max(1e-6, expected_area), 0.0, 1.0))

    x1, y1, x2, y2 = [float(value) for value in bbox]
    width = max(1.0, x2 - x1)
    height = max(1.0, y2 - y1)
    box_center = np.asarray([0.5 * (x1 + x2), 0.5 * (y1 + y2)], dtype=np.float32)
    center_shift = float(np.linalg.norm(np.asarray([cx, cy], dtype=np.float32) - box_center))
    shift_score = float(np.clip(1.0 - center_shift / max(3.0, 0.45 * max(width, height)), 0.0, 1.0))
    radius_reference = max(2.0, 0.25 * (width + height))
    if method == "appearance_ellipse":
        radius_ratio = radius / radius_reference
        if radius_ratio < 0.65 or radius_ratio > 1.20:
            return None
        if center_shift > max(3.0, 0.18 * radius_reference):
            return None
    size_score = float(np.clip(1.0 - abs(radius - radius_reference) / max(2.0, radius_reference), 0.0, 1.0))
    quality = float(
        np.clip(
            0.30 * circularity
            + 0.24 * axis_ratio
            + 0.18 * fill
            + 0.16 * shift_score
            + 0.12 * size_score,
            0.0,
            1.0,
        )
    )
    return BallGeometryEstimate(
        center_px=(float(cx), float(cy)),
        radius_px=max(2.0, float(radius)),
        quality=quality,
        method=method,
    )


def _foreground_contour(frame_bgr: np.ndarray, bbox: BBox) -> Optional[np.ndarray]:
    height, width = frame_bgr.shape[:2]
    x1, y1, x2, y2 = [float(value) for value in bbox]
    box_w = max(4.0, x2 - x1)
    box_h = max(4.0, y2 - y1)
    pad_x = max(3, int(round(box_w * 0.18)))
    pad_y = max(3, int(round(box_h * 0.18)))
    left = max(0, int(np.floor(x1)) - pad_x)
    top = max(0, int(np.floor(y1)) - pad_y)
    right = min(width, int(np.ceil(x2)) + pad_x)
    bottom = min(height, int(np.ceil(y2)) + pad_y)
    if right - left < 8 or bottom - top < 8:
        return None
    roi = frame_bgr[top:bottom, left:right]
    lab = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB).astype(np.float32)
    border_width = max(1, int(round(min(roi.shape[:2]) * 0.12)))
    border = np.zeros(roi.shape[:2], dtype=np.uint8)
    border[:border_width, :] = 1
    border[-border_width:, :] = 1
    border[:, :border_width] = 1
    border[:, -border_width:] = 1
    background_pixels = lab[border.astype(bool)]
    if background_pixels.size == 0:
        return None
    background = np.median(background_pixels, axis=0)
    color_distance = np.linalg.norm(lab - background.reshape((1, 1, 3)), axis=2)
    scaled = np.clip(color_distance * 4.0, 0.0, 255.0).astype(np.uint8)
    _, foreground = cv2.threshold(scaled, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    kernel_size = 3 if min(roi.shape[:2]) < 60 else 5
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    foreground = cv2.morphologyEx(foreground, cv2.MORPH_OPEN, kernel)
    foreground = cv2.morphologyEx(foreground, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(foreground, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None

    expected_center = np.asarray([0.5 * (x1 + x2) - left, 0.5 * (y1 + y2) - top], dtype=np.float32)
    expected_radius = max(2.0, 0.25 * (box_w + box_h))
    expected_area = np.pi * expected_radius * expected_radius
    best: Optional[np.ndarray] = None
    best_score = -1e9
    for contour in contours:
        area = float(abs(cv2.contourArea(contour)))
        if area < expected_area * 0.16 or area > expected_area * 1.8:
            continue
        moments = cv2.moments(contour)
        if abs(float(moments["m00"])) < 1e-6:
            continue
        center = np.asarray(
            [moments["m10"] / moments["m00"], moments["m01"] / moments["m00"]],
            dtype=np.float32,
        )
        distance = float(np.linalg.norm(center - expected_center))
        score = area / max(1.0, expected_area) - 0.8 * distance / max(2.0, expected_radius)
        if score > best_score:
            best_score = score
            best = contour
    if best is None:
        return None
    translated = best.astype(np.float32)
    translated[:, 0, 0] += float(left)
    translated[:, 0, 1] += float(top)
    return translated
