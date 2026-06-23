from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

from ..config import DetectorConfig
from ..schemas import Detection
from ..utils import clamp, default_inference_device, iou_xyxy

LOGGER = logging.getLogger(__name__)


class Detector(ABC):
    version = "base"

    @abstractmethod
    def detect(self, frame_bgr: np.ndarray, mask_polygon: Optional[np.ndarray] = None) -> List[Detection]:
        ...


class DisabledDetector(Detector):
    version = "disabled"

    def detect(self, frame_bgr: np.ndarray, mask_polygon: Optional[np.ndarray] = None) -> List[Detection]:
        return []


class ColorBallDetector(Detector):
    """Small deterministic detector for synthetic videos and smoke tests."""

    version = "color_ball_debug_v1"

    def __init__(self, min_area: float = 80.0):
        self.min_area = float(min_area)

    def detect(self, frame_bgr: np.ndarray, mask_polygon: Optional[np.ndarray] = None) -> List[Detection]:
        if frame_bgr is None or frame_bgr.size == 0:
            return []
        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
        masks = [
            ("cue", cv2.inRange(hsv, (0, 0, 170), (180, 55, 255))),
            ("solid", cv2.inRange(hsv, (0, 70, 90), (18, 255, 255))),
            ("stripe", cv2.inRange(hsv, (22, 70, 90), (42, 255, 255))),
        ]
        dark = cv2.inRange(hsv, (0, 0, 0), (180, 255, 45))
        masks.append(("black", dark))
        out: List[Detection] = []
        for cls_id, (name, mask) in enumerate(masks):
            if mask_polygon is not None and mask_polygon.size >= 6:
                poly_mask = np.zeros(mask.shape[:2], dtype=np.uint8)
                cv2.fillPoly(poly_mask, [mask_polygon.reshape((-1, 1, 2)).astype(np.int32)], 255)
                mask = cv2.bitwise_and(mask, poly_mask)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                area = float(cv2.contourArea(contour))
                if area < self.min_area:
                    continue
                x, y, w, h = cv2.boundingRect(contour)
                if w <= 2 or h <= 2:
                    continue
                out.append(
                    Detection(
                        bbox=(float(x), float(y), float(x + w), float(y + h)),
                        conf=float(clamp(area / 1500.0, 0.25, 0.99)),
                        cls_id=cls_id,
                        cls_name=name,
                    )
                )
        return out


class UltralyticsDetector(Detector):
    version = "ultralytics_yolo_v1"

    def __init__(
        self,
        model_path: str,
        class_names: Sequence[str],
        conf: float,
        iou: float,
        device: str,
        tile_size: int,
        tile_overlap: float,
        max_det_per_tile: int,
        batch_size: int,
    ):
        try:
            from ultralytics import YOLO
        except Exception as exc:
            raise RuntimeError("Ultralytics detector requires `pip install ultralytics`.") from exc
        self.model = YOLO(model_path)
        try:
            self.model.fuse()
        except Exception:
            pass
        self.class_names = list(class_names)
        self.conf = float(clamp(conf, 0.01, 0.99))
        self.iou = float(clamp(iou, 0.01, 0.99))
        self.device = default_inference_device() if str(device).lower() in {"", "auto"} else str(device)
        self.tile_size = int(max(320, tile_size))
        self.tile_overlap = float(clamp(tile_overlap, 0.0, 0.8))
        self.max_det_per_tile = int(max(1, max_det_per_tile))
        self.batch_size = int(max(1, batch_size))

    def _iter_tiles(self, width: int, height: int):
        step = max(64, int(self.tile_size * (1.0 - self.tile_overlap)))
        xs = list(range(0, max(1, width - self.tile_size + 1), step))
        ys = list(range(0, max(1, height - self.tile_size + 1), step))
        if not xs or xs[-1] + self.tile_size < width:
            xs.append(max(0, width - self.tile_size))
        if not ys or ys[-1] + self.tile_size < height:
            ys.append(max(0, height - self.tile_size))
        for y in ys:
            for x in xs:
                x2 = min(width, x + self.tile_size)
                y2 = min(height, y + self.tile_size)
                yield x, y, x2, y2

    @staticmethod
    def _nms(boxes: np.ndarray, scores: np.ndarray, threshold: float) -> List[int]:
        if boxes.shape[0] == 0:
            return []
        order = np.argsort(-scores)
        keep: List[int] = []
        while order.size > 0:
            i = int(order[0])
            keep.append(i)
            if order.size == 1:
                break
            rest = order[1:]
            ious = np.asarray([iou_xyxy(boxes[i], boxes[j]) for j in rest], dtype=np.float32)
            order = rest[ious < threshold]
        return keep

    def detect(self, frame_bgr: np.ndarray, mask_polygon: Optional[np.ndarray] = None) -> List[Detection]:
        h, w = frame_bgr.shape[:2]
        tiles: List[np.ndarray] = []
        offsets: List[Tuple[int, int]] = []
        for x1, y1, x2, y2 in self._iter_tiles(w, h):
            tile = frame_bgr[y1:y2, x1:x2]
            if tile.size == 0:
                continue
            tiles.append(tile)
            offsets.append((x1, y1))
        if not tiles:
            return []
        results = self.model.predict(
            source=tiles,
            conf=self.conf,
            iou=self.iou,
            max_det=self.max_det_per_tile,
            device=self.device,
            verbose=False,
            batch=min(self.batch_size, len(tiles)),
        )
        boxes_all: List[np.ndarray] = []
        scores_all: List[np.ndarray] = []
        cls_all: List[np.ndarray] = []
        for idx, result in enumerate(results):
            boxes = getattr(result, "boxes", None)
            if boxes is None or len(boxes) == 0:
                continue
            x0, y0 = offsets[idx]
            xyxy = boxes.xyxy.cpu().numpy().astype(np.float32)
            xyxy[:, [0, 2]] += x0
            xyxy[:, [1, 3]] += y0
            boxes_all.append(xyxy)
            scores_all.append(boxes.conf.cpu().numpy().astype(np.float32))
            cls_all.append(boxes.cls.cpu().numpy().astype(np.int32))
        if not boxes_all:
            return []
        boxes_np = np.concatenate(boxes_all, axis=0)
        scores_np = np.concatenate(scores_all, axis=0)
        cls_np = np.concatenate(cls_all, axis=0)
        keep: List[int] = []
        for cls_id in np.unique(cls_np):
            idx = np.where(cls_np == cls_id)[0]
            keep.extend(idx[k] for k in self._nms(boxes_np[idx], scores_np[idx], self.iou))
        out: List[Detection] = []
        for i in keep:
            x1, y1, x2, y2 = [float(v) for v in boxes_np[i]]
            cx = 0.5 * (x1 + x2)
            cy = 0.5 * (y1 + y2)
            if mask_polygon is not None and mask_polygon.size >= 6:
                inside = cv2.pointPolygonTest(mask_polygon.reshape((-1, 1, 2)).astype(np.float32), (cx, cy), False)
                if inside < 0.0:
                    continue
            cls_id = int(cls_np[i])
            cls_name = self.class_names[cls_id] if 0 <= cls_id < len(self.class_names) else str(cls_id)
            out.append(Detection(bbox=(x1, y1, x2, y2), conf=float(scores_np[i]), cls_id=cls_id, cls_name=cls_name))
        return out


def create_detector(config: DetectorConfig) -> Detector:
    backend = str(config.backend or "disabled").lower()
    if backend == "disabled":
        return DisabledDetector()
    if backend in {"debug_color", "color", "synthetic"}:
        return ColorBallDetector()
    if backend in {"ultralytics", "yolo"}:
        if not config.model_path:
            raise ValueError("detector.model_path is required for Ultralytics detector.")
        return UltralyticsDetector(
            model_path=config.model_path,
            class_names=config.class_names,
            conf=config.conf,
            iou=config.iou,
            device=config.device,
            tile_size=config.tile_size,
            tile_overlap=config.tile_overlap,
            max_det_per_tile=config.max_det_per_tile,
            batch_size=config.batch_size,
        )
    raise ValueError(f"Unsupported detector backend: {config.backend}")

