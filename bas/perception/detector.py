from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

from ..config import DetectorConfig
from .. import runtime_env
from ..runtime_env import prepare_runtime_environment
from ..schemas import Detection
from ..utils import clamp, default_inference_device, group_from_class, iou_xyxy
from .ball_geometry import BallCenterRefiner

LOGGER = logging.getLogger(__name__)
BALL_CENTER_REFINER = BallCenterRefiner()


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
                bbox = (float(x), float(y), float(x + w), float(y + h))
                geometry = BALL_CENTER_REFINER.refine(
                    frame_bgr,
                    bbox,
                    mask_polygon=contour.reshape((-1, 2)),
                )
                out.append(
                    Detection(
                        bbox=bbox,
                        conf=float(clamp(area / 1500.0, 0.25, 0.99)),
                        cls_id=cls_id,
                        cls_name=name,
                        refined_center_px=geometry.center_px,
                        refined_radius_px=geometry.radius_px,
                        geometry_quality=geometry.quality,
                        geometry_method=geometry.method,
                    )
                )
        return _filter_ball_geometry_outliers(out)


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
        prepare_runtime_environment()
        try:
            from ultralytics import YOLO
        except ModuleNotFoundError as exc:
            if exc.name != "ultralytics":
                raise RuntimeError(
                    f"Ultralytics 已安装，但缺少运行依赖：{exc.name}。"
                    "请先运行一次 Setup_Environment.cmd，或重新安装 requirements-yolo.txt。"
                ) from exc
            raise RuntimeError(
                "当前检测后端为 Ultralytics，但当前 .venv 未安装 YOLO 推理依赖。"
                "请先运行一次 Setup_Environment.cmd，或手动执行："
                r".\.venv\Scripts\python.exe -m pip install -r requirements-yolo.txt"
            ) from exc
        except Exception as exc:
            torch_detail = ""
            if runtime_env.TORCH_IMPORT_ERROR is not None:
                torch_detail = f"；torch 预加载错误：{type(runtime_env.TORCH_IMPORT_ERROR).__name__}: {runtime_env.TORCH_IMPORT_ERROR}"
            raise RuntimeError(
                "Ultralytics 已安装，但导入失败。"
                f"原始错误：{type(exc).__name__}: {exc}{torch_detail}"
            ) from exc
        if not Path(model_path).exists():
            raise FileNotFoundError(f"Ultralytics model file does not exist: {model_path}")
        self.model = YOLO(model_path)
        try:
            self.model.fuse()
        except Exception:
            pass
        self.class_names = list(class_names)
        self.conf = float(clamp(conf, 0.01, 0.99))
        self.iou = float(clamp(iou, 0.01, 0.99))
        self.device = _resolve_ultralytics_device(device)
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
        mask_polygons_all: List[Optional[np.ndarray]] = []
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
            result_masks = getattr(result, "masks", None)
            mask_xy = getattr(result_masks, "xy", None)
            for mask_index in range(xyxy.shape[0]):
                polygon = None
                if mask_xy is not None and mask_index < len(mask_xy):
                    candidate = np.asarray(mask_xy[mask_index], dtype=np.float32).reshape((-1, 2))
                    if candidate.shape[0] >= 5:
                        candidate[:, 0] += float(x0)
                        candidate[:, 1] += float(y0)
                        polygon = candidate
                mask_polygons_all.append(polygon)
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
            polygon = mask_polygons_all[i] if i < len(mask_polygons_all) else None
            geometry = None
            if group_from_class(cls_name) in {"cue", "solid", "stripe", "black"}:
                geometry = BALL_CENTER_REFINER.refine(
                    frame_bgr,
                    (x1, y1, x2, y2),
                    mask_polygon=polygon,
                )
            axis_endpoints_px = None
            axis_quality = 0.0
            if group_from_class(cls_name) == "cue_stick":
                axis_endpoints_px, axis_quality = _cue_axis_from_polygon(polygon)
            out.append(
                Detection(
                    bbox=(x1, y1, x2, y2),
                    conf=float(scores_np[i]),
                    cls_id=cls_id,
                    cls_name=cls_name,
                    refined_center_px=geometry.center_px if geometry is not None else None,
                    refined_radius_px=geometry.radius_px if geometry is not None else None,
                    geometry_quality=geometry.quality if geometry is not None else 0.45,
                    geometry_method=geometry.method if geometry is not None else "bbox",
                    axis_endpoints_px=axis_endpoints_px,
                    axis_quality=float(axis_quality),
                )
            )
        return _filter_ball_geometry_outliers(out)


def _cue_axis_from_polygon(
    polygon: Optional[np.ndarray],
) -> tuple[Optional[Tuple[Tuple[float, float], Tuple[float, float]]], float]:
    if polygon is None:
        return None, 0.0
    points = np.asarray(polygon, dtype=np.float32).reshape((-1, 2))
    if points.shape[0] < 5 or not np.all(np.isfinite(points)):
        return None, 0.0
    center = np.mean(points, axis=0)
    centered = points - center
    covariance = np.cov(centered, rowvar=False)
    if covariance.shape != (2, 2) or not np.all(np.isfinite(covariance)):
        return None, 0.0
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)
    major_value = max(0.0, float(eigenvalues[order[-1]]))
    minor_value = max(0.0, float(eigenvalues[order[0]]))
    if major_value <= 1.0e-6:
        return None, 0.0
    axis = np.asarray(eigenvectors[:, order[-1]], dtype=np.float32)
    projections = centered @ axis
    low, high = np.percentile(projections, [2.0, 98.0])
    length = float(high - low)
    if length < 12.0:
        return None, 0.0
    p1 = center + axis * float(low)
    p2 = center + axis * float(high)
    elongation = major_value / max(1.0e-6, minor_value)
    quality = float(clamp((elongation - 2.0) / 18.0, 0.0, 1.0))
    return (
        ((float(p1[0]), float(p1[1])), (float(p2[0]), float(p2[1]))),
        quality,
    )


def create_detector(config: DetectorConfig) -> Detector:
    backend = str(config.backend or "disabled").lower()
    class_names = _load_class_names(config.class_file_path) if config.class_file_path else list(config.class_names)
    if backend == "disabled":
        return DisabledDetector()
    if backend in {"debug_color", "color", "synthetic"}:
        return ColorBallDetector()
    if backend in {"ultralytics", "yolo"}:
        if not config.model_path:
            raise ValueError("detector.model_path is required for Ultralytics detector.")
        if config.class_file_path and not Path(config.class_file_path).exists():
            raise FileNotFoundError(f"Detector class file does not exist: {config.class_file_path}")
        return UltralyticsDetector(
            model_path=config.model_path,
            class_names=class_names,
            conf=config.conf,
            iou=config.iou,
            device=config.device,
            tile_size=config.tile_size,
            tile_overlap=config.tile_overlap,
            max_det_per_tile=config.max_det_per_tile,
            batch_size=config.batch_size,
        )
    raise ValueError(f"Unsupported detector backend: {config.backend}")


def _filter_ball_geometry_outliers(detections: List[Detection]) -> List[Detection]:
    ball_detections = [
        detection
        for detection in detections
        if group_from_class(detection.cls_name) in {"cue", "solid", "stripe", "black"}
        and float(detection.geometry_quality) >= 0.4
        and float(detection.conf) >= 0.6
    ]
    if len(ball_detections) < 5:
        return detections
    median_radius = float(np.median([detection.radius_px for detection in ball_detections]))
    if median_radius <= 2.0:
        return detections
    filtered: List[Detection] = []
    for detection in detections:
        if group_from_class(detection.cls_name) not in {"cue", "solid", "stripe", "black"}:
            filtered.append(detection)
            continue
        ratio = float(detection.radius_px) / median_radius
        if 0.55 <= ratio <= 1.5:
            filtered.append(detection)
            continue
        if float(detection.conf) < 0.8:
            continue
        detection.geometry_quality = min(float(detection.geometry_quality), 0.2)
        detection.geometry_method = f"{detection.geometry_method}:size_outlier"
        filtered.append(detection)
    return filtered


def _resolve_ultralytics_device(device: str) -> str:
    requested = str(device or "auto").strip().lower()
    if requested in {"", "auto"}:
        return default_inference_device()
    if requested in {"cpu", "mps"}:
        return requested
    torch = runtime_env.preload_torch()
    if torch is None:
        cuda_available = False
        cuda_count = 0
    else:
        try:
            cuda_available = bool(torch.cuda.is_available())
            cuda_count = int(torch.cuda.device_count()) if cuda_available else 0
        except Exception:
            cuda_available = False
            cuda_count = 0
    if requested.isdigit():
        idx = int(requested)
        if cuda_available and idx < cuda_count:
            return requested
        LOGGER.warning("Requested CUDA device %s is unavailable; falling back to CPU.", requested)
        return "cpu"
    if requested.startswith("cuda"):
        if cuda_available:
            return requested
        LOGGER.warning("Requested CUDA device %s is unavailable; falling back to CPU.", requested)
        return "cpu"
    return str(device)


def _load_class_names(path: str) -> List[str]:
    p = Path(path)
    if not p.exists():
        return []
    text = p.read_text(encoding="utf-8-sig").strip()
    if not text:
        return []
    if p.suffix.lower() == ".json":
        import json

        data = json.loads(text)
        if isinstance(data, dict):
            if "names" in data and isinstance(data["names"], list):
                return [str(x) for x in data["names"]]
            return [str(data[k]) for k in sorted(data.keys(), key=lambda x: int(x) if str(x).isdigit() else str(x))]
        if isinstance(data, list):
            return [str(x) for x in data]
    return [line.strip() for line in text.splitlines() if line.strip()]
