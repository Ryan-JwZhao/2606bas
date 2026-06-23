from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np


@dataclass
class StarFormulaConfig:
    enabled: bool = False
    inset_left_pct: float = 0.0
    inset_right_pct: float = 0.0
    inset_top_pct: float = 0.0
    inset_bottom_pct: float = 0.0
    label_offset_tb_pct: float = 6.0
    label_offset_lr_pct: float = 8.0
    offset_x_px: float = 0.0
    offset_y_px: float = 0.0
    scale_x_pct: float = 100.0
    scale_y_pct: float = 100.0
    angle_deg: float = 0.0

    @classmethod
    def from_mapping(cls, payload: Optional[Dict[str, Any]]) -> "StarFormulaConfig":
        data = payload or {}
        legacy = _as_float(data.get("label_offset_pct"), 6.0)
        return cls(
            enabled=_as_bool(data.get("enabled"), False),
            inset_left_pct=_as_float(data.get("inset_left_pct"), 0.0),
            inset_right_pct=_as_float(data.get("inset_right_pct"), 0.0),
            inset_top_pct=_as_float(data.get("inset_top_pct"), 0.0),
            inset_bottom_pct=_as_float(data.get("inset_bottom_pct"), 0.0),
            label_offset_tb_pct=_as_float(data.get("label_offset_tb_pct"), legacy),
            label_offset_lr_pct=_as_float(data.get("label_offset_lr_pct"), max(legacy, 8.0)),
            offset_x_px=_as_float(data.get("offset_x_px"), 0.0),
            offset_y_px=_as_float(data.get("offset_y_px"), 0.0),
            scale_x_pct=_as_float(data.get("scale_x_pct"), 100.0),
            scale_y_pct=_as_float(data.get("scale_y_pct"), 100.0),
            angle_deg=_as_float(data.get("angle_deg"), 0.0),
        )

    def clamp_for_projection(self, proj_w: int, proj_h: int) -> "StarFormulaConfig":
        max_off_x = max(400.0, float(proj_w) * 2.0)
        max_off_y = max(300.0, float(proj_h) * 2.0)
        return StarFormulaConfig(
            enabled=bool(self.enabled),
            inset_left_pct=float(np.clip(self.inset_left_pct, -20.0, 40.0)),
            inset_right_pct=float(np.clip(self.inset_right_pct, -20.0, 40.0)),
            inset_top_pct=float(np.clip(self.inset_top_pct, -20.0, 40.0)),
            inset_bottom_pct=float(np.clip(self.inset_bottom_pct, -20.0, 40.0)),
            label_offset_tb_pct=float(np.clip(self.label_offset_tb_pct, 0.0, 30.0)),
            label_offset_lr_pct=float(np.clip(self.label_offset_lr_pct, 0.0, 30.0)),
            offset_x_px=float(np.clip(self.offset_x_px, -max_off_x, max_off_x)),
            offset_y_px=float(np.clip(self.offset_y_px, -max_off_y, max_off_y)),
            scale_x_pct=float(np.clip(self.scale_x_pct, 10.0, 300.0)),
            scale_y_pct=float(np.clip(self.scale_y_pct, 10.0, 300.0)),
            angle_deg=_wrap_angle_deg(self.angle_deg),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "enabled": bool(self.enabled),
            "inset_left_pct": float(self.inset_left_pct),
            "inset_right_pct": float(self.inset_right_pct),
            "inset_top_pct": float(self.inset_top_pct),
            "inset_bottom_pct": float(self.inset_bottom_pct),
            "label_offset_tb_pct": float(self.label_offset_tb_pct),
            "label_offset_lr_pct": float(self.label_offset_lr_pct),
            "offset_x_px": float(self.offset_x_px),
            "offset_y_px": float(self.offset_y_px),
            "scale_x_pct": float(self.scale_x_pct),
            "scale_y_pct": float(self.scale_y_pct),
            "angle_deg": float(self.angle_deg),
        }


def draw_star_formula(canvas: np.ndarray, config: StarFormulaConfig) -> None:
    cfg = config.clamp_for_projection(canvas.shape[1], canvas.shape[0])
    if not cfg.enabled:
        return
    proj_w = canvas.shape[1]
    proj_h = canvas.shape[0]
    quad = _default_table_quad(proj_w, proj_h)

    left = float(8.0 * cfg.inset_left_pct / 100.0)
    right = float(8.0 - 8.0 * cfg.inset_right_pct / 100.0)
    top = float(4.0 * cfg.inset_top_pct / 100.0)
    bottom = float(4.0 - 4.0 * cfg.inset_bottom_pct / 100.0)
    if right - left <= 0.8 or bottom - top <= 0.4:
        return

    H = cv2.getPerspectiveTransform(
        np.array([[0.0, 0.0], [8.0, 0.0], [8.0, 4.0], [0.0, 4.0]], dtype=np.float32),
        quad.astype(np.float32),
    )
    base_center = np.mean(quad.astype(np.float64), axis=0)
    avg_w = 0.5 * (float(np.linalg.norm(quad[1] - quad[0])) + float(np.linalg.norm(quad[2] - quad[3])))
    avg_h = 0.5 * (float(np.linalg.norm(quad[3] - quad[0])) + float(np.linalg.norm(quad[2] - quad[1])))
    cell_size = max(12.0, min(avg_w / 8.0, avg_h / 4.0))
    mean_scale = math.sqrt(max(0.1, (cfg.scale_x_pct * cfg.scale_y_pct) / 10000.0))
    font_thickness = max(1, int(round(cell_size * mean_scale / 56.0)))
    line_thickness = max(1, int(round(float(font_thickness) / 3.0)))
    font_scale = float(np.clip(cell_size * mean_scale / 80.0, 0.225, 1.1))
    label_v = float((bottom - top) * cfg.label_offset_tb_pct / 200.0)
    label_u = float((bottom - top) * cfg.label_offset_lr_pct / 200.0)
    color = (255, 255, 255)

    angle_rad = math.radians(float(cfg.angle_deg))
    affine = np.array(
        [
            [math.cos(angle_rad) * (cfg.scale_x_pct / 100.0), -math.sin(angle_rad) * (cfg.scale_y_pct / 100.0)],
            [math.sin(angle_rad) * (cfg.scale_x_pct / 100.0), math.cos(angle_rad) * (cfg.scale_y_pct / 100.0)],
        ],
        dtype=np.float64,
    )
    offset = np.array([cfg.offset_x_px, cfg.offset_y_px], dtype=np.float64)

    def transform(u: float, v: float) -> Tuple[int, int]:
        out = cv2.perspectiveTransform(np.array([[[float(u), float(v)]]], dtype=np.float32), H)
        pt = out.reshape(2).astype(np.float64)
        warped = base_center + affine @ (pt - base_center) + offset
        return (int(round(float(warped[0]))), int(round(float(warped[1]))))

    border = np.array([transform(left, top), transform(right, top), transform(right, bottom), transform(left, bottom)], dtype=np.int32)
    cv2.polylines(canvas, [border], True, color, line_thickness, cv2.LINE_AA)

    step_u = (right - left) / 8.0
    step_v = (bottom - top) / 4.0
    for idx in range(1, 8):
        u = left + step_u * idx
        cv2.line(canvas, transform(u, top), transform(u, bottom), color, line_thickness, cv2.LINE_AA)
    for idx in range(1, 4):
        v = top + step_v * idx
        cv2.line(canvas, transform(left, v), transform(right, v), color, line_thickness, cv2.LINE_AA)

    for idx, text in enumerate(["0", "1", "2", "3", "4", "5", "6", "7", "X"]):
        _draw_centered_text(canvas, text, transform(left + step_u * idx, bottom + label_v), font_scale, font_thickness, color)
    for idx, text in enumerate(["1", "1.5", "2", "2.5", "3", "3.5", "4", "4.5", "5"]):
        _draw_centered_text(canvas, text, transform(left + step_u * idx, top - label_v), font_scale, font_thickness, color)
    for idx, text in enumerate(["X", "X", "7", "6", "5"]):
        if idx not in (0, 4):
            _draw_centered_text(canvas, text, transform(right + label_u, bottom - step_v * idx), font_scale, font_thickness, color)


def _default_table_quad(proj_w: int, proj_h: int) -> np.ndarray:
    margin = 0.08
    max_w = float(proj_w) * (1.0 - 2.0 * margin)
    max_h = float(proj_h) * (1.0 - 2.0 * margin)
    width = min(max_w, max_h * 2.0)
    height = width * 0.5
    cx = float(proj_w) * 0.5
    cy = float(proj_h) * 0.5
    return np.array(
        [[cx - width * 0.5, cy - height * 0.5], [cx + width * 0.5, cy - height * 0.5], [cx + width * 0.5, cy + height * 0.5], [cx - width * 0.5, cy + height * 0.5]],
        dtype=np.float32,
    )


def _draw_centered_text(img: np.ndarray, text: str, anchor: Tuple[int, int], font_scale: float, thickness: int, color: Tuple[int, int, int]) -> None:
    (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
    cv2.putText(img, text, (int(anchor[0] - text_w * 0.5), int(anchor[1] + text_h * 0.5)), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness, cv2.LINE_AA)


def _as_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _as_bool(value: Any, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        low = value.strip().lower()
        if low in {"1", "true", "yes", "on"}:
            return True
        if low in {"0", "false", "no", "off"}:
            return False
    return bool(default)


def _wrap_angle_deg(angle_deg: float) -> float:
    angle = float(angle_deg)
    while angle > 180.0:
        angle -= 360.0
    while angle < -180.0:
        angle += 360.0
    return angle

