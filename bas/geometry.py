from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np


@dataclass
class TableGeometry:
    outer_norm: np.ndarray = field(default_factory=lambda: np.zeros((0, 2), dtype=np.float32))
    inner_norm: np.ndarray = field(default_factory=lambda: np.zeros((0, 2), dtype=np.float32))
    inline_norm: List[np.ndarray] = field(default_factory=list)
    pockets_norm: List[np.ndarray] = field(default_factory=list)

    @property
    def is_empty(self) -> bool:
        return self.outer_norm.shape[0] < 3 and self.inner_norm.shape[0] < 3 and not self.inline_norm and not self.pockets_norm

    def scaled(self, width: int, height: int) -> Tuple[np.ndarray, np.ndarray, List[np.ndarray]]:
        def scale(arr: np.ndarray) -> np.ndarray:
            if arr.size == 0:
                return arr.copy()
            out = arr.copy().astype(np.float32)
            out[:, 0] *= float(width)
            out[:, 1] *= float(height)
            return out

        return scale(self.outer_norm), scale(self.inner_norm), [scale(p) for p in self.pockets_norm]

    def reference_scaled(self, width: int, height: int) -> Tuple[np.ndarray, List[np.ndarray], List[np.ndarray]]:
        def scale(arr: np.ndarray) -> np.ndarray:
            if arr.size == 0:
                return arr.copy()
            out = arr.copy().astype(np.float32)
            out[:, 0] *= float(width)
            out[:, 1] *= float(height)
            return out

        return scale(self.outer_norm), [scale(p) for p in self.inline_norm], [scale(p) for p in self.pockets_norm]


class TableGeometryLoader:
    @classmethod
    def load_optional(cls, outline_path: Optional[str], inline_path: Optional[str], pocket_path: Optional[str]) -> TableGeometry:
        if not any([outline_path, inline_path, pocket_path]):
            return TableGeometry()
        try:
            return cls.load(outline_path, inline_path, pocket_path)
        except Exception:
            return TableGeometry()

    @classmethod
    def load(cls, outline_path: Optional[str], inline_path: Optional[str], pocket_path: Optional[str]) -> TableGeometry:
        out_data = cls._load_json(outline_path) if outline_path else {}
        in_data = cls._load_json(inline_path) if inline_path else {}
        pk_data = cls._load_json(pocket_path) if pocket_path else {}

        src_w = float(out_data.get("imageWidth") or in_data.get("imageWidth") or pk_data.get("imageWidth") or 1920)
        src_h = float(out_data.get("imageHeight") or in_data.get("imageHeight") or pk_data.get("imageHeight") or 1080)

        outer = cls._first_shape_with_label(out_data, "outline", src_w, src_h)
        if outer.shape[0] < 3:
            outer = cls._first_any_shape(out_data, src_w, src_h)

        inline_lines = cls._all_shapes_with_label(in_data, "inline", src_w, src_h)
        pockets = cls._all_shapes_with_label(pk_data, "pocket", src_w, src_h)

        inner = np.zeros((0, 2), dtype=np.float32)
        stitch_lines = [*inline_lines, *pockets]
        if stitch_lines:
            inner = cls._stitch_lines_to_polygon(stitch_lines)
            if inner.shape[0] < 3:
                stack = np.vstack(stitch_lines).astype(np.float32)
                hull = cv2.convexHull((stack * np.array([1000.0, 1000.0], dtype=np.float32)).astype(np.float32))
                inner = hull.reshape((-1, 2)).astype(np.float32) / np.array([1000.0, 1000.0], dtype=np.float32)
        if inner.shape[0] < 3:
            inner = outer.copy()
        return TableGeometry(outer_norm=outer, inner_norm=inner, inline_norm=inline_lines, pockets_norm=pockets)

    @staticmethod
    def _load_json(path: Optional[str]) -> Dict[str, Any]:
        if not path:
            return {}
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(str(path))
        with p.open("r", encoding="utf-8-sig") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}

    @staticmethod
    def _shape_points(shape: Dict[str, Any], src_w: float, src_h: float) -> np.ndarray:
        pts: List[List[float]] = []
        for item in shape.get("points", []):
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                pts.append([float(item[0]) / max(1.0, src_w), float(item[1]) / max(1.0, src_h)])
        return np.asarray(pts, dtype=np.float32).reshape((-1, 2)) if pts else np.zeros((0, 2), dtype=np.float32)

    @classmethod
    def _first_shape_with_label(cls, data: Dict[str, Any], label_key: str, src_w: float, src_h: float) -> np.ndarray:
        for shape in data.get("shapes", []):
            if isinstance(shape, dict) and label_key in str(shape.get("label", "")).lower():
                return cls._shape_points(shape, src_w, src_h)
        return np.zeros((0, 2), dtype=np.float32)

    @classmethod
    def _first_any_shape(cls, data: Dict[str, Any], src_w: float, src_h: float) -> np.ndarray:
        shapes = data.get("shapes", [])
        if shapes and isinstance(shapes[0], dict):
            return cls._shape_points(shapes[0], src_w, src_h)
        return np.zeros((0, 2), dtype=np.float32)

    @classmethod
    def _all_shapes_with_label(cls, data: Dict[str, Any], label_key: str, src_w: float, src_h: float) -> List[np.ndarray]:
        out: List[np.ndarray] = []
        for shape in data.get("shapes", []):
            if not isinstance(shape, dict):
                continue
            if label_key not in str(shape.get("label", "")).lower():
                continue
            arr = cls._shape_points(shape, src_w, src_h)
            if arr.shape[0] >= 2:
                out.append(arr)
        return out

    @staticmethod
    def _stitch_lines_to_polygon(lines: List[np.ndarray], join_eps: float = 0.03) -> np.ndarray:
        candidates = [np.asarray(line, dtype=np.float32).reshape((-1, 2)) for line in lines if np.asarray(line).size >= 4]
        if not candidates:
            return np.zeros((0, 2), dtype=np.float32)
        best = np.zeros((0, 2), dtype=np.float32)
        best_used = -1
        best_cost = float("inf")
        for start_idx in range(len(candidates)):
            for rev in (False, True):
                used = [False] * len(candidates)
                current = candidates[start_idx][::-1] if rev else candidates[start_idx]
                parts = [current.copy()]
                used[start_idx] = True
                end = parts[-1][-1]
                cost = 0.0
                for _ in range(len(candidates) - 1):
                    best_i = -1
                    best_rev = False
                    best_dist = float("inf")
                    for idx, seg in enumerate(candidates):
                        if used[idx]:
                            continue
                        d0 = float(np.linalg.norm(seg[0] - end))
                        d1 = float(np.linalg.norm(seg[-1] - end))
                        if d0 < best_dist:
                            best_i, best_rev, best_dist = idx, False, d0
                        if d1 < best_dist:
                            best_i, best_rev, best_dist = idx, True, d1
                    if best_i < 0 or best_dist > join_eps:
                        break
                    nxt = candidates[best_i][::-1] if best_rev else candidates[best_i]
                    parts.append(nxt[1:])
                    end = parts[-1][-1]
                    used[best_i] = True
                    cost += best_dist
                merged = np.vstack(parts).astype(np.float32)
                if merged.shape[0] < 3:
                    continue
                end_gap = float(np.linalg.norm(merged[0] - merged[-1]))
                used_count = sum(1 for v in used if v)
                total = cost + end_gap
                if used_count > best_used or (used_count == best_used and total < best_cost):
                    best, best_used, best_cost = merged, used_count, total
        if best.shape[0] >= 3 and float(np.linalg.norm(best[0] - best[-1])) <= join_eps:
            return best
        return np.zeros((0, 2), dtype=np.float32)
