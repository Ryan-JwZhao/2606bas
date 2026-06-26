from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np


@dataclass
class BallCompensationModel:
    mode: str = "none"
    control_points_camera_px: np.ndarray = field(default_factory=lambda: np.zeros((0, 2), dtype=np.float64))
    delta_table_mm: np.ndarray = field(default_factory=lambda: np.zeros((0, 2), dtype=np.float64))
    max_neighbors: int = 8
    source_path: Optional[str] = None
    quality_report: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def load_json(cls, path: str | Path | None) -> "BallCompensationModel":
        if path is None:
            return cls()
        p = Path(path)
        if not p.exists():
            return cls(source_path=str(p), quality_report={"missing": True})
        with p.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return cls(
            mode=str(data.get("mode", "none")),
            control_points_camera_px=np.asarray(
                data.get("control_camera_points", data.get("control_points_camera_px", [])),
                dtype=np.float64,
            ).reshape((-1, 2)),
            delta_table_mm=np.asarray(data.get("delta_table_mm", []), dtype=np.float64).reshape((-1, 2)),
            max_neighbors=max(1, int(data.get("max_neighbors", 8) or 8)),
            source_path=str(p),
            quality_report=dict(data.get("quality_report", {})),
        )

    @property
    def is_valid(self) -> bool:
        return (
            self.control_points_camera_px.shape[0] > 0
            and self.control_points_camera_px.shape == self.delta_table_mm.shape
        )

    @property
    def version(self) -> str:
        if self.source_path:
            return Path(self.source_path).stem
        return "no_ball_compensation"

    def to_dict(self, extra_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "mode": str(self.mode or "none"),
            "control_camera_points": np.asarray(self.control_points_camera_px, dtype=np.float64).reshape((-1, 2)).tolist(),
            "delta_table_mm": np.asarray(self.delta_table_mm, dtype=np.float64).reshape((-1, 2)).tolist(),
            "max_neighbors": int(max(1, int(self.max_neighbors))),
            "quality_report": dict(self.quality_report),
        }
        if extra_data:
            payload.update(extra_data)
        return payload

    def save_json(self, path: str | Path, extra_data: Optional[Dict[str, Any]] = None) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("w", encoding="utf-8") as f:
            json.dump(self.to_dict(extra_data=extra_data), f, ensure_ascii=False, indent=2)
        self.source_path = str(p)

    def offsets_for_camera_points(self, points_camera_px: np.ndarray) -> np.ndarray:
        pts = np.asarray(points_camera_px, dtype=np.float64).reshape((-1, 2))
        if pts.size == 0 or not self.is_valid:
            return np.zeros_like(pts)
        out = np.zeros_like(pts)
        k = min(int(self.max_neighbors), self.control_points_camera_px.shape[0])
        for idx, pt in enumerate(pts):
            d2 = np.sum((self.control_points_camera_px - pt.reshape((1, 2))) ** 2, axis=1)
            nearest = int(np.argmin(d2))
            if d2[nearest] <= 1e-9:
                out[idx] = self.delta_table_mm[nearest]
                continue
            nn = np.argsort(d2)[:k]
            weights = 1.0 / np.maximum(d2[nn], 1.0)
            out[idx] = np.sum(self.delta_table_mm[nn] * weights[:, None], axis=0) / max(1e-9, float(np.sum(weights)))
        return out
