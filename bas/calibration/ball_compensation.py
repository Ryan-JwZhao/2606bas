from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from ..geometry_contract import context_compatibility_errors
from .artifact_io import atomic_write_json, load_json_object


@dataclass
class BallCompensationModel:
    mode: str = "none"
    control_points_camera_px: np.ndarray = field(default_factory=lambda: np.zeros((0, 2), dtype=np.float64))
    delta_table_mm: np.ndarray = field(default_factory=lambda: np.zeros((0, 2), dtype=np.float64))
    target_table_mm: np.ndarray = field(default_factory=lambda: np.zeros((0, 2), dtype=np.float64))
    sample_weights: np.ndarray = field(default_factory=lambda: np.zeros((0,), dtype=np.float64))
    max_neighbors: int = 8
    source_path: Optional[str] = None
    quality_report: Dict[str, Any] = field(default_factory=dict)
    calibration_context: Dict[str, Any] = field(default_factory=dict)
    compatibility_errors: tuple[str, ...] = ()

    @classmethod
    def load_json(
        cls,
        path: str | Path | None,
        *,
        expected_context: Optional[Dict[str, Any]] = None,
    ) -> "BallCompensationModel":
        if path is None:
            return cls()
        p = Path(path)
        if not p.exists():
            return cls(source_path=str(p), quality_report={"missing": True})
        try:
            data = load_json_object(p)
            samples = data.get("samples", [])
            targets = data.get("target_table_mm", [])
            weights = data.get("sample_weights", [])
            if isinstance(samples, list) and samples:
                if not targets:
                    targets = [sample.get("target_table_mm", []) for sample in samples]
                if not weights:
                    weights = [_sample_weight(sample) for sample in samples]
            stored_context = dict(data.get("calibration_context", {}))
            compatibility_errors = context_compatibility_errors(stored_context, expected_context)
            controls = _finite_array(
                data.get("control_camera_points", data.get("control_points_camera_px", [])),
                (-1, 2),
                "control_camera_points",
            )
            deltas = _finite_array(data.get("delta_table_mm", []), (-1, 2), "delta_table_mm")
            target_points = _finite_array(targets, (-1, 2), "target_table_mm")
            sample_weights = _finite_array(weights, (-1,), "sample_weights")
            if deltas.shape[0] not in {0, controls.shape[0]}:
                raise ValueError("delta_table_mm must be empty or match control point count")
            if target_points.shape[0] not in {0, controls.shape[0]}:
                raise ValueError("target_table_mm must be empty or match control point count")
            if sample_weights.shape[0] not in {0, controls.shape[0]}:
                raise ValueError("sample_weights must be empty or match control point count")
            return cls(
                mode=str(data.get("mode", "none")),
                control_points_camera_px=controls,
                delta_table_mm=deltas,
                target_table_mm=target_points,
                sample_weights=sample_weights,
                max_neighbors=max(1, int(data.get("max_neighbors", 8) or 8)),
                source_path=str(p),
                quality_report=dict(data.get("quality_report", {})),
                calibration_context=stored_context,
                compatibility_errors=compatibility_errors,
            )
        except (OSError, TypeError, ValueError) as exc:
            return cls(source_path=str(p), quality_report={"load_error": str(exc)})

    @property
    def is_valid(self) -> bool:
        return (
            self.control_points_camera_px.shape[0] > 0
            and not self.compatibility_errors
            and self.quality_report.get("quality_gate_passed") is not False
            and (
                self.control_points_camera_px.shape == self.delta_table_mm.shape
                or self.control_points_camera_px.shape == self.target_table_mm.shape
            )
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
            "target_table_mm": np.asarray(self.target_table_mm, dtype=np.float64).reshape((-1, 2)).tolist(),
            "sample_weights": np.asarray(self.sample_weights, dtype=np.float64).reshape((-1,)).tolist(),
            "max_neighbors": int(max(1, int(self.max_neighbors))),
            "quality_report": dict(self.quality_report),
            "calibration_context": dict(self.calibration_context),
        }
        if extra_data:
            payload.update(extra_data)
        return payload

    def save_json(self, path: str | Path, extra_data: Optional[Dict[str, Any]] = None) -> None:
        p = Path(path)
        atomic_write_json(p, self.to_dict(extra_data=extra_data))
        self.source_path = str(p)

    def offsets_for_camera_points(self, points_camera_px: np.ndarray) -> np.ndarray:
        pts = np.asarray(points_camera_px, dtype=np.float64).reshape((-1, 2))
        if (
            pts.size == 0
            or not self.is_valid
            or self.delta_table_mm.shape != self.control_points_camera_px.shape
        ):
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


def _sample_weight(sample: Dict[str, Any]) -> float:
    confidence = float(sample.get("detection_confidence", 1.0) or 0.0)
    spread = max(0.0, float(sample.get("stability_spread_px", 0.0) or 0.0))
    geometry_quality = float(sample.get("geometry_quality", 0.0) or 0.0)
    method = str(sample.get("geometry_method", "unknown")).strip().lower()
    method_weight = 1.0 if method.startswith("segmentation_ellipse") else 0.85 if method.startswith("appearance_ellipse") else 0.25
    return float(np.clip(confidence * geometry_quality * geometry_quality * method_weight / (1.0 + spread * spread), 0.01, 1.0))


def _finite_array(value: Any, shape: tuple[int, ...], field_name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64).reshape(shape)
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{field_name} contains non-finite values")
    return array
