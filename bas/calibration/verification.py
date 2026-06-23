from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np

from ..utils import percentile
from .service import CalibrationService


def verify_holdout_file(path: str | Path, service: CalibrationService) -> Dict[str, Any]:
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    samples = raw.get("samples", raw) if isinstance(raw, dict) else raw
    if not isinstance(samples, list):
        raise ValueError("Holdout JSON must be a list or an object with a 'samples' list.")
    return verify_holdout_samples(samples, service)


def verify_holdout_samples(samples: Iterable[Dict[str, Any]], service: CalibrationService) -> Dict[str, Any]:
    img_errors: List[float] = []
    mm_errors: List[float] = []
    distances_cm: List[float] = []
    zone_errors: Dict[str, List[float]] = {}
    used_samples = 0
    skipped_samples = 0

    for sample in samples:
        cam = _point(sample, "camera_px", "cam_px", "image_px")
        proj = _point(sample, "projector_px", "proj_px")
        world = _point(sample, "world_mm", "table_mm")
        zone = str(sample.get("zone", "unlabeled"))

        sample_used = False
        if cam is not None and proj is not None:
            pred = service.camera_px_to_projector_px(np.asarray([cam], dtype=np.float32))[0]
            img_errors.append(float(np.linalg.norm(pred - np.asarray(proj, dtype=np.float32))))
            sample_used = True

        if world is not None and (proj is not None or cam is not None):
            observed = (
                service.projector_px_to_table_mm(np.asarray([proj], dtype=np.float32))[0]
                if proj is not None
                else service.camera_px_to_table_mm(np.asarray([cam], dtype=np.float32))[0]
            )
            err = float(np.linalg.norm(observed - np.asarray(world, dtype=np.float32)))
            mm_errors.append(err)
            zone_errors.setdefault(zone, []).append(err)
            distances_cm.append(_distance_cm(sample, world))
            sample_used = True

        if sample_used:
            used_samples += 1
        else:
            skipped_samples += 1

    report: Dict[str, Any] = {
        "sample_count": used_samples,
        "skipped_count": skipped_samples,
        "image_error_px": _stats(img_errors),
        "table_error_mm": _stats(mm_errors),
        "zone_p95_mm": {zone: percentile(values, 95) for zone, values in sorted(zone_errors.items()) if values},
        "distance_slope_mm_per_cm": _fit_slope(distances_cm, mm_errors),
        "thresholds": {
            "image_mean_px": 0.20,
            "image_p95_px": 0.35,
            "mvp_table_median_mm": 1.5,
            "mvp_table_p95_mm": 3.0,
            "formal_table_median_mm": 1.0,
            "formal_table_p95_mm": 2.0,
            "pocket_zone_p95_mm": 2.5,
            "distance_slope_abs_mm_per_cm": 0.03,
        },
    }
    report["verdict"] = _verdict(report)
    return report


def format_holdout_report(report: Dict[str, Any]) -> str:
    img = report.get("image_error_px", {})
    mm = report.get("table_error_mm", {})
    slope = report.get("distance_slope_mm_per_cm")
    verdict = report.get("verdict", {})
    lines = [
        f"样本: {report.get('sample_count', 0)} 有效 / {report.get('skipped_count', 0)} 跳过",
        "图像误差: "
        f"mean={img.get('mean', 0.0):.3f}px, median={img.get('median', 0.0):.3f}px, "
        f"p95={img.get('p95', 0.0):.3f}px, max={img.get('max', 0.0):.3f}px",
        "台面误差: "
        f"median={mm.get('median', 0.0):.3f}mm, p95={mm.get('p95', 0.0):.3f}mm, max={mm.get('max', 0.0):.3f}mm",
        f"距离梯度: {slope:.4f} mm/cm" if slope is not None else "距离梯度: 样本不足",
        f"MVP: {'通过' if verdict.get('mvp') else '未通过'} | 正式: {'通过' if verdict.get('formal') else '未通过'}",
    ]
    zones = report.get("zone_p95_mm", {})
    if zones:
        zone_text = ", ".join(f"{zone}={value:.2f}mm" for zone, value in zones.items())
        lines.append(f"分区 P95: {zone_text}")
    return "\n".join(lines)


def _point(sample: Dict[str, Any], *keys: str) -> Optional[np.ndarray]:
    for key in keys:
        value = sample.get(key)
        if value is None:
            continue
        arr = np.asarray(value, dtype=np.float32).reshape((-1,))
        if arr.shape[0] >= 2:
            return arr[:2].astype(np.float32)
    return None


def _distance_cm(sample: Dict[str, Any], world: np.ndarray) -> float:
    for key in ("distance_cm", "distance_to_projector_cm"):
        if key in sample:
            return float(sample[key])
    origin = np.asarray(sample.get("projector_origin_mm", [0.0, 0.0]), dtype=np.float32).reshape((-1,))
    if origin.shape[0] < 2:
        origin = np.zeros((2,), dtype=np.float32)
    return float(np.linalg.norm(np.asarray(world, dtype=np.float32) - origin[:2]) / 10.0)


def _stats(values: List[float]) -> Dict[str, float]:
    if not values:
        return {"count": 0, "mean": 0.0, "median": 0.0, "p95": 0.0, "max": 0.0}
    arr = np.asarray(values, dtype=np.float32)
    return {
        "count": int(arr.shape[0]),
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "p95": percentile(arr, 95),
        "max": float(np.max(arr)),
    }


def _fit_slope(distances_cm: List[float], errors_mm: List[float]) -> Optional[float]:
    if len(distances_cm) < 2 or len(errors_mm) < 2 or len(distances_cm) != len(errors_mm):
        return None
    x = np.asarray(distances_cm, dtype=np.float64)
    y = np.asarray(errors_mm, dtype=np.float64)
    if float(np.max(x) - np.min(x)) < 1e-6:
        return None
    slope, _ = np.polyfit(x, y, deg=1)
    return float(slope)


def _verdict(report: Dict[str, Any]) -> Dict[str, bool]:
    img = report["image_error_px"]
    mm = report["table_error_mm"]
    slope = report["distance_slope_mm_per_cm"]
    img_ok = img["count"] == 0 or (img["mean"] < 0.20 and img["p95"] < 0.35)
    mvp_mm_ok = mm["count"] > 0 and mm["median"] < 1.5 and mm["p95"] < 3.0
    formal_mm_ok = mm["count"] > 0 and mm["median"] < 1.0 and mm["p95"] < 2.0
    slope_ok = slope is None or abs(float(slope)) < 0.03
    zones_ok = all(float(value) < 2.5 for zone, value in report["zone_p95_mm"].items() if "pocket" in str(zone).lower() or "袋" in str(zone))
    return {
        "image": bool(img_ok),
        "mvp": bool(img_ok and mvp_mm_ok and slope_ok and zones_ok),
        "formal": bool(img_ok and formal_mm_ok and slope_ok and zones_ok),
        "slope": bool(slope_ok),
        "zones": bool(zones_ok),
    }
