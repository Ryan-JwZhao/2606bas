from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np

from ..utils import percentile
from .quality_standards import (
    DISTANCE_ERROR_SLOPE_ABS_MM_PER_CM,
    FORMAL_TABLE_ERROR_MEDIAN_MM,
    FORMAL_TABLE_ERROR_P95_MM,
    IMAGE_ERROR_MEAN_PX,
    IMAGE_ERROR_P95_PX,
    MVP_TABLE_ERROR_MEDIAN_MM,
    MVP_TABLE_ERROR_P95_MM,
    POCKET_ZONE_ERROR_P95_MM,
)
from .service import CalibrationService


MVP_MIN_HOLDOUT_SAMPLES = 12
FORMAL_MIN_HOLDOUT_SAMPLES = 24
MVP_MIN_LABELED_ZONES = 4
FORMAL_MIN_LABELED_ZONES = 8
MVP_MIN_POCKET_ZONES = 2
FORMAL_MIN_POCKET_ZONES = 4


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
    zone_vectors: Dict[str, List[np.ndarray]] = {}
    used_samples = 0
    skipped_samples = 0

    for sample in samples:
        cam = _point(sample, "camera_px", "cam_px", "image_px")
        proj = _point(sample, "projector_px", "proj_px")
        world = _point(sample, "world_mm", "table_mm")
        zone = str(sample.get("zone", "unlabeled"))

        sample_used = False
        if cam is not None and proj is not None:
            pred = (
                service.ball_camera_px_to_projector_px(np.asarray([cam], dtype=np.float32))[0]
                if _is_ball_sample(sample)
                else service.camera_px_to_projector_px(np.asarray([cam], dtype=np.float32))[0]
            )
            img_errors.append(float(np.linalg.norm(pred - np.asarray(proj, dtype=np.float32))))
            sample_used = True

        if world is not None and (proj is not None or cam is not None):
            observed = (
                service.ball_camera_px_to_table_mm(np.asarray([cam], dtype=np.float32))[0]
                if cam is not None and _is_ball_sample(sample)
                else (
                    service.projector_px_to_table_mm(np.asarray([proj], dtype=np.float32))[0]
                    if proj is not None
                    else service.camera_px_to_table_mm(np.asarray([cam], dtype=np.float32))[0]
                )
            )
            error_vector = observed - np.asarray(world, dtype=np.float32)
            err = float(np.linalg.norm(error_vector))
            mm_errors.append(err)
            zone_errors.setdefault(zone, []).append(err)
            zone_vectors.setdefault(zone, []).append(error_vector.astype(np.float64))
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
        "zone_mean_vector_mm": {
            zone: np.mean(np.asarray(vectors, dtype=np.float64), axis=0).tolist()
            for zone, vectors in sorted(zone_vectors.items())
            if vectors
        },
        "distance_slope_mm_per_cm": _fit_slope(distances_cm, mm_errors),
        "coverage": _coverage_report(used_samples, zone_errors),
        "thresholds": {
            "image_mean_px": IMAGE_ERROR_MEAN_PX,
            "image_p95_px": IMAGE_ERROR_P95_PX,
            "mvp_table_median_mm": MVP_TABLE_ERROR_MEDIAN_MM,
            "mvp_table_p95_mm": MVP_TABLE_ERROR_P95_MM,
            "formal_table_median_mm": FORMAL_TABLE_ERROR_MEDIAN_MM,
            "formal_table_p95_mm": FORMAL_TABLE_ERROR_P95_MM,
            "pocket_zone_p95_mm": POCKET_ZONE_ERROR_P95_MM,
            "distance_slope_abs_mm_per_cm": DISTANCE_ERROR_SLOPE_ABS_MM_PER_CM,
            "mvp_min_samples": MVP_MIN_HOLDOUT_SAMPLES,
            "formal_min_samples": FORMAL_MIN_HOLDOUT_SAMPLES,
            "mvp_min_labeled_zones": MVP_MIN_LABELED_ZONES,
            "formal_min_labeled_zones": FORMAL_MIN_LABELED_ZONES,
            "mvp_min_pocket_zones": MVP_MIN_POCKET_ZONES,
            "formal_min_pocket_zones": FORMAL_MIN_POCKET_ZONES,
        },
    }
    report["verdict"] = _verdict(report)
    report["geometry_model"] = dict(service.geometry_quality_report)
    return report


def format_holdout_report(report: Dict[str, Any]) -> str:
    img = report.get("image_error_px", {})
    mm = report.get("table_error_mm", {})
    slope = report.get("distance_slope_mm_per_cm")
    verdict = report.get("verdict", {})
    coverage = report.get("coverage", {})
    lines = [
        f"样本: {report.get('sample_count', 0)} 有效 / {report.get('skipped_count', 0)} 跳过",
        "图像误差: "
        f"mean={img.get('mean', 0.0):.3f}px, median={img.get('median', 0.0):.3f}px, "
        f"p95={img.get('p95', 0.0):.3f}px, max={img.get('max', 0.0):.3f}px",
        "台面误差: "
        f"median={mm.get('median', 0.0):.3f}mm, p95={mm.get('p95', 0.0):.3f}mm, max={mm.get('max', 0.0):.3f}mm",
        f"距离梯度: {slope:.4f} mm/cm" if slope is not None else "距离梯度: 样本不足",
        f"MVP: {'通过' if verdict.get('mvp') else '未通过'} | 正式: {'通过' if verdict.get('formal') else '未通过'}",
        "空间覆盖: "
        f"分区={len(coverage.get('labeled_zones', []))}, "
        f"袋口分区={len(coverage.get('pocket_zones', []))}, "
        f"正式覆盖={'通过' if coverage.get('formal') else '未通过'}",
    ]
    zones = report.get("zone_p95_mm", {})
    if zones:
        zone_text = ", ".join(f"{zone}={value:.2f}mm" for zone, value in zones.items())
        lines.append(f"分区 P95: {zone_text}")
    zone_biases = _format_zone_mean_vectors(report.get("zone_mean_vector_mm", {}))
    if zone_biases:
        lines.append(f"分区平均偏差: {zone_biases}")
    return "\n".join(lines)


def _format_zone_mean_vectors(value: Any) -> str:
    if not isinstance(value, dict):
        return ""
    formatted: list[str] = []
    for zone, vector in value.items():
        try:
            point = np.asarray(vector, dtype=np.float64).reshape((-1,))
        except (TypeError, ValueError):
            continue
        if point.size < 2 or not bool(np.all(np.isfinite(point[:2]))):
            continue
        dx, dy = float(point[0]), float(point[1])
        formatted.append(f"{zone}=(dx={dx:+.2f}, dy={dy:+.2f}, |v|={float(np.hypot(dx, dy)):.2f})mm")
    return ", ".join(formatted)


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


def _is_ball_sample(sample: Dict[str, Any]) -> bool:
    kind = str(sample.get("kind", sample.get("surface", sample.get("observation_type", "")))).strip().lower()
    return kind in {"ball", "ball_center", "sphere_center"}


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
    img_ok = img["count"] == 0 or (
        img["mean"] < IMAGE_ERROR_MEAN_PX and img["p95"] < IMAGE_ERROR_P95_PX
    )
    mvp_mm_ok = (
        mm["count"] > 0
        and mm["median"] < MVP_TABLE_ERROR_MEDIAN_MM
        and mm["p95"] < MVP_TABLE_ERROR_P95_MM
    )
    formal_mm_ok = (
        mm["count"] > 0
        and mm["median"] < FORMAL_TABLE_ERROR_MEDIAN_MM
        and mm["p95"] < FORMAL_TABLE_ERROR_P95_MM
    )
    slope_ok = slope is None or abs(float(slope)) < DISTANCE_ERROR_SLOPE_ABS_MM_PER_CM
    coverage = report["coverage"]
    mvp_coverage = bool(coverage["mvp"])
    formal_coverage = bool(coverage["formal"])
    zones_ok = all(float(value) < 2.5 for zone, value in report["zone_p95_mm"].items() if "pocket" in str(zone).lower() or "袋" in str(zone))
    return {
        "image": bool(img_ok),
        "mvp": bool(img_ok and mvp_mm_ok and slope_ok and zones_ok and mvp_coverage),
        "formal": bool(img_ok and formal_mm_ok and slope_ok and zones_ok and formal_coverage),
        "slope": bool(slope_ok),
        "zones": bool(zones_ok),
        "mvp_sample_coverage": mvp_coverage,
        "sample_coverage": formal_coverage,
    }


def _coverage_report(sample_count: int, zone_errors: Dict[str, List[float]]) -> Dict[str, Any]:
    labeled = {
        str(zone).strip().lower()
        for zone, values in zone_errors.items()
        if values and str(zone).strip().lower() not in {"", "unlabeled", "unknown"}
    }
    pocket_zones = {zone for zone in labeled if "pocket" in zone or "袋" in zone}
    mvp = (
        int(sample_count) >= MVP_MIN_HOLDOUT_SAMPLES
        and len(labeled) >= MVP_MIN_LABELED_ZONES
        and len(pocket_zones) >= MVP_MIN_POCKET_ZONES
    )
    formal = (
        int(sample_count) >= FORMAL_MIN_HOLDOUT_SAMPLES
        and len(labeled) >= FORMAL_MIN_LABELED_ZONES
        and len(pocket_zones) >= FORMAL_MIN_POCKET_ZONES
    )
    return {
        "sample_count": int(sample_count),
        "labeled_zones": sorted(labeled),
        "pocket_zones": sorted(pocket_zones),
        "mvp": bool(mvp),
        "formal": bool(formal),
    }
