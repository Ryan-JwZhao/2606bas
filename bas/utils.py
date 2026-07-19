from __future__ import annotations

import math
import sys
from typing import Iterable, Optional, Tuple

import numpy as np

from .runtime_env import preload_torch


def clamp(value: float, lo: float, hi: float) -> float:
    return float(max(lo, min(hi, value)))


def unit(v: np.ndarray, fallback: Tuple[float, float] = (1.0, 0.0)) -> np.ndarray:
    arr = np.asarray(v, dtype=np.float32).reshape((2,))
    n = float(np.linalg.norm(arr))
    if n < 1e-8:
        return np.asarray(fallback, dtype=np.float32)
    return (arr / n).astype(np.float32)


def angle_deg(a: np.ndarray, b: np.ndarray) -> float:
    aa = np.asarray(a, dtype=np.float32).reshape((2,))
    bb = np.asarray(b, dtype=np.float32).reshape((2,))
    na = float(np.linalg.norm(aa))
    nb = float(np.linalg.norm(bb))
    if na < 1e-8 or nb < 1e-8:
        return 180.0
    c = clamp(float(np.dot(aa, bb) / (na * nb)), -1.0, 1.0)
    return float(np.degrees(np.arccos(c)))


def point_segment_distance(point: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    p = np.asarray(point, dtype=np.float32).reshape((2,))
    aa = np.asarray(a, dtype=np.float32).reshape((2,))
    bb = np.asarray(b, dtype=np.float32).reshape((2,))
    ab = bb - aa
    denom = float(np.dot(ab, ab))
    if denom < 1e-9:
        return float(np.linalg.norm(p - aa))
    t = clamp(float(np.dot(p - aa, ab) / denom), 0.0, 1.0)
    closest = aa + t * ab
    return float(np.linalg.norm(p - closest))


def iou_xyxy(a: Iterable[float], b: Iterable[float]) -> float:
    ax1, ay1, ax2, ay2 = [float(x) for x in a]
    bx1, by1, bx2, by2 = [float(x) for x in b]
    x1 = max(ax1, bx1)
    y1 = max(ay1, by1)
    x2 = min(ax2, bx2)
    y2 = min(ay2, by2)
    iw = max(0.0, x2 - x1)
    ih = max(0.0, y2 - y1)
    inter = iw * ih
    if inter <= 0.0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    return float(inter / max(1e-9, area_a + area_b - inter))


def percentile(values: Iterable[float], q: float) -> float:
    arr = np.asarray(list(values), dtype=np.float64)
    if arr.size == 0:
        return 0.0
    return float(np.percentile(arr, q))


def group_from_class(name: str) -> str:
    n = str(name).strip().lower()
    if n.isdigit():
        number = int(n)
        if number == 0:
            return "cue"
        if 1 <= number <= 7:
            return "solid"
        if number == 8:
            return "black"
        if 9 <= number <= 15:
            return "stripe"
    if n in {"wb", "white", "cue", "cue_ball", "white_ball"}:
        return "cue"
    if n in {"bb", "black", "eight", "8", "black_ball"}:
        return "black"
    if n in {"sob", "solid", "solid_ball", "solids"}:
        return "solid"
    if n in {"stb", "stripe", "stripe_ball", "striped", "stripes"}:
        return "stripe"
    if n in {"cue_stick", "stick", "cue-stick"}:
        return "cue_stick"
    return "other"


def parse_resolution(text: str) -> Tuple[int, int]:
    w, h = str(text).lower().split("x", maxsplit=1)
    return int(w), int(h)


def default_inference_device() -> str:
    torch = preload_torch()
    if torch is None:
        return "cpu"
    try:
        if torch.cuda.is_available():
            return "0"
        if sys.platform == "darwin" and torch.backends.mps.is_available():
            return "mps"
    except Exception:
        return "cpu"
    return "cpu"


def fourcc_to_str(value: float | int | None) -> str:
    if value is None:
        return "unknown"
    try:
        iv = int(value)
    except Exception:
        return "unknown"
    chars = []
    for i in range(4):
        c = chr((iv >> (8 * i)) & 0xFF)
        chars.append(c if 32 <= ord(c) <= 126 else "?")
    return "".join(chars)


def monotonic_ns() -> int:
    import time

    return int(time.monotonic_ns())


def wall_time_id() -> str:
    import datetime as _dt

    return _dt.datetime.now().strftime("%Y%m%d_%H%M%S_%f")


def ensure_numpy_points(points: object) -> np.ndarray:
    arr = np.asarray(points, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[1] != 2:
        return np.zeros((0, 2), dtype=np.float32)
    return arr.reshape((-1, 2)).astype(np.float32)


def optional_import_error(package_name: str, action: str) -> RuntimeError:
    return RuntimeError(
        f"{action} requires optional package '{package_name}'. "
        "Install the matching dependency, then retry."
    )
