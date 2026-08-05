from __future__ import annotations

import hashlib
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
    boundary_segments_norm: List[np.ndarray] = field(default_factory=list)
    boundary_source_count: int = 0
    boundary_complete: bool = True
    boundary_self_intersections: int = 0

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

    def boundary_scaled(self, width: int, height: int) -> Tuple[np.ndarray, List[np.ndarray]]:
        def scale(arr: np.ndarray) -> np.ndarray:
            if arr.size == 0:
                return arr.copy()
            out = arr.copy().astype(np.float32)
            out[:, 0] *= float(width)
            out[:, 1] *= float(height)
            return out

        return scale(self.inner_norm), [scale(seg) for seg in self.boundary_segments_norm]


def table_geometry_fingerprint(geometry: TableGeometry) -> str:
    """Return a stable identity for the geometry that affects runtime decisions."""

    digest = hashlib.sha256(b"bas-table-geometry-v2-endpoint-snap")
    groups = (
        ("outer", [geometry.outer_norm]),
        ("inline", geometry.inline_norm),
        ("pockets", geometry.pockets_norm),
    )
    for label, arrays in groups:
        digest.update(label.encode("ascii"))
        digest.update(len(arrays).to_bytes(4, "little"))
        for value in arrays:
            points = np.asarray(value, dtype="<f4").reshape((-1, 2))
            digest.update(points.shape[0].to_bytes(4, "little"))
            digest.update(points.tobytes(order="C"))
    return f"table_geometry_v2:{digest.hexdigest()[:20]}"


@dataclass(frozen=True)
class _GeometrySource:
    """One LabelMe document with its own coordinate domain and UI role hint."""

    role_hint: str
    data: Dict[str, Any]
    width: float
    height: float


class TableGeometryLoader:
    @classmethod
    def load_optional(cls, outline_path: Optional[str], inline_path: Optional[str], pocket_path: Optional[str]) -> TableGeometry:
        """Load configured geometry, returning empty only when no paths were configured."""

        if not any([outline_path, inline_path, pocket_path]):
            return TableGeometry()
        return cls.load(outline_path, inline_path, pocket_path)

    @classmethod
    def load(cls, outline_path: Optional[str], inline_path: Optional[str], pocket_path: Optional[str]) -> TableGeometry:
        sources = cls._load_sources(outline_path, inline_path, pocket_path)

        # Geometry identity comes from the LabelMe labels, not from whichever
        # file-picker row supplied the document.  The role hint only controls
        # precedence, so accidentally swapping inline/pocket rows cannot blank
        # the whole runtime geometry.
        outer = cls._first_shape_from_sources(sources, "outline", preferred_role="outline")
        if outer.shape[0] < 3:
            outline_source = cls._preferred_source(sources, "outline")
            if outline_source is not None and not cls._contains_geometry_label(outline_source.data):
                outer = cls._first_any_shape(
                    outline_source.data,
                    outline_source.width,
                    outline_source.height,
                )

        inline_lines = cls._all_shapes_from_sources(sources, "inline", preferred_role="inline")
        pockets = cls._all_shapes_from_sources(sources, "pocket", preferred_role="pocket")
        pocket_order = canonical_pocket_indices(pockets)
        pockets = [pockets[index] for index in pocket_order]

        # Some field annotations draw one continuous rail through a middle
        # pocket and provide the pocket lip as a separate curve.  Cut that rail
        # at the pocket endpoints so the real lip participates in the closed
        # table boundary instead of being silently discarded by the stitcher.
        inline_lines = cls._split_inline_lines_at_embedded_pockets(inline_lines, pockets)

        inner = np.zeros((0, 2), dtype=np.float32)
        boundary_segments: List[np.ndarray] = []
        boundary_self_intersections = 0
        stitch_lines = [*inline_lines, *pockets]
        boundary_complete = not stitch_lines
        if stitch_lines:
            inner, boundary_segments = cls._stitch_lines_to_polygon(stitch_lines)
            boundary_complete = bool(
                inner.shape[0] >= 3 and len(boundary_segments) == len(stitch_lines)
            )
            if inner.shape[0] >= 3:
                boundary_self_intersections = cls._count_polygon_self_intersections(inner)
            if inner.shape[0] < 3:
                stack = np.vstack(stitch_lines).astype(np.float32)
                hull = cv2.convexHull((stack * np.array([1000.0, 1000.0], dtype=np.float32)).astype(np.float32))
                inner = hull.reshape((-1, 2)).astype(np.float32) / np.array([1000.0, 1000.0], dtype=np.float32)
        if inner.shape[0] < 3:
            inner = outer.copy()
        return TableGeometry(
            outer_norm=outer,
            inner_norm=inner,
            inline_norm=inline_lines,
            pockets_norm=pockets,
            boundary_segments_norm=boundary_segments,
            boundary_source_count=len(stitch_lines),
            boundary_complete=boundary_complete,
            boundary_self_intersections=boundary_self_intersections,
        )

    @classmethod
    def _load_sources(
        cls,
        outline_path: Optional[str],
        inline_path: Optional[str],
        pocket_path: Optional[str],
    ) -> List[_GeometrySource]:
        loaded: List[tuple[str, Dict[str, Any]]] = []
        seen_paths: set[str] = set()
        for role_hint, path in (
            ("outline", outline_path),
            ("inline", inline_path),
            ("pocket", pocket_path),
        ):
            if not path:
                continue
            canonical_path = str(Path(path).resolve()).casefold()
            if canonical_path in seen_paths:
                continue
            seen_paths.add(canonical_path)
            loaded.append((role_hint, cls._load_json(path)))

        fallback_width = float(
            next((data.get("imageWidth") for _, data in loaded if data.get("imageWidth")), 1920)
        )
        fallback_height = float(
            next((data.get("imageHeight") for _, data in loaded if data.get("imageHeight")), 1080)
        )
        return [
            _GeometrySource(
                role_hint=role_hint,
                data=data,
                width=float(data.get("imageWidth") or fallback_width),
                height=float(data.get("imageHeight") or fallback_height),
            )
            for role_hint, data in loaded
        ]

    @staticmethod
    def _preferred_source(sources: List[_GeometrySource], role: str) -> Optional[_GeometrySource]:
        return next((source for source in sources if source.role_hint == role), None)

    @staticmethod
    def _sources_in_role_order(sources: List[_GeometrySource], preferred_role: str) -> List[_GeometrySource]:
        return sorted(sources, key=lambda source: source.role_hint != preferred_role)

    @classmethod
    def _first_shape_from_sources(
        cls,
        sources: List[_GeometrySource],
        label_key: str,
        *,
        preferred_role: str,
    ) -> np.ndarray:
        for source in cls._sources_in_role_order(sources, preferred_role):
            points = cls._first_shape_with_label(source.data, label_key, source.width, source.height)
            if points.shape[0] > 0:
                return points
        return np.zeros((0, 2), dtype=np.float32)

    @classmethod
    def _all_shapes_from_sources(
        cls,
        sources: List[_GeometrySource],
        label_key: str,
        *,
        preferred_role: str,
    ) -> List[np.ndarray]:
        for source in cls._sources_in_role_order(sources, preferred_role):
            shapes = cls._all_shapes_with_label(source.data, label_key, source.width, source.height)
            if shapes:
                return shapes
        return []

    @staticmethod
    def _contains_geometry_label(data: Dict[str, Any]) -> bool:
        for shape in data.get("shapes", []):
            if not isinstance(shape, dict):
                continue
            label = str(shape.get("label", "")).lower()
            if any(key in label for key in ("outline", "inline", "pocket")):
                return True
        return False

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

    @classmethod
    def _split_inline_lines_at_embedded_pockets(
        cls,
        inline_lines: List[np.ndarray],
        pockets: List[np.ndarray],
        join_eps: float = 0.03,
    ) -> List[np.ndarray]:
        fragments = [np.asarray(line, dtype=np.float32).reshape((-1, 2)).copy() for line in inline_lines]
        for pocket in pockets:
            curve = np.asarray(pocket, dtype=np.float32).reshape((-1, 2))
            if curve.shape[0] < 2:
                continue
            endpoints = (curve[0], curve[-1])
            best: Optional[
                tuple[
                    float,
                    int,
                    tuple[float, float, int, float],
                    tuple[float, float, int, float],
                ]
            ] = None
            for index, line in enumerate(fragments):
                if line.shape[0] < 2:
                    continue
                first = cls._project_point_to_polyline(endpoints[0], line)
                second = cls._project_point_to_polyline(endpoints[1], line)
                score = max(first[0], second[0])
                if score > float(join_eps):
                    continue
                if abs(first[1] - second[1]) <= float(join_eps) * 0.25:
                    continue
                candidate = (score, index, first, second)
                if best is None or candidate[0] < best[0]:
                    best = candidate
            if best is None:
                continue

            _, index, first, second = best
            line = fragments[index]
            if first[1] <= second[1]:
                early_endpoint, early_projection = endpoints[0], first
                late_endpoint, late_projection = endpoints[1], second
            else:
                early_endpoint, early_projection = endpoints[1], second
                late_endpoint, late_projection = endpoints[0], first
            prefix = cls._polyline_prefix_to_endpoint(line, early_projection, early_endpoint)
            suffix = cls._polyline_suffix_from_endpoint(line, late_projection, late_endpoint)
            replacements = [part for part in (prefix, suffix) if part.shape[0] >= 2]
            if len(replacements) == 2:
                fragments[index : index + 1] = replacements
        return fragments

    @staticmethod
    def _project_point_to_polyline(
        point: np.ndarray,
        line: np.ndarray,
    ) -> tuple[float, float, int, float]:
        best_distance = float("inf")
        best_arc = 0.0
        best_index = 0
        best_t = 0.0
        cumulative = 0.0
        for index in range(line.shape[0] - 1):
            start = line[index]
            delta = line[index + 1] - start
            length = float(np.linalg.norm(delta))
            if length <= 1e-9:
                continue
            t = float(np.clip(np.dot(point - start, delta) / (length * length), 0.0, 1.0))
            projected = start + t * delta
            distance = float(np.linalg.norm(point - projected))
            if distance < best_distance:
                best_distance = distance
                best_arc = cumulative + t * length
                best_index = index
                best_t = t
            cumulative += length
        return best_distance, best_arc, best_index, best_t

    @staticmethod
    def _polyline_prefix_to_endpoint(
        line: np.ndarray,
        projection: tuple[float, float, int, float],
        endpoint: np.ndarray,
    ) -> np.ndarray:
        segment_index = int(projection[2])
        points = [*line[: segment_index + 1], np.asarray(endpoint, dtype=np.float32)]
        return np.asarray(points, dtype=np.float32).reshape((-1, 2))

    @staticmethod
    def _polyline_suffix_from_endpoint(
        line: np.ndarray,
        projection: tuple[float, float, int, float],
        endpoint: np.ndarray,
    ) -> np.ndarray:
        segment_index = int(projection[2])
        points = [np.asarray(endpoint, dtype=np.float32), *line[segment_index + 1 :]]
        return np.asarray(points, dtype=np.float32).reshape((-1, 2))

    @staticmethod
    def _stitch_lines_to_polygon(lines: List[np.ndarray], join_eps: float = 0.03) -> Tuple[np.ndarray, List[np.ndarray]]:
        candidates = [np.asarray(line, dtype=np.float32).reshape((-1, 2)) for line in lines if np.asarray(line).size >= 4]
        if not candidates:
            return np.zeros((0, 2), dtype=np.float32), []
        best = np.zeros((0, 2), dtype=np.float32)
        best_used = -1
        best_cost = float("inf")
        best_parts: List[np.ndarray] = []
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
                    parts.append(nxt.copy())
                    end = nxt[-1]
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
                    best_parts = [part.copy() for part in parts]
        if best.shape[0] >= 3 and float(np.linalg.norm(best[0] - best[-1])) <= join_eps:
            snapped_parts = TableGeometryLoader._snap_closed_segment_endpoints(best_parts)
            merged = np.vstack(
                [snapped_parts[0], *[part[1:] for part in snapped_parts[1:]]]
            ).astype(np.float32)
            return merged, snapped_parts
        return np.zeros((0, 2), dtype=np.float32), []

    @staticmethod
    def _snap_closed_segment_endpoints(parts: List[np.ndarray]) -> List[np.ndarray]:
        """Make stitched joins exact without creating short loops at annotation gaps."""

        snapped = [np.asarray(part, dtype=np.float32).reshape((-1, 2)).copy() for part in parts]
        for index in range(len(snapped)):
            next_index = (index + 1) % len(snapped)
            joint = (snapped[index][-1] + snapped[next_index][0]) * np.float32(0.5)
            snapped[index][-1] = joint
            snapped[next_index][0] = joint
        return snapped

    @staticmethod
    def _count_polygon_self_intersections(polygon: np.ndarray) -> int:
        points = np.asarray(polygon, dtype=np.float64).reshape((-1, 2))
        count = 0
        size = points.shape[0]
        for first in range(size):
            a = points[first]
            b = points[(first + 1) % size]
            for second in range(first + 1, size):
                if second == first or second == (first + 1) % size:
                    continue
                if (second + 1) % size == first:
                    continue
                if first == 0 and second == size - 1:
                    continue
                c = points[second]
                d = points[(second + 1) % size]
                if TableGeometryLoader._segments_properly_intersect(a, b, c, d):
                    count += 1
        return count

    @staticmethod
    def _segments_properly_intersect(
        a: np.ndarray,
        b: np.ndarray,
        c: np.ndarray,
        d: np.ndarray,
        epsilon: float = 1e-10,
    ) -> bool:
        def cross(start: np.ndarray, end: np.ndarray, point: np.ndarray) -> float:
            edge = end - start
            offset = point - start
            return float(edge[0] * offset[1] - edge[1] * offset[0])

        ab_c = cross(a, b, c)
        ab_d = cross(a, b, d)
        cd_a = cross(c, d, a)
        cd_b = cross(c, d, b)
        return bool(
            ab_c * ab_d < -float(epsilon)
            and cd_a * cd_b < -float(epsilon)
        )


def pocket_arc_center(points: np.ndarray, expected_diameter: float) -> np.ndarray:
    """Return the circle center represented by an annotated pocket-jaw arc.

    Pocket annotations trace the cloth-facing jaw.  Their point mean is
    therefore biased toward the table and is not the physical pocket center.
    Legacy straight or irregular annotations retain the old point-mean
    behavior through conservative fit-quality gates.
    """

    curve = np.asarray(points, dtype=np.float64).reshape((-1, 2))
    if curve.shape[0] == 0:
        return np.zeros((2,), dtype=np.float32)
    fallback = np.mean(curve, axis=0)
    if curve.shape[0] < 3:
        return fallback.astype(np.float32)

    x = curve[:, 0]
    y = curve[:, 1]
    design = np.column_stack((2.0 * x, 2.0 * y, np.ones(curve.shape[0], dtype=np.float64)))
    target = x * x + y * y
    try:
        solution, _, rank, _ = np.linalg.lstsq(design, target, rcond=None)
    except np.linalg.LinAlgError:
        return fallback.astype(np.float32)
    if int(rank) < 3 or not np.all(np.isfinite(solution)):
        return fallback.astype(np.float32)

    center = solution[:2]
    radius_sq = float(solution[2] + np.dot(center, center))
    if not np.isfinite(radius_sq) or radius_sq <= 0.0:
        return fallback.astype(np.float32)
    radius = float(np.sqrt(radius_sq))
    radial_error = np.linalg.norm(curve - center, axis=1) - radius
    rmse = float(np.sqrt(np.mean(radial_error * radial_error)))
    diameter = max(1.0, float(expected_diameter))
    plausible_radius = diameter * 0.45 <= radius <= diameter * 2.0
    plausible_offset = float(np.linalg.norm(center - fallback)) <= diameter * 1.5
    accurate_arc = rmse <= max(1.5, radius * 0.08)
    if not (plausible_radius and plausible_offset and accurate_arc):
        return fallback.astype(np.float32)
    return center.astype(np.float32)


def canonical_pocket_indices(pockets: List[np.ndarray]) -> List[int]:
    """Return stable TL, TM, TR, BR, BM, BL indices for a six-pocket table."""

    if len(pockets) != 6:
        return list(range(len(pockets)))
    valid = [
        index
        for index, curve in enumerate(pockets)
        if np.asarray(curve, dtype=np.float32).reshape((-1, 2)).shape[0] >= 2
    ]
    if len(valid) != 6:
        return list(range(len(pockets)))
    ranked = sorted(
        valid,
        key=lambda index: float(np.mean(np.asarray(pockets[index], dtype=np.float32)[:, 1])),
    )
    top = sorted(
        ranked[:3],
        key=lambda index: float(np.mean(np.asarray(pockets[index], dtype=np.float32)[:, 0])),
    )
    bottom = sorted(
        ranked[3:],
        key=lambda index: float(np.mean(np.asarray(pockets[index], dtype=np.float32)[:, 0])),
        reverse=True,
    )
    return [*top, *bottom]
