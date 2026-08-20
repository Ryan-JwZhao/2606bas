from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Optional

import numpy as np

from ..config import StateConfig
from ..schemas import TrackObservation, TracksFrame


OBJECT_GROUPS = {"solid", "stripe", "black"}
OBSERVABLE_PHASES = {"STABLE_IDLE", "PRE_SHOT_ARMED"}


@dataclass(frozen=True)
class BreakRackEvidence:
    is_rack: bool = False
    object_count: int = 0
    clustered_count: int = 0
    outlier_count: int = 0
    stable_frames: int = 0
    centroid: Optional[tuple[float, float]] = None
    normalized_centroid: Optional[tuple[float, float]] = None
    cluster_radius: float = 0.0
    coordinate_domain: str = "unknown"
    reasons: tuple[str, ...] = field(default_factory=tuple)

    def to_payload(self) -> dict[str, object]:
        return {
            "is_rack": bool(self.is_rack),
            "object_count": int(self.object_count),
            "clustered_count": int(self.clustered_count),
            "outlier_count": int(self.outlier_count),
            "stable_frames": int(self.stable_frames),
            "centroid": list(self.centroid) if self.centroid is not None else None,
            "normalized_centroid": (
                list(self.normalized_centroid) if self.normalized_centroid is not None else None
            ),
            "cluster_radius": float(self.cluster_radius),
            "coordinate_domain": str(self.coordinate_domain),
            "reasons": list(self.reasons),
        }


class BreakShotLifecycle:
    """Recognize a prepared rack and lock the break decision at shot start.

    Shot strength is intentionally absent.  A break is identified from the
    stable, compact object-ball rack that existed before the shot.  Position is
    only a broad table-relative prior, so normal rack placement error does not
    turn into a fixed-coordinate dependency.
    """

    version = "break_lifecycle_v2"

    def __init__(self, config: StateConfig):
        self.config = config
        self._inner_polygon_mm: list[tuple[float, float]] = []
        self._ball_diameter_mm = 57.15
        self._evidence_frames = 0
        self._armed = False
        self._completed = False
        self._active_shot_is_break: Optional[bool] = None
        self.last_evidence = BreakRackEvidence()

    @property
    def break_pending(self) -> bool:
        return bool(not self._completed and (self._armed or self._active_shot_is_break is True))

    @property
    def completed(self) -> bool:
        return bool(self._completed)

    @property
    def active_shot_is_break(self) -> bool:
        return bool(self._active_shot_is_break)

    def reset(self) -> None:
        self._evidence_frames = 0
        self._armed = False
        self._completed = False
        self._active_shot_is_break = None
        self.last_evidence = BreakRackEvidence()

    def set_table_context(
        self,
        *,
        inner_polygon_mm: Optional[Iterable[tuple[float, float]]] = None,
        ball_diameter_mm: Optional[float] = None,
    ) -> None:
        if inner_polygon_mm is not None:
            self._inner_polygon_mm = [(float(x), float(y)) for x, y in inner_polygon_mm]
        if ball_diameter_mm is not None and float(ball_diameter_mm) > 0:
            self._ball_diameter_mm = float(ball_diameter_mm)

    def observe(self, frame: TracksFrame, *, phase: object) -> BreakRackEvidence:
        phase_name = str(getattr(phase, "value", phase) or "").strip().upper()
        if self._completed or self._active_shot_is_break is not None or phase_name not in OBSERVABLE_PHASES:
            return self.last_evidence

        evidence = self._rack_evidence(frame.tracks)
        if evidence.is_rack:
            self._evidence_frames += 1
        else:
            self._evidence_frames = 0
        required = max(1, int(getattr(self.config, "break_rack_stable_frames", 3)))
        if self._evidence_frames >= required:
            # Keep the decision armed through short detector dropouts and the
            # first moments of motion once a stable rack has been observed.
            self._armed = True
        self.last_evidence = BreakRackEvidence(
            is_rack=evidence.is_rack,
            object_count=evidence.object_count,
            clustered_count=evidence.clustered_count,
            outlier_count=evidence.outlier_count,
            stable_frames=self._evidence_frames,
            centroid=evidence.centroid,
            normalized_centroid=evidence.normalized_centroid,
            cluster_radius=evidence.cluster_radius,
            coordinate_domain=evidence.coordinate_domain,
            reasons=evidence.reasons,
        )
        return self.last_evidence

    def mark_shot_started(self) -> bool:
        if self._active_shot_is_break is None:
            self._active_shot_is_break = bool(self._armed and not self._completed)
        return bool(self._active_shot_is_break)

    def mark_shot_resolved(self, *, valid: bool) -> None:
        if not valid:
            self._active_shot_is_break = None
            return
        self._completed = True
        self._armed = False
        self._evidence_frames = 0
        self._active_shot_is_break = None

    def snapshot(self) -> dict[str, object]:
        return {
            "version": self.version,
            "completed": bool(self._completed),
            "armed": bool(self._armed and not self._completed),
            "evidence_frames": int(self._evidence_frames),
            "last_evidence": self.last_evidence.to_payload(),
        }

    def restore(self, payload: object) -> None:
        data = dict(payload or {}) if isinstance(payload, dict) else {}
        self._completed = bool(data.get("completed", False))
        self._armed = bool(data.get("armed", False)) and not self._completed
        self._evidence_frames = max(0, int(data.get("evidence_frames", 0))) if self._armed else 0
        self._active_shot_is_break = None

    def debug_payload(self) -> dict[str, object]:
        payload = self.snapshot()
        payload["break_pending"] = self.break_pending
        payload["active_shot_is_break"] = self.active_shot_is_break
        return payload

    def _rack_evidence(self, tracks: list[TrackObservation]) -> BreakRackEvidence:
        minimum_quality = float(getattr(self.config, "break_rack_min_quality", 0.40))
        objects = [
            track
            for track in tracks
            if str(track.group).strip().lower() in OBJECT_GROUPS
            and track.visibility == "visible"
            and bool(track.confirmed)
            and float(track.quality) >= minimum_quality
        ]
        minimum_balls = max(3, int(getattr(self.config, "break_rack_min_object_balls", 10)))
        if len(objects) < minimum_balls:
            return BreakRackEvidence(
                object_count=len(objects),
                reasons=("insufficient_object_balls",),
            )

        use_mm = sum(track.center_mm is not None for track in objects) >= minimum_balls
        selected = [track for track in objects if track.center_mm is not None] if use_mm else objects
        points = np.asarray(
            [track.center_mm if use_mm else track.center_px for track in selected],
            dtype=np.float32,
        ).reshape((-1, 2))
        centroid_arr = np.median(points, axis=0)
        distances = np.linalg.norm(points - centroid_arr, axis=1)
        if use_mm:
            diameter = max(1.0, float(self._ball_diameter_mm))
            domain = "table_mm"
        else:
            diameters = [max(2.0, float(track.radius_px) * 2.0) for track in selected]
            diameter = max(2.0, float(np.median(np.asarray(diameters, dtype=np.float32))))
            domain = "camera_px"
        radius = diameter * max(1.0, float(getattr(self.config, "break_rack_cluster_radius_diameters", 3.8)))
        clustered = int(np.count_nonzero(distances <= radius))
        outliers = max(0, len(selected) - clustered)
        required_fraction = float(np.clip(getattr(self.config, "break_rack_cluster_fraction", 0.80), 0.5, 1.0))
        maximum_outliers = max(0, int(getattr(self.config, "break_rack_max_outlier_balls", 1)))
        compact = clustered >= minimum_balls and clustered / max(1, len(selected)) >= required_fraction
        complete = outliers <= maximum_outliers

        normalized = self._normalized_table_point(centroid_arr) if use_mm else None
        position_ok = normalized is None or self._in_broad_foot_spot_region(normalized)
        reasons: list[str] = []
        if compact:
            reasons.append("compact_object_cluster")
        else:
            reasons.append("object_cluster_too_wide")
        if complete:
            reasons.append("rack_complete")
        else:
            reasons.append("too_many_scattered_object_balls")
        if normalized is None:
            reasons.append("position_unavailable")
        elif position_ok:
            reasons.append("foot_spot_region")
        else:
            reasons.append("outside_foot_spot_region")
        return BreakRackEvidence(
            is_rack=bool(compact and complete and position_ok),
            object_count=len(selected),
            clustered_count=clustered,
            outlier_count=outliers,
            centroid=(float(centroid_arr[0]), float(centroid_arr[1])),
            normalized_centroid=normalized,
            cluster_radius=float(radius),
            coordinate_domain=domain,
            reasons=tuple(reasons),
        )

    def _normalized_table_point(self, point: np.ndarray) -> Optional[tuple[float, float]]:
        if len(self._inner_polygon_mm) < 3:
            return None
        polygon = np.asarray(self._inner_polygon_mm, dtype=np.float32).reshape((-1, 2))
        x_min, y_min = np.min(polygon, axis=0)
        x_max, y_max = np.max(polygon, axis=0)
        width = float(x_max - x_min)
        height = float(y_max - y_min)
        if width <= 1e-6 or height <= 1e-6:
            return None
        return (float((point[0] - x_min) / width), float((point[1] - y_min) / height))

    def _in_broad_foot_spot_region(self, point: tuple[float, float]) -> bool:
        x, y = point
        x_tolerance = max(0.05, float(getattr(self.config, "break_rack_foot_spot_x_tolerance", 0.20)))
        y_tolerance = max(0.05, float(getattr(self.config, "break_rack_center_y_tolerance", 0.28)))
        near_either_foot_spot = min(abs(x - 0.25), abs(x - 0.75)) <= x_tolerance
        return bool(near_either_foot_spot and abs(y - 0.50) <= y_tolerance)
