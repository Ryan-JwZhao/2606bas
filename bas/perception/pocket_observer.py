from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Optional

import cv2
import numpy as np

from ..pocket_evidence import (
    VISUAL_TERMINAL_ADVANCE_DIAMETERS,
    VISUAL_TERMINAL_DEPTH_DIAMETERS,
)
from ..schemas import (
    FramePacket,
    PocketVisualObservation,
    PocketVisualObservationFrame,
    TrackObservation,
    TracksFrame,
)
from ..utils import group_from_class
from .regions import BALL_GROUPS, DetectionRegionPolicy, PocketGuardRegion


RECENT_ASSOCIATION_BASE_DISTANCE_DIAMETERS = 6.0
RECENT_ASSOCIATION_MAX_DISTANCE_DIAMETERS = 12.0
RECENT_ASSOCIATION_TERMINAL_MARGIN_DIAMETERS = 2.0


@dataclass
class _PocketHistory:
    previous_gray: Optional[np.ndarray] = None
    background_image: Optional[np.ndarray] = None
    foreground_center: Optional[tuple[float, float]] = None
    foreground_stable_since_ns: Optional[int] = None
    clear_since_ns: Optional[int] = None
    inward_latched: bool = False
    inward_latched_since_ns: Optional[int] = None
    outward_streak: int = 0
    previous_track_centers: dict[int, tuple[float, float]] = field(default_factory=dict)
    votes: Deque[tuple[int, int, str, tuple[float, float], float]] = field(default_factory=deque)


class PocketObserver:
    """Cheap visual observer that only examines calibrated pocket guard ROIs.

    Frame difference supplies the independent motion gate.  Track motion and a
    short class-vote history determine direction and ball group; weak motion
    without a ball association is deliberately diagnostic-only.
    """

    version = "pocket_observer_v2"

    def __init__(self, *, history_ms: int = 1500) -> None:
        self.history_ms = max(100, int(history_ms))
        self._history: dict[int, _PocketHistory] = {}
        self._global_votes: Deque[tuple[int, int, str, tuple[float, float], float]] = deque()

    def reset(self) -> None:
        self._history.clear()
        self._global_votes.clear()

    def update(
        self,
        frame: FramePacket,
        tracks: TracksFrame,
        policy: DetectionRegionPolicy | None,
    ) -> PocketVisualObservationFrame:
        started = time.perf_counter()
        if frame.image is None or frame.image.size == 0 or policy is None:
            return self._frame(frame, [], started)
        self._record_global_tracks(frame.ts_cam_ns, tracks.tracks)
        table_center = _polygon_center(policy.ball_polygon, frame.image.shape)
        observations = [
            self._observe_guard(frame.image, frame.ts_cam_ns, tracks.tracks, guard, table_center)
            for guard in policy.ball_guard_regions
        ]
        return self._frame(frame, observations, started)

    def _observe_guard(
        self,
        image: np.ndarray,
        ts_ns: int,
        tracks: list[TrackObservation],
        guard: PocketGuardRegion,
        table_center: tuple[float, float],
    ) -> PocketVisualObservation:
        history = self._history.setdefault(guard.pocket_index, _PocketHistory())
        crop_image, mask, offset = _guard_crop(image, guard.polygon)
        crop = cv2.cvtColor(crop_image, cv2.COLOR_BGR2GRAY)
        current_track_centers: dict[int, tuple[float, float]] = {}
        current_visible_ids: set[int] = set()
        current_visible_tracks: list[TrackObservation] = []
        associated: list[TrackObservation] = []
        for track in tracks:
            group = _track_group(track)
            if group not in BALL_GROUPS or track.visibility != "visible" or float(track.quality) <= 0.25:
                continue
            current_visible_ids.add(int(track.track_id))
            current_visible_tracks.append(track)
            if _near_guard(track.center_px, guard):
                associated.append(track)
                current_track_centers[int(track.track_id)] = (float(track.center_px[0]), float(track.center_px[1]))
                history.votes.append(
                    (ts_ns, int(track.track_id), group, current_track_centers[int(track.track_id)], float(track.confidence))
                )
        cutoff = int(ts_ns) - self.history_ms * 1_000_000
        while history.votes and history.votes[0][0] < cutoff:
            history.votes.popleft()

        motion_score, motion_centers = _motion_measure(history.previous_gray, crop, mask, guard.ball_diameter_px)
        foreground_score, foreground_center, foreground_object_present = _foreground_measure(
            history.background_image,
            crop_image,
            mask,
            guard.ball_diameter_px,
        )
        axis = _unit_vector(np.asarray(guard.center_px, dtype=np.float32) - np.asarray(table_center, dtype=np.float32))
        directional: list[tuple[float, TrackObservation]] = []
        for track in associated:
            previous = history.previous_track_centers.get(int(track.track_id))
            if previous is None:
                previous = _recent_same_group_position(history.votes, ts_ns, int(track.track_id), _track_group(track))
            if previous is None:
                previous = self._recent_global_track_position(int(track.track_id), ts_ns)
            if previous is None:
                continue
            delta = np.asarray(track.center_px, dtype=np.float32) - np.asarray(previous, dtype=np.float32)
            directional.append((float(np.dot(delta, axis)), track))

        scale = max(1.0, float(guard.ball_diameter_px))
        track_inward = [
            track
            for advance, track in directional
            if advance >= scale * 0.25 and _point_in_entry_gate(track.center_px, guard, table_center)
        ]
        track_outward = [track for advance, track in directional if advance <= -scale * 0.25]
        foreground_advance = 0.0
        if foreground_center is not None and history.foreground_center is not None:
            foreground_delta = np.asarray(foreground_center, dtype=np.float32) - np.asarray(
                history.foreground_center,
                dtype=np.float32,
            )
            foreground_advance = float(np.dot(foreground_delta, axis))
        motion_gate = motion_score >= 0.035
        foreground_gate = foreground_score >= 0.08
        inward = bool((motion_gate or foreground_gate) and track_inward)
        outward = bool((motion_gate or foreground_gate) and track_outward)

        foreground_global = None
        if foreground_center is not None:
            foreground_global = (
                float(foreground_center[0] + offset[0]),
                float(foreground_center[1] + offset[1]),
            )
        motion_globals = [
            (float(center[0] + offset[0]), float(center[1] + offset[1]))
            for center in motion_centers
        ]

        visual_points: list[tuple[float, tuple[float, float]]] = []
        if foreground_gate and foreground_global is not None and _point_in_entry_gate(
            foreground_global, guard, table_center
        ):
            depth, _ = _entry_coordinates(foreground_global, guard, table_center)
            visual_points.append((float(depth), foreground_global))
        if motion_gate:
            for motion_global in motion_globals:
                if not _point_in_entry_gate(motion_global, guard, table_center):
                    continue
                depth, _ = _entry_coordinates(motion_global, guard, table_center)
                visual_points.append((float(depth), motion_global))
        visual_points.sort(key=lambda item: item[0], reverse=True)
        entry_depth_diameters = visual_points[0][0] / scale if visual_points else None
        visual_global = visual_points[0][1] if visual_points else None

        selected = track_inward or track_outward
        group = _vote_group(selected, history.votes, ts_ns)
        associated_ids = sorted({int(track.track_id) for track in selected})
        if not associated_ids and visual_global is not None:
            # Rank a current visible ball against a very recent disappeared
            # ball. A moving, spatially closer visible ball wins; otherwise a
            # disappearance owns the terminal motion instead of a stationary
            # neighbour at the same pocket.
            visible_candidates = {
                int(track.track_id): track
                for track in (*associated, *current_visible_tracks)
                if float(
                    np.linalg.norm(
                        np.asarray(track.center_px, dtype=np.float32)
                        - np.asarray(guard.center_px, dtype=np.float32)
                    )
                )
                <= scale * 3.0
            }
            nearby = sorted(
                (
                    (
                        float(
                            np.linalg.norm(
                                np.asarray(track.center_px, dtype=np.float32)
                                - np.asarray(visual_global, dtype=np.float32)
                            )
                        ),
                        track,
                    )
                    for track in visible_candidates.values()
                ),
                key=lambda item: (item[0], int(item[1].track_id)),
            )
            visible_choice = nearby[0] if nearby and nearby[0][0] <= scale * 2.5 else None
            fallback = self._recent_global_association(
                guard,
                table_center,
                ts_ns,
                excluded_track_ids=current_visible_ids,
                max_age_ms=450.0,
            )
            fallback_distance = None
            if fallback is not None:
                fallback_distance = min(
                    float(
                        np.linalg.norm(
                            np.asarray(fallback[3], dtype=np.float32) - np.asarray(point, dtype=np.float32)
                        )
                    )
                    for _, point in visual_points
                )
            prefer_visible = bool(
                visible_choice is not None
                and (
                    fallback is None
                    or (
                        not _track_is_stationary(visible_choice[1], scale)
                        and fallback_distance is not None
                        and visible_choice[0] + scale * 0.5 < fallback_distance
                    )
                )
            )
            selected_center: tuple[float, float] | None = None
            if prefer_visible and visible_choice is not None:
                selected = [visible_choice[1]]
                selected_center = tuple(map(float, visible_choice[1].center_px))
                group = _vote_group(selected, history.votes, ts_ns)
                associated_ids = [int(visible_choice[1].track_id)]
            elif fallback is not None:
                _, fallback_id, fallback_group, fallback_center, _ = fallback
                selected_center = fallback_center
                group = fallback_group
                associated_ids = [int(fallback_id)]
            elif visible_choice is not None:
                selected = [visible_choice[1]]
                selected_center = tuple(map(float, visible_choice[1].center_px))
                group = _vote_group(selected, history.votes, ts_ns)
                associated_ids = [int(visible_choice[1].track_id)]

            if selected_center is not None and visual_points:
                track_depth, _ = _entry_coordinates(selected_center, guard, table_center)
                terminal_depth = scale * VISUAL_TERMINAL_DEPTH_DIAMETERS
                minimum_advance = scale * VISUAL_TERMINAL_ADVANCE_DIAMETERS
                advanced = any(depth >= track_depth + minimum_advance for depth, _ in visual_points)
                terminal = any(depth >= terminal_depth for depth, _ in visual_points)
                inward = bool((motion_gate or foreground_gate) and advanced and terminal)
                fallback_outward = foreground_gate and foreground_advance <= -scale * 0.12
                history.outward_streak = history.outward_streak + 1 if fallback_outward else 0
                outward = bool(outward or history.outward_streak >= 2)
        if not associated_ids:
            selected = associated
            group = _vote_group(selected, history.votes, ts_ns)
            associated_ids = sorted({int(track.track_id) for track in selected})
        if track_outward:
            history.outward_streak = max(2, history.outward_streak)
        elif not outward and associated_ids:
            history.outward_streak = 0
        if not associated_ids and group is not None and motion_gate:
            recent = [row for row in history.votes if row[2] == group]
            associated_ids = sorted({int(row[1]) for row in recent[-3:]})
        # Motion is never promoted without a class-associated ball history.
        if group is None or not associated_ids:
            inward = False
            outward = False

        local_foreground_center = None
        foreground_depth_diameters = None
        if foreground_center is not None:
            local_foreground_center = np.asarray(foreground_center, dtype=np.float32) + np.asarray(offset, dtype=np.float32)
            foreground_depth, _ = _entry_coordinates(tuple(local_foreground_center), guard, table_center)
            foreground_depth_diameters = foreground_depth / scale
        foreground_stable = False
        if foreground_center is not None:
            if history.foreground_center is None or float(
                np.linalg.norm(np.asarray(foreground_center) - np.asarray(history.foreground_center))
            ) > scale * 0.12:
                history.foreground_stable_since_ns = ts_ns
            elif history.foreground_stable_since_ns is None:
                history.foreground_stable_since_ns = ts_ns
            foreground_stable = bool(
                history.foreground_stable_since_ns is not None
                and ts_ns - history.foreground_stable_since_ns >= 350_000_000
            )
        else:
            history.foreground_stable_since_ns = None
        lip_tracks = [
            track
            for track in associated
            if _track_is_stationary(track, scale)
            and _point_on_table_side_lip(track.center_px, guard, table_center)
        ]
        lip_track_ids = sorted({int(track.track_id) for track in lip_tracks})
        foreground_on_lip = bool(
            foreground_stable
            and foreground_object_present
            and local_foreground_center is not None
            and _point_on_table_side_lip(tuple(local_foreground_center), guard, table_center)
        )
        lip_occupied = bool(lip_track_ids or foreground_on_lip)
        clear = not associated and foreground_score < 0.05 and motion_score < 0.04
        if outward:
            history.inward_latched = False
            history.inward_latched_since_ns = None
        if clear:
            if history.clear_since_ns is None:
                history.clear_since_ns = ts_ns
            elif ts_ns - history.clear_since_ns >= 500_000_000:
                history.inward_latched = False
                history.inward_latched_since_ns = None
        else:
            history.clear_since_ns = None
        latch_expired = bool(
            history.inward_latched_since_ns is not None
            and ts_ns - history.inward_latched_since_ns >= self.history_ms * 1_000_000
        )
        emit_inward = bool(inward and (not history.inward_latched or latch_expired))
        if emit_inward:
            history.inward_latched = True
            history.inward_latched_since_ns = ts_ns
        sources: list[str] = []
        if motion_gate:
            sources.append("frame_difference")
            sources.append("ball_sized_motion")
        if foreground_gate:
            sources.append("foreground_motion")
        if history.votes:
            sources.append("track_history")
        if associated:
            sources.append("visible_track")

        if history.background_image is None or history.background_image.shape != crop_image.shape:
            history.background_image = crop_image.astype(np.float32)
        else:
            cv2.accumulateWeighted(crop_image, history.background_image, 0.01, mask=mask)
        history.previous_gray = crop.copy()
        history.foreground_center = foreground_center
        history.previous_track_centers = current_track_centers
        return PocketVisualObservation(
            pocket_index=int(guard.pocket_index),
            inward_crossing=emit_inward,
            outward_crossing=outward,
            lip_occupied=lip_occupied,
            clear=clear,
            group=group,
            confidence=float(min(0.98, 0.45 + motion_score + (0.25 if associated_ids else 0.0))),
            associated_track_ids=associated_ids,
            lip_track_ids=lip_track_ids,
            evidence_sources=sources,
            motion_score=float(motion_score),
            foreground_score=float(foreground_score),
            foreground_center_px=(
                (float(local_foreground_center[0]), float(local_foreground_center[1]))
                if local_foreground_center is not None
                else None
            ),
            foreground_depth_diameters=(
                float(foreground_depth_diameters) if foreground_depth_diameters is not None else None
            ),
            entry_depth_diameters=(float(entry_depth_diameters) if entry_depth_diameters is not None else None),
        )

    def _record_global_tracks(self, ts_ns: int, tracks: list[TrackObservation]) -> None:
        for track in tracks:
            group = _track_group(track)
            if group not in BALL_GROUPS or track.visibility != "visible" or float(track.quality) <= 0.25:
                continue
            self._global_votes.append(
                (
                    int(ts_ns),
                    int(track.track_id),
                    group,
                    (float(track.center_px[0]), float(track.center_px[1])),
                    float(track.confidence),
                )
            )
        cutoff = int(ts_ns) - self.history_ms * 1_000_000
        while self._global_votes and self._global_votes[0][0] < cutoff:
            self._global_votes.popleft()

    def _recent_global_association(
        self,
        guard: PocketGuardRegion,
        table_center: tuple[float, float],
        now_ns: int,
        *,
        excluded_track_ids: set[int] | None = None,
        max_age_ms: float | None = None,
    ) -> Optional[tuple[int, int, str, tuple[float, float], float]]:
        latest_by_track: dict[int, tuple[int, int, str, tuple[float, float], float]] = {}
        for row in self._global_votes:
            latest_by_track[int(row[1])] = row
        candidates: list[tuple[float, float, tuple[int, int, str, tuple[float, float], float]]] = []
        center = np.asarray(guard.center_px, dtype=np.float32)
        base_distance = max(
            1.0,
            float(guard.ball_diameter_px) * RECENT_ASSOCIATION_BASE_DISTANCE_DIAMETERS,
        )
        maximum_distance = max(
            base_distance,
            float(guard.ball_diameter_px) * RECENT_ASSOCIATION_MAX_DISTANCE_DIAMETERS,
        )
        excluded = excluded_track_ids or set()
        for row in latest_by_track.values():
            if int(row[1]) in excluded:
                continue
            age_ms = max(0.0, (int(now_ns) - int(row[0])) / 1_000_000.0)
            if age_ms > float(self.history_ms):
                continue
            if max_age_ms is not None and age_ms > float(max_age_ms):
                continue
            track_rows = [candidate for candidate in self._global_votes if int(candidate[1]) == int(row[1])]
            candidate_distance_limit = base_distance
            previous_distinct = next(
                (
                    candidate
                    for candidate in reversed(track_rows[:-1])
                    if float(
                        np.linalg.norm(
                            np.asarray(row[3], dtype=np.float32)
                            - np.asarray(candidate[3], dtype=np.float32)
                        )
                    )
                    >= float(guard.ball_diameter_px) * 0.25
                ),
                None,
            )
            if previous_distinct is not None:
                travel = np.asarray(row[3], dtype=np.float32) - np.asarray(previous_distinct[3], dtype=np.float32)
                travel_norm = float(np.linalg.norm(travel))
                if travel_norm > 1e-6:
                    heading = travel / travel_norm
                    remaining = center - np.asarray(row[3], dtype=np.float32)
                    forward = float(np.dot(remaining, heading))
                    lateral = float(abs(remaining[0] * heading[1] - remaining[1] * heading[0]))
                    if forward <= 0.0 or lateral > float(guard.ball_diameter_px) * 2.5:
                        continue
                    elapsed_ms = max(1e-3, (int(row[0]) - int(previous_distinct[0])) / 1_000_000.0)
                    measured_speed_px_ms = travel_norm / elapsed_ms
                    projected_reach = measured_speed_px_ms * age_ms
                    terminal_margin = (
                        float(guard.ball_diameter_px) * RECENT_ASSOCIATION_TERMINAL_MARGIN_DIAMETERS
                    )
                    candidate_distance_limit = min(
                        maximum_distance,
                        max(base_distance, projected_reach + terminal_margin),
                    )
            if age_ms > 300.0 and max_age_ms is None:
                previous = track_rows[-2] if len(track_rows) >= 2 else None
                if previous is None:
                    continue
                current_depth, _ = _entry_coordinates(row[3], guard, table_center)
                previous_depth, _ = _entry_coordinates(previous[3], guard, table_center)
                short_detection = len(track_rows) <= 2
                if not short_detection and current_depth < previous_depth + float(guard.ball_diameter_px) * 0.08:
                    continue
            distance = float(np.linalg.norm(np.asarray(row[3], dtype=np.float32) - center))
            if distance <= candidate_distance_limit:
                candidates.append((distance, age_ms, row))
        if not candidates:
            return None
        candidates.sort(key=lambda item: (item[0], item[1], item[2][1]))
        if len(candidates) > 1 and candidates[1][0] <= candidates[0][0] + float(guard.ball_diameter_px) * 0.5:
            return None
        return candidates[0][2]

    def _recent_global_track_position(self, track_id: int, now_ns: int) -> Optional[tuple[float, float]]:
        skipped_current = False
        for ts_ns, candidate_id, _, center, _ in reversed(self._global_votes):
            if candidate_id != track_id:
                continue
            if ts_ns >= now_ns and not skipped_current:
                skipped_current = True
                continue
            if ts_ns < now_ns:
                return center
        return None

    def _frame(
        self,
        frame: FramePacket,
        observations: list[PocketVisualObservation],
        started: float,
    ) -> PocketVisualObservationFrame:
        return PocketVisualObservationFrame(
            frame_id=frame.frame_id,
            ts_cam_ns=frame.ts_cam_ns,
            observations=observations,
            latency_ms=float((time.perf_counter() - started) * 1000.0),
            observer_version=self.version,
        )


def _guard_crop(image: np.ndarray, polygon: np.ndarray) -> tuple[np.ndarray, np.ndarray, tuple[int, int]]:
    height, width = image.shape[:2]
    x, y, w, h = cv2.boundingRect(np.asarray(polygon, dtype=np.float32).astype(np.int32))
    x1, y1 = max(0, x), max(0, y)
    x2, y2 = min(width, x + w), min(height, y + h)
    if x2 <= x1 or y2 <= y1:
        shape = (1, 1, image.shape[2]) if image.ndim == 3 else (1, 1)
        return np.zeros(shape, dtype=image.dtype), np.zeros((1, 1), dtype=np.uint8), (x1, y1)
    crop = image[y1:y2, x1:x2]
    local = np.asarray(polygon, dtype=np.float32) - np.asarray([x1, y1], dtype=np.float32)
    mask = np.zeros(crop.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [local.astype(np.int32)], 255)
    return crop, mask, (x1, y1)


def _motion_measure(
    previous: Optional[np.ndarray],
    current: np.ndarray,
    mask: np.ndarray,
    diameter_px: float,
) -> tuple[float, list[tuple[float, float]]]:
    if previous is None or previous.shape != current.shape or current.size <= 1:
        return 0.0, []
    diff = cv2.absdiff(previous, current)
    threshold = max(10.0, float(np.percentile(diff[mask > 0], 75)) if np.any(mask > 0) else 10.0)
    binary = np.where((diff >= threshold) & (mask > 0), 255, 0).astype(np.uint8)
    kernel = np.ones((3, 3), dtype=np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    ball_area = max(1.0, np.pi * (max(1.0, float(diameter_px)) * 0.5) ** 2)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates: list[tuple[float, tuple[float, float]]] = []
    for contour in contours:
        area = float(cv2.contourArea(contour))
        if area < ball_area * 0.03 or area > ball_area * 6.0:
            continue
        _, _, width, height = cv2.boundingRect(contour)
        aspect = max(width, height) / max(1.0, min(width, height))
        if aspect > 8.0 or min(width, height) < max(2.0, float(diameter_px) * 0.10):
            continue
        moments = cv2.moments(contour)
        if abs(float(moments["m00"])) <= 1e-6:
            continue
        center = (float(moments["m10"] / moments["m00"]), float(moments["m01"] / moments["m00"]))
        candidates.append((area, center))
    if not candidates:
        return 0.0, []
    candidates.sort(key=lambda item: item[0], reverse=True)
    area = candidates[0][0]
    return min(1.0, area / ball_area), [center for _, center in candidates]


def _foreground_measure(
    background: Optional[np.ndarray],
    current: np.ndarray,
    mask: np.ndarray,
    diameter_px: float,
) -> tuple[float, Optional[tuple[float, float]], bool]:
    if background is None or background.shape != current.shape or current.size <= 1:
        return 0.0, None, False
    background_u8 = background.astype(np.uint8)
    color_diff = cv2.absdiff(background_u8, current)
    diff = color_diff
    if diff.ndim == 3:
        diff = np.max(diff, axis=2)
    binary = np.where((diff >= 18) & (mask > 0), 255, 0).astype(np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, np.ones((3, 3), dtype=np.uint8))
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    ball_area = max(1.0, np.pi * (max(1.0, float(diameter_px)) * 0.5) ** 2)
    candidates: list[tuple[float, tuple[float, float], np.ndarray]] = []
    for contour in contours:
        area = float(cv2.contourArea(contour))
        if area < ball_area * 0.08 or area > ball_area * 3.2:
            continue
        x, y, width, height = cv2.boundingRect(contour)
        aspect = max(width, height) / max(1.0, min(width, height))
        if aspect > 3.2:
            continue
        moments = cv2.moments(contour)
        if abs(float(moments["m00"])) <= 1e-6:
            continue
        center = (float(moments["m10"] / moments["m00"]), float(moments["m01"] / moments["m00"]))
        candidates.append((area, center, contour))
    if not candidates:
        return 0.0, None, False
    area, center, contour = max(candidates, key=lambda item: item[0])
    current_object_present = _foreground_represents_current_object(
        background_u8,
        current,
        mask,
        contour,
    )
    return min(1.0, area / ball_area), center, current_object_present


def _foreground_represents_current_object(
    background: np.ndarray,
    current: np.ndarray,
    mask: np.ndarray,
    contour: np.ndarray,
) -> bool:
    contour_mask = np.zeros(mask.shape, dtype=np.uint8)
    cv2.drawContours(contour_mask, [contour], -1, 255, thickness=-1)
    object_pixels = contour_mask > 0
    felt_pixels = (mask > 0) & ~object_pixels
    if not np.any(object_pixels) or np.count_nonzero(felt_pixels) < 16:
        return False

    if current.ndim == 3:
        felt_color = np.median(current[felt_pixels], axis=0).astype(np.float32)
        current_color = np.median(current[object_pixels], axis=0).astype(np.float32)
        background_color = np.median(background[object_pixels], axis=0).astype(np.float32)
    else:
        felt_color = np.asarray([np.median(current[felt_pixels])], dtype=np.float32)
        current_color = np.asarray([np.median(current[object_pixels])], dtype=np.float32)
        background_color = np.asarray([np.median(background[object_pixels])], dtype=np.float32)
    current_distance = float(np.linalg.norm(current_color - felt_color))
    background_distance = float(np.linalg.norm(background_color - felt_color))
    return current_distance >= background_distance + 8.0


def _near_guard(point: tuple[float, float], guard: PocketGuardRegion) -> bool:
    inside = cv2.pointPolygonTest(
        np.asarray(guard.polygon, dtype=np.float32).reshape((-1, 1, 2)),
        (float(point[0]), float(point[1])),
        False,
    )
    if inside >= 0:
        return True
    distance = float(np.linalg.norm(np.asarray(point, dtype=np.float32) - np.asarray(guard.center_px, dtype=np.float32)))
    return distance <= max(1.0, float(guard.ball_diameter_px) * 2.25)


def _point_in_entry_gate(
    point: tuple[float, float],
    guard: PocketGuardRegion,
    table_center: tuple[float, float],
) -> bool:
    depth, lateral = _entry_coordinates(point, guard, table_center)
    diameter = max(1.0, float(guard.ball_diameter_px))
    return depth >= -diameter * 1.05 and lateral <= diameter * 1.55


def _entry_coordinates(
    point: tuple[float, float],
    guard: PocketGuardRegion,
    table_center: tuple[float, float],
) -> tuple[float, float]:
    axis = _unit_vector(np.asarray(guard.center_px, dtype=np.float32) - np.asarray(table_center, dtype=np.float32))
    tangent = np.asarray([-axis[1], axis[0]], dtype=np.float32)
    offset = np.asarray(point, dtype=np.float32) - np.asarray(guard.center_px, dtype=np.float32)
    depth = float(np.dot(offset, axis))
    lateral = abs(float(np.dot(offset, tangent)))
    return depth, lateral


def _point_on_table_side_lip(
    point: tuple[float, float],
    guard: PocketGuardRegion,
    table_center: tuple[float, float],
) -> bool:
    depth, lateral = _entry_coordinates(point, guard, table_center)
    diameter = max(1.0, float(guard.ball_diameter_px))
    return -diameter * 2.4 <= depth <= -diameter * 0.08 and lateral <= diameter * 1.5


def _track_group(track: TrackObservation) -> str:
    group = str(track.group or "").strip().lower()
    if group in BALL_GROUPS:
        return group
    return str(group_from_class(track.cls_name) or "").strip().lower()


def _recent_same_group_position(
    votes: Deque[tuple[int, int, str, tuple[float, float], float]],
    now_ns: int,
    track_id: int,
    group: str,
) -> Optional[tuple[float, float]]:
    for ts_ns, previous_id, previous_group, center, _ in reversed(votes):
        if ts_ns >= now_ns:
            continue
        if previous_id == track_id or previous_group == group:
            return center
    return None


def _vote_group(
    tracks: list[TrackObservation],
    votes: Deque[tuple[int, int, str, tuple[float, float], float]],
    now_ns: int,
) -> Optional[str]:
    scores: dict[str, float] = {}
    for track in tracks:
        group = _track_group(track)
        if group in BALL_GROUPS:
            scores[group] = scores.get(group, 0.0) + max(0.1, float(track.confidence)) * 2.0
    if scores:
        return max(scores, key=scores.get)
    for ts_ns, _, group, _, confidence in votes:
        age_ms = max(0.0, (now_ns - ts_ns) / 1_000_000.0)
        scores[group] = scores.get(group, 0.0) + max(0.05, confidence) * max(0.05, 1.0 - age_ms / 1500.0)
    return max(scores, key=scores.get) if scores else None


def _track_is_stationary(track: TrackObservation, diameter_px: float) -> bool:
    speed = float(np.hypot(*track.velocity_px_s))
    return speed <= max(3.0, diameter_px * 0.75)


def _polygon_center(polygon: Optional[np.ndarray], shape: tuple[int, ...]) -> tuple[float, float]:
    if polygon is None or np.asarray(polygon).size < 6:
        return (float(shape[1]) * 0.5, float(shape[0]) * 0.5)
    points = np.asarray(polygon, dtype=np.float32).reshape((-1, 2))
    return (float(np.mean(points[:, 0])), float(np.mean(points[:, 1])))


def _unit_vector(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    return vector / norm if norm > 1e-6 else np.asarray([0.0, -1.0], dtype=np.float32)


__all__ = ["PocketObserver"]
