from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

from ..schemas import Point, ShotCandidate


STATIONARY_ROUTE_PHASES = {"STABLE_IDLE", "PRE_SHOT_ARMED"}


@dataclass(frozen=True)
class BallPositionMeasurement:
    """One current-frame ball position in the table-coordinate planning domain."""

    track_id: int
    center_mm: Point
    velocity_mm_s: Point = (0.0, 0.0)
    quality: float = 1.0
    uncertainty_mm: float = 0.0


@dataclass
class _PositionState:
    position_mm: np.ndarray
    velocity_mm_s: np.ndarray
    last_measurement_mm: np.ndarray
    last_ts_ns: int


class PlanningPositionStabilizer:
    """Filter planning inputs while preserving a fresh result on every frame.

    The module never stores or returns a route.  It only estimates current ball
    centres; all ghost-ball, collision, pocket-clearance and overlay geometry is
    recomputed by the planner from the returned positions.
    """

    version = "planning_position_stability_v1"

    def __init__(self, config) -> None:
        self.config = config
        self._states: dict[int, _PositionState] = {}
        self._last_update_ts_ns: int | None = None
        self.last_status = "idle"

    def reset(self) -> None:
        self._states.clear()
        self._last_update_ts_ns = None
        self.last_status = "reset"

    def update(
        self,
        ts_cam_ns: int,
        measurements: Sequence[BallPositionMeasurement],
        *,
        phase: str = "STABLE_IDLE",
    ) -> dict[int, np.ndarray]:
        ts_ns = int(ts_cam_ns)
        if not bool(getattr(self.config, "route_stability_enabled", True)):
            self.reset()
            self.last_status = "disabled"
            return {
                int(item.track_id): _finite_point(item.center_mm)
                for item in measurements
            }

        if self._last_update_ts_ns is not None and ts_ns < self._last_update_ts_ns:
            self.reset()
            self.last_status = "timestamp_rewind"

        stationary_phase = str(phase or "").strip().upper() in STATIONARY_ROUTE_PHASES
        shown: dict[int, np.ndarray] = {}
        visible_ids: set[int] = set()
        seeded = 0
        reset_count = 0
        motion_count = 0

        for measurement in measurements:
            track_id = int(measurement.track_id)
            visible_ids.add(track_id)
            observed = _finite_point(measurement.center_mm)
            measured_velocity = _finite_point(measurement.velocity_mm_s, fallback=(0.0, 0.0))
            state = self._states.get(track_id)
            if state is None:
                self._states[track_id] = _PositionState(
                    position_mm=observed.copy(),
                    velocity_mm_s=measured_velocity.copy(),
                    last_measurement_mm=observed.copy(),
                    last_ts_ns=ts_ns,
                )
                shown[track_id] = observed.copy()
                seeded += 1
                continue

            dt_s = (ts_ns - int(state.last_ts_ns)) / 1_000_000_000.0
            reset_gap_s = max(0.001, float(getattr(self.config, "route_stability_reset_gap_ms", 500.0)) / 1000.0)
            innovation = observed - state.position_mm
            innovation_mm = float(np.linalg.norm(innovation))
            reset_distance_mm = max(1.0, float(getattr(self.config, "route_stability_reset_distance_mm", 80.0)))
            if dt_s > reset_gap_s or innovation_mm >= reset_distance_mm:
                state.position_mm = observed.copy()
                state.velocity_mm_s = measured_velocity.copy()
                state.last_measurement_mm = observed.copy()
                state.last_ts_ns = ts_ns
                shown[track_id] = observed.copy()
                reset_count += 1
                continue
            if dt_s <= 0.0:
                shown[track_id] = self._predicted_output(
                    state,
                    measured_speed=float(np.linalg.norm(measured_velocity)),
                    motion_factor=0.0,
                    stationary_phase=stationary_phase,
                )
                continue

            raw_velocity = (observed - state.last_measurement_mm) / dt_s
            measured_speed = float(np.linalg.norm(measured_velocity))
            deadband_mm = max(0.0, float(getattr(self.config, "route_stability_deadband_mm", 2.5)))
            response_distance_mm = max(
                deadband_mm + 1e-3,
                float(getattr(self.config, "route_stability_response_distance_mm", 7.0)),
            )
            quiet_speed = max(0.0, float(getattr(self.config, "route_stability_quiet_speed_mm_s", 12.0)))
            fast_speed = max(
                quiet_speed + 1e-3,
                float(getattr(self.config, "route_stability_fast_speed_mm_s", 160.0)),
            )
            speed_factor = _smoothstep(quiet_speed, fast_speed, measured_speed)
            distance_factor = _smoothstep(deadband_mm, response_distance_mm, innovation_mm)
            motion_factor = max(speed_factor, distance_factor)
            if not stationary_phase:
                motion_factor = 1.0
            if motion_factor > 0.01:
                motion_count += 1

            stationary_tau_s = max(
                0.01,
                float(getattr(self.config, "route_stability_stationary_tau_ms", 3000.0)) / 1000.0,
            )
            motion_tau_s = max(
                0.005,
                float(getattr(self.config, "route_stability_motion_tau_ms", 60.0)) / 1000.0,
            )
            tau_s = math.exp(
                math.log(stationary_tau_s) * (1.0 - motion_factor)
                + math.log(motion_tau_s) * motion_factor
            )
            quality = float(np.clip(float(measurement.quality), 0.0, 1.0))
            uncertainty = max(0.0, float(measurement.uncertainty_mm))
            if motion_factor < 1.0:
                tau_s *= 1.0 + 0.25 * (1.0 - quality) + 0.04 * min(5.0, uncertainty)
            position_gain = _time_gain(dt_s, tau_s)
            if stationary_phase and measured_speed <= quiet_speed and innovation_mm <= deadband_mm:
                micro_gain = float(np.clip(float(getattr(self.config, "route_stability_micro_gain", 0.08)), 1e-4, 1.0))
                position_gain *= micro_gain
            state.position_mm = state.position_mm + position_gain * innovation

            velocity_target = measured_velocity
            if not np.all(np.isfinite(velocity_target)):
                velocity_target = raw_velocity
            if stationary_phase and measured_speed <= quiet_speed and innovation_mm <= deadband_mm:
                velocity_target = np.zeros((2,), dtype=np.float64)
            velocity_tau_s = 0.12 if motion_factor >= 0.5 else 0.45
            velocity_gain = _time_gain(dt_s, velocity_tau_s)
            state.velocity_mm_s = state.velocity_mm_s + velocity_gain * (velocity_target - state.velocity_mm_s)
            state.last_measurement_mm = observed.copy()
            state.last_ts_ns = ts_ns
            shown[track_id] = self._predicted_output(
                state,
                measured_speed=measured_speed,
                motion_factor=motion_factor,
                stationary_phase=stationary_phase,
            )

        self._prune_missing(ts_ns, visible_ids)
        self._last_update_ts_ns = ts_ns
        self.last_status = (
            f"continuous visible={len(shown)} motion={motion_count} "
            f"seed={seeded} reset={reset_count}"
        )
        return shown

    def _predicted_output(
        self,
        state: _PositionState,
        *,
        measured_speed: float,
        motion_factor: float,
        stationary_phase: bool,
    ) -> np.ndarray:
        quiet_speed = max(0.0, float(getattr(self.config, "route_stability_quiet_speed_mm_s", 12.0)))
        if stationary_phase and measured_speed <= quiet_speed:
            return state.position_mm.copy()
        horizon_s = max(0.0, float(getattr(self.config, "route_stability_prediction_ms", 70.0)) / 1000.0)
        lead = state.velocity_mm_s * horizon_s * float(np.clip(motion_factor, 0.0, 1.0))
        max_lead_mm = max(0.0, float(getattr(self.config, "route_stability_prediction_max_mm", 12.0)))
        lead_norm = float(np.linalg.norm(lead))
        if max_lead_mm > 0.0 and lead_norm > max_lead_mm:
            lead = lead * (max_lead_mm / lead_norm)
        return (state.position_mm + lead).astype(np.float64, copy=False)

    def _prune_missing(self, now_ns: int, visible_ids: set[int]) -> None:
        reset_gap_ns = int(
            max(1.0, float(getattr(self.config, "route_stability_reset_gap_ms", 500.0)))
            * 1_000_000
        )
        expired = [
            track_id
            for track_id, state in self._states.items()
            if track_id not in visible_ids and int(now_ns) - int(state.last_ts_ns) > reset_gap_ns
        ]
        for track_id in expired:
            self._states.pop(track_id, None)


class RouteTopologyContinuity:
    """Choose only among current-frame valid candidates; never retain coordinates."""

    version = "route_topology_continuity_v1"

    def __init__(self, config) -> None:
        self.config = config
        self.reset()

    def reset(self) -> None:
        self._preferred_signature: tuple | None = None
        self._pending_signature: tuple | None = None
        self._pending_since_ns: int | None = None
        self._last_ts_ns: int | None = None
        self.last_status = "reset"

    def select(
        self,
        candidates: Sequence[ShotCandidate],
        *,
        ts_cam_ns: int,
        shot_mode: str,
    ) -> list[ShotCandidate]:
        current = list(candidates)
        ts_ns = int(ts_cam_ns)
        if self._last_ts_ns is not None and ts_ns < self._last_ts_ns:
            self.reset()
        self._last_ts_ns = ts_ns
        if not bool(getattr(self.config, "route_topology_continuity_enabled", True)):
            self._preferred_signature = _candidate_topology(current[0], shot_mode) if current else None
            self._pending_signature = None
            self._pending_since_ns = None
            self.last_status = "disabled"
            return current
        if not current:
            self._preferred_signature = None
            self._pending_signature = None
            self._pending_since_ns = None
            self.last_status = "clear_no_valid_route"
            return []

        signatures: Mapping[tuple, ShotCandidate] = {
            _candidate_topology(candidate, shot_mode): candidate
            for candidate in current
        }
        ranked_best = current[0]
        ranked_signature = _candidate_topology(ranked_best, shot_mode)
        had_preferred = self._preferred_signature is not None
        preferred = signatures.get(self._preferred_signature) if had_preferred else None
        if preferred is None:
            self._preferred_signature = ranked_signature
            self._pending_signature = None
            self._pending_since_ns = None
            self.last_status = "switch_invalid" if had_preferred else "seed"
            return _move_first(current, ranked_best)
        if ranked_signature == self._preferred_signature:
            self._pending_signature = None
            self._pending_since_ns = None
            self.last_status = "follow_ranked"
            return _move_first(current, preferred)

        score_gain = float(ranked_best.score) - float(preferred.score)
        minimum_gain = max(0.0, float(getattr(self.config, "route_topology_switch_score_delta", 0.18)))
        if not math.isfinite(score_gain) or score_gain < minimum_gain:
            self._pending_signature = None
            self._pending_since_ns = None
            self.last_status = f"hold_topology gain={score_gain:.3f}"
            return _move_first(current, preferred)

        if self._pending_signature != ranked_signature or self._pending_since_ns is None:
            self._pending_signature = ranked_signature
            self._pending_since_ns = ts_ns
        confirm_ns = int(
            max(0.0, float(getattr(self.config, "route_topology_switch_confirm_ms", 220.0)))
            * 1_000_000
        )
        elapsed_ns = max(0, ts_ns - int(self._pending_since_ns))
        if elapsed_ns < confirm_ns:
            self.last_status = f"switch_pending {elapsed_ns / 1_000_000.0:.0f}/{confirm_ns / 1_000_000.0:.0f}ms"
            return _move_first(current, preferred)

        self._preferred_signature = ranked_signature
        self._pending_signature = None
        self._pending_since_ns = None
        self.last_status = f"switch_commit gain={score_gain:.3f}"
        return _move_first(current, ranked_best)


def _candidate_topology(candidate: ShotCandidate, shot_mode: str) -> tuple:
    explanation = dict(getattr(candidate, "explanation", {}) or {})
    rails = tuple(str(value) for value in (explanation.get("target_shot_rails") or ()))
    rebounds = int(explanation.get("target_shot_rebounds", len(rails)) or 0)
    rail_assist = str(explanation.get("rail_assist_rail") or "")
    return (
        str(shot_mode or "rule").strip().lower(),
        int(candidate.target_track_id),
        str(candidate.target_group),
        int(candidate.pocket_index),
        rebounds,
        rails,
        rail_assist,
    )


def _move_first(candidates: list[ShotCandidate], selected: ShotCandidate) -> list[ShotCandidate]:
    if candidates and candidates[0] is selected:
        return candidates
    return [selected, *(candidate for candidate in candidates if candidate is not selected)]


def _finite_point(value, *, fallback: Point = (0.0, 0.0)) -> np.ndarray:
    try:
        point = np.asarray(value, dtype=np.float64).reshape((2,))
    except (TypeError, ValueError):
        point = np.asarray(fallback, dtype=np.float64)
    if not np.all(np.isfinite(point)):
        point = np.asarray(fallback, dtype=np.float64)
    return point.copy()


def _smoothstep(lower: float, upper: float, value: float) -> float:
    if upper <= lower:
        return 1.0 if value > lower else 0.0
    t = float(np.clip((float(value) - lower) / (upper - lower), 0.0, 1.0))
    return t * t * (3.0 - 2.0 * t)


def _time_gain(dt_s: float, tau_s: float) -> float:
    if dt_s <= 0.0:
        return 0.0
    return float(np.clip(1.0 - math.exp(-dt_s / max(1e-6, tau_s)), 0.0, 1.0))


__all__ = [
    "BallPositionMeasurement",
    "PlanningPositionStabilizer",
    "RouteTopologyContinuity",
]
