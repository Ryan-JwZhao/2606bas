from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from ..config import StateConfig
from ..schemas import Event, MatchPhase, PocketVisualObservationFrame, TracksFrame
from ..state.pocket import PerBallPocketFSM


@dataclass(frozen=True)
class TrainingPocketJudgment:
    """Number-aware view of the shared rule-mode pocket evaluator."""

    events: tuple[Event, ...]
    detected_events: tuple[Event, ...]

    @property
    def detected_numbers(self) -> tuple[int, ...]:
        return tuple(
            number
            for event in self.detected_events
            if (number := _event_ball_number(event)) is not None
        )


class TrainingPocketJudge:
    """Adapts the rule-mode per-ball pocket FSM to numbered training balls."""

    version = PerBallPocketFSM.__name__

    def __init__(self, state_config: StateConfig):
        self._fsm = PerBallPocketFSM(state_config)
        self._pockets_mm: list[tuple[float, float]] = []

    @property
    def has_table_context(self) -> bool:
        return bool(self._pockets_mm)

    def reset(self) -> None:
        self._fsm.reset()

    def set_table_context(
        self,
        *,
        inner_polygon_mm: Sequence[tuple[float, float]] | None = None,
        table_edge_polygon_mm: Sequence[tuple[float, float]] | None = None,
        ball_center_reachable_polygon_mm: Sequence[tuple[float, float]] | None = None,
        pockets_mm: Sequence[tuple[float, float]] | None = None,
        ball_diameter_mm: float | None = None,
        pocket_curves_mm: Sequence[Sequence[tuple[float, float]]] | None = None,
    ) -> None:
        if pockets_mm is not None:
            self._pockets_mm = [(float(x), float(y)) for x, y in pockets_mm]
        self._fsm.set_table_context(
            inner_polygon_mm=_points(inner_polygon_mm),
            table_edge_polygon_mm=_points(table_edge_polygon_mm),
            ball_center_reachable_polygon_mm=_points(ball_center_reachable_polygon_mm),
            pockets_mm=_points(pockets_mm),
            ball_diameter_mm=ball_diameter_mm,
            pocket_curves_mm=(
                [_points(curve) or [] for curve in pocket_curves_mm]
                if pocket_curves_mm is not None
                else None
            ),
        )

    def update(
        self,
        tracks_frame: TracksFrame,
        *,
        active: bool,
        pocket_observations: PocketVisualObservationFrame | None = None,
    ) -> TrainingPocketJudgment:
        phase = MatchPhase.SHOT_ACTIVE if active else MatchPhase.STABLE_IDLE
        events = tuple(
            self._fsm.update(
                tracks_frame,
                phase,
                pocket_observations if active else None,
            )
        )
        detected_events = tuple(event for event in events if event.name == "POCKET_DETECTED")

        decision_ids = [
            str(event.payload.get("decision_id", ""))
            for event in detected_events
            if str(event.payload.get("decision_id", "")).strip()
        ]
        if decision_ids:
            self._fsm.mark_confirmed(decision_ids)

        return TrainingPocketJudgment(
            events=events,
            detected_events=detected_events,
        )


def _event_ball_number(event: Event) -> int | None:
    return _ball_number(event.payload.get("track_id"))


def _ball_number(value: object) -> int | None:
    try:
        number = int(value)
    except (TypeError, ValueError):
        return None
    return number if 0 <= number <= 15 else None


def _points(
    values: Sequence[tuple[float, float]] | None,
) -> list[tuple[float, float]] | None:
    if values is None:
        return None
    return [(float(x), float(y)) for x, y in values]


__all__ = ["TrainingPocketJudge", "TrainingPocketJudgment"]
