from __future__ import annotations

from dataclasses import replace
from typing import Sequence

from ..config import StateConfig, TrainingConfig
from ..schemas import Event, PocketVisualObservationFrame, TrackObservation, TracksFrame
from .models import TrainingScenario, TrainingStateFrame
from .numbered_tracker import ball_number_from_track
from .pocket_judge import TrainingPocketJudge
from .prompts import completion_message, mode_hint, rule_decision_message, running_message
from .rules import (
    ACTION_ACCEPT,
    ACTION_FAIL,
    ACTION_NONE,
    ACTION_RESPOT_CONTINUE,
    TrainingRuleDecision,
    get_training_rule_set,
)
from .scenarios import get_training_scenario, validate_scenario_setup


class TrainingSession:
    version = "numbered_training_session_v3_rule_sets"

    def __init__(
        self,
        config: TrainingConfig,
        *,
        state_config: StateConfig | None = None,
        ball_diameter_mm: float = 57.15,
    ):
        self.config = config
        self.ball_diameter_mm = float(ball_diameter_mm)
        self.scenario: TrainingScenario = get_training_scenario(config.scenario_id)
        self.rules = get_training_rule_set(self.scenario.rule_set_id)
        self._pocket_judge = TrainingPocketJudge(state_config or StateConfig(engine="modern"))
        self._latest_tracks: list[TrackObservation] = []
        self._potted: list[int] = []
        self._error_count = 0
        self._respot_required: set[int] = set()
        self._cue_respot_required = False
        self._attempt = 0
        self._started_ts_ns = 0
        self._ball_center_reachable_polygon_mm: list[tuple[float, float]] = []
        self._pockets_mm: list[tuple[float, float]] = []
        self._state = self._new_state(0, 0)

    @property
    def state(self) -> TrainingStateFrame:
        return self._state

    def select_scenario(self, scenario_id: str) -> TrainingStateFrame:
        self.scenario = get_training_scenario(scenario_id)
        self.rules = get_training_rule_set(self.scenario.rule_set_id)
        self.config.scenario_id = self.scenario.scenario_id
        return self.reset()

    def reset(self) -> TrainingStateFrame:
        frame_id = self._state.frame_id
        ts_cam_ns = self._state.ts_cam_ns
        self._potted.clear()
        self._error_count = 0
        self._respot_required.clear()
        self._cue_respot_required = False
        self._started_ts_ns = 0
        self._pocket_judge.reset()
        self._state = self._new_state(frame_id, ts_cam_ns)
        if self._latest_tracks:
            current_tracks = TracksFrame(
                frame_id=frame_id,
                ts_cam_ns=ts_cam_ns,
                tracks=list(self._latest_tracks),
            )
            self._pocket_judge.update(
                self.pocket_observer_tracks(current_tracks),
                active=False,
            )
            self._state = self._setup_state(frame_id, ts_cam_ns, self._latest_tracks)
        return self._state

    def start(self) -> tuple[bool, TrainingStateFrame]:
        ready, message = self._validate_setup(self._latest_tracks)
        if not ready:
            self._state = replace(self._state, phase="setup", setup_ready=False, message=message, failure_reason=None)
            return False, self._state
        self._pocket_judge.reset()
        current_tracks = TracksFrame(
            frame_id=int(self._state.frame_id),
            ts_cam_ns=int(self._state.ts_cam_ns),
            tracks=list(self._latest_tracks),
        )
        self._pocket_judge.update(
            self.pocket_observer_tracks(current_tracks),
            active=False,
        )
        self._attempt += 1
        self._potted.clear()
        self._error_count = 0
        self._respot_required.clear()
        self._cue_respot_required = False
        self._started_ts_ns = int(self._state.ts_cam_ns)
        self._state = replace(
            self._state,
            phase="running",
            setup_ready=True,
            message=self._running_message(),
            expected_numbers=list(self._expected_numbers()),
            potted_numbers=[],
            progress_current=0,
            attempt=self._attempt,
            elapsed_s=0.0,
            failure_reason=None,
            error_count=0,
            remaining_numbers=list(self.scenario.required_balls),
            mode_hint=self._mode_hint(),
            respot_required_numbers=[],
            events=[],
        )
        return True, self._state

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
        if ball_diameter_mm is not None:
            self.ball_diameter_mm = float(ball_diameter_mm)
        if ball_center_reachable_polygon_mm is not None:
            self._ball_center_reachable_polygon_mm = [
                (float(x), float(y)) for x, y in ball_center_reachable_polygon_mm
            ]
        elif inner_polygon_mm is not None and not self._ball_center_reachable_polygon_mm:
            self._ball_center_reachable_polygon_mm = [(float(x), float(y)) for x, y in inner_polygon_mm]
        if pockets_mm is not None:
            self._pockets_mm = [(float(x), float(y)) for x, y in pockets_mm]
        self._pocket_judge.set_table_context(
            inner_polygon_mm=inner_polygon_mm,
            table_edge_polygon_mm=table_edge_polygon_mm,
            ball_center_reachable_polygon_mm=ball_center_reachable_polygon_mm,
            pockets_mm=pockets_mm,
            ball_diameter_mm=ball_diameter_mm,
            pocket_curves_mm=pocket_curves_mm,
        )

    def update(
        self,
        tracks_frame: TracksFrame,
        pocket_observations: PocketVisualObservationFrame | None = None,
    ) -> TrainingStateFrame:
        self._latest_tracks = list(tracks_frame.tracks)
        frame_id = int(tracks_frame.frame_id)
        ts_cam_ns = int(tracks_frame.ts_cam_ns)
        pocket_judgment = self._pocket_judge.update(
            self.pocket_observer_tracks(tracks_frame),
            active=self._state.phase == "running",
            pocket_observations=pocket_observations,
        )
        if self._state.phase not in {"running", "passed", "failed"}:
            self._state = self._setup_state(frame_id, ts_cam_ns, self._latest_tracks)
            return self._state
        if self._state.phase in {"passed", "failed"}:
            self._state = replace(
                self._state,
                frame_id=frame_id,
                ts_cam_ns=ts_cam_ns,
                visible_numbers=self._visible_numbers(self._latest_tracks),
                elapsed_s=self._elapsed_s(ts_cam_ns),
                events=[],
            )
            return self._state

        visible = set(self._visible_numbers(self._latest_tracks))
        if self._cue_respot_required and 0 in visible:
            self._cue_respot_required = False
        self._respot_required.difference_update(visible)
        if not self._pocket_judge.has_table_context:
            events = [
                self._event(
                    "TRAINING_FAILED",
                    frame_id,
                    ts_cam_ns,
                    reason="pocket_context_unavailable",
                )
            ]
            self._state = self._failed(
                frame_id,
                ts_cam_ns,
                "袋口标定不可用，训练已停止",
                "pocket_context_unavailable",
                visible,
                events,
            )
            return self._state
        pending_numbers = set(self.scenario.required_balls) - set(self._potted)
        watched = pending_numbers | ({0} if self.scenario.require_cue_ball else set())
        confirmed = [number for number in pocket_judgment.detected_numbers if number in watched]

        events: list[Event] = []
        decision = self.rules.evaluate_pocket_result(confirmed, self._expected_numbers())
        if decision.action == ACTION_RESPOT_CONTINUE:
            self._error_count += 1
            if decision.cue_ball_pocketed:
                self._cue_respot_required = True
            self._respot_required.update(decision.balls)
            events.append(
                self._event(
                    "TRAINING_WARNING",
                    frame_id,
                    ts_cam_ns,
                    **self._decision_payload(decision),
                )
            )
            self._prime_pocket_judge(tracks_frame)
            self._state = self._running_state(
                frame_id,
                ts_cam_ns,
                visible,
                events,
                message=rule_decision_message(decision),
            )
            return self._state
        if decision.action == ACTION_FAIL:
            events.append(
                self._event(
                    "TRAINING_FAILED",
                    frame_id,
                    ts_cam_ns,
                    **self._decision_payload(decision),
                )
            )
            self._state = self._failed(
                frame_id,
                ts_cam_ns,
                rule_decision_message(decision),
                str(decision.reason or "rule_failed"),
                visible,
                events,
            )
            return self._state
        if decision.action == ACTION_ACCEPT and decision.ball is not None:
            number = int(decision.ball)
            self._potted.append(number)
            detected_event = next(
                (
                    event
                    for event in pocket_judgment.detected_events
                    if int(event.payload.get("track_id", -1)) == number
                ),
                None,
            )
            events.append(
                self._event(
                    "TRAINING_BALL_POTTED",
                    frame_id,
                    ts_cam_ns,
                    ball=number,
                    pocket_index=(detected_event.payload.get("pocket_index") if detected_event else None),
                    decision_id=(detected_event.payload.get("decision_id") if detected_event else None),
                    judgment="rules_pocket_fsm",
                )
            )
            if len(self._potted) == len(self.scenario.required_balls):
                events.append(self._event("TRAINING_PASSED", frame_id, ts_cam_ns, scenario=self.scenario.scenario_id))
                self._state = replace(
                    self._state,
                    frame_id=frame_id,
                    ts_cam_ns=ts_cam_ns,
                    phase="passed",
                    message=completion_message(self.scenario.stages),
                    expected_numbers=[],
                    visible_numbers=sorted(visible),
                    potted_numbers=list(self._potted),
                    progress_current=len(self._potted),
                    elapsed_s=self._elapsed_s(ts_cam_ns),
                    error_count=self._error_count,
                    remaining_numbers=[],
                    mode_hint=self._mode_hint(),
                    respot_required_numbers=[],
                    events=events,
                )
                return self._state
        elif decision.action != ACTION_NONE:
            raise RuntimeError(f"训练规则返回了无法处理的动作: {decision.action}")

        self._state = self._running_state(frame_id, ts_cam_ns, visible, events)
        return self._state

    def _new_state(self, frame_id: int, ts_cam_ns: int) -> TrainingStateFrame:
        return TrainingStateFrame(
            frame_id=frame_id,
            ts_cam_ns=ts_cam_ns,
            scenario_id=self.scenario.scenario_id,
            scenario_title=self.scenario.display_title,
            required_numbers=list(self.scenario.required_balls),
            progress_total=len(self.scenario.required_balls),
            attempt=self._attempt,
            remaining_numbers=list(self.scenario.required_balls),
            mode_hint=self._mode_hint(),
        )

    def _setup_state(self, frame_id: int, ts_cam_ns: int, tracks: Sequence[TrackObservation]) -> TrainingStateFrame:
        ready, message = self._validate_setup(tracks)
        return TrainingStateFrame(
            frame_id=frame_id,
            ts_cam_ns=ts_cam_ns,
            scenario_id=self.scenario.scenario_id,
            scenario_title=self.scenario.display_title,
            phase="ready" if ready else "setup",
            message=message,
            setup_ready=ready,
            required_numbers=list(self.scenario.required_balls),
            expected_numbers=list(self.scenario.stages[0]),
            visible_numbers=self._visible_numbers(tracks),
            potted_numbers=[],
            progress_total=len(self.scenario.required_balls),
            attempt=self._attempt,
            error_count=0,
            remaining_numbers=list(self.scenario.required_balls),
            mode_hint=self._mode_hint(),
        )

    def pocket_observer_tracks(self, tracks_frame: TracksFrame) -> TracksFrame:
        allowed = set(self.scenario.required_balls)
        if self.scenario.require_cue_ball:
            allowed.add(0)
        tracks = [
            track
            for track in tracks_frame.tracks
            if (number := ball_number_from_track(track)) is not None and number in allowed
        ]
        return replace(tracks_frame, tracks=tracks)

    def _validate_setup(self, tracks: Sequence[TrackObservation]) -> tuple[bool, str]:
        if not self._pocket_judge.has_table_context:
            return False, "袋口标定不可用，无法开始训练"
        return validate_scenario_setup(
            self.scenario,
            tracks,
            ball_diameter_mm=self.ball_diameter_mm,
            ball_center_reachable_polygon_mm=self._ball_center_reachable_polygon_mm,
            pockets_mm=self._pockets_mm,
        )

    def _expected_numbers(self) -> tuple[int, ...]:
        potted = set(self._potted)
        for stage in self.scenario.stages:
            remaining = tuple(number for number in stage if number not in potted)
            if remaining:
                return remaining
        return ()

    def _running_message(self) -> str:
        return running_message(
            self._expected_numbers(),
            cue_respot_required=self._cue_respot_required,
            respot_required_numbers=self._respot_required,
        )

    def _running_state(
        self,
        frame_id: int,
        ts_cam_ns: int,
        visible: set[int],
        events: list[Event],
        *,
        message: str | None = None,
    ) -> TrainingStateFrame:
        remaining = [number for number in self.scenario.required_balls if number not in self._potted]
        respot = sorted(self._respot_required | ({0} if self._cue_respot_required else set()))
        return replace(
            self._state,
            frame_id=frame_id,
            ts_cam_ns=ts_cam_ns,
            phase="running",
            message=message or self._running_message(),
            expected_numbers=list(self._expected_numbers()),
            visible_numbers=sorted(visible),
            potted_numbers=list(self._potted),
            progress_current=len(self._potted),
            elapsed_s=self._elapsed_s(ts_cam_ns),
            failure_reason=None,
            error_count=self._error_count,
            remaining_numbers=remaining,
            mode_hint=self._mode_hint(),
            respot_required_numbers=respot,
            events=events,
        )

    def _prime_pocket_judge(self, tracks_frame: TracksFrame) -> None:
        self._pocket_judge.reset()
        self._pocket_judge.update(
            self.pocket_observer_tracks(tracks_frame),
            active=False,
        )

    def _mode_hint(self) -> str:
        return mode_hint(self.scenario.stages)

    @staticmethod
    def _decision_payload(decision: TrainingRuleDecision) -> dict[str, object]:
        payload: dict[str, object] = {
            "reason": str(decision.reason or ""),
            "action": decision.action,
        }
        if decision.ball is not None:
            payload["ball"] = int(decision.ball)
        if decision.balls:
            payload["balls"] = list(decision.balls)
        if decision.expected:
            payload["expected"] = list(decision.expected)
        return payload

    def _failed(
        self,
        frame_id: int,
        ts_cam_ns: int,
        message: str,
        reason: str,
        visible: set[int],
        events: list[Event],
    ) -> TrainingStateFrame:
        return replace(
            self._state,
            frame_id=frame_id,
            ts_cam_ns=ts_cam_ns,
            phase="failed",
            message=message,
            visible_numbers=sorted(visible),
            potted_numbers=list(self._potted),
            progress_current=len(self._potted),
            elapsed_s=self._elapsed_s(ts_cam_ns),
            failure_reason=reason,
            error_count=self._error_count,
            remaining_numbers=[number for number in self.scenario.required_balls if number not in self._potted],
            mode_hint=self._mode_hint(),
            respot_required_numbers=sorted(self._respot_required | ({0} if self._cue_respot_required else set())),
            events=events,
        )

    def _elapsed_s(self, ts_cam_ns: int) -> float:
        if self._started_ts_ns <= 0:
            return 0.0
        return max(0.0, (int(ts_cam_ns) - self._started_ts_ns) / 1e9)

    @staticmethod
    def _visible_numbers(tracks: Sequence[TrackObservation]) -> list[int]:
        numbers = {
            number
            for track in tracks
            if track.visibility == "visible" and (number := ball_number_from_track(track)) is not None
        }
        return sorted(numbers)

    @staticmethod
    def _event(name: str, frame_id: int, ts_cam_ns: int, **payload: object) -> Event:
        return Event(name=name, frame_id=frame_id, ts_cam_ns=ts_cam_ns, payload=dict(payload))
