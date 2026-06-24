from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional

from ..config import LearningConfig
from ..schemas import Event, MatchStateFrame, ShotPlan, to_jsonable
from ..utils import wall_time_id


@dataclass
class _ActiveShot:
    sample_id: str
    start_frame_id: int
    start_ts_cam_ns: int
    pre_state: MatchStateFrame
    pre_plan: ShotPlan
    events: list[Event] = field(default_factory=list)


class LearningSampleRecorder:
    """Writes shot-level JSONL samples for the offline learning toolkit."""

    format_version = "bas_shot_sample_v1"

    def __init__(self, config: LearningConfig):
        self.config = config
        self.session_id = f"learning_{wall_time_id()}"
        self.session_dir = Path(config.samples_directory) / self.session_id
        self.session_dir.mkdir(parents=True, exist_ok=True)
        self.path = self.session_dir / "shot_samples.jsonl"
        self._jsonl = self.path.open("a", encoding="utf-8")
        self._last_reference_state: Optional[MatchStateFrame] = None
        self._last_reference_plan: Optional[ShotPlan] = None
        self._active: Optional[_ActiveShot] = None
        self.sample_count = 0

    def observe(self, state: MatchStateFrame, plan: ShotPlan) -> None:
        event_names = {event.name for event in state.events}
        if self._active is None and self._is_reference_frame(state, plan, event_names):
            self._last_reference_state = state
            self._last_reference_plan = plan

        if self._active is None and self._starts_shot(event_names):
            self._start(state, plan)

        if self._active is None:
            return

        self._active.events.extend(state.events)
        if self._ends_shot(event_names, state.phase):
            self._finalize(state)

    def close(self) -> None:
        self._jsonl.close()

    def _start(self, state: MatchStateFrame, plan: ShotPlan) -> None:
        pre_state = self._last_reference_state or state
        pre_plan = self._last_reference_plan or plan
        if len(pre_plan.candidates) < max(1, int(self.config.min_candidates)):
            return
        self._active = _ActiveShot(
            sample_id=f"shot_{pre_state.frame_id}_{state.frame_id}_{wall_time_id()}",
            start_frame_id=state.frame_id,
            start_ts_cam_ns=state.ts_cam_ns,
            pre_state=pre_state,
            pre_plan=pre_plan,
        )

    def _finalize(self, end_state: MatchStateFrame) -> None:
        active = self._active
        if active is None:
            return
        labels = self._labels(active.pre_plan, active.events)
        sample = {
            "format": self.format_version,
            "sample_id": active.sample_id,
            "session_id": self.session_id,
            "source": "bas_online_pipeline",
            "start_frame_id": active.start_frame_id,
            "start_ts_cam_ns": active.start_ts_cam_ns,
            "end_frame_id": end_state.frame_id,
            "end_ts_cam_ns": end_state.ts_cam_ns,
            "pre_state": to_jsonable(active.pre_state),
            "plan": to_jsonable(active.pre_plan),
            "events": [to_jsonable(event) for event in active.events],
            "end_state": to_jsonable(end_state),
            "labels": labels,
        }
        self._jsonl.write(json.dumps(sample, ensure_ascii=False) + "\n")
        self._jsonl.flush()
        self.sample_count += 1
        self._active = None
        self._last_reference_state = end_state
        self._last_reference_plan = None

    def _labels(self, plan: ShotPlan, events: Iterable[Event]) -> dict[str, Any]:
        potted: list[dict[str, Any]] = []
        event_names: list[str] = []
        for event in events:
            event_names.append(event.name)
            if event.name == "POT_PROBABLE":
                potted.append(dict(event.payload))
        potted_ids = {int(item["track_id"]) for item in potted if "track_id" in item}
        potted_groups = [str(item.get("group", "")) for item in potted]
        scratch = any(group == "cue" for group in potted_groups)
        anomaly = any(name in {"ANOMALY", "ANOMALY_RECOVERED"} for name in event_names)
        best_target = plan.best.target_track_id if plan.best is not None else None
        best_success = bool(best_target is not None and int(best_target) in potted_ids and not scratch and not anomaly)
        return {
            "potted": potted,
            "potted_track_ids": sorted(potted_ids),
            "potted_groups": potted_groups,
            "scratch": scratch,
            "foul": bool(scratch or anomaly),
            "system_best_candidate_id": plan.best.candidate_id if plan.best is not None else None,
            "system_best_target_track_id": best_target,
            "best_candidate_success": best_success,
            "adopted_system_suggestion": None,
            "adopted_success": None,
        }

    def _is_reference_frame(self, state: MatchStateFrame, plan: ShotPlan, event_names: set[str]) -> bool:
        if self._starts_shot(event_names) or self._ends_shot(event_names, state.phase):
            return False
        return state.phase in {"STABLE_IDLE", "PRE_SHOT_ARMED"} and len(plan.candidates) >= max(1, int(self.config.min_candidates))

    @staticmethod
    def _starts_shot(event_names: set[str]) -> bool:
        return bool({"SHOT_STARTED", "SHOT_START_VOTED"} & event_names)

    @staticmethod
    def _ends_shot(event_names: set[str], phase: str) -> bool:
        return bool({"TURN_RESOLVE", "ANOMALY", "ANOMALY_RECOVERED"} & event_names) or phase == "TURN_RESOLVE"
