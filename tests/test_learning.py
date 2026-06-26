from __future__ import annotations

import json

from bas.config import LearningConfig
from bas.learning import LearningSampleRecorder
from bas.planning.learning import JsonLearningRanker
from bas.schemas import Event, MatchStateFrame, ShotCandidate, ShotPlan, TrackObservation
from rl.dataset import build_training_data


def _candidate(candidate_id: str, target_id: int, *, score: float, risk: float) -> ShotCandidate:
    return ShotCandidate(
        candidate_id=candidate_id,
        cue_track_id=1,
        target_track_id=target_id,
        target_group="solid",
        pocket_index=2,
        cue_ball=(100.0, 100.0),
        object_ball=(400.0, 100.0),
        ghost_ball=(342.0, 100.0),
        pocket_point=(1000.0, 0.0),
        aim_line=[(100.0, 100.0), (342.0, 100.0)],
        object_line=[(400.0, 100.0), (1000.0, 0.0)],
        cut_angle_deg=10.0,
        cue_distance_mm=242.0,
        object_distance_mm=608.0,
        score=score,
        risk=risk,
        explanation={"cue_clearance_mm": 100.0, "object_clearance_mm": 100.0},
    )


def _track(track_id: int, group: str) -> TrackObservation:
    return TrackObservation(
        track_id=track_id,
        bbox=(0, 0, 10, 10),
        center_px=(100.0 + track_id, 100.0),
        radius_px=5.0,
        cls_name=group,
        group=group,
        confidence=0.9,
        center_mm=(100.0 + track_id, 100.0),
        quality=0.9,
    )


def test_json_learning_ranker_can_rerank(tmp_path) -> None:
    model_path = tmp_path / "ranker.json"
    model_path.write_text(
        json.dumps(
            {
                "format": "bas_linear_ranker_v1",
                "model_version": "test",
                "feature_names": ["base_risk"],
                "normalization": {"mean": [0.0], "std": [1.0]},
                "output_names": ["rank_score"],
                "weights": [10.0],
                "bias": -5.0,
                "score_weights": {"rank": 2.0, "risk": 0.0},
            }
        ),
        encoding="utf-8",
    )
    ranker = JsonLearningRanker(model_path, table_width_mm=1000, table_height_mm=500, score_blend=1.0)
    low_model_score = _candidate("low", 2, score=2.0, risk=0.0)
    high_model_score = _candidate("high", 3, score=0.0, risk=1.0)

    ranked = ranker.rerank([low_model_score, high_model_score])

    assert ranked[0].candidate_id == "high"
    assert ranked[0].explanation["learning_ranker"] == "learning_json:test"


def test_learning_sample_recorder_writes_training_rows(tmp_path) -> None:
    recorder = LearningSampleRecorder(LearningConfig(collect_enabled=True, samples_directory=str(tmp_path)))
    plan = ShotPlan(
        plan_id="p1",
        frame_id=1,
        ts_cam_ns=1,
        candidates=[_candidate("c1", 2, score=1.0, risk=0.1)],
        best=_candidate("c1", 2, score=1.0, risk=0.1),
    )
    stable = MatchStateFrame(frame_id=1, ts_cam_ns=1, phase="STABLE_IDLE", layout=[_track(1, "cue"), _track(2, "solid")])
    started = MatchStateFrame(frame_id=2, ts_cam_ns=2, phase="SHOT_ACTIVE", events=[Event("SHOT_STARTED", 2, 2)], layout=stable.layout)
    ended = MatchStateFrame(
        frame_id=3,
        ts_cam_ns=3,
        phase="TURN_RESOLVE",
        events=[Event("POT_PROBABLE", 3, 3, payload={"track_id": 2, "group": "solid", "pocket_index": 2}), Event("TURN_RESOLVE", 3, 3)],
        layout=[_track(1, "cue")],
    )

    recorder.observe(stable, plan)
    recorder.observe(started, plan)
    recorder.observe(ended, plan)
    recorder.close()

    rows = list(recorder.session_dir.glob("shot_samples.jsonl"))
    assert rows
    data = build_training_data(rows[0])
    assert data.features.shape[0] == 1
    assert data.pot.tolist() == [1.0]
    assert data.rank.tolist() == [1.0]


def test_learning_sample_recorder_does_not_create_empty_session_file(tmp_path) -> None:
    recorder = LearningSampleRecorder(LearningConfig(collect_enabled=True, samples_directory=str(tmp_path)))

    assert not recorder.path.exists()
    assert not recorder.session_dir.exists()

    recorder.close()

    assert not recorder.path.exists()
    assert not recorder.session_dir.exists()
