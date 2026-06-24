from __future__ import annotations

import argparse
import json
from pathlib import Path


RL_ROOT = Path(__file__).resolve().parent


def main() -> int:
    parser = argparse.ArgumentParser(description="Create tiny demo samples for checking the RL toolkit wiring.")
    parser.add_argument("--out", default=str(RL_ROOT / "data" / "samples" / "demo" / "shot_samples.jsonl"))
    args = parser.parse_args()
    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    samples = [_sample(i, success=(i % 2 == 0)) for i in range(12)]
    with path.open("w", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")
    print(path)
    return 0


def _sample(index: int, *, success: bool) -> dict:
    target_id = 2 + index
    cue_x = 250.0 + index * 8.0
    object_x = 720.0 - index * 5.0
    potted = [{"track_id": target_id, "group": "solid", "pocket_index": 2}] if success else []
    return {
        "format": "bas_shot_sample_v1",
        "sample_id": f"demo_{index}",
        "session_id": "demo",
        "source": "demo",
        "start_frame_id": index * 10,
        "start_ts_cam_ns": index * 1_000_000,
        "end_frame_id": index * 10 + 8,
        "end_ts_cam_ns": index * 1_000_000 + 800_000,
        "pre_state": {
            "frame_id": index * 10,
            "ts_cam_ns": index * 1_000_000,
            "phase": "STABLE_IDLE",
            "layout": [
                _track(1, "cue", cue_x, 635.0),
                _track(target_id, "solid", object_x, 635.0),
                _track(100 + index, "stripe", 900.0, 420.0),
            ],
        },
        "plan": {
            "plan_id": f"demo_plan_{index}",
            "frame_id": index * 10,
            "ts_cam_ns": index * 1_000_000,
            "candidates": [
                _candidate(f"c_good_{index}", target_id, cue_x, object_x, cut=8.0 + index, risk=0.15, score=1.4),
                _candidate(f"c_bad_{index}", target_id + 1000, cue_x, object_x + 80.0, cut=55.0, risk=0.72, score=0.2),
            ],
            "best": _candidate(f"c_good_{index}", target_id, cue_x, object_x, cut=8.0 + index, risk=0.15, score=1.4),
        },
        "events": [],
        "end_state": {
            "frame_id": index * 10 + 8,
            "ts_cam_ns": index * 1_000_000 + 800_000,
            "phase": "TURN_RESOLVE",
            "layout": [_track(1, "cue", cue_x + 120.0, 635.0)],
        },
        "labels": {
            "potted": potted,
            "potted_track_ids": [target_id] if success else [],
            "potted_groups": ["solid"] if success else [],
            "scratch": False,
            "foul": False,
            "system_best_candidate_id": f"c_good_{index}",
            "system_best_target_track_id": target_id,
            "best_candidate_success": success,
            "adopted_system_suggestion": None,
            "adopted_success": None,
        },
    }


def _track(track_id: int, group: str, x: float, y: float) -> dict:
    return {
        "track_id": track_id,
        "group": group,
        "center_px": [x, y],
        "center_mm": [x, y],
        "radius_px": 15.0,
        "radius_mm": 28.575,
        "quality": 0.9,
        "visibility": "visible",
    }


def _candidate(candidate_id: str, target_id: int, cue_x: float, object_x: float, *, cut: float, risk: float, score: float) -> dict:
    return {
        "candidate_id": candidate_id,
        "cue_track_id": 1,
        "target_track_id": target_id,
        "target_group": "solid",
        "pocket_index": 2,
        "cue_ball": [cue_x, 635.0],
        "object_ball": [object_x, 635.0],
        "ghost_ball": [object_x - 57.15, 635.0],
        "pocket_point": [2540.0, 0.0],
        "aim_line": [[cue_x, 635.0], [object_x - 57.15, 635.0]],
        "object_line": [[object_x, 635.0], [2540.0, 0.0]],
        "cut_angle_deg": cut,
        "cue_distance_mm": abs(object_x - cue_x),
        "object_distance_mm": abs(2540.0 - object_x),
        "score": score,
        "risk": risk,
        "explanation": {
            "cue_clearance_mm": 120.0,
            "object_clearance_mm": 130.0,
            "cut_penalty": cut / 80.0,
            "distance_penalty": 0.25,
            "pocket_angle_penalty": 0.2,
        },
    }


if __name__ == "__main__":
    raise SystemExit(main())
