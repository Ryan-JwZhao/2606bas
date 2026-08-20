from __future__ import annotations

import json
from pathlib import Path

from bas.config import StateConfig
from bas.schemas import TrackObservation, TracksFrame
from bas.state.break_detection import BreakShotLifecycle


FIXTURE = Path(__file__).parent / "fixtures" / "break_rack_video_samples.json"


def _track(track_id: int, group: str, center_mm: tuple[float, float]) -> TrackObservation:
    return TrackObservation(
        track_id=track_id,
        bbox=(0.0, 0.0, 10.0, 10.0),
        center_px=center_mm,
        center_mm=center_mm,
        radius_px=10.0,
        radius_mm=57.15 * 0.5,
        cls_name=group,
        group=group,
        confidence=0.9,
        quality=0.9,
        confirmed=True,
        visibility="visible",
    )


def _frame(sample: dict[str, object], frame_id: int, *, offset: tuple[float, float] = (0.0, 0.0)) -> TracksFrame:
    dx, dy = offset
    cue = sample["cue_center_mm"]
    objects = sample["object_centers_mm"]
    tracks = [_track(1, "cue", (float(cue[0]) + dx, float(cue[1]) + dy))]
    groups = ["black", *("solid" for _ in range(7)), *("stripe" for _ in range(7))]
    tracks.extend(
        _track(index + 2, groups[index % len(groups)], (float(point[0]) + dx, float(point[1]) + dy))
        for index, point in enumerate(objects)
    )
    return TracksFrame(frame_id=frame_id, ts_cam_ns=frame_id * 100_000_000, tracks=tracks)


def _detector(payload: dict[str, object]) -> BreakShotLifecycle:
    detector = BreakShotLifecycle(StateConfig(engine="modern", break_rack_stable_frames=3))
    detector.set_table_context(
        inner_polygon_mm=[tuple(point) for point in payload["table_polygon_mm"]],
        ball_diameter_mm=float(payload["ball_diameter_mm"]),
    )
    return detector


def test_both_real_video_racks_arm_break_without_using_shot_strength() -> None:
    payload = json.loads(FIXTURE.read_text(encoding="utf-8"))

    for sample in payload["samples"]:
        detector = _detector(payload)
        for frame_id in range(1, 4):
            detector.observe(_frame(sample, frame_id), phase="STABLE_IDLE")

        assert detector.break_pending is True, sample["video"]
        assert detector.last_evidence.is_rack is True, sample["video"]
        assert detector.mark_shot_started() is True, sample["video"]


def test_rack_position_is_a_broad_prior_not_a_fixed_coordinate() -> None:
    payload = json.loads(FIXTURE.read_text(encoding="utf-8"))
    sample = payload["samples"][0]
    detector = _detector(payload)

    for frame_id in range(1, 4):
        detector.observe(_frame(sample, frame_id, offset=(-250.0, 120.0)), phase="STABLE_IDLE")

    assert detector.break_pending is True
    assert "foot_spot_region" in detector.last_evidence.reasons


def test_armed_rack_survives_one_degraded_frame_before_shot() -> None:
    payload = json.loads(FIXTURE.read_text(encoding="utf-8"))
    sample = payload["samples"][0]
    detector = _detector(payload)
    for frame_id in range(1, 4):
        detector.observe(_frame(sample, frame_id), phase="STABLE_IDLE")

    detector.observe(TracksFrame(4, 400_000_000, []), phase="STABLE_IDLE")

    assert detector.break_pending is True
    assert detector.mark_shot_started() is True


def test_distributed_midgame_balls_do_not_arm_break() -> None:
    payload = json.loads(FIXTURE.read_text(encoding="utf-8"))
    sample = {
        "cue_center_mm": [300.0, 635.0],
        "object_centers_mm": [
            [300.0 + (index % 5) * 430.0, 150.0 + (index // 5) * 420.0]
            for index in range(15)
        ],
    }
    detector = _detector(payload)

    for frame_id in range(1, 8):
        detector.observe(_frame(sample, frame_id), phase="STABLE_IDLE")

    assert detector.break_pending is False
    assert detector.mark_shot_started() is False


def test_real_light_break_settled_layout_does_not_rearm_after_cold_start() -> None:
    payload = json.loads(FIXTURE.read_text(encoding="utf-8"))
    sample = payload["post_break_samples"][0]
    detector = _detector(payload)

    for frame_id in range(1, 5):
        detector.observe(_frame(sample, frame_id), phase="STABLE_IDLE")

    assert detector.break_pending is False
    assert detector.last_evidence.is_rack is False
    assert detector.mark_shot_started() is False


def test_valid_first_nonbreak_shot_prevents_later_cluster_from_rearming_break() -> None:
    payload = json.loads(FIXTURE.read_text(encoding="utf-8"))
    sample = payload["samples"][0]
    detector = _detector(payload)

    assert detector.mark_shot_started() is False
    detector.mark_shot_resolved(valid=True)
    for frame_id in range(1, 5):
        detector.observe(_frame(sample, frame_id), phase="STABLE_IDLE")

    assert detector.break_pending is False
