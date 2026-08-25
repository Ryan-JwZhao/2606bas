from __future__ import annotations

import cv2
import numpy as np

from bas.config import StateConfig
from bas.perception.pocket_observer import PocketObserver
from bas.perception.regions import DetectionRegionPolicy, PocketGuardRegion
from bas.schemas import FramePacket, MatchPhase, TrackObservation, TracksFrame
from bas.state.pocket import PerBallPocketFSM


def _track(track_id: int, group: str, x: float, y: float, vy: float) -> TrackObservation:
    return TrackObservation(
        track_id=track_id,
        bbox=(x - 6, y - 6, x + 6, y + 6),
        center_px=(x, y),
        radius_px=6,
        cls_name=group,
        group=group,
        confidence=0.95,
        velocity_px_s=(0.0, vy),
        quality=0.95,
    )


def _image(y: int | None) -> np.ndarray:
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    if y is not None:
        cv2.circle(image, (50, y), 6, (240, 240, 240), -1)
    return image


def _policy() -> DetectionRegionPolicy:
    inner = np.array([[10, 20], [90, 20], [90, 90], [10, 90]], dtype=np.float32)
    guard = PocketGuardRegion(
        pocket_index=0,
        polygon=np.array([[25, 0], [75, 0], [75, 50], [25, 50]], dtype=np.float32),
        center_px=(50.0, 8.0),
        ball_diameter_px=12.0,
    )
    return DetectionRegionPolicy(global_polygon=inner, ball_polygon=inner, ball_guard_regions=(guard,))


def test_observer_reports_class_associated_inward_crossing() -> None:
    observer = PocketObserver(history_ms=1500)
    policy = _policy()
    observer.update(
        FramePacket(1, 1_000_000_000, "cam", image=_image(45)),
        TracksFrame(1, 1_000_000_000, [_track(7, "solid", 50, 45, -300)]),
        policy,
    )

    out = observer.update(
        FramePacket(2, 1_100_000_000, "cam", image=_image(20)),
        TracksFrame(2, 1_100_000_000, [_track(7, "solid", 50, 20, -300)]),
        policy,
    )

    observed = out.observations[0]
    assert observed.inward_crossing is True
    assert observed.group == "solid"
    assert observed.associated_track_ids == [7]
    assert "frame_difference" in observed.evidence_sources


def test_observer_does_not_promote_unassociated_tiny_motion() -> None:
    observer = PocketObserver(history_ms=1500)
    policy = _policy()
    first = np.zeros((100, 100, 3), dtype=np.uint8)
    second = first.copy()
    second[10:12, 49:51] = 255
    observer.update(FramePacket(1, 1_000_000_000, "cam", image=first), TracksFrame(1, 1_000_000_000, []), policy)

    out = observer.update(FramePacket(2, 1_050_000_000, "cam", image=second), TracksFrame(2, 1_050_000_000, []), policy)

    assert out.observations[0].inward_crossing is False


def test_fast_crossing_prefers_recently_disappeared_ball_over_static_lip_ball() -> None:
    """Regression: shot 4 crossed the ROI between detections while another ball sat nearby."""

    observer = PocketObserver(history_ms=1500)
    policy = _policy()
    before = np.zeros((100, 100, 3), dtype=np.uint8)
    cv2.circle(before, (35, 25), 6, (120, 120, 120), -1)
    crossing = before.copy()
    cv2.circle(crossing, (50, 4), 6, (240, 240, 240), -1)
    static = _track(4, "solid", 25, 45, 0)
    target = _track(14, "stripe", 50, 75, 0)

    observer.update(
        FramePacket(1, 1_000_000_000, "cam", image=before),
        TracksFrame(1, 1_000_000_000, [static, target]),
        policy,
    )
    target.visibility = "occluded"
    target.lost_frames = 1
    out = observer.update(
        FramePacket(2, 1_100_000_000, "cam", image=crossing),
        TracksFrame(2, 1_100_000_000, [static, target]),
        policy,
    )

    observed = out.observations[0]
    assert observed.inward_crossing is True
    assert observed.group == "stripe"
    assert observed.associated_track_ids == [14]


def test_deep_non_ball_motion_does_not_hijack_tangential_disappeared_track() -> None:
    """Regression: a cue entering a pocket ROI must not pot a ball moving past the pocket."""

    observer = PocketObserver(history_ms=1500)
    policy = _policy()
    empty = np.zeros((100, 100, 3), dtype=np.uint8)
    observer.update(
        FramePacket(1, 1_000_000_000, "cam", image=empty),
        TracksFrame(1, 1_000_000_000, [_track(45, "stripe", 10, 75, 0)]),
        policy,
    )
    observer.update(
        FramePacket(2, 1_100_000_000, "cam", image=empty),
        TracksFrame(2, 1_100_000_000, [_track(45, "stripe", 30, 75, 0)]),
        policy,
    )
    crossing = empty.copy()
    cv2.circle(crossing, (50, 4), 6, (240, 240, 240), -1)
    vanished = _track(45, "stripe", 30, 75, 0)
    vanished.visibility = "occluded"
    vanished.lost_frames = 1

    out = observer.update(
        FramePacket(3, 1_200_000_000, "cam", image=crossing),
        TracksFrame(3, 1_200_000_000, [vanished]),
        policy,
    )

    assert out.observations[0].inward_crossing is False


def test_aligned_ball_occluded_at_mouth_can_still_cross() -> None:
    """Regression: shot 23 vanished just before the visual centroid reached pocket centre."""

    observer = PocketObserver(history_ms=1500)
    policy = _policy()
    empty = np.zeros((100, 100, 3), dtype=np.uint8)
    observer.update(
        FramePacket(1, 1_000_000_000, "cam", image=empty),
        TracksFrame(1, 1_000_000_000, [_track(11, "stripe", 50, 75, -300)]),
        policy,
    )
    observer.update(
        FramePacket(2, 1_100_000_000, "cam", image=empty),
        TracksFrame(2, 1_100_000_000, [_track(11, "stripe", 50, 35, -300)]),
        policy,
    )
    mouth = empty.copy()
    cv2.circle(mouth, (50, 9), 6, (240, 240, 240), -1)
    vanished = _track(11, "stripe", 50, 35, -300)
    vanished.visibility = "occluded"
    vanished.lost_frames = 1

    out = observer.update(
        FramePacket(3, 1_200_000_000, "cam", image=mouth),
        TracksFrame(3, 1_200_000_000, [vanished]),
        policy,
    )

    observed = out.observations[0]
    assert observed.inward_crossing is True
    assert observed.group == "stripe"
    assert observed.associated_track_ids == [11]


def test_latch_does_not_rearm_immediately_when_class_vote_changes() -> None:
    """One physical crossing must not duplicate when its short class vote changes."""

    observer = PocketObserver(history_ms=1500)
    policy = _policy()

    def image_with(*balls: tuple[int, int]) -> np.ndarray:
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        for x, y in balls:
            cv2.circle(image, (x, y), 6, (240, 240, 240), -1)
        return image

    observer.update(
        FramePacket(1, 1_000_000_000, "cam", image=image_with((35, 45))),
        TracksFrame(1, 1_000_000_000, [_track(7, "cue", 35, 45, -300)]),
        policy,
    )
    first = observer.update(
        FramePacket(2, 1_100_000_000, "cam", image=image_with((35, 20))),
        TracksFrame(2, 1_100_000_000, [_track(7, "cue", 35, 20, -300)]),
        policy,
    )
    observer.update(
        FramePacket(3, 1_200_000_000, "cam", image=image_with((35, 20), (60, 45))),
        TracksFrame(
            3,
            1_200_000_000,
            [_track(7, "cue", 35, 20, 0), _track(11, "stripe", 60, 45, -300)],
        ),
        policy,
    )
    second = observer.update(
        FramePacket(4, 1_300_000_000, "cam", image=image_with((35, 20), (60, 4))),
        TracksFrame(
            4,
            1_300_000_000,
            [_track(7, "cue", 35, 20, 0), _track(11, "stripe", 60, 4, -300)],
        ),
        policy,
    )

    assert first.observations[0].inward_crossing is True
    assert first.observations[0].group == "cue"
    assert second.observations[0].inward_crossing is False


def test_visual_centroid_prefers_nearby_visible_ball_over_disappeared_fallback() -> None:
    """Regression: shot 23's stripe was visible near the motion while a cue track was stale."""

    observer = PocketObserver(history_ms=1500)
    policy = _policy()
    empty = np.zeros((100, 100, 3), dtype=np.uint8)
    cue = _track(997, "cue", 10, 75, 0)
    stripe_far = _track(11, "stripe", 76, 75, -300)
    observer.update(
        FramePacket(1, 1_000_000_000, "cam", image=empty),
        TracksFrame(1, 1_000_000_000, [cue, stripe_far]),
        policy,
    )
    # The real shot-23 track is outside the normal guard association but close
    # to the visual crossing centroid.  Keep this fixture just outside both
    # the polygon and the normal 2.25-diameter radius.
    stripe_near = _track(11, "stripe", 76, 23, -300)
    observer.update(
        FramePacket(2, 1_100_000_000, "cam", image=empty),
        TracksFrame(2, 1_100_000_000, [stripe_near]),
        policy,
    )
    mouth = empty.copy()
    cv2.circle(mouth, (50, 9), 6, (240, 240, 240), -1)

    out = observer.update(
        FramePacket(3, 1_200_000_000, "cam", image=mouth),
        TracksFrame(3, 1_200_000_000, [stripe_near]),
        policy,
    )

    observed = out.observations[0]
    assert observed.inward_crossing is True
    assert observed.group == "stripe"
    assert observed.associated_track_ids == [11]


def test_low_fps_terminal_crossing_can_emit_before_reaching_pocket_center() -> None:
    """A ball may advance into the terminal corridor and disappear between source frames."""

    observer = PocketObserver(history_ms=1500)
    policy = _policy()
    empty = _image(None)
    observer.update(FramePacket(1, 1_000_000_000, "cam", image=empty), TracksFrame(1, 1_000_000_000, []), policy)
    observer.update(
        FramePacket(2, 1_100_000_000, "cam", image=_image(35)),
        TracksFrame(2, 1_100_000_000, [_track(31, "stripe", 50, 35, 0)]),
        policy,
    )
    vanished = _track(31, "stripe", 50, 35, 0)
    vanished.visibility = "occluded"
    vanished.lost_frames = 1

    out = observer.update(
        FramePacket(3, 1_216_000_000, "cam", image=_image(16)),
        TracksFrame(3, 1_216_000_000, [vanished]),
        policy,
    )

    observed = out.observations[0]
    assert observed.inward_crossing is True
    assert observed.group == "stripe"
    assert observed.associated_track_ids == [31]
    assert observed.entry_depth_diameters is not None
    assert observed.entry_depth_diameters >= -0.75


def test_recently_disappeared_crossing_beats_stationary_visible_lip_ball() -> None:
    """The moving disappearance owns the crossing when another ball remains by the pocket."""

    observer = PocketObserver(history_ms=1500)
    policy = _policy()

    def image_with(*balls: tuple[int, int, int]) -> np.ndarray:
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        for x, y, value in balls:
            cv2.circle(image, (x, y), 6, (value, value, value), -1)
        return image

    static = _track(4, "solid", 58, 22, 0)
    target = _track(14, "stripe", 50, 35, 0)
    observer.update(
        FramePacket(1, 1_000_000_000, "cam", image=image_with((58, 22, 120), (50, 35, 240))),
        TracksFrame(1, 1_000_000_000, [static, target]),
        policy,
    )
    target.visibility = "occluded"
    target.lost_frames = 1

    out = observer.update(
        FramePacket(2, 1_116_000_000, "cam", image=image_with((58, 22, 120), (50, 16, 240))),
        TracksFrame(2, 1_116_000_000, [static, target]),
        policy,
    )

    observed = out.observations[0]
    assert observed.inward_crossing is True
    assert observed.group == "stripe"
    assert observed.associated_track_ids == [14]
    assert observed.lip_track_ids == [4]


def test_departed_background_ball_does_not_become_unassociated_lip_occupancy() -> None:
    """Absolute background difference after a ball leaves is clear evidence, not a live lip ball."""

    observer = PocketObserver(history_ms=1500)
    policy = _policy()
    observer.update(
        FramePacket(1, 1_000_000_000, "cam", image=_image(20)),
        TracksFrame(1, 1_000_000_000, [_track(9, "solid", 50, 20, 0)]),
        policy,
    )
    observer.update(FramePacket(2, 1_100_000_000, "cam", image=_image(None)), TracksFrame(2, 1_100_000_000, []), policy)

    out = observer.update(
        FramePacket(3, 1_500_000_000, "cam", image=_image(None)),
        TracksFrame(3, 1_500_000_000, []),
        policy,
    )

    observed = out.observations[0]
    assert observed.foreground_score > 0.08
    assert observed.lip_occupied is False
    assert observed.lip_track_ids == []


def test_terminal_motion_does_not_relay_to_distant_disappeared_track() -> None:
    """A remote history vote cannot claim unrelated ball-sized pocket motion."""

    observer = PocketObserver(history_ms=1500)
    policy = _policy()
    empty = _image(None)
    observer.update(FramePacket(1, 1_000_000_000, "cam", image=empty), TracksFrame(1, 1_000_000_000, []), policy)
    observer.update(
        FramePacket(2, 1_100_000_000, "cam", image=empty),
        TracksFrame(2, 1_100_000_000, [_track(77, "solid", 50, 90, 0)]),
        policy,
    )
    vanished = _track(77, "solid", 50, 90, 0)
    vanished.visibility = "occluded"
    vanished.lost_frames = 1

    out = observer.update(
        FramePacket(3, 1_216_000_000, "cam", image=_image(16)),
        TracksFrame(3, 1_216_000_000, [vanished]),
        policy,
    )

    observed = out.observations[0]
    assert observed.inward_crossing is False
    assert observed.associated_track_ids == []


def test_fast_aligned_history_can_extend_recent_association_beyond_base_distance() -> None:
    """A measured high-speed heading may extend the relay beyond the conservative base radius."""

    observer = PocketObserver(history_ms=1500)
    policy = _policy()
    empty = _image(None)
    observer.update(
        FramePacket(1, 1_000_000_000, "cam", image=empty),
        TracksFrame(1, 1_000_000_000, [_track(88, "stripe", 50, 160, -700)]),
        policy,
    )
    observer.update(
        FramePacket(2, 1_100_000_000, "cam", image=empty),
        TracksFrame(2, 1_100_000_000, [_track(88, "stripe", 50, 90, -700)]),
        policy,
    )
    vanished = _track(88, "stripe", 50, 90, -700)
    vanished.visibility = "occluded"
    vanished.lost_frames = 1

    out = observer.update(
        FramePacket(3, 1_216_000_000, "cam", image=_image(16)),
        TracksFrame(3, 1_216_000_000, [vanished]),
        policy,
    )

    observed = out.observations[0]
    assert observed.inward_crossing is True
    assert observed.associated_track_ids == [88]


def test_untracked_ball_still_on_lip_blocks_state_detection() -> None:
    """Current-image lip occupancy remains a veto after the detector loses the ball."""

    observer = PocketObserver(history_ms=1500)
    policy = _policy()
    fsm = PerBallPocketFSM(StateConfig(engine="modern", pocket_visual_confirmation_ms=1300))
    fsm.set_table_context(
        inner_polygon_mm=[(10, 20), (90, 20), (90, 90), (10, 90)],
        pockets_mm=[(50, 8)],
        ball_diameter_mm=12,
    )
    observer.update(
        FramePacket(0, 900_000_000, "cam", image=_image(None)),
        TracksFrame(0, 900_000_000, []),
        policy,
    )
    visible = _track(31, "stripe", 50, 35, 0)
    first_tracks = TracksFrame(1, 1_000_000_000, [visible])
    first_visual = observer.update(FramePacket(1, 1_000_000_000, "cam", image=_image(35)), first_tracks, policy)
    fsm.update(first_tracks, MatchPhase.SHOT_ACTIVE, first_visual)
    vanished = _track(31, "stripe", 50, 35, 0)
    vanished.visibility = "occluded"
    vanished.lost_frames = 1
    crossing_tracks = TracksFrame(2, 1_100_000_000, [vanished])
    crossing_visual = observer.update(
        FramePacket(2, 1_100_000_000, "cam", image=_image(16)),
        crossing_tracks,
        policy,
    )
    candidate = fsm.update(crossing_tracks, MatchPhase.SHOT_ACTIVE, crossing_visual)
    lip_tracks = TracksFrame(3, 1_500_000_000, [])
    lip_visual = observer.update(FramePacket(3, 1_500_000_000, "cam", image=_image(16)), lip_tracks, policy)
    fsm.update(lip_tracks, MatchPhase.SHOT_ACTIVE, lip_visual)
    final_tracks = TracksFrame(4, 2_400_000_000, [])
    final_visual = observer.update(FramePacket(4, 2_400_000_000, "cam", image=_image(16)), final_tracks, policy)
    final = fsm.update(final_tracks, MatchPhase.SHOT_ACTIVE, final_visual)

    assert any(event.name == "POCKET_CANDIDATE" for event in candidate)
    assert lip_visual.observations[0].lip_occupied is True
    assert not any(event.name == "POCKET_DETECTED" for event in final)
