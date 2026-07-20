from __future__ import annotations

import cv2
import numpy as np

from bas.perception.pocket_observer import PocketObserver
from bas.perception.regions import DetectionRegionPolicy, PocketGuardRegion
from bas.schemas import FramePacket, TrackObservation, TracksFrame


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
