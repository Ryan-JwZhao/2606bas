from __future__ import annotations

import re
import time
from dataclasses import dataclass
from typing import Dict

import numpy as np

from ..config import TrackerConfig
from ..schemas import Detection, DetectionsFrame, TrackObservation, TracksFrame


_BALL_NUMBER_RE = re.compile(r"(?:ball[_\- ]*)?(\d{1,2})$", re.IGNORECASE)


def parse_ball_number(value: object, *, cls_id: int | None = None) -> int | None:
    match = _BALL_NUMBER_RE.fullmatch(str(value or "").strip())
    if match is not None:
        number = int(match.group(1))
        return number if 0 <= number <= 15 else None
    if cls_id is not None and 0 <= int(cls_id) <= 15:
        return int(cls_id)
    return None


def numbered_ball_group(number: int) -> str:
    if number == 0:
        return "cue"
    if 1 <= number <= 7:
        return "solid"
    if number == 8:
        return "black"
    if 9 <= number <= 15:
        return "stripe"
    return "other"


def ball_number_from_track(track: TrackObservation) -> int | None:
    return parse_ball_number(track.cls_name)


@dataclass
class _NumberTrack:
    number: int
    bbox: tuple[float, float, float, float]
    center: np.ndarray
    radius_px: float
    confidence: float
    last_ts_ns: int
    velocity: np.ndarray
    age: int = 1
    lost_frames: int = 0


class NumberedBallTracker:
    """Identity-first tracker for a model where each ball number is a unique class."""

    version = "numbered_ball_identity_v1"

    def __init__(self, config: TrackerConfig):
        self.config = config
        self._tracks: Dict[int, _NumberTrack] = {}

    def reset(self) -> None:
        self._tracks.clear()

    def update(self, detections_frame: DetectionsFrame) -> TracksFrame:
        started = time.perf_counter()
        selected: dict[int, Detection] = {}
        for detection in detections_frame.detections:
            if float(detection.conf) < float(self.config.low_conf):
                continue
            number = parse_ball_number(detection.cls_name, cls_id=detection.cls_id)
            if number is None:
                continue
            previous = selected.get(number)
            if previous is None or float(detection.conf) > float(previous.conf):
                selected[number] = detection

        for number, track in list(self._tracks.items()):
            detection = selected.pop(number, None)
            if detection is None:
                track.lost_frames += 1
                if track.lost_frames > int(self.config.max_lost_frames):
                    del self._tracks[number]
                continue
            self._update(track, detection, detections_frame.ts_cam_ns)

        for number, detection in selected.items():
            if float(detection.conf) < float(self.config.high_conf):
                continue
            center = np.asarray(detection.center, dtype=np.float32)
            self._tracks[number] = _NumberTrack(
                number=number,
                bbox=tuple(float(value) for value in detection.bbox),
                center=center,
                radius_px=float(detection.radius_px),
                confidence=float(detection.conf),
                last_ts_ns=int(detections_frame.ts_cam_ns),
                velocity=np.zeros((2,), dtype=np.float32),
            )

        observations = [self._observation(track) for _, track in sorted(self._tracks.items())]
        return TracksFrame(
            frame_id=detections_frame.frame_id,
            ts_cam_ns=detections_frame.ts_cam_ns,
            tracks=observations,
            tracker_version=self.version,
            latency_ms=float((time.perf_counter() - started) * 1000.0),
        )

    def _update(self, track: _NumberTrack, detection: Detection, ts_ns: int) -> None:
        center = np.asarray(detection.center, dtype=np.float32)
        dt = max(1e-6, (int(ts_ns) - int(track.last_ts_ns)) / 1e9)
        instant_velocity = (center - track.center) / dt
        alpha = max(0.0, min(1.0, float(self.config.velocity_smoothing)))
        track.velocity = (1.0 - alpha) * track.velocity + alpha * instant_velocity
        track.center = center
        track.bbox = tuple(float(value) for value in detection.bbox)
        track.radius_px = float(detection.radius_px)
        track.confidence = float(detection.conf)
        track.last_ts_ns = int(ts_ns)
        track.age += 1
        track.lost_frames = 0

    @staticmethod
    def _observation(track: _NumberTrack) -> TrackObservation:
        quality = float(track.confidence) * (0.72 ** max(0, track.lost_frames))
        return TrackObservation(
            track_id=int(track.number),
            bbox=track.bbox,
            center_px=(float(track.center[0]), float(track.center[1])),
            radius_px=float(track.radius_px),
            cls_name=str(track.number),
            group=numbered_ball_group(track.number),
            confidence=float(track.confidence),
            velocity_px_s=(float(track.velocity[0]), float(track.velocity[1])),
            quality=quality,
            age=int(track.age),
            lost_frames=int(track.lost_frames),
            visibility="visible" if track.lost_frames == 0 else "occluded",
        )
