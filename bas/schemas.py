from __future__ import annotations

from dataclasses import asdict, dataclass, field, is_dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

Point = Tuple[float, float]
BBox = Tuple[float, float, float, float]


class BallGroup(str, Enum):
    CUE = "cue"
    SOLID = "solid"
    STRIPE = "stripe"
    BLACK = "black"
    OTHER = "other"


class MatchPhase(str, Enum):
    STABLE_IDLE = "STABLE_IDLE"
    PRE_SHOT_ARMED = "PRE_SHOT_ARMED"
    SHOT_ACTIVE = "SHOT_ACTIVE"
    SETTLING = "SETTLING"
    TURN_RESOLVE = "TURN_RESOLVE"
    ANOMALY_RECOVERY = "ANOMALY_RECOVERY"


@dataclass
class FramePacket:
    frame_id: int
    ts_cam_ns: int
    camera_id: str
    image: Optional[np.ndarray] = None
    image_uri: Optional[str] = None
    exposure_meta: Dict[str, Any] = field(default_factory=dict)
    calib_version: str = "unversioned"


@dataclass
class Detection:
    bbox: BBox
    conf: float
    cls_id: int
    cls_name: str

    @property
    def center(self) -> Point:
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) * 0.5, (y1 + y2) * 0.5)

    @property
    def radius_px(self) -> float:
        x1, y1, x2, y2 = self.bbox
        return max(2.0, 0.25 * ((x2 - x1) + (y2 - y1)))


@dataclass
class DetectionsFrame:
    frame_id: int
    ts_cam_ns: int
    detections: List[Detection] = field(default_factory=list)
    detector_version: str = "disabled"
    latency_ms: float = 0.0


@dataclass
class TrackObservation:
    track_id: int
    bbox: BBox
    center_px: Point
    radius_px: float
    cls_name: str
    group: str
    confidence: float
    velocity_px_s: Point = (0.0, 0.0)
    quality: float = 1.0
    age: int = 1
    lost_frames: int = 0
    visibility: str = "visible"


@dataclass
class TracksFrame:
    frame_id: int
    ts_cam_ns: int
    tracks: List[TrackObservation] = field(default_factory=list)
    tracker_version: str = "centroid_v1"
    latency_ms: float = 0.0


@dataclass
class TableModel:
    width_mm: float
    height_mm: float
    ball_diameter_mm: float
    inner_polygon_mm: List[Point]
    pockets_mm: List[Point]


@dataclass
class Event:
    name: str
    ts_cam_ns: int
    frame_id: int
    payload: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 1.0


@dataclass
class MatchStateFrame:
    frame_id: int
    ts_cam_ns: int
    phase: str
    events: List[Event] = field(default_factory=list)
    layout: List[TrackObservation] = field(default_factory=list)
    confidence: float = 1.0
    state_version: str = "temporal_state_v1"


@dataclass
class ShotCandidate:
    candidate_id: str
    cue_track_id: int
    target_track_id: int
    target_group: str
    pocket_index: int
    cue_ball: Point
    object_ball: Point
    ghost_ball: Point
    pocket_point: Point
    aim_line: List[Point]
    object_line: List[Point]
    cut_angle_deg: float
    cue_distance_mm: float
    object_distance_mm: float
    score: float
    risk: float
    explanation: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ShotPlan:
    plan_id: str
    frame_id: int
    ts_cam_ns: int
    candidates: List[ShotCandidate] = field(default_factory=list)
    best: Optional[ShotCandidate] = None
    planner_version: str = "geometry_physics_mvp_v1"


@dataclass
class OverlayLine:
    points: List[Point]
    color: Tuple[int, int, int] = (0, 255, 128)
    width: int = 3
    label: Optional[str] = None


@dataclass
class ProjectionOverlay:
    overlay_id: str
    frame_id: int
    projector_size: Tuple[int, int]
    lines: List[OverlayLine] = field(default_factory=list)
    circles: List[Tuple[Point, float, Tuple[int, int, int]]] = field(default_factory=list)
    labels: List[Tuple[Point, str, Tuple[int, int, int]]] = field(default_factory=list)


def to_jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value):
        return {k: to_jsonable(v) for k, v in asdict(value).items()}
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    return value

