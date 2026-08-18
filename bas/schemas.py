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
    geometry_version: str = "unversioned"


@dataclass
class Detection:
    bbox: BBox
    conf: float
    cls_id: int
    cls_name: str
    refined_center_px: Optional[Point] = None
    refined_radius_px: Optional[float] = None
    geometry_quality: float = 0.45
    geometry_method: str = "bbox"

    @property
    def center(self) -> Point:
        if self.refined_center_px is not None:
            return (float(self.refined_center_px[0]), float(self.refined_center_px[1]))
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) * 0.5, (y1 + y2) * 0.5)

    @property
    def radius_px(self) -> float:
        if self.refined_radius_px is not None:
            return max(2.0, float(self.refined_radius_px))
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
    center_mm: Optional[Point] = None
    velocity_mm_s: Optional[Point] = None
    radius_mm: Optional[float] = None
    quality: float = 1.0
    age: int = 1
    confirmed: bool = True
    lost_frames: int = 0
    visibility: str = "visible"
    geometry_quality: float = 1.0
    geometry_method: str = "unknown"


@dataclass
class TracksFrame:
    frame_id: int
    ts_cam_ns: int
    tracks: List[TrackObservation] = field(default_factory=list)
    tracker_version: str = "centroid_v1"
    latency_ms: float = 0.0


@dataclass
class PocketVisualObservation:
    """Compact, tracker-associated visual evidence for one physical pocket."""

    pocket_index: int
    inward_crossing: bool = False
    outward_crossing: bool = False
    lip_occupied: bool = False
    clear: bool = False
    group: Optional[str] = None
    confidence: float = 0.0
    associated_track_ids: List[int] = field(default_factory=list)
    evidence_sources: List[str] = field(default_factory=list)
    motion_score: float = 0.0
    foreground_score: float = 0.0
    foreground_center_px: Optional[Point] = None
    foreground_depth_diameters: Optional[float] = None


@dataclass
class PocketVisualObservationFrame:
    frame_id: int
    ts_cam_ns: int
    observations: List[PocketVisualObservation] = field(default_factory=list)
    latency_ms: float = 0.0
    observer_version: str = "pocket_observer_v1"


@dataclass
class TableModel:
    width_mm: float
    height_mm: float
    ball_diameter_mm: float
    inner_polygon_mm: List[Point]
    pockets_mm: List[Point]
    projection_visible_polygon_mm: List[Point] = field(default_factory=list)
    center_playable_polygon_mm: List[Point] = field(default_factory=list)
    projection_visible_pockets_mm: List[Point] = field(default_factory=list)
    # Shot planning uses raw fitted pocket-arc centres.  Keep this separate from
    # pockets_mm because that legacy field is also consumed by pocket judging.
    planning_pockets_mm: List[Point] = field(default_factory=list)


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
    turn_target_group: Optional[str] = None
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
    shot_mode: str = "rule"
    hook_status: str = "off"
    planner_version: str = "geometry_physics_mvp_v1"
    locked_target_id: Optional[int] = None
    target_lock_status: str = "off"
    target_shot_status: str = "off"


@dataclass
class OverlayLine:
    points: List[Point]
    color: Tuple[int, int, int] = (0, 255, 128)
    width: int = 3
    label: Optional[str] = None
    style: str = "solid"
    arrow: bool = False


@dataclass
class OverlayCircle:
    center: Point
    radius: float
    color: Tuple[int, int, int] = (255, 255, 255)
    width: int = 2
    radius_y: Optional[float] = None
    rotation_deg: float = 0.0
    stability_key: Optional[str] = None


@dataclass
class OverlayText:
    position: Point
    text: str
    color: Tuple[int, int, int] = (255, 255, 255)
    font_size_px: float = 36.0
    max_width_ratio: float = 0.9
    outline_width_px: float = 2.0
    background_alpha: int = 110
    rotation_deg: float = 0.0


@dataclass
class ProjectionOverlay:
    overlay_id: str
    frame_id: int
    projector_size: Tuple[int, int]
    lines: List[OverlayLine] = field(default_factory=list)
    circles: List[OverlayCircle] = field(default_factory=list)
    labels: List[Tuple[Point, str, Tuple[int, int, int]]] = field(default_factory=list)
    texts: List[OverlayText] = field(default_factory=list)
    suppress_star_formula: bool = False


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
