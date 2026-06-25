from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from .paths import DEFAULT_CONFIG_PATH, PROJECT_ROOT, resolve_path


@dataclass
class CameraConfig:
    backend: str = "auto"  # auto | nori | opencv | video | synthetic
    device_index: int = 0
    camera_id: str = "overhead_main"
    width: int = 1920
    height: int = 1080
    fps: int = 30
    video_path: Optional[str] = None
    nori_sdk_root: Optional[str] = None
    nori_device_id: Optional[int] = None
    exposure_auto: Optional[bool] = False
    exposure_level: Optional[int] = -5
    distortion_correction_enabled: bool = True
    distortion_correction_file: Optional[str] = "C:/CodeProject/2606BAS/calib/intrinsics_opencv.yaml"


@dataclass
class DetectorConfig:
    backend: str = "disabled"  # disabled | ultralytics
    model_path: Optional[str] = None
    class_file_path: Optional[str] = None
    class_names: List[str] = field(default_factory=lambda: ["cue", "solid", "stripe", "black", "cue_stick"])
    conf: float = 0.25
    iou: float = 0.5
    device: str = "auto"
    tile_size: int = 1280
    tile_overlap: float = 0.15
    max_det_per_tile: int = 128
    batch_size: int = 4
    detect_interval_frames: int = 1
    detect_fps_limit_hz: float = 10.0


@dataclass
class TrackerConfig:
    high_conf: float = 0.45
    low_conf: float = 0.12
    max_lost_frames: int = 18
    vote_window: int = 24
    match_distance_px: float = 90.0
    match_iou: float = 0.08
    velocity_smoothing: float = 0.35


@dataclass
class CalibrationConfig:
    camera_file: Optional[str] = None
    projection_file: Optional[str] = None
    table_width_mm: float = 2540.0
    table_height_mm: float = 1270.0
    ball_diameter_mm: float = 57.15
    projection_visible_inset_top_mm: float = 0.0
    projection_visible_inset_right_mm: float = 0.0
    projection_visible_inset_bottom_mm: float = 0.0
    projection_visible_inset_left_mm: float = 0.0
    physical_rail_inset_top_mm: float = 0.0
    physical_rail_inset_right_mm: float = 0.0
    physical_rail_inset_bottom_mm: float = 0.0
    physical_rail_inset_left_mm: float = 0.0
    physical_middle_pocket_relief_top_mm: float = 0.0
    physical_middle_pocket_relief_bottom_mm: float = 0.0
    center_reachable_extra_margin_mm: float = 2.0
    ball_center_compensation_enabled: bool = False
    ball_center_compensation_auto_reference: bool = True
    ball_center_compensation_ref_x_px: float = 0.0
    ball_center_compensation_ref_y_px: float = 0.0
    ball_center_compensation_scale_x_pct: float = 0.0
    ball_center_compensation_scale_y_pct: float = 0.0


@dataclass
class GeometryConfig:
    outline_path: Optional[str] = None
    inline_path: Optional[str] = None
    pocket_path: Optional[str] = None


@dataclass
class StateConfig:
    still_speed_px_s: float = 25.0
    moving_speed_px_s: float = 65.0
    still_speed_mm_s: float = 8.0
    moving_speed_mm_s: float = 22.0
    stable_frames: int = 12
    armed_frames: int = 4
    settle_frames: int = 18
    anomaly_frames: int = 8
    collision_epsilon_mm: float = 10.0
    rail_epsilon_mm: float = 14.0
    pocket_funnel_radius_mm: float = 95.0
    event_cooldown_frames: int = 5
    shot_tip_radius_multiplier: float = 3.0
    shot_speed_jump_mm_s: float = 18.0
    shot_accel_mm_s2: float = 120.0


@dataclass
class PlannerConfig:
    enabled: bool = True
    shot_mode: str = "rule"  # rule | free
    max_cut_angle_deg: float = 80.0
    top_k: int = 5
    cue_path_margin_mm: float = 4.0
    object_path_margin_mm: float = 4.0
    collision_padding_mm: float = 2.0
    free_max_collisions: int = 2


@dataclass
class LearningConfig:
    ranker_enabled: bool = True
    ranker_model_path: Optional[str] = None
    score_blend: float = 0.65
    collect_enabled: bool = False
    samples_directory: str = "rl/data/samples"
    min_candidates: int = 1


@dataclass
class ProjectionConfig:
    enabled: bool = True
    projector_width: int = 1280
    projector_height: int = 800
    screen_index: int = 1
    fullscreen: bool = True
    geometry_reference_enabled: bool = True


@dataclass
class ReplayConfig:
    enabled: bool = True
    directory: str = "replays"
    write_video: bool = False
    write_debug_frames: bool = False


@dataclass
class LoggingConfig:
    directory: str = "logs"
    level: str = "INFO"


@dataclass
class AppConfig:
    camera: CameraConfig = field(default_factory=CameraConfig)
    detector: DetectorConfig = field(default_factory=DetectorConfig)
    tracker: TrackerConfig = field(default_factory=TrackerConfig)
    calibration: CalibrationConfig = field(default_factory=CalibrationConfig)
    geometry: GeometryConfig = field(default_factory=GeometryConfig)
    state: StateConfig = field(default_factory=StateConfig)
    planner: PlannerConfig = field(default_factory=PlannerConfig)
    learning: LearningConfig = field(default_factory=LearningConfig)
    projection: ProjectionConfig = field(default_factory=ProjectionConfig)
    replay: ReplayConfig = field(default_factory=ReplayConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)

    @classmethod
    def load(cls, path: str | Path | None = None) -> "AppConfig":
        cfg_path = Path(path) if path else DEFAULT_CONFIG_PATH
        if not cfg_path.exists():
            return cls()
        with cfg_path.open("r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
        return cls.from_dict(raw)

    @classmethod
    def from_dict(cls, raw: Dict[str, Any]) -> "AppConfig":
        def section(name: str, typ: type) -> Any:
            data = raw.get(name, {})
            if data is None:
                data = {}
            return typ(**{k: v for k, v in dict(data).items() if k in typ.__dataclass_fields__})

        return cls(
            camera=section("camera", CameraConfig),
            detector=section("detector", DetectorConfig),
            tracker=section("tracker", TrackerConfig),
            calibration=section("calibration", CalibrationConfig),
            geometry=section("geometry", GeometryConfig),
            state=section("state", StateConfig),
            planner=section("planner", PlannerConfig),
            learning=section("learning", LearningConfig),
            projection=section("projection", ProjectionConfig),
            replay=section("replay", ReplayConfig),
            logging=section("logging", LoggingConfig),
        )

    def resolve_paths(self) -> "AppConfig":
        base = PROJECT_ROOT
        if self.camera.video_path:
            p = resolve_path(self.camera.video_path, base=base)
            self.camera.video_path = str(p) if p else None
        if self.camera.nori_sdk_root:
            p = resolve_path(self.camera.nori_sdk_root, base=base)
            self.camera.nori_sdk_root = str(p) if p else None
        if self.camera.distortion_correction_file:
            p = resolve_path(self.camera.distortion_correction_file, base=base)
            self.camera.distortion_correction_file = str(p) if p else None
        if self.detector.model_path:
            p = resolve_path(self.detector.model_path, base=base)
            self.detector.model_path = str(p) if p else None
        if self.detector.class_file_path:
            p = resolve_path(self.detector.class_file_path, base=base)
            self.detector.class_file_path = str(p) if p else None
        if self.calibration.camera_file:
            p = resolve_path(self.calibration.camera_file, base=base)
            self.calibration.camera_file = str(p) if p else None
        if self.calibration.projection_file:
            p = resolve_path(self.calibration.projection_file, base=base)
            self.calibration.projection_file = str(p) if p else None
        if self.geometry.outline_path:
            p = resolve_path(self.geometry.outline_path, base=base)
            self.geometry.outline_path = str(p) if p else None
        if self.geometry.inline_path:
            p = resolve_path(self.geometry.inline_path, base=base)
            self.geometry.inline_path = str(p) if p else None
        if self.geometry.pocket_path:
            p = resolve_path(self.geometry.pocket_path, base=base)
            self.geometry.pocket_path = str(p) if p else None
        if self.learning.ranker_model_path:
            p = resolve_path(self.learning.ranker_model_path, base=base)
            self.learning.ranker_model_path = str(p) if p else None
        self.learning.samples_directory = str(resolve_path(self.learning.samples_directory, base=base) or Path(self.learning.samples_directory))
        self.logging.directory = str(resolve_path(self.logging.directory, base=base) or Path(self.logging.directory))
        self.replay.directory = str(resolve_path(self.replay.directory, base=base) or Path(self.replay.directory))
        return self
