from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

from .config import AppConfig
from .paths import PROJECT_ROOT


SETTINGS_PATH = PROJECT_ROOT / "local_settings" / "user_settings.json"


@dataclass
class UserSettings:
    camera_backend: Optional[str] = None
    camera_device_index: Optional[int] = None
    nori_device_id: Optional[int] = None
    width: Optional[int] = None
    height: Optional[int] = None
    fps: Optional[int] = None
    video_path: Optional[str] = None
    nori_sdk_root: Optional[str] = None
    exposure_auto: Optional[bool] = None
    exposure_level: Optional[int] = None
    white_balance_auto: Optional[bool] = None
    white_balance_value: Optional[int] = None
    distortion_correction_enabled: Optional[bool] = None
    distortion_correction_file: Optional[str] = None
    detector_backend: Optional[str] = None
    model_path: Optional[str] = None
    class_file_path: Optional[str] = None
    detect_interval_frames: Optional[int] = None
    detect_fps_limit_hz: Optional[float] = None
    outline_path: Optional[str] = None
    inline_path: Optional[str] = None
    pocket_path: Optional[str] = None
    camera_calibration_file: Optional[str] = None
    projection_calibration_file: Optional[str] = None
    projection_mode: Optional[str] = None
    legacy_projection_calibration_file: Optional[str] = None
    engineered_plane_projection_file: Optional[str] = None
    engineered_ball_compensation_file: Optional[str] = None
    projection_screen_index: Optional[int] = None
    projection_width: Optional[int] = None
    projection_height: Optional[int] = None
    projection_visible_inset_top_mm: Optional[float] = None
    projection_visible_inset_right_mm: Optional[float] = None
    projection_visible_inset_bottom_mm: Optional[float] = None
    projection_visible_inset_left_mm: Optional[float] = None
    physical_rail_inset_top_mm: Optional[float] = None
    physical_rail_inset_right_mm: Optional[float] = None
    physical_rail_inset_bottom_mm: Optional[float] = None
    physical_rail_inset_left_mm: Optional[float] = None
    physical_middle_pocket_relief_top_mm: Optional[float] = None
    physical_middle_pocket_relief_bottom_mm: Optional[float] = None
    center_reachable_extra_margin_mm: Optional[float] = None
    ball_center_compensation_enabled: Optional[bool] = None
    ball_center_compensation_auto_reference: Optional[bool] = None
    ball_center_compensation_ref_x_px: Optional[float] = None
    ball_center_compensation_ref_y_px: Optional[float] = None
    ball_center_compensation_scale_x_pct: Optional[float] = None
    ball_center_compensation_scale_y_pct: Optional[float] = None
    projection_geometry_reference_enabled: Optional[bool] = None
    ui_geometry_reference_enabled: Optional[bool] = None
    replay_enabled: Optional[bool] = None
    state_machine_engine: Optional[str] = None
    shot_mode: Optional[str] = None
    planner_route_freeze_enabled: Optional[bool] = None
    planner_route_freeze_enter_frames: Optional[int] = None
    planner_route_freeze_release_frames: Optional[int] = None
    planner_route_freeze_same_route_refresh_mm: Optional[float] = None
    planner_route_freeze_same_route_refresh_score_delta: Optional[float] = None
    planner_route_freeze_switch_confirm_frames: Optional[int] = None
    planner_route_freeze_switch_min_distance_mm: Optional[float] = None
    planner_route_freeze_switch_min_score_delta: Optional[float] = None
    planner_cue_sector_correction_enabled: Optional[bool] = None
    planner_cue_sector_angle_deg: Optional[float] = None
    planner_cue_sector_edge_margin_deg: Optional[float] = None
    planner_cue_sector_corridor_width_px: Optional[float] = None
    planner_cue_sector_switch_confirm_frames: Optional[int] = None
    planner_cue_sector_switch_min_score_delta: Optional[float] = None
    planner_cue_sector_lock_margin_px: Optional[float] = None
    planner_cue_sector_lock_forward_tolerance_px: Optional[float] = None
    planner_cue_sector_lock_release_frames: Optional[int] = None
    planner_cue_sector_min_stick_quality: Optional[float] = None
    planner_cue_sector_require_balls_stationary: Optional[bool] = None
    planner_cue_sector_stationary_speed_mm_s: Optional[float] = None
    planner_cue_sector_stationary_speed_px_s: Optional[float] = None
    planner_target_lock_enabled: Optional[bool] = None
    planner_target_lock_confirm_frames: Optional[int] = None
    planner_target_lock_switch_confirm_frames: Optional[int] = None
    planner_target_shot_enabled: Optional[bool] = None
    planner_target_shot_trigger_frames: Optional[int] = None
    learning_ranker_enabled: Optional[bool] = None
    learning_ranker_model_path: Optional[str] = None
    learning_score_blend: Optional[float] = None
    learning_collect_enabled: Optional[bool] = None
    learning_samples_directory: Optional[str] = None
    star_formula: Dict[str, Any] = field(default_factory=dict)
    _provided_keys: set[str] = field(default_factory=set, repr=False, compare=False)
    _loaded_from_file: bool = field(default=False, repr=False, compare=False)

    @classmethod
    def load(cls, path: Path = SETTINGS_PATH) -> "UserSettings":
        if not path.exists():
            return cls()
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            return cls()
        allowed = {k for k in cls.__dataclass_fields__.keys() if not k.startswith("_")}
        settings = cls(**{k: v for k, v in data.items() if k in allowed})
        settings._provided_keys = {k for k in data.keys() if k in allowed}
        settings._loaded_from_file = True
        return settings

    def save(self, path: Path = SETTINGS_PATH) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        data = asdict(self)
        data.pop("_provided_keys", None)
        data.pop("_loaded_from_file", None)
        with path.open("w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def apply_to_config(self, config: AppConfig) -> AppConfig:
        if self._has("camera_backend") and self.camera_backend:
            config.camera.backend = self.camera_backend
        if self.camera_device_index is not None:
            config.camera.device_index = int(self.camera_device_index)
        if self._has("nori_device_id"):
            config.camera.nori_device_id = int(self.nori_device_id) if self.nori_device_id is not None else None
        elif self.nori_device_id is not None:
            config.camera.nori_device_id = int(self.nori_device_id)
        if self.width is not None:
            config.camera.width = int(self.width)
        if self.height is not None:
            config.camera.height = int(self.height)
        if self.fps is not None:
            config.camera.fps = int(self.fps)
        if self._has("video_path"):
            config.camera.video_path = _clean_optional_text(self.video_path)
        if self._has("nori_sdk_root"):
            config.camera.nori_sdk_root = _clean_optional_text(self.nori_sdk_root)
        if self.exposure_auto is not None:
            config.camera.exposure_auto = bool(self.exposure_auto)
        if self.exposure_level is not None:
            config.camera.exposure_level = max(-10, min(0, int(self.exposure_level)))
        if self.white_balance_auto is not None:
            config.camera.white_balance_auto = bool(self.white_balance_auto)
        if self.white_balance_value is not None:
            config.camera.white_balance_value = max(1000, min(15000, int(self.white_balance_value)))
        if self.distortion_correction_enabled is not None:
            config.camera.distortion_correction_enabled = bool(self.distortion_correction_enabled)
        if self._has("distortion_correction_file"):
            config.camera.distortion_correction_file = _clean_optional_text(self.distortion_correction_file)
        if self._has("detector_backend") and self.detector_backend:
            config.detector.backend = self.detector_backend
        if self._has("model_path"):
            config.detector.model_path = _clean_optional_text(self.model_path)
        if self._has("class_file_path"):
            config.detector.class_file_path = _clean_optional_text(self.class_file_path)
        if self.detect_interval_frames is not None:
            config.detector.detect_interval_frames = max(1, int(self.detect_interval_frames))
        if self.detect_fps_limit_hz is not None:
            config.detector.detect_fps_limit_hz = max(0.0, float(self.detect_fps_limit_hz))
        if self._has("outline_path"):
            config.geometry.outline_path = _clean_optional_text(self.outline_path)
        if self._has("inline_path"):
            config.geometry.inline_path = _clean_optional_text(self.inline_path)
        if self._has("pocket_path"):
            config.geometry.pocket_path = _clean_optional_text(self.pocket_path)
        if self._has("camera_calibration_file"):
            config.calibration.camera_file = _clean_optional_text(self.camera_calibration_file)
        if self._has("projection_mode") and self.projection_mode:
            config.calibration.set_projection_mode(self.projection_mode)
        if self._has("legacy_projection_calibration_file"):
            config.calibration.legacy_projection_file = _clean_optional_text(self.legacy_projection_calibration_file)
        if self._has("engineered_plane_projection_file"):
            config.calibration.engineered_plane_projection_file = _clean_optional_text(self.engineered_plane_projection_file)
        if self._has("engineered_ball_compensation_file"):
            config.calibration.engineered_ball_compensation_file = _clean_optional_text(self.engineered_ball_compensation_file)
        if self._has("projection_calibration_file"):
            config.calibration.projection_file = _clean_optional_text(self.projection_calibration_file)
            if config.calibration.projection_file:
                if not self._has("legacy_projection_calibration_file") and config.calibration.normalized_projection_mode() == "legacy":
                    config.calibration.legacy_projection_file = config.calibration.projection_file
                if not self._has("engineered_plane_projection_file") and config.calibration.normalized_projection_mode() == "engineered":
                    config.calibration.engineered_plane_projection_file = config.calibration.projection_file
            elif not self._has("legacy_projection_calibration_file") and config.calibration.normalized_projection_mode() == "legacy":
                config.calibration.legacy_projection_file = None
            elif not self._has("engineered_plane_projection_file") and config.calibration.normalized_projection_mode() == "engineered":
                config.calibration.engineered_plane_projection_file = None
        config.calibration.sync_projection_file_alias()
        if self.projection_screen_index is not None:
            config.projection.screen_index = int(self.projection_screen_index)
        if self.projection_width is not None:
            config.projection.projector_width = int(self.projection_width)
        if self.projection_height is not None:
            config.projection.projector_height = int(self.projection_height)
        if self.projection_visible_inset_top_mm is not None:
            config.calibration.projection_visible_inset_top_mm = max(0.0, float(self.projection_visible_inset_top_mm))
        if self.projection_visible_inset_right_mm is not None:
            config.calibration.projection_visible_inset_right_mm = max(0.0, float(self.projection_visible_inset_right_mm))
        if self.projection_visible_inset_bottom_mm is not None:
            config.calibration.projection_visible_inset_bottom_mm = max(0.0, float(self.projection_visible_inset_bottom_mm))
        if self.projection_visible_inset_left_mm is not None:
            config.calibration.projection_visible_inset_left_mm = max(0.0, float(self.projection_visible_inset_left_mm))
        if self.physical_rail_inset_top_mm is not None:
            config.calibration.physical_rail_inset_top_mm = max(0.0, float(self.physical_rail_inset_top_mm))
        if self.physical_rail_inset_right_mm is not None:
            config.calibration.physical_rail_inset_right_mm = max(0.0, float(self.physical_rail_inset_right_mm))
        if self.physical_rail_inset_bottom_mm is not None:
            config.calibration.physical_rail_inset_bottom_mm = max(0.0, float(self.physical_rail_inset_bottom_mm))
        if self.physical_rail_inset_left_mm is not None:
            config.calibration.physical_rail_inset_left_mm = max(0.0, float(self.physical_rail_inset_left_mm))
        if self.physical_middle_pocket_relief_top_mm is not None:
            config.calibration.physical_middle_pocket_relief_top_mm = max(0.0, float(self.physical_middle_pocket_relief_top_mm))
        if self.physical_middle_pocket_relief_bottom_mm is not None:
            config.calibration.physical_middle_pocket_relief_bottom_mm = max(0.0, float(self.physical_middle_pocket_relief_bottom_mm))
        if self.center_reachable_extra_margin_mm is not None:
            config.calibration.center_reachable_extra_margin_mm = max(0.0, float(self.center_reachable_extra_margin_mm))
        if self.ball_center_compensation_enabled is not None:
            config.calibration.ball_center_compensation_enabled = bool(self.ball_center_compensation_enabled)
        if self.ball_center_compensation_auto_reference is not None:
            config.calibration.ball_center_compensation_auto_reference = bool(self.ball_center_compensation_auto_reference)
        elif self._has("ball_center_compensation_ref_x_px") or self._has("ball_center_compensation_ref_y_px"):
            ref_x = float(self.ball_center_compensation_ref_x_px) if self.ball_center_compensation_ref_x_px is not None else 0.0
            ref_y = float(self.ball_center_compensation_ref_y_px) if self.ball_center_compensation_ref_y_px is not None else 0.0
            config.calibration.ball_center_compensation_auto_reference = bool(ref_x == -1.0 and ref_y == -1.0)
        if self.ball_center_compensation_ref_x_px is not None:
            config.calibration.ball_center_compensation_ref_x_px = float(self.ball_center_compensation_ref_x_px)
        if self.ball_center_compensation_ref_y_px is not None:
            config.calibration.ball_center_compensation_ref_y_px = float(self.ball_center_compensation_ref_y_px)
        if self.ball_center_compensation_scale_x_pct is not None:
            config.calibration.ball_center_compensation_scale_x_pct = float(self.ball_center_compensation_scale_x_pct)
        if self.ball_center_compensation_scale_y_pct is not None:
            config.calibration.ball_center_compensation_scale_y_pct = float(self.ball_center_compensation_scale_y_pct)
        if self.ui_geometry_reference_enabled is not None:
            config.projection.geometry_reference_enabled = bool(self.ui_geometry_reference_enabled)
        elif self.projection_geometry_reference_enabled is not None:
            config.projection.geometry_reference_enabled = bool(self.projection_geometry_reference_enabled)
        if self.replay_enabled is not None:
            config.replay.enabled = bool(self.replay_enabled)
        if self._has("state_machine_engine") and self.state_machine_engine:
            engine = str(self.state_machine_engine).strip().lower()
            config.state.engine = "modern" if engine in {"modern", "new", "v2", "state_machine_new"} else "legacy"
        if self._has("shot_mode") and self.shot_mode:
            mode = str(self.shot_mode).strip().lower()
            config.planner.shot_mode = "free" if mode in {"free", "free_shot"} else "rule"
        if self.planner_route_freeze_enabled is not None:
            config.planner.route_freeze_enabled = bool(self.planner_route_freeze_enabled)
        if self.planner_route_freeze_enter_frames is not None:
            config.planner.route_freeze_enter_frames = max(1, int(self.planner_route_freeze_enter_frames))
        if self.planner_route_freeze_release_frames is not None:
            config.planner.route_freeze_release_frames = max(1, int(self.planner_route_freeze_release_frames))
        if self.planner_route_freeze_same_route_refresh_mm is not None:
            config.planner.route_freeze_same_route_refresh_mm = max(0.0, float(self.planner_route_freeze_same_route_refresh_mm))
        if self.planner_route_freeze_same_route_refresh_score_delta is not None:
            config.planner.route_freeze_same_route_refresh_score_delta = max(
                0.0,
                float(self.planner_route_freeze_same_route_refresh_score_delta),
            )
        if self.planner_route_freeze_switch_confirm_frames is not None:
            config.planner.route_freeze_switch_confirm_frames = max(1, int(self.planner_route_freeze_switch_confirm_frames))
        if self.planner_route_freeze_switch_min_distance_mm is not None:
            config.planner.route_freeze_switch_min_distance_mm = max(0.0, float(self.planner_route_freeze_switch_min_distance_mm))
        if self.planner_route_freeze_switch_min_score_delta is not None:
            config.planner.route_freeze_switch_min_score_delta = max(0.0, float(self.planner_route_freeze_switch_min_score_delta))
        if self.planner_cue_sector_correction_enabled is not None:
            config.planner.cue_sector_correction_enabled = bool(self.planner_cue_sector_correction_enabled)
        if self.planner_cue_sector_angle_deg is not None:
            config.planner.cue_sector_angle_deg = max(1.0, min(120.0, float(self.planner_cue_sector_angle_deg)))
        if self.planner_cue_sector_edge_margin_deg is not None:
            config.planner.cue_sector_edge_margin_deg = max(0.0, min(30.0, float(self.planner_cue_sector_edge_margin_deg)))
        if self.planner_cue_sector_corridor_width_px is not None:
            config.planner.cue_sector_corridor_width_px = max(1.0, float(self.planner_cue_sector_corridor_width_px))
        if self.planner_cue_sector_switch_confirm_frames is not None:
            config.planner.cue_sector_switch_confirm_frames = max(1, int(self.planner_cue_sector_switch_confirm_frames))
        if self.planner_cue_sector_switch_min_score_delta is not None:
            config.planner.cue_sector_switch_min_score_delta = max(0.0, float(self.planner_cue_sector_switch_min_score_delta))
        if self.planner_cue_sector_lock_margin_px is not None:
            config.planner.cue_sector_lock_margin_px = max(0.0, float(self.planner_cue_sector_lock_margin_px))
        if self.planner_cue_sector_lock_forward_tolerance_px is not None:
            config.planner.cue_sector_lock_forward_tolerance_px = max(0.0, float(self.planner_cue_sector_lock_forward_tolerance_px))
        if self.planner_cue_sector_lock_release_frames is not None:
            config.planner.cue_sector_lock_release_frames = max(1, int(self.planner_cue_sector_lock_release_frames))
        if self.planner_cue_sector_min_stick_quality is not None:
            config.planner.cue_sector_min_stick_quality = max(0.0, min(1.0, float(self.planner_cue_sector_min_stick_quality)))
        if self.planner_cue_sector_require_balls_stationary is not None:
            config.planner.cue_sector_require_balls_stationary = bool(self.planner_cue_sector_require_balls_stationary)
        if self.planner_cue_sector_stationary_speed_mm_s is not None:
            config.planner.cue_sector_stationary_speed_mm_s = max(0.0, float(self.planner_cue_sector_stationary_speed_mm_s))
        if self.planner_cue_sector_stationary_speed_px_s is not None:
            config.planner.cue_sector_stationary_speed_px_s = max(0.0, float(self.planner_cue_sector_stationary_speed_px_s))
        if self.planner_target_lock_enabled is not None:
            config.planner.target_lock_enabled = bool(self.planner_target_lock_enabled)
        if self.planner_target_lock_confirm_frames is not None:
            config.planner.target_lock_confirm_frames = max(1, int(self.planner_target_lock_confirm_frames))
        if self.planner_target_lock_switch_confirm_frames is not None:
            config.planner.target_lock_switch_confirm_frames = max(1, int(self.planner_target_lock_switch_confirm_frames))
        if self.planner_target_shot_enabled is not None:
            config.planner.target_shot_enabled = bool(self.planner_target_shot_enabled)
        if self.planner_target_shot_trigger_frames is not None:
            config.planner.target_shot_trigger_frames = max(1, int(self.planner_target_shot_trigger_frames))
        if self.learning_ranker_enabled is not None:
            config.learning.ranker_enabled = bool(self.learning_ranker_enabled)
        if self._has("learning_ranker_model_path"):
            config.learning.ranker_model_path = _clean_optional_text(self.learning_ranker_model_path)
        if self.learning_score_blend is not None:
            config.learning.score_blend = max(0.0, min(1.0, float(self.learning_score_blend)))
        if self.learning_collect_enabled is not None:
            config.learning.collect_enabled = bool(self.learning_collect_enabled)
        if self._has("learning_samples_directory") and self.learning_samples_directory:
            config.learning.samples_directory = self.learning_samples_directory
        return config

    def _has(self, name: str) -> bool:
        if self._loaded_from_file:
            return name in self._provided_keys
        return getattr(self, name) is not None

    @classmethod
    def from_config(cls, config: AppConfig, star_formula: Optional[Dict[str, Any]] = None) -> "UserSettings":
        return cls(
            camera_backend=config.camera.backend,
            camera_device_index=config.camera.device_index,
            nori_device_id=config.camera.nori_device_id,
            width=config.camera.width,
            height=config.camera.height,
            fps=config.camera.fps,
            video_path=config.camera.video_path,
            nori_sdk_root=config.camera.nori_sdk_root,
            exposure_auto=config.camera.exposure_auto,
            exposure_level=config.camera.exposure_level,
            white_balance_auto=config.camera.white_balance_auto,
            white_balance_value=config.camera.white_balance_value,
            distortion_correction_enabled=config.camera.distortion_correction_enabled,
            distortion_correction_file=config.camera.distortion_correction_file,
            detector_backend=config.detector.backend,
            model_path=config.detector.model_path,
            class_file_path=config.detector.class_file_path,
            detect_interval_frames=config.detector.detect_interval_frames,
            detect_fps_limit_hz=config.detector.detect_fps_limit_hz,
            outline_path=config.geometry.outline_path,
            inline_path=config.geometry.inline_path,
            pocket_path=config.geometry.pocket_path,
            camera_calibration_file=config.calibration.camera_file,
            projection_calibration_file=config.calibration.active_projection_file(),
            projection_mode=config.calibration.normalized_projection_mode(),
            legacy_projection_calibration_file=config.calibration.legacy_projection_file,
            engineered_plane_projection_file=config.calibration.engineered_plane_projection_file,
            engineered_ball_compensation_file=config.calibration.engineered_ball_compensation_file,
            projection_screen_index=config.projection.screen_index,
            projection_width=config.projection.projector_width,
            projection_height=config.projection.projector_height,
            projection_visible_inset_top_mm=config.calibration.projection_visible_inset_top_mm,
            projection_visible_inset_right_mm=config.calibration.projection_visible_inset_right_mm,
            projection_visible_inset_bottom_mm=config.calibration.projection_visible_inset_bottom_mm,
            projection_visible_inset_left_mm=config.calibration.projection_visible_inset_left_mm,
            physical_rail_inset_top_mm=config.calibration.physical_rail_inset_top_mm,
            physical_rail_inset_right_mm=config.calibration.physical_rail_inset_right_mm,
            physical_rail_inset_bottom_mm=config.calibration.physical_rail_inset_bottom_mm,
            physical_rail_inset_left_mm=config.calibration.physical_rail_inset_left_mm,
            physical_middle_pocket_relief_top_mm=config.calibration.physical_middle_pocket_relief_top_mm,
            physical_middle_pocket_relief_bottom_mm=config.calibration.physical_middle_pocket_relief_bottom_mm,
            center_reachable_extra_margin_mm=config.calibration.center_reachable_extra_margin_mm,
            ball_center_compensation_enabled=config.calibration.ball_center_compensation_enabled,
            ball_center_compensation_auto_reference=config.calibration.ball_center_compensation_auto_reference,
            ball_center_compensation_ref_x_px=config.calibration.ball_center_compensation_ref_x_px,
            ball_center_compensation_ref_y_px=config.calibration.ball_center_compensation_ref_y_px,
            ball_center_compensation_scale_x_pct=config.calibration.ball_center_compensation_scale_x_pct,
            ball_center_compensation_scale_y_pct=config.calibration.ball_center_compensation_scale_y_pct,
            ui_geometry_reference_enabled=config.projection.geometry_reference_enabled,
            replay_enabled=config.replay.enabled,
            state_machine_engine=config.state.engine,
            shot_mode=config.planner.shot_mode,
            planner_route_freeze_enabled=config.planner.route_freeze_enabled,
            planner_route_freeze_enter_frames=config.planner.route_freeze_enter_frames,
            planner_route_freeze_release_frames=config.planner.route_freeze_release_frames,
            planner_route_freeze_same_route_refresh_mm=config.planner.route_freeze_same_route_refresh_mm,
            planner_route_freeze_same_route_refresh_score_delta=config.planner.route_freeze_same_route_refresh_score_delta,
            planner_route_freeze_switch_confirm_frames=config.planner.route_freeze_switch_confirm_frames,
            planner_route_freeze_switch_min_distance_mm=config.planner.route_freeze_switch_min_distance_mm,
            planner_route_freeze_switch_min_score_delta=config.planner.route_freeze_switch_min_score_delta,
            planner_cue_sector_correction_enabled=config.planner.cue_sector_correction_enabled,
            planner_cue_sector_angle_deg=config.planner.cue_sector_angle_deg,
            planner_cue_sector_edge_margin_deg=config.planner.cue_sector_edge_margin_deg,
            planner_cue_sector_corridor_width_px=config.planner.cue_sector_corridor_width_px,
            planner_cue_sector_switch_confirm_frames=config.planner.cue_sector_switch_confirm_frames,
            planner_cue_sector_switch_min_score_delta=config.planner.cue_sector_switch_min_score_delta,
            planner_cue_sector_lock_margin_px=config.planner.cue_sector_lock_margin_px,
            planner_cue_sector_lock_forward_tolerance_px=config.planner.cue_sector_lock_forward_tolerance_px,
            planner_cue_sector_lock_release_frames=config.planner.cue_sector_lock_release_frames,
            planner_cue_sector_min_stick_quality=config.planner.cue_sector_min_stick_quality,
            planner_cue_sector_require_balls_stationary=config.planner.cue_sector_require_balls_stationary,
            planner_cue_sector_stationary_speed_mm_s=config.planner.cue_sector_stationary_speed_mm_s,
            planner_cue_sector_stationary_speed_px_s=config.planner.cue_sector_stationary_speed_px_s,
            planner_target_lock_enabled=config.planner.target_lock_enabled,
            planner_target_lock_confirm_frames=config.planner.target_lock_confirm_frames,
            planner_target_lock_switch_confirm_frames=config.planner.target_lock_switch_confirm_frames,
            planner_target_shot_enabled=config.planner.target_shot_enabled,
            planner_target_shot_trigger_frames=config.planner.target_shot_trigger_frames,
            learning_ranker_enabled=config.learning.ranker_enabled,
            learning_ranker_model_path=config.learning.ranker_model_path,
            learning_score_blend=config.learning.score_blend,
            learning_collect_enabled=config.learning.collect_enabled,
            learning_samples_directory=config.learning.samples_directory,
            star_formula=dict(star_formula or {}),
        )


def _clean_optional_text(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    return text or None
