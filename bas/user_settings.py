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
    projection_screen_index: Optional[int] = None
    projection_width: Optional[int] = None
    projection_height: Optional[int] = None
    projection_geometry_reference_enabled: Optional[bool] = None
    ui_geometry_reference_enabled: Optional[bool] = None
    replay_enabled: Optional[bool] = None
    shot_mode: Optional[str] = None
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
        if self._has("projection_calibration_file"):
            config.calibration.projection_file = _clean_optional_text(self.projection_calibration_file)
        if self.projection_screen_index is not None:
            config.projection.screen_index = int(self.projection_screen_index)
        if self.projection_width is not None:
            config.projection.projector_width = int(self.projection_width)
        if self.projection_height is not None:
            config.projection.projector_height = int(self.projection_height)
        if self.ui_geometry_reference_enabled is not None:
            config.projection.geometry_reference_enabled = bool(self.ui_geometry_reference_enabled)
        elif self.projection_geometry_reference_enabled is not None:
            config.projection.geometry_reference_enabled = bool(self.projection_geometry_reference_enabled)
        if self.replay_enabled is not None:
            config.replay.enabled = bool(self.replay_enabled)
        if self._has("shot_mode") and self.shot_mode:
            mode = str(self.shot_mode).strip().lower()
            config.planner.shot_mode = "free" if mode in {"free", "free_shot"} else "rule"
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
            projection_calibration_file=config.calibration.projection_file,
            projection_screen_index=config.projection.screen_index,
            projection_width=config.projection.projector_width,
            projection_height=config.projection.projector_height,
            ui_geometry_reference_enabled=config.projection.geometry_reference_enabled,
            replay_enabled=config.replay.enabled,
            shot_mode=config.planner.shot_mode,
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
