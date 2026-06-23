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
    replay_enabled: Optional[bool] = None
    star_formula: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def load(cls, path: Path = SETTINGS_PATH) -> "UserSettings":
        if not path.exists():
            return cls()
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            return cls()
        allowed = cls.__dataclass_fields__.keys()
        return cls(**{k: v for k, v in data.items() if k in allowed})

    def save(self, path: Path = SETTINGS_PATH) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(asdict(self), f, ensure_ascii=False, indent=2)

    def apply_to_config(self, config: AppConfig) -> AppConfig:
        if self.camera_backend:
            config.camera.backend = self.camera_backend
        if self.camera_device_index is not None:
            config.camera.device_index = int(self.camera_device_index)
        if self.nori_device_id is not None:
            config.camera.nori_device_id = int(self.nori_device_id)
        if self.width is not None:
            config.camera.width = int(self.width)
        if self.height is not None:
            config.camera.height = int(self.height)
        if self.fps is not None:
            config.camera.fps = int(self.fps)
        if self.video_path:
            config.camera.video_path = self.video_path
        if self.nori_sdk_root:
            config.camera.nori_sdk_root = self.nori_sdk_root
        if self.exposure_auto is not None:
            config.camera.exposure_auto = bool(self.exposure_auto)
        if self.exposure_level is not None:
            config.camera.exposure_level = max(-10, min(0, int(self.exposure_level)))
        if self.distortion_correction_enabled is not None:
            config.camera.distortion_correction_enabled = bool(self.distortion_correction_enabled)
        if self.distortion_correction_file:
            config.camera.distortion_correction_file = self.distortion_correction_file
        if self.detector_backend:
            config.detector.backend = self.detector_backend
        if self.model_path:
            config.detector.model_path = self.model_path
        if self.class_file_path:
            config.detector.class_file_path = self.class_file_path
        if self.detect_interval_frames is not None:
            config.detector.detect_interval_frames = max(1, int(self.detect_interval_frames))
        if self.detect_fps_limit_hz is not None:
            config.detector.detect_fps_limit_hz = max(0.0, float(self.detect_fps_limit_hz))
        if self.outline_path:
            config.geometry.outline_path = self.outline_path
        if self.inline_path:
            config.geometry.inline_path = self.inline_path
        if self.pocket_path:
            config.geometry.pocket_path = self.pocket_path
        if self.camera_calibration_file:
            config.calibration.camera_file = self.camera_calibration_file
        if self.projection_calibration_file:
            config.calibration.projection_file = self.projection_calibration_file
        if self.projection_screen_index is not None:
            config.projection.screen_index = int(self.projection_screen_index)
        if self.projection_width is not None:
            config.projection.projector_width = int(self.projection_width)
        if self.projection_height is not None:
            config.projection.projector_height = int(self.projection_height)
        if self.replay_enabled is not None:
            config.replay.enabled = bool(self.replay_enabled)
        return config

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
            replay_enabled=config.replay.enabled,
            star_formula=dict(star_formula or {}),
        )
