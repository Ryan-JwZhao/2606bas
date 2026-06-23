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
    detector_backend: Optional[str] = None
    model_path: Optional[str] = None
    class_file_path: Optional[str] = None
    outline_path: Optional[str] = None
    inline_path: Optional[str] = None
    pocket_path: Optional[str] = None
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
        if self.detector_backend:
            config.detector.backend = self.detector_backend
        if self.model_path:
            config.detector.model_path = self.model_path
        if self.class_file_path:
            config.detector.class_file_path = self.class_file_path
        if self.outline_path:
            config.geometry.outline_path = self.outline_path
        if self.inline_path:
            config.geometry.inline_path = self.inline_path
        if self.pocket_path:
            config.geometry.pocket_path = self.pocket_path
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
            detector_backend=config.detector.backend,
            model_path=config.detector.model_path,
            class_file_path=config.detector.class_file_path,
            outline_path=config.geometry.outline_path,
            inline_path=config.geometry.inline_path,
            pocket_path=config.geometry.pocket_path,
            projection_screen_index=config.projection.screen_index,
            projection_width=config.projection.projector_width,
            projection_height=config.projection.projector_height,
            replay_enabled=config.replay.enabled,
            star_formula=dict(star_formula or {}),
        )

