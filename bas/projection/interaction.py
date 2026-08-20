from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Deque, Optional

import cv2
import numpy as np

from ..calibration.service import CalibrationService
from ..paths import PROJECT_ROOT
from ..schemas import MatchStateFrame, OverlayText, ProjectionOverlay, ShotPlan
from .overlay import render_overlay_with_star
from .star_formula import StarFormulaConfig
from .text import draw_overlay_texts

DEFAULT_NOTICE_DURATION_S = 3.0
DEFAULT_ANIMATION_FPS = 12.0
MIN_ANIMATION_FPS = 12.0
MAX_ANIMATION_FPS = 16.0
MAX_VISIBLE_NOTICES = 3


@dataclass(frozen=True)
class _Notice:
    text: str
    expires_at: float


@dataclass(frozen=True)
class _SequenceAsset:
    name: str
    frame_paths: tuple[Path, ...]


@dataclass
class _ActiveSequence:
    asset: _SequenceAsset
    started_at: float
    fps: float
    cached_index: int = -1
    cached_frame: Optional[np.ndarray] = None


class ProjectionInteractionController:
    def __init__(
        self,
        *,
        asset_root: Path | None = None,
        time_source: Callable[[], float] = time.monotonic,
        auto_pocket_animation_enabled: bool = True,
        auto_victory_animation_enabled: bool = True,
    ) -> None:
        self.asset_root = Path(asset_root or (PROJECT_ROOT / "example" / "motion"))
        self._time = time_source
        self._notices: Deque[_Notice] = deque(maxlen=24)
        self._active_sequence: Optional[_ActiveSequence] = None
        self._asset_cache: dict[str, Optional[_SequenceAsset]] = {}
        self._seen_event_keys: Deque[tuple[object, ...]] = deque()
        self._seen_event_key_set: set[tuple[object, ...]] = set()
        self._active_target_shot_id: Optional[int] = None
        self._boot_notice_sent = False
        self._auto_pocket_animation_enabled = bool(auto_pocket_animation_enabled)
        self._auto_victory_animation_enabled = bool(auto_victory_animation_enabled)

    def set_auto_triggers(
        self,
        *,
        pocket_enabled: Optional[bool] = None,
        victory_enabled: Optional[bool] = None,
    ) -> None:
        if pocket_enabled is not None:
            self._auto_pocket_animation_enabled = bool(pocket_enabled)
        if victory_enabled is not None:
            self._auto_victory_animation_enabled = bool(victory_enabled)

    def reset_display_geometry(self) -> None:
        # Route continuity now lives in table-space planning inputs.  Rendering
        # intentionally keeps no independent line-endpoint history.
        return None

    def notify(self, text: str, *, duration_s: float = DEFAULT_NOTICE_DURATION_S) -> bool:
        message = str(text or "").strip()
        if not message:
            return False
        now = self._time()
        self._prune_notices(now)
        self._notices.append(_Notice(text=message, expires_at=now + max(0.1, float(duration_s))))
        return True

    def notify_boot_ready(self) -> bool:
        if self._boot_notice_sent:
            return False
        self._boot_notice_sent = True
        return self.notify("系统已开机")

    def notify_init_success(self) -> bool:
        return self.notify("系统初始化成功")

    def trigger_pocket_animation(self, pocket_index: int, *, fps_hint: Optional[float] = None) -> bool:
        asset = self._asset(f"pocket:{int(pocket_index)}")
        if asset is None:
            return False
        self._active_sequence = _ActiveSequence(
            asset=asset,
            started_at=self._time(),
            fps=self._suggest_fps(fps_hint),
        )
        return True

    def trigger_victory_animation(self, *, fps_hint: Optional[float] = None) -> bool:
        asset = self._asset("victory")
        if asset is None:
            return False
        self._active_sequence = _ActiveSequence(
            asset=asset,
            started_at=self._time(),
            fps=self._suggest_fps(fps_hint),
        )
        return True

    def stop_animation(self) -> None:
        self._active_sequence = None

    def has_active_content(self) -> bool:
        now = self._time()
        self._prune_notices(now)
        return bool(self._notices) or self._active_sequence is not None

    def observe_output(
        self,
        *,
        state: MatchStateFrame,
        plan: ShotPlan,
        fps_hint: Optional[float] = None,
    ) -> bool:
        changed = False
        pocket_events = [event for event in state.events if event.name == "POCKET_CONFIRMED"]
        victory_events = [event for event in state.events if event.name == "GAME_STATUS_CHANGED"]
        for event in [*pocket_events, *victory_events]:
            payload = dict(event.payload or {})
            if event.name == "POCKET_CONFIRMED":
                key = (
                    "POCKET_EVENT",
                    payload.get("shot_id"),
                    payload.get("decision_id"),
                    payload.get("track_id"),
                    payload.get("pocket_index"),
                )
                if not self._remember_event_key(key):
                    continue
                pocket_index = self._safe_int(payload.get("pocket_index"))
                if pocket_index is not None and self._auto_pocket_animation_enabled:
                    changed = self.trigger_pocket_animation(pocket_index, fps_hint=fps_hint) or changed
            elif event.name == "GAME_STATUS_CHANGED":
                key = (
                    "GAME_STATUS_EVENT",
                    payload.get("shot_id"),
                    payload.get("decision_id"),
                    payload.get("from_status"),
                    payload.get("to_status"),
                )
                if not self._remember_event_key(key):
                    continue
                if self._auto_victory_animation_enabled:
                    changed = self.trigger_victory_animation(fps_hint=fps_hint) or changed

        active_target_id = int(plan.locked_target_id) if plan.shot_mode == "target" and plan.locked_target_id is not None else None
        if active_target_id is not None and active_target_id != self._active_target_shot_id:
            changed = self.notify(f"目标球已锁定 #{active_target_id}") or changed
        self._active_target_shot_id = active_target_id
        return changed

    def compose_frame(
        self,
        overlay: ProjectionOverlay,
        *,
        star_formula: StarFormulaConfig,
        calibration: CalibrationService | None = None,
    ) -> np.ndarray:
        image = render_overlay_with_star(overlay, star_formula)
        animation = self._current_animation_frame(overlay.projector_size)
        if animation is not None:
            image = _alpha_blend_fullscreen(image, animation)
        notices = self._active_notices()
        if notices:
            anchor = self._notice_anchor(overlay.projector_size, calibration=calibration)
            _draw_notices(image, notices, anchor)
        return image

    def _asset(self, key: str) -> Optional[_SequenceAsset]:
        if key in self._asset_cache:
            return self._asset_cache[key]
        if key == "victory":
            asset = self._sequence_from_dir(self.asset_root / "Win", "victory")
        elif key.startswith("pocket:"):
            pocket_index = int(key.split(":", maxsplit=1)[1])
            asset = self._sequence_from_dir(self.asset_root / "Goal" / f"pocket{pocket_index}", f"pocket{pocket_index}")
        else:
            asset = None
        self._asset_cache[key] = asset
        return asset

    @staticmethod
    def _sequence_from_dir(path: Path, name: str) -> Optional[_SequenceAsset]:
        if not path.exists():
            return None
        frame_paths = tuple(sorted(p for p in path.glob("*.png") if p.is_file()))
        if not frame_paths:
            return None
        return _SequenceAsset(name=name, frame_paths=frame_paths)

    def _remember_event_key(self, key: tuple[object, ...]) -> bool:
        if key in self._seen_event_key_set:
            return False
        while len(self._seen_event_keys) >= 96:
            old = self._seen_event_keys.popleft()
            self._seen_event_key_set.discard(old)
        self._seen_event_keys.append(key)
        self._seen_event_key_set.add(key)
        return True

    def _current_animation_frame(self, projector_size: tuple[int, int]) -> Optional[np.ndarray]:
        active = self._active_sequence
        if active is None:
            return None
        frame_count = len(active.asset.frame_paths)
        if frame_count <= 0:
            self._active_sequence = None
            return None
        elapsed = max(0.0, self._time() - active.started_at)
        frame_index = int(elapsed * max(MIN_ANIMATION_FPS, float(active.fps)))
        if frame_index >= frame_count:
            self._active_sequence = None
            return None
        if active.cached_frame is None or active.cached_index != frame_index:
            frame = cv2.imread(str(active.asset.frame_paths[frame_index]), cv2.IMREAD_UNCHANGED)
            if frame is None:
                self._active_sequence = None
                return None
            active.cached_frame = _ensure_bgra(frame)
            active.cached_index = frame_index
        frame = active.cached_frame
        if frame is None:
            return None
        width, height = [max(1, int(v)) for v in projector_size]
        if frame.shape[1] != width or frame.shape[0] != height:
            frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_LINEAR)
        return frame

    def _active_notices(self) -> list[str]:
        now = self._time()
        self._prune_notices(now)
        if not self._notices:
            return []
        return [notice.text for notice in list(self._notices)[-MAX_VISIBLE_NOTICES:]]

    def _prune_notices(self, now: float) -> None:
        keep = [notice for notice in self._notices if notice.expires_at > now]
        if len(keep) == len(self._notices):
            return
        self._notices.clear()
        self._notices.extend(keep)

    def _notice_anchor(
        self,
        projector_size: tuple[int, int],
        *,
        calibration: CalibrationService | None,
    ) -> tuple[int, int]:
        width, height = [max(1, int(v)) for v in projector_size]
        if calibration is None:
            return (40, 42)
        table = calibration.table
        for attr_name in ("center_playable_polygon_mm", "projection_visible_polygon_mm", "inner_polygon_mm"):
            polygon = np.asarray(getattr(table, attr_name, []) or [], dtype=np.float32).reshape((-1, 2))
            if polygon.shape[0] < 3:
                continue
            try:
                proj = calibration.table_mm_to_projector_px(polygon).astype(np.float32)
            except Exception:
                continue
            x = int(round(max(18.0, min(float(np.min(proj[:, 0])) + 14.0, width - 180.0))))
            y = int(round(max(28.0, min(float(np.min(proj[:, 1])) + 24.0, height - 80.0))))
            return (x, y)
        return (40, 42)

    @staticmethod
    def _safe_int(value: object) -> Optional[int]:
        try:
            return int(value)  # type: ignore[arg-type]
        except Exception:
            return None

    @staticmethod
    def _suggest_fps(fps_hint: Optional[float]) -> float:
        try:
            hint = float(fps_hint or 0.0)
        except Exception:
            hint = 0.0
        if hint <= 0.0:
            return DEFAULT_ANIMATION_FPS
        return float(max(MIN_ANIMATION_FPS, min(MAX_ANIMATION_FPS, round(hint * 0.5))))


def _ensure_bgra(frame: np.ndarray) -> np.ndarray:
    if frame.ndim == 2:
        bgr = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        alpha = np.full((frame.shape[0], frame.shape[1], 1), 255, dtype=np.uint8)
        return np.concatenate([bgr, alpha], axis=2)
    if frame.shape[2] == 4:
        return frame
    if frame.shape[2] == 3:
        alpha = np.full((frame.shape[0], frame.shape[1], 1), 255, dtype=np.uint8)
        return np.concatenate([frame, alpha], axis=2)
    raise ValueError(f"Unsupported animation frame shape: {frame.shape}")


def _alpha_blend_fullscreen(base_bgr: np.ndarray, overlay_bgra: np.ndarray) -> np.ndarray:
    if base_bgr.shape[:2] != overlay_bgra.shape[:2]:
        raise ValueError("Overlay size must match base image size.")
    alpha = overlay_bgra[:, :, 3:4].astype(np.float32) / 255.0
    if float(np.max(alpha)) <= 0.0:
        return base_bgr
    foreground = overlay_bgra[:, :, :3].astype(np.float32)
    background = base_bgr.astype(np.float32)
    blended = foreground * alpha + background * (1.0 - alpha)
    return np.clip(blended, 0.0, 255.0).astype(np.uint8)


def _draw_notices(image: np.ndarray, notices: list[str], anchor: tuple[int, int]) -> None:
    x, y = [int(v) for v in anchor]
    line_gap = 22
    items = [
        OverlayText(
            position=(float(x + 120), float(y + index * line_gap)),
            text=text,
            font_size_px=16.0,
            max_width_ratio=0.35,
            outline_width_px=1.0,
            background_alpha=0,
        )
        for index, text in enumerate(notices)
    ]
    draw_overlay_texts(image, items)
