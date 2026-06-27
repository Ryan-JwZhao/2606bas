from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Deque, Optional

import numpy as np

from .config import InstantReplayConfig
from .media_capture import FfmpegH264Recorder, concat_mp4_segments
from .utils import wall_time_id


@dataclass(frozen=True)
class InstantReplayTriggerResult:
    accepted: bool
    message: str
    output_path: Optional[Path] = None
    cooldown_remaining_s: float = 0.0


@dataclass(frozen=True)
class _SegmentMeta:
    path: Path
    start_ns: int
    end_ns: int
    frames: int


class InstantReplayBuffer:
    def __init__(
        self,
        config: InstantReplayConfig,
        *,
        recorder_factory=FfmpegH264Recorder,
        concat_func=concat_mp4_segments,
        monotonic_time=time.monotonic,
        async_exports: bool = True,
    ):
        self.config = config
        self._recorder_factory = recorder_factory
        self._concat_func = concat_func
        self._monotonic_time = monotonic_time
        self._async_exports = bool(async_exports)
        self._lock = threading.RLock()
        self._events: Deque[str] = deque(maxlen=64)
        self._session_dir: Optional[Path] = None
        self._segments_dir: Optional[Path] = None
        self._exports_dir: Optional[Path] = None
        self._segments: Deque[_SegmentMeta] = deque()
        self._current_recorder: Optional[FfmpegH264Recorder] = None
        self._current_path: Optional[Path] = None
        self._current_start_ns: int = 0
        self._current_last_ts_ns: int = 0
        self._current_frames: int = 0
        self._segment_index: int = 0
        self._width: int = 0
        self._height: int = 0
        self._fps: float = 0.0
        self._bitrate_kbps: int = 0
        self._running = False
        self._export_in_progress = 0
        self._last_trigger_monotonic = float("-inf")

    @property
    def enabled(self) -> bool:
        return bool(getattr(self.config, "enabled", False))

    @property
    def is_running(self) -> bool:
        return self._running and self._current_recorder is not None

    def start(self, *, width: int, height: int, fps: float) -> None:
        if not self.enabled:
            return
        with self._lock:
            if self._running:
                return
            self._width = max(1, int(width))
            self._height = max(1, int(height))
            self._fps = max(1.0, float(fps))
            self._bitrate_kbps = max(500, int(getattr(self.config, "bitrate_kbps", 6000)))
            self._prepare_session_dirs_locked()
            self._running = True

    def stop(self) -> None:
        with self._lock:
            self._close_current_segment_locked(end_ns=self._current_last_ts_ns or self._current_start_ns)
            self._running = False

    def write(self, frame_bgr: np.ndarray, *, ts_ns: Optional[int] = None) -> None:
        if not self.enabled:
            return
        with self._lock:
            if not self._running:
                raise RuntimeError("Instant replay buffer must be started before writing frames.")
            effective_ts = self._effective_ts_ns(ts_ns)
            if self._current_recorder is None:
                self._open_segment_locked(effective_ts)
            elif self._current_frames > 0 and effective_ts - self._current_start_ns >= self._segment_duration_ns():
                self._close_current_segment_locked(end_ns=self._current_last_ts_ns or effective_ts)
                self._open_segment_locked(effective_ts)
            if self._current_recorder is None:
                raise RuntimeError("Instant replay segment recorder is unavailable.")
            self._current_recorder.write(frame_bgr)
            self._current_frames += 1
            self._current_last_ts_ns = effective_ts

    def request_export(self, *, trigger_ts_ns: Optional[int] = None) -> InstantReplayTriggerResult:
        if not self.enabled:
            return InstantReplayTriggerResult(False, "前60秒纯净视频缓存已关闭。")
        with self._lock:
            if not self._running:
                return InstantReplayTriggerResult(False, "前60秒纯净视频缓存尚未启动。")
            now = self._monotonic_time()
            cooldown_remaining = self._cooldown_remaining_locked(now)
            if cooldown_remaining > 0.0:
                return InstantReplayTriggerResult(
                    False,
                    f"触发过于频繁，请在 {cooldown_remaining:.1f} 秒后重试。",
                    cooldown_remaining_s=cooldown_remaining,
                )
            effective_trigger_ns = self._effective_ts_ns(trigger_ts_ns)
            self._flush_for_export_locked(effective_trigger_ns)
            selected = self._select_segments_locked(effective_trigger_ns)
            if not selected:
                return InstantReplayTriggerResult(False, "当前缓存片段不足，无法导出前60秒纯净视频。")
            self._last_trigger_monotonic = now
            output_path = self._export_path_locked()
            self._export_in_progress += 1
            if self._async_exports:
                worker = threading.Thread(
                    target=self._run_export_job,
                    args=(selected, output_path, int(getattr(self.config, "export_seconds", 60))),
                    daemon=True,
                )
                worker.start()
            else:
                self._run_export_job(selected, output_path, int(getattr(self.config, "export_seconds", 60)))
            return InstantReplayTriggerResult(
                True,
                f"已接受导出请求，将保存最近 {int(getattr(self.config, 'export_seconds', 60))} 秒纯净视频。",
                output_path=output_path,
            )

    def drain_events(self) -> list[str]:
        with self._lock:
            items = list(self._events)
            self._events.clear()
            return items

    def status_detail(self) -> str:
        with self._lock:
            if not self.enabled:
                return "回看关闭"
            buffered_seconds = self._buffered_seconds_locked()
            buffered_text = f"回看 {buffered_seconds:.0f}s/{int(getattr(self.config, 'buffer_seconds', 120))}s"
            parts = [buffered_text]
            if self._export_in_progress > 0:
                parts.append("导出中")
            cooldown_remaining = self._cooldown_remaining_locked(self._monotonic_time())
            if cooldown_remaining > 0.0:
                parts.append(f"冷却 {cooldown_remaining:.0f}s")
            elif self._running:
                parts.append("可触发")
            else:
                parts.append("待机")
            return " / ".join(parts)

    def _prepare_session_dirs_locked(self) -> None:
        base = Path(getattr(self.config, "directory", "local_settings/instant_replay"))
        self._session_dir = base / f"session_{wall_time_id()}"
        self._segments_dir = self._session_dir / "segments"
        self._exports_dir = self._session_dir / "exports"
        self._segments_dir.mkdir(parents=True, exist_ok=True)
        self._exports_dir.mkdir(parents=True, exist_ok=True)
        self._segments.clear()
        self._current_recorder = None
        self._current_path = None
        self._current_start_ns = 0
        self._current_last_ts_ns = 0
        self._current_frames = 0
        self._segment_index = 0
        self._export_in_progress = 0
        self._last_trigger_monotonic = float("-inf")
        self._events.clear()

    def _segment_duration_ns(self) -> int:
        return int(max(1, int(getattr(self.config, "segment_seconds", 1))) * 1_000_000_000)

    def _buffer_duration_ns(self) -> int:
        return int(max(1, int(getattr(self.config, "buffer_seconds", 120))) * 1_000_000_000)

    def _export_duration_ns(self) -> int:
        return int(max(1, int(getattr(self.config, "export_seconds", 60))) * 1_000_000_000)

    def _effective_ts_ns(self, value: Optional[int]) -> int:
        if value is not None and int(value) > 0:
            return int(value)
        return time.time_ns()

    def _open_segment_locked(self, start_ns: int) -> None:
        if self._segments_dir is None:
            raise RuntimeError("Instant replay session directory is not prepared.")
        path = self._segments_dir / f"segment_{self._segment_index:06d}.mp4"
        self._segment_index += 1
        self._current_recorder = self._recorder_factory(
            path,
            width=self._width,
            height=self._height,
            fps=self._fps,
            bitrate_kbps=self._bitrate_kbps,
        )
        self._current_path = path
        self._current_start_ns = int(start_ns)
        self._current_last_ts_ns = int(start_ns)
        self._current_frames = 0

    def _close_current_segment_locked(self, *, end_ns: int) -> None:
        recorder = self._current_recorder
        path = self._current_path
        frames = int(self._current_frames)
        self._current_recorder = None
        self._current_path = None
        start_ns = int(self._current_start_ns)
        self._current_start_ns = 0
        self._current_last_ts_ns = 0
        self._current_frames = 0
        if recorder is None or path is None:
            return
        try:
            code = recorder.close()
        except Exception as exc:
            code = -1
            self._push_event_locked(f"前60秒纯净视频片段关闭失败: {exc}")
        if frames <= 0 or code != 0:
            self._discard_segment_files_locked(path)
            return
        segment = _SegmentMeta(path=path, start_ns=start_ns, end_ns=max(start_ns, int(end_ns)), frames=frames)
        self._segments.append(segment)
        self._prune_segments_locked()

    def _flush_for_export_locked(self, trigger_ns: int) -> None:
        if self._current_recorder is not None:
            self._close_current_segment_locked(end_ns=trigger_ns)
        if self._running:
            self._open_segment_locked(trigger_ns + 1)

    def _prune_segments_locked(self) -> None:
        if not self._segments:
            return
        latest_end_ns = self._segments[-1].end_ns
        cutoff_ns = latest_end_ns - self._buffer_duration_ns()
        while self._segments and self._segments[0].end_ns < cutoff_ns:
            segment = self._segments.popleft()
            self._discard_segment_files_locked(segment.path)

    def _select_segments_locked(self, trigger_ns: int) -> list[Path]:
        cutoff_ns = trigger_ns - self._export_duration_ns()
        return [segment.path for segment in self._segments if segment.end_ns >= cutoff_ns]

    def _export_path_locked(self) -> Path:
        if self._exports_dir is None:
            raise RuntimeError("Instant replay export directory is not prepared.")
        return self._exports_dir / f"retro_clip_{wall_time_id()}.mp4"

    def _run_export_job(self, segments: list[Path], output_path: Path, export_seconds: int) -> None:
        try:
            code = self._concat_func(segments, output_path, reencode=False, bitrate_kbps=self._bitrate_kbps)
            if code != 0:
                code = self._concat_func(segments, output_path, reencode=True, bitrate_kbps=self._bitrate_kbps)
            if code == 0 and output_path.exists():
                message = f"前{export_seconds}秒纯净视频已导出: {output_path}"
            else:
                message = f"前{export_seconds}秒纯净视频导出失败，ffmpeg code={code}"
                try:
                    output_path.unlink(missing_ok=True)
                except Exception:
                    pass
            with self._lock:
                self._push_event_locked(message)
        except Exception as exc:
            with self._lock:
                self._push_event_locked(f"前60秒纯净视频导出异常: {exc}")
        finally:
            with self._lock:
                self._export_in_progress = max(0, self._export_in_progress - 1)

    def _discard_segment_files_locked(self, path: Path) -> None:
        try:
            path.unlink(missing_ok=True)
        except Exception:
            pass
        log_path = path.with_suffix(path.suffix + ".ffmpeg.log")
        try:
            log_path.unlink(missing_ok=True)
        except Exception:
            pass

    def _buffered_seconds_locked(self) -> float:
        start_ns = self._segments[0].start_ns if self._segments else None
        end_ns = self._segments[-1].end_ns if self._segments else None
        if self._current_frames > 0 and self._current_start_ns > 0:
            start_ns = self._current_start_ns if start_ns is None else min(start_ns, self._current_start_ns)
            end_ns = max(end_ns or 0, self._current_last_ts_ns or self._current_start_ns)
        if start_ns is None or end_ns is None:
            return 0.0
        return max(0.0, (end_ns - start_ns) / 1_000_000_000.0)

    def _cooldown_remaining_locked(self, now: float) -> float:
        cooldown = max(0.0, float(getattr(self.config, "cooldown_seconds", 30)))
        if cooldown <= 0.0 or self._last_trigger_monotonic == float("-inf"):
            return 0.0
        return max(0.0, cooldown - (float(now) - self._last_trigger_monotonic))

    def _push_event_locked(self, message: str) -> None:
        self._events.append(str(message))
