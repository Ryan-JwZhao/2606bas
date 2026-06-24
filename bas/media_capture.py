from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional

import cv2
import numpy as np


class FfmpegH264Recorder:
    def __init__(
        self,
        path: str | Path,
        *,
        width: int,
        height: int,
        fps: float,
        bitrate_kbps: int = 6000,
    ):
        self.path = Path(path)
        self.width = int(width)
        self.height = int(height)
        self.fps = max(1.0, float(fps))
        self.bitrate_kbps = int(bitrate_kbps)
        self.frames_written = 0
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._stderr_path = self.path.with_suffix(self.path.suffix + ".ffmpeg.log")
        self._stderr_file = self._stderr_path.open("ab")
        self._process: Optional[subprocess.Popen[bytes]] = subprocess.Popen(
            self._command(),
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=self._stderr_file,
            creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0,
        )

    def write(self, frame_bgr: np.ndarray) -> None:
        process = self._process
        if process is None or process.stdin is None:
            raise RuntimeError("ffmpeg recorder is already closed")
        if process.poll() is not None:
            raise RuntimeError(f"ffmpeg exited with code {process.returncode}; see {self._stderr_path}")
        frame = self._normalize_frame(frame_bgr)
        try:
            process.stdin.write(frame.tobytes())
        except BrokenPipeError as exc:
            raise RuntimeError(f"ffmpeg pipe closed; see {self._stderr_path}") from exc
        self.frames_written += 1

    def close(self) -> int:
        process = self._process
        self._process = None
        if process is None:
            self._close_log()
            return 0
        if process.stdin is not None:
            try:
                process.stdin.close()
            except Exception:
                pass
        try:
            code = int(process.wait(timeout=8.0))
        except subprocess.TimeoutExpired:
            process.kill()
            code = int(process.wait(timeout=2.0))
        self._close_log()
        return code

    def _close_log(self) -> None:
        if self._stderr_file is not None:
            self._stderr_file.close()
            self._stderr_file = None

    def _normalize_frame(self, frame_bgr: np.ndarray) -> np.ndarray:
        frame = np.asarray(frame_bgr)
        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError("frame must be a BGR uint8 image")
        if frame.shape[1] != self.width or frame.shape[0] != self.height:
            frame = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_AREA)
        if frame.dtype != np.uint8:
            frame = frame.astype(np.uint8)
        return np.ascontiguousarray(frame)

    def _command(self) -> list[str]:
        fps_text = f"{self.fps:.3f}".rstrip("0").rstrip(".")
        bitrate = f"{self.bitrate_kbps}k"
        return [
            _ffmpeg_executable(),
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "bgr24",
            "-s:v",
            f"{self.width}x{self.height}",
            "-r",
            fps_text,
            "-i",
            "pipe:0",
            "-an",
            "-c:v",
            "libx264",
            "-b:v",
            bitrate,
            "-maxrate",
            bitrate,
            "-bufsize",
            f"{self.bitrate_kbps * 2}k",
            "-preset",
            "veryfast",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(self.path),
        ]


def _ffmpeg_executable() -> str:
    for env_name in ("BAS_FFMPEG", "FFMPEG_BINARY"):
        value = os.environ.get(env_name)
        if value and Path(value).exists():
            return value
    found = shutil.which("ffmpeg")
    if found:
        return found
    candidates = [
        Path("C:/Program Files/Shutter Encoder/Library/ffmpeg.exe"),
        Path("C:/ffmpeg/bin/ffmpeg.exe"),
        Path("C:/Program Files/ffmpeg/bin/ffmpeg.exe"),
        Path("C:/Program Files (x86)/ffmpeg/bin/ffmpeg.exe"),
    ]
    for path in candidates:
        if path.exists():
            return str(path)
    return "ffmpeg"
