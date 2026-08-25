from __future__ import annotations

import io
import subprocess
from pathlib import Path

from bas.media_capture import FfmpegH264Recorder


class _FakeStdin:
    def __init__(self) -> None:
        self.closed = False

    def close(self) -> None:
        self.closed = True


class _SlowFinalizingProcess:
    def __init__(self) -> None:
        self.stdin = _FakeStdin()
        self.killed = False
        self.wait_timeouts: list[float | None] = []

    def wait(self, timeout: float | None = None) -> int:
        self.wait_timeouts.append(timeout)
        if timeout is not None:
            raise subprocess.TimeoutExpired("ffmpeg", timeout)
        return 0

    def kill(self) -> None:
        self.killed = True


def _unstarted_recorder(path: Path) -> FfmpegH264Recorder:
    recorder = object.__new__(FfmpegH264Recorder)
    recorder.path = path
    recorder.width = 1920
    recorder.height = 1080
    recorder.fps = 8.644
    recorder.bitrate_kbps = 6000
    recorder.frames_written = 0
    return recorder


def test_recorder_close_allows_ffmpeg_to_finish_mp4_index() -> None:
    recorder = _unstarted_recorder(Path("capture.mp4"))
    process = _SlowFinalizingProcess()
    recorder._process = process
    recorder._stderr_file = io.BytesIO()

    code = recorder.close()

    assert code == 0
    assert process.stdin.closed
    assert process.wait_timeouts == [None]
    assert not process.killed
    assert recorder._stderr_file is None


def test_live_recorder_skips_faststart_relocation() -> None:
    recorder = _unstarted_recorder(Path("capture.mp4"))

    command = recorder._command()

    assert "-movflags" not in command
    assert "+faststart" not in command
    assert command[-1] == "capture.mp4"
