from __future__ import annotations

from pathlib import Path

import numpy as np

from bas.config import InstantReplayConfig
from bas.instant_replay import InstantReplayBuffer


class _FakeClock:
    def __init__(self, start: float = 0.0) -> None:
        self.value = float(start)

    def __call__(self) -> float:
        return self.value

    def advance(self, seconds: float) -> None:
        self.value += float(seconds)


class _FakeRecorder:
    def __init__(
        self,
        path: str | Path,
        *,
        width: int,
        height: int,
        fps: float,
        bitrate_kbps: int = 6000,
    ) -> None:
        self.path = Path(path)
        self.width = int(width)
        self.height = int(height)
        self.fps = float(fps)
        self.bitrate_kbps = int(bitrate_kbps)
        self.frames_written = 0
        self.closed = False
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_bytes(b"")

    def write(self, frame_bgr: np.ndarray) -> None:
        if self.closed:
            raise RuntimeError("recorder already closed")
        frame = np.asarray(frame_bgr)
        assert frame.shape[:2] == (self.height, self.width)
        self.frames_written += 1
        with self.path.open("ab") as f:
            f.write(bytes([self.frames_written % 251]))

    def close(self) -> int:
        self.closed = True
        return 0


def _frame(value: int = 0) -> np.ndarray:
    return np.full((4, 6, 3), value, dtype=np.uint8)


def test_instant_replay_prunes_old_segments_and_exports_recent_window(tmp_path: Path) -> None:
    concat_calls: list[dict[str, object]] = []
    base_ts_ns = 10_000_000_000

    def fake_concat(
        paths: list[str | Path],
        output_path: str | Path,
        *,
        reencode: bool = False,
        bitrate_kbps: int = 6000,
    ) -> int:
        concat_calls.append(
            {
                "paths": [Path(path) for path in paths],
                "output_path": Path(output_path),
                "reencode": bool(reencode),
                "bitrate_kbps": int(bitrate_kbps),
            }
        )
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(b"retro")
        return 0

    config = InstantReplayConfig(
        directory=str(tmp_path / "instant_replay"),
        segment_seconds=1,
        buffer_seconds=3,
        export_seconds=2,
        cooldown_seconds=30,
        bitrate_kbps=4500,
    )
    buffer = InstantReplayBuffer(
        config,
        recorder_factory=_FakeRecorder,
        concat_func=fake_concat,
        monotonic_time=_FakeClock(start=100.0),
        async_exports=False,
    )

    buffer.start(width=6, height=4, fps=30.0)
    for idx, ts_ns in enumerate(
        [
            base_ts_ns + 0,
            base_ts_ns + 1_000_000_000,
            base_ts_ns + 2_000_000_000,
            base_ts_ns + 3_000_000_000,
            base_ts_ns + 4_000_000_000,
        ]
    ):
        buffer.write(_frame(idx), ts_ns=ts_ns)

    result = buffer.request_export(trigger_ts_ns=base_ts_ns + 4_500_000_000)

    assert result.accepted is True
    assert result.output_path is not None
    assert result.output_path.exists() is True
    assert result.output_path.parent.name == "exports"
    assert len(concat_calls) == 1
    assert concat_calls[0]["reencode"] is False
    assert concat_calls[0]["bitrate_kbps"] == 4500
    assert [path.name for path in concat_calls[0]["paths"]] == ["segment_000003.mp4", "segment_000004.mp4"]

    session_dir = next((tmp_path / "instant_replay").glob("session_*"))
    segments_dir = session_dir / "segments"
    assert (segments_dir / "segment_000000.mp4").exists() is False
    assert (segments_dir / "segment_000001.mp4").exists() is False
    assert (segments_dir / "segment_000002.mp4").exists() is True
    assert (segments_dir / "segment_000003.mp4").exists() is True
    assert (segments_dir / "segment_000004.mp4").exists() is True
    assert buffer.drain_events()


def test_instant_replay_trigger_respects_cooldown(tmp_path: Path) -> None:
    concat_calls: list[list[Path]] = []
    clock = _FakeClock(start=200.0)
    base_ts_ns = 20_000_000_000

    def fake_concat(
        paths: list[str | Path],
        output_path: str | Path,
        *,
        reencode: bool = False,
        bitrate_kbps: int = 6000,
    ) -> int:
        concat_calls.append([Path(path) for path in paths])
        Path(output_path).write_bytes(b"ok")
        return 0

    config = InstantReplayConfig(
        directory=str(tmp_path / "instant_replay"),
        segment_seconds=1,
        buffer_seconds=10,
        export_seconds=2,
        cooldown_seconds=30,
    )
    buffer = InstantReplayBuffer(
        config,
        recorder_factory=_FakeRecorder,
        concat_func=fake_concat,
        monotonic_time=clock,
        async_exports=False,
    )

    buffer.start(width=6, height=4, fps=25.0)
    buffer.write(_frame(1), ts_ns=base_ts_ns + 0)
    buffer.write(_frame(2), ts_ns=base_ts_ns + 1_100_000_000)

    first = buffer.request_export(trigger_ts_ns=base_ts_ns + 1_500_000_000)
    second = buffer.request_export(trigger_ts_ns=base_ts_ns + 1_500_000_000)
    clock.advance(30.1)
    third = buffer.request_export(trigger_ts_ns=base_ts_ns + 1_500_000_000)

    assert first.accepted is True
    assert second.accepted is False
    assert second.cooldown_remaining_s > 29.0
    assert third.accepted is True
    assert len(concat_calls) == 2


def test_instant_replay_export_reencodes_after_copy_failure(tmp_path: Path) -> None:
    concat_modes: list[bool] = []
    base_ts_ns = 30_000_000_000

    def fake_concat(
        paths: list[str | Path],
        output_path: str | Path,
        *,
        reencode: bool = False,
        bitrate_kbps: int = 6000,
    ) -> int:
        concat_modes.append(bool(reencode))
        if not reencode:
            return 1
        Path(output_path).write_bytes(b"reencoded")
        return 0

    config = InstantReplayConfig(
        directory=str(tmp_path / "instant_replay"),
        segment_seconds=1,
        buffer_seconds=10,
        export_seconds=2,
        cooldown_seconds=0,
    )
    buffer = InstantReplayBuffer(
        config,
        recorder_factory=_FakeRecorder,
        concat_func=fake_concat,
        async_exports=False,
    )

    buffer.start(width=6, height=4, fps=25.0)
    buffer.write(_frame(1), ts_ns=base_ts_ns + 0)
    buffer.write(_frame(2), ts_ns=base_ts_ns + 1_100_000_000)

    result = buffer.request_export(trigger_ts_ns=base_ts_ns + 1_500_000_000)

    assert result.accepted is True
    assert result.output_path is not None
    assert result.output_path.exists() is True
    assert concat_modes == [False, True]
