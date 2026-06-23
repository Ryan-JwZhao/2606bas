from __future__ import annotations

from bas.config import ReplayConfig
from bas.replay import ReplayReader, ReplayRecorder
from bas.schemas import FramePacket


def test_replay_recorder_writes_jsonl(tmp_path) -> None:
    cfg = ReplayConfig(enabled=True, directory=str(tmp_path), write_video=False, write_debug_frames=False)
    rec = ReplayRecorder(cfg)
    rec.write_frame_packet(FramePacket(frame_id=1, ts_cam_ns=123, camera_id="test"))
    path = rec.session_dir / "events.jsonl"
    rec.close()
    events = list(ReplayReader(path).iter_events())
    assert len(events) == 1
    assert events[0]["type"] == "frame"

