from __future__ import annotations

from pathlib import Path

import pytest

from bas.remote_control import RemoteCommandQueue, normalize_remote_action


def test_remote_command_queue_roundtrip(tmp_path: Path) -> None:
    queue = RemoteCommandQueue(tmp_path / "queue")

    written = queue.enqueue("free-shot-once", source="test")
    drained = queue.drain()

    assert written.exists() is False
    assert len(drained) == 1
    assert drained[0].action == "free_shot_once"
    assert drained[0].source == "test"


def test_remote_action_normalization_rejects_unknown_command() -> None:
    with pytest.raises(ValueError):
        normalize_remote_action("not-a-real-command")


def test_remote_action_normalization_accepts_retro_clip_alias() -> None:
    assert normalize_remote_action("save-retro-clip") == "save_retro_clip"
