from __future__ import annotations

from pathlib import Path

import pytest

from bas.remote_control import RemoteCommand, RemoteCommandQueue, is_stale_transient_command, normalize_remote_action


def test_remote_command_queue_roundtrip(tmp_path: Path) -> None:
    queue = RemoteCommandQueue(tmp_path / "queue")

    written = queue.enqueue("hook-shot-once", source="test")
    drained = queue.drain()

    assert written.exists() is False
    assert len(drained) == 1
    assert drained[0].action == "hook_shot_once"
    assert drained[0].source == "test"


def test_remote_action_normalization_rejects_unknown_command() -> None:
    with pytest.raises(ValueError):
        normalize_remote_action("not-a-real-command")


def test_remote_action_normalization_accepts_retro_clip_alias() -> None:
    assert normalize_remote_action("save-retro-clip") == "save_retro_clip"


def test_remote_action_normalization_accepts_hook_alias() -> None:
    assert normalize_remote_action("hook-shot-once") == "hook_shot_once"


def test_old_one_shot_command_is_stale_but_start_capture_is_not() -> None:
    old_hook = RemoteCommand("hook", "hook_shot_once", created_at_ms=1_000, source="test")
    old_start = RemoteCommand("start", "start_capture", created_at_ms=1_000, source="test")

    assert is_stale_transient_command(old_hook, now_ms=61_001, max_age_ms=60_000) is True
    assert is_stale_transient_command(old_start, now_ms=61_001, max_age_ms=60_000) is False
