from __future__ import annotations

import os
from pathlib import Path

import pytest

from bas import cli
from bas.single_instance import acquire_runtime_single_instance


def test_ui_command_returns_silently_when_runtime_is_already_running(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cli, "acquire_runtime_single_instance", lambda: None)

    def _unexpected_load(*args, **kwargs):
        raise AssertionError("User settings should not load when another BAS instance is already running.")

    monkeypatch.setattr(cli.UserSettings, "load", classmethod(_unexpected_load))

    assert cli.main(["ui"]) == 0


def test_remote_control_command_skips_single_instance_guard(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    class _Queue:
        def enqueue(self, action: str, *, args=None, source: str = "cli") -> Path:
            return Path(f"C:/tmp/{action}_{source}.json")

    monkeypatch.setattr(cli, "acquire_runtime_single_instance", lambda: (_ for _ in ()).throw(AssertionError("guard should not run")))
    monkeypatch.setattr(cli.UserSettings, "load", classmethod(lambda cls, *args, **kwargs: cls()))
    monkeypatch.setattr(cli.AppConfig, "load", classmethod(lambda cls, *args, **kwargs: cls()))
    monkeypatch.setattr(cli, "prepare_runtime_environment", lambda: None)
    monkeypatch.setattr("bas.remote_control.RemoteCommandQueue", _Queue)

    assert cli.main(["remote-control", "start-capture"]) == 0
    assert "start-capture_cli.json" in capsys.readouterr().out


@pytest.mark.skipif(os.name != "nt", reason="Windows named mutex behavior is only available on Windows.")
def test_runtime_single_instance_rejects_second_acquire() -> None:
    first = acquire_runtime_single_instance()
    assert first is not None
    try:
        second = acquire_runtime_single_instance()
        assert second is None
    finally:
        first.release()

    third = acquire_runtime_single_instance()
    assert third is not None
    third.release()
