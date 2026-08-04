from __future__ import annotations

from pathlib import Path

from bas.ui import main_window


def test_projection_config_path_from_input_resolves_relative_to_current_directory() -> None:
    current = r"C:\calib\projection_calibration.json"
    out = main_window.projection_config_path_from_input("next.json", current)
    assert out == Path(r"C:\calib\next.json")


def test_timestamped_projection_output_path_replaces_existing_timestamp(monkeypatch) -> None:
    monkeypatch.setattr(main_window.time, "strftime", lambda _fmt: "20260625_120102")
    out = main_window.timestamped_projection_output_path(
        r"C:\calib\projection_calibration_20260802_173306.json"
    )
    assert out == Path(r"C:\CodeProject\2606BAS\local_settings\calibrations\projection_calibration_20260625_120102.json")
