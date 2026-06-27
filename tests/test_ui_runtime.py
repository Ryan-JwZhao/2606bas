from __future__ import annotations

import os
import threading
import time
from types import SimpleNamespace

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt5 import QtWidgets

from bas.config import AppConfig
from bas import runtime_env

runtime_env.preload_torch_for_backend = lambda backend: None

from bas.ui import main_window


def _app() -> QtWidgets.QApplication:
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication([])
    return app


def test_start_pipeline_and_tick_use_same_thread(monkeypatch) -> None:
    app = _app()
    window = main_window.OperatorWindow(AppConfig())
    created: list[object] = []

    class _FakePipeline:
        def __init__(self, config, star_formula=None, control_state=None):
            self.creator_thread_id = threading.get_ident()
            self.step_thread_id: int | None = None
            self.state_machine = SimpleNamespace(turn_target_group=None, operator_hold=False)
            self.capture = SimpleNamespace(info=lambda: SimpleNamespace(backend="fake", width=1920, height=1080, fps=30.0))
            self.calibration = SimpleNamespace(
                projection=SimpleNamespace(is_valid=False),
                camera=SimpleNamespace(is_valid=False),
            )
            self.detector = SimpleNamespace(detector=SimpleNamespace(version="fake_detector"))
            self.tracker = SimpleNamespace(version="fake_tracker")
            self.planner = SimpleNamespace(version="fake_planner", learning_ranker=SimpleNamespace(version="fake_ranker"))
            self.recorder = None
            self.last_timings_ms = {}
            created.append(self)

        def step(self):
            self.step_thread_id = threading.get_ident()
            return None

        def close(self) -> None:
            return None

    monkeypatch.setattr(main_window, "RuntimePipeline", _FakePipeline)
    monkeypatch.setattr(window, "_save_user_settings", lambda: None)

    window.start_pipeline()

    deadline = time.time() + 1.0
    while not created and time.time() < deadline:
        app.processEvents()
    assert created

    while window.pipeline is None and time.time() < deadline:
        app.processEvents()
    assert window.pipeline is created[0]

    window._tick()

    fake = created[0]
    assert fake.creator_thread_id == fake.step_thread_id
    window.close()


def test_tick_runtime_exception_stops_pipeline_instead_of_crashing(monkeypatch) -> None:
    window = main_window.OperatorWindow.__new__(main_window.OperatorWindow)
    monkeypatch.setattr(main_window.QtWidgets.QMessageBox, "critical", lambda *args, **kwargs: 0)
    monkeypatch.setattr(main_window.LOGGER, "exception", lambda *args, **kwargs: None)

    class _ExplodingPipeline:
        def __init__(self) -> None:
            self.closed = False
            self.state_machine = SimpleNamespace(turn_target_group=None)

        def step(self):
            raise RuntimeError("boom")

        def close(self) -> None:
            self.closed = True

    class _Timer:
        def stop(self) -> None:
            return None

        def start(self, *_args) -> None:
            return None

    pipeline = _ExplodingPipeline()
    window.timer = _Timer()
    window.pipeline = pipeline
    window._frame_busy = False
    window._append_log = lambda _message: None
    window._capture_timer_interval_ms = lambda *, elapsed_ms=0.0: 1

    def _stop_pipeline() -> None:
        pipeline.close()
        window.pipeline = None

    window.stop_pipeline = _stop_pipeline

    main_window.OperatorWindow._tick(window)

    assert pipeline.closed is True
    assert window.pipeline is None
    assert window._frame_busy is False
