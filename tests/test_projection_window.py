from __future__ import annotations

from PyQt5 import QtCore

from bas.projection.window import bind_window_to_screen


def test_projection_window_explicitly_binds_native_window_to_configured_screen() -> None:
    class _NativeWindow:
        def __init__(self) -> None:
            self.bound_screen = None

        def setScreen(self, screen) -> None:
            self.bound_screen = screen

    class _Screen:
        def geometry(self) -> QtCore.QRect:
            return QtCore.QRect(-1280, 0, 1280, 800)

    class _Window:
        def __init__(self) -> None:
            self.native_window = _NativeWindow()
            self.created = False
            self.rect = None

        def winId(self) -> int:
            self.created = True
            return 1

        def windowHandle(self):
            return self.native_window

        def setGeometry(self, rect: QtCore.QRect) -> None:
            self.rect = rect

    native_window = _NativeWindow()
    window = _Window()
    window.native_window = native_window
    screen = _Screen()

    bind_window_to_screen(window, screen)

    assert window.created is True
    assert native_window.bound_screen is screen
    assert window.rect == QtCore.QRect(-1280, 0, 1280, 800)
