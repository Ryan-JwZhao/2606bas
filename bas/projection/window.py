from __future__ import annotations

import logging
from typing import Optional

import cv2
import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets

from ..config import ProjectionConfig
from ..display_geometry import DisplayGeometryStabilizer, stabilize_projection_overlay
from ..schemas import ProjectionOverlay
from .frame_transform import ProjectionFrameTransform
from .overlay import render_overlay_with_star
from .star_formula import StarFormulaConfig

LOGGER = logging.getLogger(__name__)


def bind_window_to_screen(window, screen) -> None:
    """Bind a native Qt window to one display before applying its geometry.

    Moving a widget is not sufficient after Windows display hot-plug events: the
    native window can remain associated with the old QScreen and Windows then
    scales its stale full-screen surface.  Creating the native handle first and
    setting its QScreen makes the pixel domain explicit.
    """

    window.winId()
    handle = window.windowHandle()
    if handle is not None:
        handle.setScreen(screen)
    window.setGeometry(screen.geometry())


class ProjectionWindow(QtWidgets.QWidget):
    def __init__(self, config: ProjectionConfig):
        super().__init__()
        self.config = config
        self.setWindowTitle("BAS Projection")
        self.setStyleSheet("background: black;")
        self._label = QtWidgets.QLabel(self)
        self._label.setAlignment(QtCore.Qt.AlignCenter)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._label)
        self.resize(config.projector_width, config.projector_height)
        self.star_formula = StarFormulaConfig()
        self._calibration_mode = False
        self._source_image_bgr: Optional[np.ndarray] = None
        self._source_pixmap: Optional[QtGui.QPixmap] = None
        self._bound_screen: Optional[QtGui.QScreen] = None
        self._screen_sync_pending = False
        self._screen_topology_connected = False
        self._overlay_geometry = DisplayGeometryStabilizer()

    def show_on_configured_screen(self) -> None:
        app = QtWidgets.QApplication.instance()
        if app is not None:
            self._connect_screen_topology(app)
        self._sync_to_configured_screen()
        if self.config.fullscreen:
            self.showFullScreen()
        else:
            self.show()

    def _configured_screen(self) -> Optional[QtGui.QScreen]:
        app = QtWidgets.QApplication.instance()
        screens = list(app.screens()) if app is not None else []
        if not screens:
            return None
        idx = int(self.config.screen_index)
        selected = screens[idx] if 0 <= idx < len(screens) else None
        expected = (int(self.config.projector_width), int(self.config.projector_height))
        if selected is not None:
            geo = selected.geometry()
            if (geo.width(), geo.height()) == expected:
                return selected
        exact_matches = [
            screen
            for screen in screens
            if (screen.geometry().width(), screen.geometry().height()) == expected
        ]
        if len(exact_matches) == 1:
            return exact_matches[0]
        return selected or screens[0]

    def _connect_screen_topology(self, app: QtWidgets.QApplication) -> None:
        if self._screen_topology_connected:
            return
        app.screenAdded.connect(self._schedule_screen_sync)
        app.screenRemoved.connect(self._schedule_screen_sync)
        if hasattr(app, "primaryScreenChanged"):
            app.primaryScreenChanged.connect(self._schedule_screen_sync)
        self._screen_topology_connected = True

    def _watch_screen(self, screen: QtGui.QScreen) -> None:
        if screen is self._bound_screen:
            return
        previous = self._bound_screen
        if previous is not None:
            for signal_name in (
                "geometryChanged",
                "logicalDotsPerInchChanged",
                "physicalDotsPerInchChanged",
            ):
                try:
                    getattr(previous, signal_name).disconnect(self._schedule_screen_sync)
                except (RuntimeError, TypeError):
                    pass
        self._bound_screen = screen
        for signal_name in (
            "geometryChanged",
            "logicalDotsPerInchChanged",
            "physicalDotsPerInchChanged",
        ):
            getattr(screen, signal_name).connect(self._schedule_screen_sync)

    def _schedule_screen_sync(self, *_args) -> None:
        if self._screen_sync_pending:
            return
        self._screen_sync_pending = True
        QtCore.QTimer.singleShot(0, self._run_scheduled_screen_sync)

    def _run_scheduled_screen_sync(self) -> None:
        self._screen_sync_pending = False
        self._sync_to_configured_screen()
        if self.config.fullscreen and self.isVisible():
            self.showFullScreen()

    def _sync_to_configured_screen(self) -> None:
        screen = self._configured_screen()
        if screen is None:
            return
        self._watch_screen(screen)
        bind_window_to_screen(self, screen)
        self._update_scaled_pixmap()

    def set_overlay(self, overlay: ProjectionOverlay) -> None:
        shown_overlay = stabilize_projection_overlay(overlay, self._overlay_geometry)
        img = render_overlay_with_star(shown_overlay, self.star_formula)
        self.set_image(img)

    def set_star_formula(self, config: StarFormulaConfig) -> None:
        self.star_formula = config

    def set_calibration_mode(self, enabled: bool) -> None:
        enabled = bool(enabled)
        if self._calibration_mode == enabled:
            return
        self._calibration_mode = enabled
        if self._source_image_bgr is not None:
            self._render_source_image()

    def set_image(self, image_bgr: np.ndarray) -> None:
        self._source_image_bgr = np.ascontiguousarray(image_bgr).copy()
        self._render_source_image()

    def _render_source_image(self) -> None:
        if self._source_image_bgr is None:
            return
        transform = ProjectionFrameTransform(
            calibration_rotation_degrees=self.config.legacy_calibration_rotation_degrees,
            output_rotation_degrees=self.config.legacy_output_rotation_degrees,
        )
        display_bgr = transform.apply(
            self._source_image_bgr,
            calibration_mode=self._calibration_mode,
        )
        rgb = cv2.cvtColor(display_bgr, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        qimg = QtGui.QImage(rgb.data, w, h, 3 * w, QtGui.QImage.Format_RGB888).copy()
        self._source_pixmap = QtGui.QPixmap.fromImage(qimg)
        self._update_scaled_pixmap()

    def _update_scaled_pixmap(self) -> None:
        if self._source_pixmap is None:
            return
        self._label.setPixmap(
            self._source_pixmap.scaled(
                self.size(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation
            )
        )

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:
        self._update_scaled_pixmap()
        super().resizeEvent(event)
