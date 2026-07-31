from __future__ import annotations

import logging
from typing import Optional

import cv2
import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets

from ..config import ProjectionConfig
from ..schemas import ProjectionOverlay
from .frame_transform import ProjectionFrameTransform
from .overlay import render_overlay_with_star
from .star_formula import StarFormulaConfig

LOGGER = logging.getLogger(__name__)


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

    def show_on_configured_screen(self) -> None:
        app = QtWidgets.QApplication.instance()
        screens = app.screens() if app is not None else []
        idx = int(self.config.screen_index)
        if 0 <= idx < len(screens):
            geo = screens[idx].geometry()
            self.move(geo.x(), geo.y())
            self.resize(geo.width(), geo.height())
        if self.config.fullscreen:
            self.showFullScreen()
        else:
            self.show()

    def set_overlay(self, overlay: ProjectionOverlay) -> None:
        img = render_overlay_with_star(overlay, self.star_formula)
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
            calibration_rotation_degrees=self.config.calibration_rotation_degrees,
            output_rotation_degrees=self.config.output_rotation_degrees,
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
