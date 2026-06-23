from __future__ import annotations

import logging
from typing import Optional

import cv2
import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets

from ..config import ProjectionConfig
from ..schemas import ProjectionOverlay
from .overlay import render_overlay_image

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
        img = render_overlay_image(overlay)
        self.set_image(img)

    def set_image(self, image_bgr: np.ndarray) -> None:
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        qimg = QtGui.QImage(rgb.data, w, h, 3 * w, QtGui.QImage.Format_RGB888).copy()
        pix = QtGui.QPixmap.fromImage(qimg)
        self._label.setPixmap(pix.scaled(self.size(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation))

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:
        pix = self._label.pixmap()
        if pix is not None:
            self._label.setPixmap(pix.scaled(self.size(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation))
        super().resizeEvent(event)

