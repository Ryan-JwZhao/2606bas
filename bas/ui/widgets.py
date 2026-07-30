from __future__ import annotations

from PyQt5 import QtCore, QtGui, QtWidgets


class CollapsibleSection(QtWidgets.QFrame):
    expandedChanged = QtCore.pyqtSignal(bool)

    def __init__(self, title: str, *, expanded: bool = True, parent: QtWidgets.QWidget | None = None):
        super().__init__(parent)
        self.setObjectName("collapsibleSection")

        self._toggle = QtWidgets.QToolButton()
        self._toggle.setObjectName("sectionToggle")
        self._toggle.setText(title)
        self._toggle.setCheckable(True)
        self._toggle.setChecked(bool(expanded))
        self._toggle.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self._toggle.setToolButtonStyle(QtCore.Qt.ToolButtonTextBesideIcon)
        self._toggle.toggled.connect(self.setExpanded)

        self._body = QtWidgets.QWidget()
        self._body_layout = QtWidgets.QVBoxLayout(self._body)
        self._body_layout.setContentsMargins(0, 0, 0, 0)
        self._body_layout.setSpacing(0)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self._toggle)
        layout.addWidget(self._body)

        self.setExpanded(bool(expanded))

    def contentLayout(self) -> QtWidgets.QVBoxLayout:
        return self._body_layout

    def contentWidget(self) -> QtWidgets.QWidget:
        return self._body

    def isExpanded(self) -> bool:
        return bool(self._toggle.isChecked())

    @QtCore.pyqtSlot(bool)
    def setExpanded(self, expanded: bool) -> None:
        expanded = bool(expanded)
        if self._toggle.isChecked() != expanded:
            self._toggle.blockSignals(True)
            self._toggle.setChecked(expanded)
            self._toggle.blockSignals(False)
        self._toggle.setArrowType(QtCore.Qt.DownArrow if expanded else QtCore.Qt.RightArrow)
        self._body.setVisible(expanded)
        self.expandedChanged.emit(expanded)
        self.updateGeometry()


class AspectRatioPreviewFrame(QtWidgets.QFrame):
    viewportChanged = QtCore.pyqtSignal()
    zoomChanged = QtCore.pyqtSignal(float)

    MIN_ZOOM_FACTOR = 0.25
    MAX_ZOOM_FACTOR = 3.0
    ZOOM_STEP = 1.25

    def __init__(self, aspect_ratio: float = 16.0 / 9.0, parent: QtWidgets.QWidget | None = None):
        super().__init__(parent)
        self.setObjectName("previewViewport")
        self._aspect_ratio = max(0.1, float(aspect_ratio))
        self._zoom_factor = 1.0
        width = 680
        self._preferred_size = QtCore.QSize(width, int(round(width / self._aspect_ratio)))

        self._label = QtWidgets.QLabel("等待画面", self)
        self._label.setObjectName("preview")
        self._label.setAlignment(QtCore.Qt.AlignCenter)

        size_policy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        size_policy.setHeightForWidth(True)
        self.setSizePolicy(size_policy)

    def label(self) -> QtWidgets.QLabel:
        return self._label

    def zoomFactor(self) -> float:
        return self._zoom_factor

    def setZoomFactor(self, factor: float) -> None:
        factor = max(self.MIN_ZOOM_FACTOR, min(self.MAX_ZOOM_FACTOR, float(factor)))
        if abs(factor - self._zoom_factor) < 1e-6:
            return
        self._zoom_factor = factor
        self._layout_preview()
        self.zoomChanged.emit(factor)
        self.viewportChanged.emit()

    def zoomIn(self) -> None:
        self.setZoomFactor(self._zoom_factor * self.ZOOM_STEP)

    def zoomOut(self) -> None:
        self.setZoomFactor(self._zoom_factor / self.ZOOM_STEP)

    def fitToViewport(self) -> None:
        self.setZoomFactor(1.0)

    def setPreferredSize(self, width: int, height: int | None = None) -> None:
        preferred_height = int(height) if height is not None else self.heightForWidth(int(width))
        self._preferred_size = QtCore.QSize(int(width), max(1, preferred_height))
        self.updateGeometry()

    def hasHeightForWidth(self) -> bool:
        return True

    def heightForWidth(self, width: int) -> int:
        return max(1, int(round(max(1, width) / self._aspect_ratio)))

    def sizeHint(self) -> QtCore.QSize:
        return QtCore.QSize(self._preferred_size)

    def minimumSizeHint(self) -> QtCore.QSize:
        width = 320
        return QtCore.QSize(width, self.heightForWidth(width))

    def _layout_preview(self) -> None:
        rect = self.contentsRect()
        if rect.width() <= 0 or rect.height() <= 0:
            self._label.setGeometry(rect)
            return

        fitted_width = rect.width()
        fitted_height = self.heightForWidth(fitted_width)
        if fitted_height > rect.height():
            fitted_height = rect.height()
            fitted_width = max(1, int(round(fitted_height * self._aspect_ratio)))

        target_width = max(1, int(round(fitted_width * self._zoom_factor)))
        target_height = max(1, int(round(fitted_height * self._zoom_factor)))
        fitted_x = rect.x() + (rect.width() - fitted_width) // 2
        fitted_y = rect.y()
        x = fitted_x + (fitted_width - target_width) // 2
        y = fitted_y + (fitted_height - target_height) // 2
        self._label.setGeometry(x, y, max(1, target_width), max(1, target_height))

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:
        self._layout_preview()
        self.viewportChanged.emit()
        super().resizeEvent(event)


class CompactButtonGrid(QtWidgets.QWidget):
    def __init__(self, columns: int = 2, parent: QtWidgets.QWidget | None = None):
        super().__init__(parent)
        self._columns = max(1, int(columns))
        self._next_index = 0
        self._grid = QtWidgets.QGridLayout(self)
        self._grid.setContentsMargins(0, 0, 0, 0)
        self._grid.setHorizontalSpacing(6)
        self._grid.setVerticalSpacing(6)

    def gridLayout(self) -> QtWidgets.QGridLayout:
        return self._grid

    def addButton(
        self,
        button: QtWidgets.QPushButton,
        *,
        row: int | None = None,
        column: int | None = None,
        column_span: int = 1,
    ) -> None:
        span = max(1, min(self._columns, int(column_span)))
        if row is None or column is None:
            row = self._next_index // self._columns
            column = self._next_index % self._columns
            if column + span > self._columns:
                row += 1
                column = 0
        self._grid.addWidget(button, row, column, 1, span)
        self._next_index = max(self._next_index, row * self._columns + column + span)
