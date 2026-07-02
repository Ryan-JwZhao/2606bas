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

    def __init__(self, aspect_ratio: float = 16.0 / 9.0, parent: QtWidgets.QWidget | None = None):
        super().__init__(parent)
        self.setObjectName("previewViewport")
        self._aspect_ratio = max(0.1, float(aspect_ratio))
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

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:
        rect = self.contentsRect()
        if rect.width() <= 0 or rect.height() <= 0:
            self._label.setGeometry(rect)
            super().resizeEvent(event)
            return

        target_width = rect.width()
        target_height = self.heightForWidth(target_width)
        if target_height > rect.height():
            target_height = rect.height()
            target_width = max(1, int(round(target_height * self._aspect_ratio)))

        x = rect.x() + (rect.width() - target_width) // 2
        y = rect.y()
        self._label.setGeometry(x, y, max(1, target_width), max(1, target_height))
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
