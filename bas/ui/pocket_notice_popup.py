from __future__ import annotations

from collections.abc import Sequence

from PyQt5 import QtCore, QtWidgets


class OperatorPocketNoticePopup(QtWidgets.QDialog):
    """Non-blocking pocket notice owned exclusively by the operator console."""

    DEFAULT_VISIBLE_MS = 4500

    def __init__(
        self,
        parent: QtWidgets.QWidget,
        *,
        visible_ms: int = DEFAULT_VISIBLE_MS,
    ) -> None:
        super().__init__(parent)
        self._visible_ms = max(500, int(visible_ms))
        self.setObjectName("operatorPocketNoticePopup")
        self.setWindowTitle("进球提示")
        self.setModal(False)
        self.setWindowModality(QtCore.Qt.NonModal)
        self.setWindowFlags(
            QtCore.Qt.Tool
            | QtCore.Qt.FramelessWindowHint
            | QtCore.Qt.WindowStaysOnTopHint
        )
        self.setAttribute(QtCore.Qt.WA_ShowWithoutActivating, True)
        self.setMinimumWidth(420)

        self._title_label = QtWidgets.QLabel("进球！", self)
        self._title_label.setObjectName("operatorPocketNoticeTitle")
        self._title_label.setAlignment(QtCore.Qt.AlignCenter)
        self._message_label = QtWidgets.QLabel("", self)
        self._message_label.setObjectName("operatorPocketNoticeMessage")
        self._message_label.setAlignment(QtCore.Qt.AlignCenter)
        self._message_label.setWordWrap(True)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(30, 20, 30, 22)
        layout.setSpacing(8)
        layout.addWidget(self._title_label)
        layout.addWidget(self._message_label)

        self.setStyleSheet(
            "QDialog#operatorPocketNoticePopup {"
            "background-color: #08120e;"
            "border: 4px solid #5cde97;"
            "border-radius: 16px;"
            "}"
            "QLabel#operatorPocketNoticeTitle {"
            "color: #5cde97;"
            "font-size: 32px;"
            "font-weight: 900;"
            "}"
            "QLabel#operatorPocketNoticeMessage {"
            "color: #f5fff9;"
            "font-size: 26px;"
            "font-weight: 800;"
            "}"
        )

        self._hide_timer = QtCore.QTimer(self)
        self._hide_timer.setSingleShot(True)
        self._hide_timer.timeout.connect(self.hide)
        self.hide()

    def text(self) -> str:
        return self._message_label.text()

    def show_messages(self, messages: Sequence[str]) -> None:
        shown = [str(message).strip() for message in messages if str(message).strip()]
        if not shown:
            return
        self._message_label.setText(" · ".join(shown))
        self._bind_to_operator_screen()
        self.adjustSize()
        self._position_over_operator()
        self.show()
        self.raise_()
        self._hide_timer.start(self._visible_ms)

    def clear(self) -> None:
        self._hide_timer.stop()
        self.hide()

    def _position_over_operator(self) -> None:
        owner = self.parentWidget()
        if owner is None:
            return
        self.adjustSize()
        owner_top_left = owner.mapToGlobal(owner.rect().topLeft())
        x = owner_top_left.x() + max(12, (owner.width() - self.width()) // 2)
        y = owner_top_left.y() + max(56, int(round(owner.height() * 0.12)))
        self.move(x, y)

    def _bind_to_operator_screen(self) -> None:
        owner = self.parentWidget()
        if owner is None:
            return
        owner_handle = owner.windowHandle()
        if owner_handle is None or owner_handle.screen() is None:
            return
        self.winId()
        popup_handle = self.windowHandle()
        if popup_handle is not None:
            popup_handle.setScreen(owner_handle.screen())


__all__ = ["OperatorPocketNoticePopup"]
