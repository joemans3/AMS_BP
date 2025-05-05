from PyQt6.QtCore import QPropertyAnimation, QSize, Qt, pyqtSignal
from PyQt6.QtGui import QBrush, QColor, QPainter
from PyQt6.QtWidgets import QWidget


class ToggleSwitch(QWidget):
    toggled = pyqtSignal(bool)

    def __init__(self, parent=None, checked=False):
        super().__init__(parent)
        self.setFixedSize(50, 28)
        self._checked = checked
        self._circle_position = 2 if not checked else 24

        self.animation = QPropertyAnimation(self, b"")
        self.animation.setDuration(200)

    def sizeHint(self):
        return QSize(50, 28)

    def mousePressEvent(self, event):
        self._checked = not self._checked
        self.animate_toggle()
        self.toggled.emit(self._checked)
        self.update()

    def animate_toggle(self):
        start = self._circle_position
        end = 24 if self._checked else 2
        self.animation.stop()
        self.animation.setStartValue(start)
        self.animation.setEndValue(end)
        self.animation.valueChanged.connect(self.set_circle_position)
        self.animation.start()

    def set_circle_position(self, val):
        self._circle_position = val
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        bg_color = QColor("#dddddd") if self._checked else QColor("#333333")
        painter.setBrush(QBrush(bg_color))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawRoundedRect(0, 0, self.width(), self.height(), 14, 14)

        painter.setBrush(QBrush(QColor("#008000")))
        painter.drawEllipse(int(self._circle_position), 2, 24, 24)

    def isChecked(self):
        return self._checked

    def setChecked(self, checked: bool):
        self._checked = checked
        self._circle_position = 24 if checked else 2
        self.update()
