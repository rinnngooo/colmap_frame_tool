"""
選択した抽出画像(current)と、1つ前の時刻の抽出画像(prev)を並べて表示するウィジェット。
"""

from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QPixmap
from PyQt6.QtWidgets import QWidget, QHBoxLayout, QVBoxLayout, QLabel


class _ImagePane(QWidget):
    def __init__(self, title: str, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)

        self.title_label = QLabel(title)
        self.title_label.setStyleSheet("font-weight: bold;")

        self.timestamp_label = QLabel("--:--")

        self.image_label = QLabel()
        self.image_label.setMinimumSize(320, 240)
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setStyleSheet("background-color: #222;")

        layout.addWidget(self.title_label)
        layout.addWidget(self.timestamp_label)
        layout.addWidget(self.image_label, stretch=1)

    def set_image(self, image_path: str | None, timestamp_label: str = "--:--"):
        self.timestamp_label.setText(timestamp_label)
        if image_path is None:
            self.image_label.setText("(画像なし)")
            self.image_label.setPixmap(QPixmap())
            return

        pixmap = QPixmap(image_path)
        if pixmap.isNull():
            self.image_label.setText("(読み込み失敗)")
            return

        scaled = pixmap.scaled(
            self.image_label.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.image_label.setPixmap(scaled)


class ImageCompareWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QHBoxLayout(self)

        self.prev_pane = _ImagePane("1つ前の抽出画像 (prev)")
        self.current_pane = _ImagePane("選択中の抽出画像 (current)")

        layout.addWidget(self.prev_pane)
        layout.addWidget(self.current_pane)

    def set_prev(self, image_path: str | None, timestamp_label: str = "--:--"):
        self.prev_pane.set_image(image_path, timestamp_label)

    def set_current(self, image_path: str | None, timestamp_label: str = "--:--"):
        self.current_pane.set_image(image_path, timestamp_label)
