"""
「画像を抽出」ボタンを押した際に表示する、抽出方式選択ダイアログ。

- fps指定で抽出: 従来通り、指定fpsで等間隔にフレームを抽出する
- 指標で抽出: 直近保存フレームからORB Homography IoU / Feature Coverageが
  目標値を下回る直前のフレームを保存していく(Min/Max Frame Gapの範囲内で)
"""

from __future__ import annotations

from typing import Optional

from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QFormLayout, QGroupBox, QRadioButton,
    QDoubleSpinBox, QSpinBox, QDialogButtonBox, QLabel, QWidget,
)


class ExtractionOptionsDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("画像を抽出します")
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("画像を抽出します"))

        # --- fps指定で抽出 ---
        self.radio_fps = QRadioButton("fps指定で抽出")
        self.radio_fps.setChecked(True)
        self.spin_fps = QDoubleSpinBox()
        self.spin_fps.setRange(0.1, 60.0)
        self.spin_fps.setSingleStep(0.1)
        self.spin_fps.setDecimals(2)
        self.spin_fps.setValue(1.0)

        fps_row = QHBoxLayout()
        fps_row.addWidget(self.radio_fps)
        fps_row.addStretch(1)
        fps_row.addWidget(self.spin_fps)
        layout.addLayout(fps_row)

        # --- 指標で抽出 ---
        self.radio_metrics = QRadioButton("指標で抽出")
        layout.addWidget(self.radio_metrics)

        metrics_panel = QWidget()
        form = QFormLayout(metrics_panel)

        self.spin_target_iou = QDoubleSpinBox()
        self.spin_target_iou.setRange(0.0, 1.0)
        self.spin_target_iou.setSingleStep(0.01)
        self.spin_target_iou.setDecimals(2)
        self.spin_target_iou.setValue(0.85)
        form.addRow("Target ORB Homography IoU", self.spin_target_iou)

        self.spin_target_coverage = QDoubleSpinBox()
        self.spin_target_coverage.setRange(0.0, 1.0)
        self.spin_target_coverage.setSingleStep(0.01)
        self.spin_target_coverage.setDecimals(2)
        self.spin_target_coverage.setValue(0.95)
        form.addRow("Target Feature Coverage", self.spin_target_coverage)

        self.spin_min_gap = QSpinBox()
        self.spin_min_gap.setRange(1, 100000)
        self.spin_min_gap.setValue(5)
        form.addRow("Min Frame Gap", self.spin_min_gap)

        self.spin_max_gap = QSpinBox()
        self.spin_max_gap.setRange(1, 100000)
        self.spin_max_gap.setValue(150)
        form.addRow("Max Frame Gap", self.spin_max_gap)

        layout.addWidget(metrics_panel)

        # ラジオボタンの選択に応じて有効/無効を切り替える
        def _update_enabled():
            self.spin_fps.setEnabled(self.radio_fps.isChecked())
            metrics_panel.setEnabled(self.radio_metrics.isChecked())

        self.radio_fps.toggled.connect(_update_enabled)
        self.radio_metrics.toggled.connect(_update_enabled)
        _update_enabled()

        # --- OK/キャンセル ---
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def get_result(self) -> Optional[dict]:
        """OKで閉じられた場合に選択内容を辞書で返す。呼び出し前にexec()すること。"""
        if self.radio_fps.isChecked():
            return {"mode": "fps", "fps": self.spin_fps.value()}
        return {
            "mode": "metrics",
            "target_iou": self.spin_target_iou.value(),
            "target_coverage": self.spin_target_coverage.value(),
            "min_gap": self.spin_min_gap.value(),
            "max_gap": self.spin_max_gap.value(),
        }
