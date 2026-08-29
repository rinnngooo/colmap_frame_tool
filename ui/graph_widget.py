"""
時系列指標グラフウィジェット。

構成:
    - 下段: オーバービュー(動画全体、常時表示)。ドラッグで範囲選択(LinearRegionItem)。
      選択範囲は半透明色で塗りつぶされる(pyqtgraphのLinearRegionItemの標準挙動)。
    - 上段: 詳細ビュー。オーバービューで選択した範囲のみ拡大表示。
      指標はチェックボックスで表示/非表示を切り替え、各系列は最大1に正規化し、
      正規化倍率をレジェンドの系列名に併記する。
      詳細ビュー上の点をクリックするとそのフレームをcurrentとして選択するシグナルを発行する。

役割分担(操作の競合回避):
    - オーバービュー: ドラッグ専用(範囲選択)
    - 詳細ビュー: クリック専用(フレーム選択)
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pyqtgraph as pg
from PyQt6.QtCore import Qt, pyqtSignal, QRect
from PyQt6.QtGui import QCursor
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QCheckBox, QLabel, QGroupBox, QToolTip

METRIC_LABELS = {
    "orb_homography_iou": "ORB Homography IoU",
    "sp_lg_fmat_hull_area": "SP+LG F-mat Hull Area",
    "inlier_count": "Inlier Count",
    "inlier_ratio": "Inlier Ratio",
    "parallax_angle_deg": "Parallax Angle",
    "feature_coverage": "Feature Coverage",
    "blur_score": "Blur Score",
}

DEFAULT_COLORS = [
    (255, 99, 71), (30, 144, 255), (60, 179, 113), (238, 130, 238),
    (255, 215, 0), (0, 206, 209), (255, 140, 0),
]


def ms_to_mmss(ms: float) -> str:
    total_seconds = int(ms // 1000)
    m, s = divmod(total_seconds, 60)
    return f"{m:02d}:{s:02d}"


class TimeAxisItem(pg.AxisItem):
    """横軸をmm:ss表示にするカスタムAxisItem(値は内部的に秒で保持)。"""

    def tickStrings(self, values, scale, spacing):
        return [ms_to_mmss(v * 1000) for v in values]


class GraphWidget(QWidget):
    # フレーム(x軸上の時刻[秒])がクリックで選択されたときに発行
    frameSelected = pyqtSignal(float)

    def __init__(self, parent=None):
        super().__init__(parent)

        self._raw_series: dict[str, tuple[np.ndarray, np.ndarray]] = {}  # name -> (x_sec, y_raw)
        self._curves_detail: dict[str, pg.PlotDataItem] = {}
        self._curves_overview: dict[str, pg.PlotDataItem] = {}
        self._checkboxes: dict[str, QCheckBox] = {}
        self._scatter_points: Optional[pg.ScatterPlotItem] = None

        self._build_ui()

    def _build_ui(self):
        main_layout = QHBoxLayout(self)

        # --- グラフ側(詳細+オーバービュー縦積み) ---
        graph_layout = QVBoxLayout()

        self.detail_plot = pg.PlotWidget(axisItems={"bottom": TimeAxisItem(orientation="bottom")})
        self.detail_plot.setLabel("left", "正規化指標値")
        self.detail_plot.setLabel("bottom", "時刻 (mm:ss)")
        self.detail_plot.addLegend(offset=(10, 10))
        self.detail_plot.scene().sigMouseClicked.connect(self._on_detail_click)
        self.detail_plot.scene().sigMouseMoved.connect(self._on_detail_hover)
        self.detail_plot.setMouseEnabled(x=True, y=False)  # y軸方向のマウス操作(パン/ズーム)を無効化

        self.overview_plot = pg.PlotWidget(axisItems={"bottom": TimeAxisItem(orientation="bottom")})
        self.overview_plot.setMaximumHeight(120)
        self.overview_plot.setLabel("bottom", "時刻 (mm:ss、全体)")
        self.overview_plot.setMouseEnabled(x=True, y=False)

        self.region = pg.LinearRegionItem(brush=pg.mkBrush(100, 100, 255, 60))
        self.overview_plot.addItem(self.region)
        self.region.sigRegionChanged.connect(self._on_region_changed)

        graph_layout.addWidget(self.detail_plot, stretch=4)
        graph_layout.addWidget(self.overview_plot, stretch=1)

        # --- 右側: 指標選択チェックボックス ---
        side_box = QGroupBox("表示する指標")
        side_layout = QVBoxLayout()
        side_box.setLayout(side_layout)
        side_box.setMaximumWidth(220)
        self._side_layout = side_layout

        main_layout.addLayout(graph_layout, stretch=4)
        main_layout.addWidget(side_box, stretch=1)

    # -----------------------------------------------------------------
    # データ投入
    # -----------------------------------------------------------------

    def set_series(self, name: str, x_sec: np.ndarray, y_raw: np.ndarray):
        """指標データをセット(またはNoneを含む配列を渡して更新)。"""
        self._raw_series[name] = (np.asarray(x_sec, dtype=float), np.asarray(y_raw, dtype=float))
        if name not in self._checkboxes:
            self._add_checkbox(name)
        self._redraw()

    def _add_checkbox(self, name: str):
        label = METRIC_LABELS.get(name, name)
        cb = QCheckBox(label)
        cb.setChecked(True)
        cb.stateChanged.connect(self._redraw)
        self._side_layout.addWidget(cb)
        self._checkboxes[name] = cb

    def _redraw(self, *_):
        color_map = {name: DEFAULT_COLORS[i % len(DEFAULT_COLORS)] for i, name in enumerate(self._raw_series)}

        for name, (x_sec, y_raw) in self._raw_series.items():
            visible = self._checkboxes[name].isChecked()

            valid = ~np.isnan(y_raw)
            if valid.sum() == 0:
                scale = 1.0
                y_norm = y_raw
            else:
                max_abs = np.nanmax(np.abs(y_raw[valid]))
                scale = 1.0 / max_abs if max_abs > 1e-12 else 1.0
                y_norm = y_raw * scale

            label_text = f"{METRIC_LABELS.get(name, name)} (\u00d7{1.0 / scale:.4g})"
            color = color_map[name]

            if name not in self._curves_detail:
                self._curves_detail[name] = self.detail_plot.plot(
                    x_sec, y_norm, pen=pg.mkPen(color=color, width=2), name=label_text,
                    symbol="o", symbolSize=6, symbolBrush=color,
                )
                self._curves_overview[name] = self.overview_plot.plot(
                    x_sec, y_norm, pen=pg.mkPen(color=color, width=1)
                )
            else:
                self._curves_detail[name].setData(x_sec, y_norm)
                self._curves_detail[name].opts["name"] = label_text
                self._curves_overview[name].setData(x_sec, y_norm)

            self._curves_detail[name].setVisible(visible)
            self._curves_overview[name].setVisible(visible)

        # オーバービューのregionが未設定なら全体を初期選択範囲にする
        all_x = np.concatenate([x for x, _ in self._raw_series.values()]) if self._raw_series else np.array([0, 1])
        if all_x.size:
            lo, hi = float(np.min(all_x)), float(np.max(all_x))
            if self.region.getRegion() == (0, 1):
                self.region.setRegion((lo, hi))

    # -----------------------------------------------------------------
    # インタラクション
    # -----------------------------------------------------------------

    def _on_region_changed(self):
        lo, hi = self.region.getRegion()
        self.detail_plot.setXRange(lo, hi, padding=0.02)

    def _on_detail_click(self, event):
        if not self._raw_series:
            return
        vb = self.detail_plot.getPlotItem().vb
        mouse_point = vb.mapSceneToView(event.scenePos())
        clicked_x = mouse_point.x()

        # 全系列のうち最も近い点のx(時刻)を選択したフレームとみなす
        nearest_x = None
        nearest_dist = float("inf")
        for x_sec, _ in self._raw_series.values():
            if x_sec.size == 0:
                continue
            idx = int(np.argmin(np.abs(x_sec - clicked_x)))
            dist = abs(x_sec[idx] - clicked_x)
            if dist < nearest_dist:
                nearest_dist = dist
                nearest_x = x_sec[idx]

        if nearest_x is not None:
            self.frameSelected.emit(nearest_x)

    def _on_detail_hover(self, scene_pos):
        if not self._raw_series:
            return
        # プロット領域外にカーソルがある場合はツールチップを出さない
        if not self.detail_plot.sceneBoundingRect().contains(scene_pos):
            QToolTip.hideText()
            return

        vb = self.detail_plot.getPlotItem().vb
        mouse_point = vb.mapSceneToView(scene_pos)
        hover_x = mouse_point.x()

        # カーソルに最も近いフレーム(時刻)を探す(表示中の系列に限らず、全系列のx軸から探す)
        nearest_x = None
        nearest_dist = float("inf")
        for x_sec, _ in self._raw_series.values():
            if x_sec.size == 0:
                continue
            idx = int(np.argmin(np.abs(x_sec - hover_x)))
            dist = abs(x_sec[idx] - hover_x)
            if dist < nearest_dist:
                nearest_dist = dist
                nearest_x = x_sec[idx]

        if nearest_x is None:
            QToolTip.hideText()
            return

        lines = [f"時刻: {ms_to_mmss(nearest_x * 1000)}"]
        for name, (x_sec, y_raw) in self._raw_series.items():
            if name not in self._checkboxes or not self._checkboxes[name].isChecked():
                continue
            if x_sec.size == 0:
                continue
            idx = int(np.argmin(np.abs(x_sec - nearest_x)))
            if abs(x_sec[idx] - nearest_x) > 1e-6:
                continue  # この系列にはその時刻のデータ点が無い
            value = y_raw[idx]
            label = METRIC_LABELS.get(name, name)
            value_text = "-" if np.isnan(value) else f"{value:.4g}"
            lines.append(f"{label}: {value_text}")

        # msecShowTimeを明示指定しないと(デフォルト-1)、テキストの長さから自動算出される
        # 表示時間が短くなり、すぐに消えてしまうことがある。マウスが動いている間は
        # このハンドラが呼ばれるたびに再表示されるが、マウスが止まっている間も
        # 消えないよう十分に長い時間(60秒)を指定する。
        QToolTip.showText(QCursor.pos(), "\n".join(lines), self.detail_plot, QRect(), 60000)

    def highlight_selected(self, x_sec: float):
        """選択中フレームの位置に縦線を表示する。"""
        if not hasattr(self, "_selection_line"):
            self._selection_line = pg.InfiniteLine(angle=90, pen=pg.mkPen("w", width=2, style=Qt.PenStyle.DashLine))
            self.detail_plot.addItem(self._selection_line)
        self._selection_line.setPos(x_sec)
