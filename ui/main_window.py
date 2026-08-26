"""
メインウィンドウ。

レイアウト:
    上段: ImageCompareWidget (prev / current 並列表示)
    下段: GraphWidget (オーバービュー+詳細ビュー、指標チェックボックス)
    ツールバー: 動画を開く / fps指定抽出 / 画像追加 / 画像削除 / プロジェクト保存

操作フロー:
    1. 動画を開く
    2. fpsを指定して抽出 -> images_dir に保存、ProjectStoreへ初期登録、
       全隣接エッジの指標を計算してグラフ表示
    3. グラフ(詳細ビュー)をクリックしてcurrentフレームを選択 -> 画像比較表示更新
    4. 「画像追加」: prevとcurrentの間の動画フレームを追加、影響エッジのみ再計算
    5. 「画像削除」: current画像を削除、影響エッジのみ再計算

キーボード操作:
    ← / →   : 選択中フレームを1つ前/次のフレームへ移動
    Insert  : 画像を追加(prev-current間)
    Delete  : 現在の画像を削除
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QFileDialog, QInputDialog, QMessageBox, QSplitter, QProgressDialog,
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QKeySequence, QShortcut

from project_store import ProjectStore, FrameRecord
from video_io import VideoReader
import metrics as metrics_mod
from ui.graph_widget import GraphWidget
from ui.image_compare_widget import ImageCompareWidget


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("COLMAP Frame Extraction Tool")
        self.resize(1200, 800)

        self.store: Optional[ProjectStore] = None
        self.reader: Optional[VideoReader] = None
        self.project_dir: Optional[Path] = None
        self.current_filename: Optional[str] = None
        self.use_sp_lg: bool = True   # SuperPoint+LightGlueを使う(未インストール等ならORBへ自動フォールバック)
        self._sp_lg_fallback_warned: bool = False

        self._build_ui()
        self._setup_shortcuts()

    # -----------------------------------------------------------------
    # キーボード操作
    # -----------------------------------------------------------------

    def _setup_shortcuts(self):
        """← / → で選択フレーム移動、Insertで追加、Deleteで削除。

        QShortcutはウィンドウ内でフォーカスされているウィジェットに関わらず
        (WindowShortcutコンテキスト、デフォルト)反応する。
        """
        self.shortcut_prev = QShortcut(QKeySequence(Qt.Key.Key_Left), self)
        self.shortcut_prev.activated.connect(self.select_prev_frame)

        self.shortcut_next = QShortcut(QKeySequence(Qt.Key.Key_Right), self)
        self.shortcut_next.activated.connect(self.select_next_frame)

        self.shortcut_insert = QShortcut(QKeySequence(Qt.Key.Key_Insert), self)
        self.shortcut_insert.activated.connect(self.add_frame_between)

        self.shortcut_delete = QShortcut(QKeySequence(Qt.Key.Key_Delete), self)
        self.shortcut_delete.activated.connect(self.remove_current_frame)

    def select_prev_frame(self):
        if self.store is None or self.current_filename is None or not self.store.frames:
            return
        idx = next(i for i, f in enumerate(self.store.frames) if f.filename == self.current_filename)
        if idx > 0:
            self.current_filename = self.store.frames[idx - 1].filename
            self._refresh_image_compare()

    def select_next_frame(self):
        if self.store is None or self.current_filename is None or not self.store.frames:
            return
        idx = next(i for i, f in enumerate(self.store.frames) if f.filename == self.current_filename)
        if idx < len(self.store.frames) - 1:
            self.current_filename = self.store.frames[idx + 1].filename
            self._refresh_image_compare()

    # -----------------------------------------------------------------
    # UI構築
    # -----------------------------------------------------------------

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root_layout = QVBoxLayout(central)

        # ツールバー相当のボタン列
        button_row = QHBoxLayout()
        self.btn_open_video = QPushButton("動画を開く")
        self.btn_extract = QPushButton("fps指定で抽出")
        self.btn_add = QPushButton("画像を追加(prev-current間)")
        self.btn_remove = QPushButton("現在の画像を削除")
        self.btn_save = QPushButton("プロジェクト保存")
        self.btn_load = QPushButton("プロジェクトを開く")

        self.btn_open_video.clicked.connect(self.open_video)
        self.btn_extract.clicked.connect(self.extract_frames)
        self.btn_add.clicked.connect(self.add_frame_between)
        self.btn_remove.clicked.connect(self.remove_current_frame)
        self.btn_save.clicked.connect(self.save_project)
        self.btn_load.clicked.connect(self.load_project)

        for b in (self.btn_open_video, self.btn_extract, self.btn_add, self.btn_remove, self.btn_save, self.btn_load):
            button_row.addWidget(b)
        root_layout.addLayout(button_row)

        self.status_label = QLabel("動画未読み込み")
        root_layout.addWidget(self.status_label)

        splitter = QSplitter(Qt.Orientation.Vertical)
        self.image_compare = ImageCompareWidget()
        self.graph_widget = GraphWidget()
        self.graph_widget.frameSelected.connect(self.on_frame_selected)

        splitter.addWidget(self.image_compare)
        splitter.addWidget(self.graph_widget)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 1)

        root_layout.addWidget(splitter, stretch=1)

        self._set_actions_enabled(video_loaded=False, project_loaded=False)

    def _set_actions_enabled(self, video_loaded: bool, project_loaded: bool):
        self.btn_extract.setEnabled(video_loaded)
        self.btn_add.setEnabled(project_loaded)
        self.btn_remove.setEnabled(project_loaded)
        self.btn_save.setEnabled(project_loaded)

    # -----------------------------------------------------------------
    # 動画を開く / 抽出
    # -----------------------------------------------------------------

    def open_video(self):
        path, _ = QFileDialog.getOpenFileName(self, "動画を選択", "", "Video Files (*.mp4 *.mov *.avi *.mkv)")
        if not path:
            return
        try:
            self.reader = VideoReader(path)
        except Exception as e:
            QMessageBox.critical(self, "エラー", f"動画を開けませんでした: {e}")
            return

        self.video_path = path
        self.status_label.setText(
            f"動画読み込み完了: {Path(path).name} (fps={self.reader.fps:.2f}, frames={self.reader.frame_count})"
        )
        self._set_actions_enabled(video_loaded=True, project_loaded=False)

    def extract_frames(self):
        if self.reader is None:
            return

        fps, ok = QInputDialog.getDouble(self, "抽出fps", "抽出するfpsを入力してください", 2.0, 0.1, 60.0, 2)
        if not ok:
            return

        out_dir = QFileDialog.getExistingDirectory(self, "プロジェクト出力先フォルダを選択")
        if not out_dir:
            return
        self.project_dir = Path(out_dir)
        images_dir = self.project_dir / "images"
        images_dir.mkdir(parents=True, exist_ok=True)

        self.store = ProjectStore(
            video_path=self.video_path,
            video_fps=self.reader.fps,
            extract_fps=fps,
            images_dir="images",
        )

        frame_records = []
        samples = list(self.reader.extract_at_fps(fps))

        progress = QProgressDialog("フレーム抽出中...", "キャンセル", 0, len(samples), self)
        progress.setWindowModality(Qt.WindowModality.WindowModal)

        for i, sample in enumerate(samples):
            if progress.wasCanceled():
                break
            record = FrameRecord.create(sample.frame_index, sample.timestamp_ms)
            record.blur_score = metrics_mod.blur_score(sample.image)
            record.exposure_ratio = metrics_mod.exposure_ratio(sample.image)
            cv2.imwrite(str(images_dir / record.filename), sample.image)
            frame_records.append(record)
            progress.setValue(i + 1)

        progress.close()

        affected_edges = self.store.initial_extract(frame_records)
        self._recompute_edges(affected_edges, progress_title="エッジ指標を計算中...")

        self.status_label.setText(f"{len(frame_records)}枚を抽出しました ({fps} fps)")
        self._set_actions_enabled(video_loaded=True, project_loaded=True)
        self._refresh_graph()

        if self.store.frames:
            self.current_filename = self.store.frames[-1].filename
            self._refresh_image_compare()

    # -----------------------------------------------------------------
    # エッジ指標の(再)計算
    # -----------------------------------------------------------------

    def _load_frame_image(self, filename: str) -> np.ndarray:
        path = self.project_dir / self.store.images_dir / filename
        img = cv2.imread(str(path))
        if img is None:
            raise IOError(f"画像を読み込めませんでした: {path}")
        return img

    def _recompute_edges(self, edges, progress_title: str = "計算中..."):
        if not edges:
            return
        progress = QProgressDialog(progress_title, None, 0, len(edges), self)
        progress.setWindowModality(Qt.WindowModality.WindowModal)

        for i, edge in enumerate(edges):
            prev_img = self._load_frame_image(edge.from_filename)
            curr_img = self._load_frame_image(edge.to_filename)
            values = self._compute_edge_metrics_with_fallback(prev_img, curr_img)
            for k, v in values.items():
                setattr(edge, k, v)
            progress.setValue(i + 1)
        progress.close()

    def _compute_edge_metrics_with_fallback(self, prev_img, curr_img) -> dict:
        """SuperPoint+LightGlueが使えればそれを使い、未インストール等で失敗する場合は
        ORBに自動フォールバックする(初回のみユーザーに通知)。
        """
        if self.use_sp_lg:
            try:
                return metrics_mod.compute_edge_metrics(
                    prev_img, curr_img, K=None, use_superpoint_lightglue=True
                )
            except ImportError as e:
                self.use_sp_lg = False
                if not self._sp_lg_fallback_warned:
                    self._sp_lg_fallback_warned = True
                    QMessageBox.warning(
                        self, "SuperPoint+LightGlue利用不可",
                        f"SuperPoint+LightGlueが使用できないため、以後ORBで計算します。\n詳細: {e}",
                    )
        return metrics_mod.compute_edge_metrics(prev_img, curr_img, K=None, use_superpoint_lightglue=False)

    # -----------------------------------------------------------------
    # グラフ / 画像比較の表示更新
    # -----------------------------------------------------------------

    def _refresh_graph(self):
        if self.store is None:
            return

        # エッジ指標(prev-current間) -> x軸はcurrent(to側)フレームの時刻
        x_sec = np.array([f.timestamp_ms / 1000 for f in self.store.frames[1:]])

        edge_metric_names = [
            "orb_homography_iou", "sp_lg_fmat_hull_area", "inlier_ratio",
            "parallax_angle_deg", "feature_coverage",
        ]
        for name in edge_metric_names:
            y = np.array([
                getattr(e, name) if getattr(e, name) is not None else np.nan
                for e in self.store.edges
            ])
            self.graph_widget.set_series(name, x_sec, y)

        # フレーム単体の指標(ブレ)
        x_all = np.array([f.timestamp_ms / 1000 for f in self.store.frames])
        blur = np.array([f.blur_score if f.blur_score is not None else np.nan for f in self.store.frames])
        self.graph_widget.set_series("blur_score", x_all, blur)

    def _refresh_image_compare(self):
        if self.store is None or self.current_filename is None:
            return
        idx = next(i for i, f in enumerate(self.store.frames) if f.filename == self.current_filename)
        curr = self.store.frames[idx]
        images_dir = self.project_dir / self.store.images_dir

        self.image_compare.set_current(str(images_dir / curr.filename), self._mmss(curr.timestamp_ms))

        if idx > 0:
            prev = self.store.frames[idx - 1]
            self.image_compare.set_prev(str(images_dir / prev.filename), self._mmss(prev.timestamp_ms))
        else:
            self.image_compare.set_prev(None)

        self.graph_widget.highlight_selected(curr.timestamp_ms / 1000)

    @staticmethod
    def _mmss(ms: int) -> str:
        total_seconds = int(ms // 1000)
        m, s = divmod(total_seconds, 60)
        return f"{m:02d}:{s:02d}"

    # -----------------------------------------------------------------
    # インタラクション: フレーム選択 / 追加 / 削除
    # -----------------------------------------------------------------

    def on_frame_selected(self, x_sec: float):
        if self.store is None:
            return
        target_ms = x_sec * 1000
        nearest = min(self.store.frames, key=lambda f: abs(f.timestamp_ms - target_ms))
        self.current_filename = nearest.filename
        self._refresh_image_compare()

    def add_frame_between(self):
        if self.store is None or self.current_filename is None or self.reader is None:
            return
        idx = next(i for i, f in enumerate(self.store.frames) if f.filename == self.current_filename)
        if idx == 0:
            QMessageBox.information(self, "情報", "先頭フレームより前には追加できません")
            return

        prev = self.store.frames[idx - 1]
        curr = self.store.frames[idx]
        mid_frame_index = (prev.frame_index + curr.frame_index) // 2
        if mid_frame_index in (prev.frame_index, curr.frame_index):
            QMessageBox.information(self, "情報", "これ以上細かく間に挿入できるフレームがありません")
            return

        sample = self.reader.get_frame(mid_frame_index)
        images_dir = self.project_dir / self.store.images_dir
        new_record, affected = self.store.add_frame(sample.frame_index, sample.timestamp_ms)
        new_record.blur_score = metrics_mod.blur_score(sample.image)
        new_record.exposure_ratio = metrics_mod.exposure_ratio(sample.image)
        cv2.imwrite(str(images_dir / new_record.filename), sample.image)

        self._recompute_edges(affected, progress_title="追加分の指標を再計算中...")
        self.current_filename = new_record.filename
        self._refresh_graph()
        self._refresh_image_compare()

    def remove_current_frame(self):
        if self.store is None or self.current_filename is None:
            return
        removed_filename = self.current_filename
        idx = next(i for i, f in enumerate(self.store.frames) if f.filename == removed_filename)

        affected = self.store.remove_frame(removed_filename)
        images_dir = self.project_dir / self.store.images_dir
        (images_dir / removed_filename).unlink(missing_ok=True)

        self._recompute_edges(affected, progress_title="削除後の指標を再計算中...")

        if self.store.frames:
            next_idx = min(idx, len(self.store.frames) - 1)
            self.current_filename = self.store.frames[next_idx].filename
        else:
            self.current_filename = None

        self._refresh_graph()
        self._refresh_image_compare()

    # -----------------------------------------------------------------
    # プロジェクト保存
    # -----------------------------------------------------------------

    def save_project(self):
        if self.store is None or self.project_dir is None:
            return
        self.store.save(self.project_dir / "project.json")
        QMessageBox.information(self, "保存完了", f"{self.project_dir / 'project.json'} に保存しました")

    def load_project(self):
        path, _ = QFileDialog.getOpenFileName(self, "プロジェクトファイルを選択", "", "Project JSON (project.json);;JSON Files (*.json)")
        if not path:
            return

        try:
            self.store = ProjectStore.load(path)
        except Exception as e:
            QMessageBox.critical(self, "エラー", f"プロジェクトを読み込めませんでした: {e}")
            return

        self.project_dir = Path(path).parent
        images_dir = self.project_dir / self.store.images_dir
        if not images_dir.exists():
            QMessageBox.warning(
                self, "警告",
                f"画像フォルダが見つかりません: {images_dir}\n"
                "プロジェクトファイルと images フォルダは同じ場所に置いてください。",
            )

        # 動画の再オープンを試みる(画像追加にはシークが必要なため)。
        # 動画が見つからない/開けない場合でも、既存画像の閲覧・削除は可能にする。
        self.reader = None
        video_loaded = False
        try:
            self.reader = VideoReader(self.store.video_path)
            self.video_path = self.store.video_path
            video_loaded = True
        except Exception as e:
            QMessageBox.warning(
                self, "動画が見つかりません",
                f"元動画を開けませんでした(画像の追加はできません): {e}\n"
                f"video_path: {self.store.video_path}",
            )

        self.status_label.setText(
            f"プロジェクト読み込み完了: {path} (frames={len(self.store.frames)}, "
            f"動画{'あり' if video_loaded else 'なし'})"
        )
        self._set_actions_enabled(video_loaded=video_loaded, project_loaded=True)
        # 既存プロジェクトを開いた後の「fps指定で抽出」は、frames/edgesを丸ごと
        # 上書きしてしまい手動追加/削除の結果が失われるため、常に無効化しておく。
        self.btn_extract.setEnabled(False)

        if self.store.frames:
            self.current_filename = self.store.frames[0].filename
        self._refresh_graph()
        self._refresh_image_compare()
