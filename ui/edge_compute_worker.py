"""
重い計算(SuperPoint+LightGlue推論を含むエッジ指標計算、モデル事前ロード)を
バックグラウンドスレッド(QThread)で実行するためのワーカー。

GUIスレッドをブロックしないよう、画像I/Oや指標計算はすべてこのワーカー内で行い、
結果はシグナル経由でメインスレッドに渡す(共有オブジェクトへの直接書き込みはしない)。
"""

from __future__ import annotations

import cv2
from PyQt6.QtCore import QObject, pyqtSignal

import metrics as metrics_mod


class EdgeComputeWorker(QObject):
    # (done, total)
    progress = pyqtSignal(int, int)
    # (from_filename, to_filename, metrics_dict)
    edge_result = pyqtSignal(str, str, dict)
    # SuperPoint+LightGlueが使えず自動フォールバックした場合に1度だけ発行
    fallback_warning = pyqtSignal(str)
    finished = pyqtSignal()

    def __init__(self, jobs: list[tuple[str, str, str, str]], use_sp_lg: bool):
        """
        jobs: [(from_filename, to_filename, from_image_path, to_image_path), ...]
        use_sp_lg: SuperPoint+LightGlueを試すかどうか(失敗時はORBに自動フォールバック)
        """
        super().__init__()
        self.jobs = jobs
        self.use_sp_lg = use_sp_lg

    def run(self):
        total = len(self.jobs)
        for i, (from_fn, to_fn, from_path, to_path) in enumerate(self.jobs):
            prev_img = cv2.imread(from_path)
            curr_img = cv2.imread(to_path)

            values = None
            if prev_img is None or curr_img is None:
                values = {}
            elif self.use_sp_lg:
                try:
                    values = metrics_mod.compute_edge_metrics(
                        prev_img, curr_img, K=None, use_superpoint_lightglue=True
                    )
                except ImportError as e:
                    self.use_sp_lg = False
                    self.fallback_warning.emit(str(e))

            if values is None:
                values = metrics_mod.compute_edge_metrics(prev_img, curr_img, K=None, use_superpoint_lightglue=False)

            self.edge_result.emit(from_fn, to_fn, values)
            self.progress.emit(i + 1, total)

        self.finished.emit()


class WarmupWorker(QObject):
    """SuperPoint+LightGlueモデルの事前ロードだけを行うワーカー。"""

    finished = pyqtSignal(bool)  # ロードできたか

    def run(self):
        ok = metrics_mod.warmup()
        self.finished.emit(ok)
