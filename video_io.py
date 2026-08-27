"""
動画の読み込み・任意フレームへの正確なシークを抽象化するモジュール。

decordはランダムアクセス(フレーム番号指定の取得)が正確かつ高速なため優先的に使う。
未インストール・非対応環境ではOpenCVにフォールバックする。
OpenCVフォールバック時はCAP_PROP_POS_FRAMESのシーク精度がコーデックに依存する点に注意
(mp4/H.264でもGOP構造によっては数フレームずれることがある)。
シビアな精度が必要な場合はdecordの導入を推奨。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Optional

import numpy as np

try:
    import decord
    _HAS_DECORD = True
except ImportError:
    _HAS_DECORD = False

import cv2


@dataclass
class FrameSample:
    frame_index: int
    timestamp_ms: int
    image: np.ndarray  # BGR, HxWx3


class VideoReader:
    def __init__(self, video_path: str):
        self.video_path = video_path
        self._backend = "decord" if _HAS_DECORD else "opencv"

        if self._backend == "decord":
            self._vr = decord.VideoReader(video_path)
            self.fps = float(self._vr.get_avg_fps())
            self.frame_count = len(self._vr)
        else:
            self._cap = cv2.VideoCapture(video_path)
            if not self._cap.isOpened():
                raise IOError(f"動画を開けませんでした: {video_path}")
            self.fps = float(self._cap.get(cv2.CAP_PROP_FPS))
            self.frame_count = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))

    def close(self) -> None:
        if self._backend == "opencv":
            self._cap.release()
        # decordはVideoReaderに明示的なcloseは不要

    def get_frame(self, frame_index: int) -> FrameSample:
        """frame_index(元動画基準)のフレームをBGRで取得する。"""
        frame_index = max(0, min(frame_index, self.frame_count - 1))
        timestamp_ms = int(round(frame_index / self.fps * 1000))

        if self._backend == "decord":
            frame_rgb = self._vr[frame_index].asnumpy()  # RGB
            image = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        else:
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, image = self._cap.read()
            if not ok:
                raise IOError(f"フレーム取得に失敗しました: index={frame_index}")

        return FrameSample(frame_index=frame_index, timestamp_ms=timestamp_ms, image=image)

    def estimate_extract_count(self, target_fps: float) -> int:
        """extract_at_fps(target_fps)が生成するフレーム数を、実際にフレームを
        読み込まずに見積もる(進捗バーの範囲設定などに使う)。
        """
        if target_fps <= 0:
            raise ValueError("target_fps は正の値である必要があります")
        step = self.fps / target_fps
        num_samples = int(self.frame_count / step) + 1
        count = 0
        for i in range(num_samples):
            frame_index = int(round(i * step))
            if frame_index >= self.frame_count:
                break
            count += 1
        return count

    def extract_at_fps(self, target_fps: float) -> Iterator[FrameSample]:
        """指定fpsで等間隔にフレームを抽出する(元動画基準のframe_indexで返す)。"""
        if target_fps <= 0:
            raise ValueError("target_fps は正の値である必要があります")

        step = self.fps / target_fps  # 元動画フレーム換算での間隔
        num_samples = int(self.frame_count / step) + 1

        for i in range(num_samples):
            frame_index = int(round(i * step))
            if frame_index >= self.frame_count:
                break
            yield self.get_frame(frame_index)

    def frame_index_from_timestamp_ms(self, timestamp_ms: int) -> int:
        return int(round(timestamp_ms / 1000 * self.fps))
