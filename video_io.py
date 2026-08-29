"""
動画の読み込み・任意フレームへの正確なシークを抽象化するモジュール。

decordはランダムアクセス(フレーム番号指定の取得)が正確かつ高速なため優先的に使う。
未インストール・非対応環境ではOpenCVにフォールバックする。
OpenCVフォールバック時はCAP_PROP_POS_FRAMESのシーク精度がコーデックに依存する点に注意
(mp4/H.264でもGOP構造によっては数フレームずれることがある)。
シビアな精度が必要な場合はdecordの導入を推奨。

大きなシークの回避策:
    decordは離れた位置、特に後方への大きなランダムシークで内部的に大量のメモリを
    消費することがある(実測で確認済み: fps指定抽出後、遠く離れた位置でInsertすると
    メモリが急増する事象があった)。extract_at_fps()による連続的な前方アクセスでは
    問題ないため、get_frame()単体で「直前のアクセス位置から大きく離れた」呼び出しが
    あった場合だけ、decordの代わりに使い捨てのcv2.VideoCaptureでその1フレームだけを
    取得するようにしている(こちらはFFmpegの通常のシーク処理に任せるため、メモリの
    使われ方がより予測しやすい)。

メモリ診断:
    環境変数 COLMAP_TOOL_DEBUG_MEMORY=1 を立てると、get_frame()呼び出しのたびに
    「直前にアクセスしたフレーム番号からのシーク距離」と「RSSの変化」を標準出力に記録する。
    fps指定抽出(常に前方への連続アクセス)では出ないが、Insertでの追加(任意の位置への
    ランダムアクセス、特に後方への大きなシーク)でメモリが急増するようであれば、
    ここが原因である可能性が高い。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Optional

import numpy as np

from mem_diag import DEBUG_MEMORY, current_rss_mb, release_unused_memory

try:
    import decord
    _HAS_DECORD = True
except ImportError:
    _HAS_DECORD = False

import cv2

# decord使用時、直前のアクセス位置からこのフレーム数以上離れた場所への
# シークが発生した場合は、decordではなく使い捨てのcv2.VideoCaptureに切り替える。
# (前方連続アクセスであるfps指定抽出では通常このしきい値を超えない)
LARGE_SEEK_THRESHOLD_FRAMES = 60


@dataclass
class FrameSample:
    frame_index: int
    timestamp_ms: int
    image: np.ndarray  # BGR, HxWx3


class VideoReader:
    def __init__(self, video_path: str):
        self.video_path = video_path
        self._backend = "decord" if _HAS_DECORD else "opencv"
        self._last_accessed_frame_index: Optional[int] = None  # シーク距離計測用

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

        rss_before = current_rss_mb() if DEBUG_MEMORY else None
        seek_distance = (
            frame_index - self._last_accessed_frame_index
            if self._last_accessed_frame_index is not None else None
        )
        is_large_seek = seek_distance is not None and abs(seek_distance) > LARGE_SEEK_THRESHOLD_FRAMES

        used_backend = self._backend
        if self._backend == "decord" and is_large_seek:
            # 大きな(特に後方への)シークはdecordではなく使い捨てのcv2.VideoCaptureで処理する
            image = self._get_frame_via_disposable_opencv(frame_index)
            used_backend = "opencv(large-seek-fallback)"
        elif self._backend == "decord":
            frame_rgb = self._vr[frame_index].asnumpy()  # RGB
            image = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        else:
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, image = self._cap.read()
            if not ok:
                raise IOError(f"フレーム取得に失敗しました: index={frame_index}")

        if DEBUG_MEMORY:
            rss_after = current_rss_mb()
            direction = "" if seek_distance is None else ("fwd" if seek_distance >= 0 else "back")
            print(
                f"[video_io.DEBUG_MEMORY] backend={used_backend} frame_index={frame_index} "
                f"seek_distance={seek_distance}({direction}) large_seek={is_large_seek} "
                f"RSS: {rss_before:.1f}MB -> {rss_after:.1f}MB (delta{rss_after - rss_before:+.1f}MB)",
                flush=True,
            )
            if is_large_seek:
                release_unused_memory()

        self._last_accessed_frame_index = frame_index
        return FrameSample(frame_index=frame_index, timestamp_ms=timestamp_ms, image=image)

    def _get_frame_via_disposable_opencv(self, frame_index: int) -> np.ndarray:
        """decordの状態に触れず、使い捨てのcv2.VideoCaptureで1フレームだけ取得する。

        毎回動画を開き直すためdecordの継続アクセスより遅いが、大きなシーク1回だけの
        コストなので実用上問題にならないはず。decordの内部状態(seek位置等)を
        変化させないため、その後の連続アクセス(fps指定抽出等)の挙動にも影響しない。
        """
        cap = cv2.VideoCapture(self.video_path)
        try:
            if not cap.isOpened():
                raise IOError(f"動画を開けませんでした: {self.video_path}")
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, image = cap.read()
            if not ok:
                raise IOError(f"フレーム取得に失敗しました: index={frame_index}")
            return image
        finally:
            cap.release()

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
