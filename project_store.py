"""
プロジェクトのメタデータ(JSON)を管理するモジュール。

命名規則:
    frame_f{frame_index:07d}_t{timestamp_ms:08d}.png

データモデル:
    FrameRecord - 1枚の抽出/追加画像に対応する情報
    EdgeMetrics - 隣接する2枚のFrameRecord間のSfM指標
    ProjectStore - frames/edgesを保持し、JSONへの永続化と
                   追加/削除時の「再計算が必要なエッジ」の特定を行う

設計方針:
    - frames は常に frame_index 昇順にソートされた状態を保つ。
      これが「現在SfMに使う画像セット」そのものを表す。
    - edges は frames の隣接ペア (frames[i], frames[i+1]) に対応する
      指標キャッシュ。frames が変化したら edges もそれに追従させる。
    - 追加/削除操作では、変化するのは高々2つのエッジなので、
      呼び出し側(UI)は返り値の「再計算が必要なエッジのリスト」だけ
      metrics.py で計算し直せばよい。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional


FILENAME_TEMPLATE = "frame_f{frame_index:07d}_t{timestamp_ms:08d}.png"


def make_filename(frame_index: int, timestamp_ms: int) -> str:
    return FILENAME_TEMPLATE.format(frame_index=frame_index, timestamp_ms=timestamp_ms)


@dataclass
class FrameRecord:
    filename: str
    frame_index: int
    timestamp_ms: int
    blur_score: Optional[float] = None
    exposure_ratio: Optional[float] = None

    @staticmethod
    def create(frame_index: int, timestamp_ms: int) -> "FrameRecord":
        return FrameRecord(
            filename=make_filename(frame_index, timestamp_ms),
            frame_index=frame_index,
            timestamp_ms=timestamp_ms,
        )


@dataclass
class EdgeMetrics:
    from_filename: str
    to_filename: str
    # 全てNoneの場合は「未計算」を意味する。UI側はNoneのエッジを
    # 見つけたらmetrics.pyで計算してここを埋める。
    orb_homography_iou: Optional[float] = None
    sp_lg_fmat_hull_area: Optional[float] = None
    inlier_count: Optional[int] = None
    inlier_ratio: Optional[float] = None
    translation_vector: Optional[list] = None  # [tx, ty, tz] (スケール不定)
    parallax_angle_deg: Optional[float] = None  # EXIF概算Kによる相対比較用
    feature_coverage: Optional[float] = None

    def is_computed(self) -> bool:
        return self.orb_homography_iou is not None

    def clear(self) -> None:
        for k in (
            "orb_homography_iou",
            "sp_lg_fmat_hull_area",
            "inlier_count",
            "inlier_ratio",
            "translation_vector",
            "parallax_angle_deg",
            "feature_coverage",
        ):
            setattr(self, k, None)


@dataclass
class ProjectStore:
    video_path: str
    video_fps: float
    extract_fps: float
    images_dir: str = "images"
    frames: list = field(default_factory=list)   # list[FrameRecord]
    edges: list = field(default_factory=list)    # list[EdgeMetrics]

    # ---------- 永続化 ----------

    def to_dict(self) -> dict:
        return {
            "video_path": self.video_path,
            "video_fps": self.video_fps,
            "extract_fps": self.extract_fps,
            "images_dir": self.images_dir,
            "frames": [asdict(f) for f in self.frames],
            "edges": [asdict(e) for e in self.edges],
        }

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.write_text(
            json.dumps(self.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    @staticmethod
    def load(path: str | Path) -> "ProjectStore":
        path = Path(path)
        data = json.loads(path.read_text(encoding="utf-8"))
        store = ProjectStore(
            video_path=data["video_path"],
            video_fps=data["video_fps"],
            extract_fps=data["extract_fps"],
            images_dir=data.get("images_dir", "images"),
        )
        store.frames = [FrameRecord(**f) for f in data.get("frames", [])]
        store.edges = [EdgeMetrics(**e) for e in data.get("edges", [])]
        return store

    # ---------- フレーム操作 ----------

    def _sort_frames(self) -> None:
        self.frames.sort(key=lambda f: f.frame_index)

    def _rebuild_edges_from_frames(self) -> None:
        """frames配列からedges配列の骨格(from/toのペア)を作り直す。
        既存の計算済み指標は、同じfrom/toペアが残っていれば引き継ぐ。
        """
        existing = {(e.from_filename, e.to_filename): e for e in self.edges}
        new_edges = []
        for a, b in zip(self.frames, self.frames[1:]):
            key = (a.filename, b.filename)
            if key in existing:
                new_edges.append(existing[key])
            else:
                new_edges.append(EdgeMetrics(from_filename=a.filename, to_filename=b.filename))
        self.edges = new_edges

    def initial_extract(self, frame_records: list[FrameRecord]) -> list[EdgeMetrics]:
        """fps指定抽出直後の初期化。全frames/edgesをセットし、
        計算が必要な全エッジを返す。
        """
        self.frames = sorted(frame_records, key=lambda f: f.frame_index)
        self.edges = [
            EdgeMetrics(from_filename=a.filename, to_filename=b.filename)
            for a, b in zip(self.frames, self.frames[1:])
        ]
        return list(self.edges)

    def add_frame(self, frame_index: int, timestamp_ms: int) -> tuple[FrameRecord, list[EdgeMetrics]]:
        """prevとcurrentの間に動画フレームを追加する。

        戻り値: (追加したFrameRecord, 再計算が必要なエッジのリスト(最大2つ))
        """
        new_frame = FrameRecord.create(frame_index, timestamp_ms)
        if any(f.frame_index == frame_index for f in self.frames):
            raise ValueError(f"frame_index {frame_index} は既に存在します")

        self.frames.append(new_frame)
        self._sort_frames()
        self._rebuild_edges_from_frames()

        idx = next(i for i, f in enumerate(self.frames) if f.filename == new_frame.filename)
        affected = []
        if idx - 1 >= 0:
            affected.append(self.edges[idx - 1])  # prev -> new
        if idx < len(self.edges):
            affected.append(self.edges[idx])      # new -> next(旧current)
        for e in affected:
            e.clear()
        return new_frame, affected

    def remove_frame(self, filename: str) -> list[EdgeMetrics]:
        """currentフレームを削除する。

        戻り値: 再計算が必要なエッジのリスト(最大1つ、prev-nextの直結エッジ)
        """
        idx = next((i for i, f in enumerate(self.frames) if f.filename == filename), None)
        if idx is None:
            raise ValueError(f"{filename} が見つかりません")

        del self.frames[idx]
        self._rebuild_edges_from_frames()

        affected = []
        # 削除後、idx-1 と idx (旧idx+1) が新しく直結したエッジになる
        if 0 <= idx - 1 < len(self.edges):
            affected.append(self.edges[idx - 1])
            self.edges[idx - 1].clear()
        return affected

    # ---------- 参照 ----------

    def get_edge(self, from_filename: str, to_filename: str) -> Optional[EdgeMetrics]:
        for e in self.edges:
            if e.from_filename == from_filename and e.to_filename == to_filename:
                return e
        return None

    def frame_by_filename(self, filename: str) -> Optional[FrameRecord]:
        for f in self.frames:
            if f.filename == filename:
                return f
        return None
