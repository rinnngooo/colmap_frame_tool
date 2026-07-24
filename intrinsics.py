"""
EXIFの35mm換算焦点距離からピクセル単位の焦点距離を概算するモジュール。

前提: カメラキャリブレーション未実施(EXIFの概算値のみ利用可能)。
ここで得られる焦点距離はあくまで近似値であり、絶対的な3次元復元には
使わず、「フレームペア間の視差角(parallax angle)の相対的な大小比較」用途に限定する。
低い視差角(≒純回転に近い動き)を検出してユーザーに提示するのが目的で、
COLMAPの本番SfMでは実際のカメラパラメータ推定はCOLMAP自身に任せる想定。

動画由来のフレームにはEXIFが無いことが多いため、
- 動画のメタデータ(コンテナ情報)に焦点距離が無い場合
- 静止画として書き出した際にEXIFが引き継がれない場合
はNoneを返し、呼び出し側(metrics.py)は「視差角は計算不可」として扱う。
"""

from __future__ import annotations

from typing import Optional

try:
    import piexif
    _HAS_PIEXIF = True
except ImportError:
    _HAS_PIEXIF = False


DEFAULT_SENSOR_WIDTH_MM = 36.0  # 35mm判換算の前提


def estimate_focal_length_px(
    image_path: str,
    image_width_px: int,
    sensor_width_mm: float = DEFAULT_SENSOR_WIDTH_MM,
) -> Optional[float]:
    """EXIFの35mm換算焦点距離からピクセル単位の焦点距離を概算する。

    取得できない場合はNoneを返す。
    """
    if not _HAS_PIEXIF:
        return None

    try:
        exif_dict = piexif.load(image_path)
    except Exception:
        return None

    focal_35mm = exif_dict.get("Exif", {}).get(piexif.ExifIFD.FocalLengthIn35mmFilm)
    if focal_35mm is None:
        return None

    # focal_px = focal_mm(35mm換算) / sensor_width_mm(35mm判) * image_width_px
    focal_px = float(focal_35mm) / sensor_width_mm * image_width_px
    return focal_px


def build_intrinsics_matrix(focal_px: float, image_width_px: int, image_height_px: int):
    """概算焦点距離から簡易的な内部パラメータ行列Kを作る(主点は画像中心と仮定)。"""
    import numpy as np

    cx = image_width_px / 2.0
    cy = image_height_px / 2.0
    K = np.array(
        [
            [focal_px, 0, cx],
            [0, focal_px, cy],
            [0, 0, 1],
        ],
        dtype=np.float64,
    )
    return K
