"""
フレームペア(prev, current)間のSfM向け指標を計算するモジュール。

指標一覧(project_store.EdgeMetrics に対応):
    - orb_homography_iou     : ORB特徴点+Homography推定によるIoU(粗いスクリーニング用)
    - sp_lg_fmat_hull_area   : SuperPoint+LightGlue+F行列+凸包面積(精密判定用) ※要移植
    - inlier_count / ratio   : マッチの信頼性
    - translation_vector     : E行列分解による並進方向(スケール不定)
    - parallax_angle_deg     : 視差角。EXIF概算Kが無ければNone(相対比較専用)
    - feature_coverage       : 特徴点の画像内グリッド分散
    - blur_score / exposure_ratio : 単一フレームの品質指標

設計方針:
    特徴点マッチング処理は FeatureMatchResult という共通データ構造に集約し、
    その後段の指標計算(IoU/inlier/並進/視差角/カバレッジ)は
    マッチング手法(ORB or SuperPoint+LightGlue)によらず共通コードで計算する。
    これにより、SuperPoint+LightGlue版を後から実装しても
    IoU計算等のロジックを二重に書かずに済む。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# 共通データ構造
# ---------------------------------------------------------------------------

@dataclass
class FeatureMatchResult:
    pts1: np.ndarray  # (N, 2) prev画像上の座標
    pts2: np.ndarray  # (N, 2) current画像上の座標
    inlier_mask: Optional[np.ndarray] = None  # F行列RANSAC後のinlierマスク(N,) bool


# ---------------------------------------------------------------------------
# マッチング(手法ごとに差し替え可能)
# ---------------------------------------------------------------------------

def orb_match(img1: np.ndarray, img2: np.ndarray, max_features: int = 2000) -> FeatureMatchResult:
    """ORB特徴点によるマッチング(粗いスクリーニング用、軽量)。"""
    orb = cv2.ORB_create(nfeatures=max_features)
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)

    if des1 is None or des2 is None or len(kp1) < 8 or len(kp2) < 8:
        return FeatureMatchResult(pts1=np.zeros((0, 2)), pts2=np.zeros((0, 2)))

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    raw_matches = bf.knnMatch(des1, des2, k=2)

    good = []
    for pair in raw_matches:
        if len(pair) != 2:
            continue
        m, n = pair
        if m.distance < 0.75 * n.distance:  # Loweのratio test
            good.append(m)

    if len(good) < 8:
        return FeatureMatchResult(pts1=np.zeros((0, 2)), pts2=np.zeros((0, 2)))

    pts1 = np.float32([kp1[m.queryIdx].pt for m in good])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in good])
    return FeatureMatchResult(pts1=pts1, pts2=pts2)


def superpoint_lightglue_match(img1: np.ndarray, img2: np.ndarray) -> FeatureMatchResult:
    """SuperPoint+LightGlueによる高精度マッチング。

    TODO: extract_keyframes_v2.py の SuperPoint+LightGlue+F行列推定ロジックを移植する。
    移植時は最終的に FeatureMatchResult(pts1, pts2, inlier_mask) を返す形にすること。
    """
    raise NotImplementedError(
        "SuperPoint+LightGlueのマッチングは未実装です。"
        "extract_keyframes_v2.py のロジックをここに移植してください。"
    )


# ---------------------------------------------------------------------------
# F行列推定・inlier
# ---------------------------------------------------------------------------

def estimate_fmatrix_inliers(match: FeatureMatchResult) -> FeatureMatchResult:
    """F行列をRANSACで推定し、inlier_maskを埋めて返す。"""
    if len(match.pts1) < 8:
        match.inlier_mask = np.zeros((len(match.pts1),), dtype=bool)
        return match

    F, mask = cv2.findFundamentalMat(
        match.pts1, match.pts2, method=cv2.FM_RANSAC, ransacReprojThreshold=1.0, confidence=0.99
    )
    if mask is None:
        match.inlier_mask = np.zeros((len(match.pts1),), dtype=bool)
    else:
        match.inlier_mask = mask.ravel().astype(bool)
    return match


def inlier_count_and_ratio(match: FeatureMatchResult) -> tuple[int, float]:
    if match.inlier_mask is None:
        match = estimate_fmatrix_inliers(match)
    total = len(match.pts1)
    if total == 0:
        return 0, 0.0
    count = int(match.inlier_mask.sum())
    return count, count / total


# ---------------------------------------------------------------------------
# IoU (Homography)
# ---------------------------------------------------------------------------

def homography_iou(match: FeatureMatchResult, img_shape: tuple[int, int]) -> Optional[float]:
    """Homography推定→prev画像をcurrent側へ射影した領域とcurrent画像全体のIoU。"""
    if len(match.pts1) < 4:
        return None

    H, mask = cv2.findHomography(match.pts1, match.pts2, cv2.RANSAC, 5.0)
    if H is None:
        return None

    h, w = img_shape[:2]
    corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
    warped = cv2.perspectiveTransform(corners, H)

    mask_img = np.zeros((h, w), dtype=np.uint8)
    mask_full = np.ones((h, w), dtype=np.uint8)
    cv2.fillConvexPoly(mask_img, warped.astype(np.int32), 1)

    intersection = np.logical_and(mask_img, mask_full).sum()
    union = np.logical_or(mask_img, mask_full).sum()
    if union == 0:
        return None
    return float(intersection / union)


# ---------------------------------------------------------------------------
# 凸包面積(F行列inlier点ベース、画像全体に対する正規化面積)
# ---------------------------------------------------------------------------

def inlier_convex_hull_area_ratio(match: FeatureMatchResult, img_shape: tuple[int, int]) -> Optional[float]:
    if match.inlier_mask is None:
        match = estimate_fmatrix_inliers(match)
    pts = match.pts2[match.inlier_mask]
    if len(pts) < 3:
        return None

    hull = cv2.convexHull(pts.astype(np.float32))
    hull_area = cv2.contourArea(hull)
    h, w = img_shape[:2]
    return float(hull_area / (h * w))


# ---------------------------------------------------------------------------
# 並進ベクトル・視差角(E行列分解、Kが必要)
# ---------------------------------------------------------------------------

def translation_and_parallax(
    match: FeatureMatchResult,
    K: Optional[np.ndarray],
) -> tuple[Optional[list], Optional[float]]:
    """E行列分解による並進方向(スケール不定)と視差角(度)を返す。

    Kが与えられない場合は (None, None) を返す
    (呼び出し側でNoneのまま「計算不可」として扱う想定)。
    """
    if K is None:
        return None, None
    if match.inlier_mask is None:
        match = estimate_fmatrix_inliers(match)

    pts1 = match.pts1[match.inlier_mask]
    pts2 = match.pts2[match.inlier_mask]
    if len(pts1) < 8:
        return None, None

    E, mask = cv2.findEssentialMat(pts1, pts2, K, method=cv2.RANSAC, prob=0.999, threshold=1.0)
    if E is None:
        return None, None

    _, R, t, _ = cv2.recoverPose(E, pts1, pts2, K, mask=mask)
    t = t.ravel()
    translation_vector = t.tolist()

    # 視差角: 並進ベクトルとカメラ光軸(z軸)のなす角の余角的な指標として、
    # 「回転を除いた純粋な並進成分の大きさ」を角度で表現する簡易指標。
    # ここでは並進ベクトルとz軸方向の角度差を視差の目安として使う。
    z_axis = np.array([0.0, 0.0, 1.0])
    cos_angle = np.clip(np.dot(t, z_axis) / (np.linalg.norm(t) + 1e-9), -1.0, 1.0)
    angle_from_forward = np.degrees(np.arccos(cos_angle))
    # 前方方向からの偏角が大きい(=横方向の並進が大きい)ほど視差が付きやすい
    parallax_angle_deg = float(angle_from_forward)

    return translation_vector, parallax_angle_deg


# ---------------------------------------------------------------------------
# 特徴点カバレッジ(グリッド分散)
# ---------------------------------------------------------------------------

def feature_coverage(match: FeatureMatchResult, img_shape: tuple[int, int], grid: int = 4) -> Optional[float]:
    if match.inlier_mask is None:
        match = estimate_fmatrix_inliers(match)
    pts = match.pts2[match.inlier_mask]
    if len(pts) == 0:
        return 0.0

    h, w = img_shape[:2]
    cell_w, cell_h = w / grid, h / grid
    occupied = set()
    for x, y in pts:
        cx, cy = int(x // cell_w), int(y // cell_h)
        occupied.add((min(cx, grid - 1), min(cy, grid - 1)))
    return len(occupied) / (grid * grid)


# ---------------------------------------------------------------------------
# 単一フレーム品質指標
# ---------------------------------------------------------------------------

def blur_score(img: np.ndarray) -> float:
    """Laplacianの分散。値が小さいほどブレ/ボケが大きい。"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def exposure_ratio(img: np.ndarray, low_thresh: int = 5, high_thresh: int = 250) -> float:
    """画素値が極端に低い/高いピクセルの割合(露出過多・不足の目安)。"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    clipped = np.logical_or(gray <= low_thresh, gray >= high_thresh).sum()
    return float(clipped / gray.size)


# ---------------------------------------------------------------------------
# エッジ指標をまとめて計算するエントリポイント
# ---------------------------------------------------------------------------

def compute_edge_metrics(
    prev_img: np.ndarray,
    curr_img: np.ndarray,
    K: Optional[np.ndarray] = None,
    use_superpoint_lightglue: bool = False,
) -> dict:
    """project_store.EdgeMetrics に詰める値を辞書で返す。"""
    orb_result = orb_match(prev_img, curr_img)
    orb_iou = homography_iou(orb_result, curr_img.shape)

    if use_superpoint_lightglue:
        precise_match = superpoint_lightglue_match(prev_img, curr_img)
    else:
        precise_match = orb_result  # SuperPoint+LightGlue未実装の間はORB結果で代用

    precise_match = estimate_fmatrix_inliers(precise_match)
    hull_area = inlier_convex_hull_area_ratio(precise_match, curr_img.shape)
    inlier_count, inlier_ratio = inlier_count_and_ratio(precise_match)
    translation_vector, parallax_angle_deg = translation_and_parallax(precise_match, K)
    coverage = feature_coverage(precise_match, curr_img.shape)

    return {
        "orb_homography_iou": orb_iou,
        "sp_lg_fmat_hull_area": hull_area,
        "inlier_count": inlier_count,
        "inlier_ratio": inlier_ratio,
        "translation_vector": translation_vector,
        "parallax_angle_deg": parallax_angle_deg,
        "feature_coverage": coverage,
    }
