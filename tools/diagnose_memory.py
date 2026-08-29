#!/usr/bin/env python3
"""
torch / SuperPoint / LightGlue の各段階でのメモリ使用量(RSS)を計測する診断スクリプト。

GUIアプリを介さずに単体で実行することで、「動画を開いた瞬間」「1回の推論」の
どちらがどれだけメモリを消費しているのかを切り分けられる。

使い方:
    python3 tools/diagnose_memory.py [画像パスA] [画像パスB]

    画像パスを省略した場合はランダムなダミー画像(640x480)でテストする。
    実際にメモリが跳ね上がったフレームペアの画像を指定すると、
    そのペア固有の原因(キーポイント数が非常に多い、等)を確認できる。
"""

import os
import sys


def report(label: str, before: float):
    from mem_diag import current_rss_mb
    after = current_rss_mb()
    print(f"[{label:32s}] RSS: {after:9.1f} MB   (delta {after - before:+8.1f} MB)")
    return after


def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, base_dir)
    from mem_diag import current_rss_mb

    base = current_rss_mb()
    print(f"[{'baseline':32s}] RSS: {base:9.1f} MB")

    import numpy as np
    base = report("import numpy", base)

    import cv2
    base = report("import cv2", base)

    import torch
    base = report("import torch", base)
    print(f"    torch.cuda.is_available() = {torch.cuda.is_available()}")

    from lightglue import LightGlue, SuperPoint
    base = report("import lightglue", base)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    extractor = SuperPoint(max_num_keypoints=512).eval().to(device)
    base = report("SuperPoint() モデル構築", base)

    matcher = LightGlue(features="superpoint").eval().to(device)
    base = report("LightGlue() モデル構築", base)

    # --- テスト画像の準備 ---
    if len(sys.argv) >= 3:
        img1 = cv2.imread(sys.argv[1])
        img2 = cv2.imread(sys.argv[2])
        print(f"    テスト画像: {sys.argv[1]}, {sys.argv[2]} (shape={img1.shape})")
    else:
        img1 = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        img2 = np.roll(img1, 5, axis=1)
        print("    テスト画像: ランダムダミー(640x480) ※実際の画像を渡すと精度の高い診断になります")

    import metrics as metrics_mod
    metrics_mod.DEBUG_MEMORY = True  # キーポイント数とRSSをmatch_pair内から出力させる

    base = report("推論前", base)

    for i in range(3):
        result = metrics_mod.superpoint_lightglue_match(img1, img2)
        base = report(f"superpoint_lightglue_match呼び出し {i+1}回目", base)
        print(f"    マッチ数: {len(result.pts1)}")

    print()
    print("解釈のヒント:")
    print("  - 'import torch' の増分が数百MB規模なら、torch自体の読み込みコストが支配的")
    print("    (torch.cuda.is_available()がTrueの場合、CUDA初期化分も上乗せされる)")
    print("  - 'モデル構築' の増分が大きい場合、モデルロード自体が重い(通常は数十MB程度のはず)")
    print("  - 'match_pair呼び出し' で1回目だけ増分が大きく、2回目以降ほぼ増えないなら、")
    print("    PyTorchのキャッシングアロケータによる初回のバッファ確保が原因")
    print("  - 実際の画像(特にメモリが跳ね上がったときのフレームペア)を渡して")
    print("    キーポイント数が多いほど増分が大きいなら、テクスチャの多いペアでの")
    print("    LightGlueのattention計算コストが原因(max_num_keypointsを下げると軽減可能)")
    print("  - 動画のシーク(特にInsert時の後方シーク)が原因かどうかは、このスクリプトでは")
    print("    切り分けられない。COLMAP_TOOL_DEBUG_MEMORY=1 でGUIアプリを起動し、")
    print("    [video_io.DEBUG_MEMORY] のログでseek_distanceとRSSの変化を確認すること。")


if __name__ == "__main__":
    main()
