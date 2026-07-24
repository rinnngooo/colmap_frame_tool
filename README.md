# colmap_frame_tool

動画からCOLMAP用のSfM画像セットを作成するための、フレーム抽出＋マニュアル調整デスクトップアプリの雛形。

## ディレクトリ構成

```
colmap_frame_tool/
  main.py              # エントリポイント
  video_io.py          # 動画読み込み・フレームシーク(decord優先、OpenCVフォールバック)
  metrics.py           # SfM向け指標計算(ORB/SuperPoint+LightGlue/ブレ/視差角 等)
  intrinsics.py        # EXIFからの内部パラメータ概算推定
  project_store.py     # プロジェクトのメタデータ(JSON)管理・フレーム/エッジ操作
  ui/
    __init__.py
    main_window.py         # メインウィンドウ(メニュー・レイアウト)
    graph_widget.py        # オーバービュー+詳細ビューの時系列グラフ(pyqtgraph)
    image_compare_widget.py # current/prev 画像の並列表示
  requirements.txt
```

## ファイル命名規則

```
frame_f{元動画フレーム番号:07d}_t{タイムスタンプms:08d}.png
例: frame_f0000123_t00004106.png
```

- 元動画のフレーム番号とタイムスタンプ(ms)をファイル名に埋め込む。
- 抽出順の連番にしないことで、後からprevとcurrentの間に任意のフレームを挿入しても
  リネームが不要（文字列ソート = 時系列ソートが保たれる）。

## プロジェクトメタデータ(JSON)スキーマ

`project_store.py` の `ProjectStore` が読み書きする。詳細は同ファイルのdocstring参照。

```jsonc
{
  "video_path": "/path/to/video.mp4",
  "video_fps": 29.97,
  "extract_fps": 2.0,
  "images_dir": "images",
  "frames": [
    {
      "filename": "frame_f0000000_t00000000.png",
      "frame_index": 0,
      "timestamp_ms": 0,
      "blur_score": null,
      "exposure_ratio": null
    }
  ],
  "edges": [
    {
      "from_filename": "frame_f0000000_t00000000.png",
      "to_filename": "frame_f0000123_t00004106.png",
      "metrics": {
        "orb_homography_iou": null,
        "sp_lg_fmat_hull_area": null,
        "inlier_count": null,
        "inlier_ratio": null,
        "translation_vector": null,
        "parallax_angle_deg": null,
        "feature_coverage": null
      }
    }
  ]
}
```

`frames` は現在のフレーム番号順にソートされた「今、SfMに使う画像セット」を表す。
`edges` は `frames` の隣接ペアに対応する指標キャッシュで、frames配列と自動的に整合するよう
`ProjectStore` 側で管理する（フレーム追加/削除時は影響するエッジのみ再計算される想定）。

## 現状の実装状態（このコミット時点）

- [x] ディレクトリ構成・雛形
- [x] `project_store.py`: フレーム/エッジのデータモデルとJSON永続化、追加/削除時の
      「差分再計算対象エッジ」の特定ロジック
- [x] `video_io.py`: decord優先・OpenCVフォールバックの動画読み込み抽象化
- [x] `intrinsics.py`: EXIF概算による内部パラメータ推定のスタブ
- [ ] `metrics.py`: 関数シグネチャとダミー実装のみ。ORB/SuperPoint+LightGlue部分は
      既存の `extract_keyframes_v2.py` のロジックを移植して実装する必要あり
- [x] `ui/`: PyQt6 + pyqtgraph によるウィンドウ・グラフ・画像比較の骨格
      （グラフのオーバービュー+詳細ビュー、正規化+倍率表示付きレジェンド、
      ドラッグ範囲選択の色付けまで動作する状態）

## 次のステップ（提案）

1. `metrics.py` に既存の `extract_keyframes_v2.py` のSuperPoint+LightGlue+F行列ロジックを移植
2. `video_io.py` を実際の動画で動作確認（decordのインストール可否も含む）
3. UIから「動画を開く→fps指定して抽出→プロジェクトJSON生成」の一連を通しで確認
4. add/delete操作と指標の差分再計算をUIから実際に動かして確認

## 依存パッケージ

`requirements.txt` 参照。PyQt6を採用しているが、PySide6でも
`from PyQt6 import ...` を `from PySide6 import ...` に置換するだけで
ほぼそのまま移行可能な書き方にしてある。
