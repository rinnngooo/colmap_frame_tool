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

## キーボード操作

- `←` / `→` : 選択中フレームを1つ前/次のフレームへ移動
- `Insert`  : 画像を追加(prev-current間)
- `Delete`  : 現在の画像を削除

## 現状の実装状態（このコミット時点）

- [x] ディレクトリ構成・雛形
- [x] `project_store.py`: フレーム/エッジのデータモデルとJSON永続化、追加/削除時の
      「差分再計算対象エッジ」の特定ロジック
- [x] `video_io.py`: decord優先・OpenCVフォールバックの動画読み込み抽象化
- [x] `intrinsics.py`: EXIF概算による内部パラメータ推定のスタブ
- [x] `metrics.py`: ORB系に加え、`extract_keyframes_v2.py`のSuperPoint+LightGlue+F行列
      (USAC_MAGSAC)+凸包面積ロジックを移植済み。torch/lightglue未インストール環境では
      自動的にORBにフォールバックする
- [x] `ui/`: PyQt6 + pyqtgraph によるウィンドウ・グラフ・画像比較の骨格
      （グラフのオーバービュー+詳細ビュー、正規化+倍率表示付きレジェンド、
      ドラッグ範囲選択の色付けまで動作する状態）
- [x] 動画からのfps指定抽出、手動追加/削除、プロジェクトの保存・読み込み
- [x] キーボード操作(←/→で選択移動、Insert/Deleteで追加/削除)

## SuperPoint+LightGlueについて

`metrics.py`の`superpoint_lightglue_match()`は`extract_keyframes_v2.py`の
`OverlapEstimator`を移植したもの。元スクリプトと同様に`resize_width`(既定640px)に
縮小した画像でマッチングを行うが、他の指標(IoU/凸包面積/カバレッジ等)は元解像度の
座標系を前提にしているため、マッチング後の座標を元解像度へスケールし直して返す。

F行列推定は元スクリプトの知見を反映し`USAC_MAGSAC`を優先(失敗時`FM_RANSAC`に
フォールバック)。ORBマッチ結果に対しても同じ関数を使うため、この安定性向上は
両方の経路に効いている。

torch/lightglueが未インストールの場合、UIは自動的にORBでの計算にフォールバックし、
初回のみダイアログで通知する(`MainWindow._compute_edge_metrics_with_fallback`)。

## プロジェクトの保存・読み込み

- 保存: `project.json`としてimages_dirと同じ階層に保存
- 読み込み: `project.json`を選択するとframes/edgesを復元し、グラフ・画像比較表示に反映
  - 元動画(`video_path`)が見つかる場合は再オープンし、画像追加が可能な状態になる
  - 元動画が見つからない場合は警告を表示した上で、既存画像の閲覧・削除のみ可能な状態で開く
  - 読み込んだプロジェクトに対する「fps指定で抽出」は無効化される
    (再実行するとframes/edgesが丸ごと上書きされ、手動追加/削除の結果が失われるため)

## 次のステップ（提案）

1. 実際の動画・実機のディスプレイ環境でUI全体を通しで操作確認
   （矢印キー/Insert/Deleteの実際のキーイベント、グラフのドラッグ範囲選択など）
2. torch/lightglueを実際にインストールしたうえでSuperPoint+LightGlue経路の実測・速度確認
3. 必要であれば、視差角計算(`intrinsics.py`)を動画コンテナのメタデータからも
   推定できるように拡張(現状はEXIFのみ対応)

## 依存パッケージ

`requirements.txt` 参照。PyQt6を採用しているが、PySide6でも
`from PyQt6 import ...` を `from PySide6 import ...` に置換するだけで
ほぼそのまま移行可能な書き方にしてある。
