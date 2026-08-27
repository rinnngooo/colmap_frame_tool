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
t{総秒数:09.3f}s_{分:02d}.{秒:02d}.{ミリ秒:03d}s_f{frame_index:07d}.png
例: t00496.005s_08.16.005s_f0014867.png
    (496.005秒 = 8分16.005秒地点、元動画の14867フレーム目)
```

- 先頭が総秒数(固定幅ゼロ埋め)のため、文字列ソート=時系列ソートが保たれる。
- 末尾のframe_indexはソートには影響しないが、元動画のどのフレームかをファイル名だけから復元できる。
- 抽出時もマニュアル追加時も、元動画の時刻・フレーム番号さえ分かれば命名できるためリネーム不要。

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
- [x] 起動時に最大化状態で表示、画像比較:グラフ = 75:25の縦分割
- [x] グラフのY軸マウス操作(パン/ズーム)を無効化(X軸のみ操作可能)
- [x] 終了時にプロジェクトを自動保存(project_dir設定済みの場合)
- [x] 画像比較パネルにタイムスタンプ+ファイル名を表示
- [x] エッジ指標計算(SuperPoint+LightGlue等)をQThreadでバックグラウンド実行し、
      GUIがフリーズしないよう修正。動画を開いた直後にモデルを事前ロード(ウォームアップ)
- [x] フレーム抽出時にジェネレータを`list()`で一括展開しないよう修正
      (大きなメモリスパイクの原因だったため。詳細は下記「メモリ使用量について」)

## フリーズ対策(バックグラウンド計算化)

以前のバージョンでは、SuperPoint+LightGlueによるエッジ指標計算がGUIスレッドで
同期実行されており、以下の2つの理由で操作中に数十秒フリーズすることがあった。

1. 計算中はQtのイベントループが止まるため、その間は再描画もキー入力も効かない
2. SuperPoint+LightGlueモデルは初回呼び出し時に遅延ロードされ、このロード自体が
   数十秒かかることがある(モデルの重みの読み込み・デバイスへの転送など)

対策として `ui/edge_compute_worker.py` に `EdgeComputeWorker` (エッジ指標計算) と
`WarmupWorker` (モデル事前ロード) を実装し、`QThread` 上で実行するようにした。
`MainWindow`側は計算中 `_busy=True` として関連ボタンを無効化しつつ、画像の選択切り替え
(add/delete後の画像比較表示)は指標計算の完了を待たずに即座に反映する。
動画を開いた直後にも `_start_warmup()` でモデルを事前ロードしておくことで、
実際のAdd/Delete操作時に遅延ロードのコストを払わずに済むようにしている。

## メモリ使用量について

フレーム抽出処理(`extract_frames`)で、以前は以下のように動画から抽出した
全フレームを`list()`で一括してメモリへ展開してから保存処理を行っていた。

```python
samples = list(self.reader.extract_at_fps(fps))  # 修正前: 全フレームを一括展開
```

`VideoReader.extract_at_fps()`は本来1枚ずつ返すジェネレータだが、`list()`で包むと
抽出対象フレーム数 × 1枚あたりのサイズ分のメモリを一度に消費してしまう
(例: 1920x1080のBGR画像1枚 ≈ 6.2MB、2fpsで15分の動画なら1800枚 ≈ 11GB)。
関数を抜けて`samples`が破棄されるとメモリは解放されるため、
「実行直後にメモリが跳ね上がり、しばらくすると元に戻る」という挙動になっていた。

修正後はジェネレータのまま1枚ずつ処理し、保存が終わった画像はすぐに参照を手放すように
した。進捗バー表示用の総枚数は、実際にフレームを生成せずに計算する
`VideoReader.estimate_extract_count()`で取得する。
テスト(640x480, 300フレームの動画)では、ピークメモリ増分が **265MB → 3.3MB**
に改善することを確認済み(実際の高解像度・長尺動画ではこの差はさらに大きくなる)。

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
