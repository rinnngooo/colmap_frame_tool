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
t{分:02d}.{秒(四捨五入・小数1桁):04.1f}s_t{総秒数:09.3f}s_f{frame_index:07d}.png
例: t00.19.0s_t00018.983s_f0000569.png
    (0分18.983秒地点、元動画の569フレーム目)
```

- 前半は「分.秒(小数1桁、パッと見て分かりやすい表示用)」、後半は「総秒数(小数3桁、
  ソート用の固定幅表現)」の2つの時刻表現を両方含む。
- 後半の総秒数部分が固定幅ゼロ埋めのため、文字列ソート=時系列ソートが保たれる。
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

## Insertキー(画像追加)でのメモリ急増について

fps指定の一括抽出(多数のエッジを計算)では平均して1エッジあたり数十MB程度の増加だったのに対し、
Insertキー(1〜2エッジのみ再計算)で一気に数GB増えるのは「特定の1回の計算だけが突出して重い」
パターンだと考えられる。有力な原因は次の2つ。

1. **LightGlueの計算コストはキーポイント数に対して非線形に増える**
   実際に検出されるキーポイント数は画像の内容(テクスチャの多さ)次第でばらつくため、
   たまたまテクスチャが多く両画像とも上限(`max_num_keypoints`)近くまで検出されたペアだけ、
   attention計算のコストが突出して大きくなることがある。

2. **PyTorchのキャッシングアロケータは一度確保した大きなバッファを解放せずに保持する**
   そのペアで一時的に大きなメモリを確保すると、その後は使われなくなってもプロセスの
   RSSとしては高止まりし続けやすい。

### 対策として実施したこと

- `metrics.py`: `max_num_keypoints`のデフォルトを1024→512に削減(コストの上限を下げる)
- `metrics.py`: 各`match_pair()`呼び出し後に中間テンソルの参照を明示的に切り、
  `gc.collect()` / `torch.cuda.empty_cache()` / `malloc_trim(0)`(Linux)でOSへの
  メモリ返却を促す`_release_unused_memory()`を追加
- `ui/edge_compute_worker.py`: 各エッジ処理後にprev_img/curr_imgへの参照を明示的に破棄

### 「Insertの時だけ」という特徴について(重要)

**エッジ指標の一括計算は、保存済みのPNG画像を読むだけで動画ファイルには一切触れない**
(`EdgeComputeWorker`は`cv2.imread()`で既に保存済みの画像を読むだけ)。一方、
**Insertによる画像追加だけが、抽出完了後にもう一度「動画そのもの」をシークして
1フレーム取り出す**(`main_window.py`の`add_frame_between()`内の
`self.reader.get_frame(mid_frame_index)`)。

fps指定抽出時は`extract_at_fps()`がフレーム番号の昇順にしか動画へアクセスしない
(常に前方への連続アクセス)のに対し、Insertでは「選択したフレームの位置」次第で、
動画中のかなり離れた位置、特に**後方への巻き戻し**が発生し得る。decordやFFmpeg系の
デコーダは、離れた位置・特に後方への大きなランダムシークで内部的に大量のフレームを
デコード/保持してしまう既知の弱点があるため、これが最有力候補と考えている。

この仮説を検証するため、`video_io.py`の`get_frame()`にも`metrics.py`と同じ
`COLMAP_TOOL_DEBUG_MEMORY=1`環境変数で有効になる診断ログを追加した。
「直前にアクセスしたフレーム番号からのシーク距離(前方/後方)」と「RSSの変化」を
呼び出しのたびに記録する。

```bash
COLMAP_TOOL_DEBUG_MEMORY=1 python3 main.py
```

抽出中は`seek_distance`が一定の小さな正の値(前方)で並ぶはずだが、Insert実行時の
ログで`seek_distance`が大きな負の値(後方への大ジャンプ)になっており、かつその
`get_frame()`呼び出し1回でRSSが大きく増えていれば、動画デコーダのランダムシークが
原因であることが確定する。逆に`get_frame()`のRSS増分が小さく、その後の
`metrics.py`側の`match_pair呼び出し`のログで増えているなら、SuperPoint+LightGlue側
(特定ペアのキーポイント数)が原因ということになる。

ログを取得したうえで、`seek_distance`とRSS変化の値を教えていただければ、
原因をどちらか一方に確定できる。もし動画デコーダ側が原因と確定した場合は、
「Insert専用に、都度使い捨ての軽量なOpenCVリーダーで1フレームだけ取得する」
「decordの代わりにOpenCVを強制する」などの対策を追加する想定。



`COLMAP_TOOL_DEBUG_MEMORY=1`環境変数を立てて実行すると、SuperPoint+LightGlue呼び出しの
たびに検出/マッチしたキーポイント数とRSSの変化が標準出力に出るようになる。

```bash
COLMAP_TOOL_DEBUG_MEMORY=1 python3 main.py
```

また、実際にメモリが跳ね上がったフレームペアの画像を指定して単体で計測できる
診断スクリプトも用意した。

```bash
python3 tools/diagnose_memory.py path/to/frameA.png path/to/frameB.png
```

`import torch`自体の増分、モデル構築の増分、推論1回目とそれ以降の増分を段階的に
表示するので、「torchの読み込みコストが支配的なのか」「特定のペアのキーポイント数が
多いことが原因なのか」を切り分けられる。もし依然として数GB規模の増加が特定のペアで
再現する場合は、そのペアの画像とキーポイント数(DEBUG_MEMORY出力)を教えていただければ
さらに原因を絞り込める。

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

## Windows実機ログで判明した追加原因と対策(重要)

実機(Windows)で`COLMAP_TOOL_DEBUG_MEMORY=1`のログを取得したところ、次の2点が判明した。

1. **`resource`モジュールはWindowsに存在せず、RSSが常に`-1.0MB`(測定不能)になっていた。**
   これまでの実装は`resource.getrusage()`を直接呼んでおり、例外を握りつぶして-1.0を
   返していたため、Windows環境では診断ログのメモリ量が一切機能していなかった。
   → `mem_diag.py`を新設し、Windowsでは`GetProcessMemoryInfo`(psapi.dll)経由で
   ワーキングセットサイズを取得するようにした。`metrics.py`/`video_io.py`/
   `tools/diagnose_memory.py`は全てこの共通モジュールを使うよう統一した。

2. **Insert時のログで`seek_distance=-1050`という大きな後方シークが実際に発生していた。**
   fps指定抽出でframe_index=1499まで進んだ後、Insertで選択したフレームが449だったため、
   1050フレーム分の後方シークが発生していた。これは「Insertだけが動画デコーダに
   再アクセスする」という仮説と一致する実測結果であり、decordの大きな(特に後方への)
   ランダムシークがメモリ急増の原因である可能性が高いと判断した。

これを受けて`video_io.py`に対策を追加した。

- `get_frame()`で、直前のアクセス位置から`LARGE_SEEK_THRESHOLD_FRAMES`(60フレーム)を
  超えるシークが発生した場合、decordではなく**使い捨てのcv2.VideoCapture**でその
  1フレームだけを取得するように変更した。fps指定抽出のような連続的な前方アクセスは
  従来通りdecordの高速パスを使い続けるため、抽出速度への影響はない。
- ログの表記もWindowsコンソールでの文字化け(cp932環境での`Δ`や全角文字)を避けるため、
  `delta`/`fwd`/`back`のASCII表記に統一した。

動作確認(Linux環境、フェイクのdecordオブジェクトを使った単体テスト)は以下の通り。

- 直前位置から60フレーム以内の小さいシーク → decordの経路がそのまま使われる
- 60フレームを超える大きいシーク(後方1400フレーム等) → decordを迂回し、
  cv2.VideoCaptureで正常に取得できる(フェイクdecordの`__getitem__`は呼ばれない)

引き続き実機でInsertを試していただき、`[video_io.DEBUG_MEMORY]`のログで
`backend=opencv(large-seek-fallback)`と表示され、かつRSSの急増が収まっているかを
確認していただきたい(今回の修正でRSSも実測値が出るようになっているはず)。
収まっていればdecordの大きなランダムシークが原因で確定する。もしそれでもメモリが
急増する場合は、原因は別の場所(SuperPoint+LightGlue側等)にあることになるので、
その際のログを共有してほしい。

### 追記: RSSがまだ-1.0MBのままだった件(ctypes実装の不具合)

Windows実機で試したところ、`backend=opencv(large-seek-fallback)`への切り替え自体は
成功しメモリ急増も収まったが、RSSの値は依然として`-1.0MB`のままだった。ctypes経由の
`GetProcessMemoryInfo`呼び出しが何らかの理由で失敗していたと考えられる
(型指定を省略していたため、失敗しても例外にならず`-1.0`を返し続けていた)。

`mem_diag.py`を、実績のある`psutil`ライブラリを最優先で使うように変更した
(`pip install psutil --break-system-packages`が必要。`requirements.txt`にも追加済み)。
psutilが無い場合のみ、型指定を厳密化し失敗理由も`DEBUG_MEMORY`時に出力する
ctypesフォールバックを使う。psutilを導入した状態で再度お試しいただきたい。

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

## 画像を抽出(fps指定 / 指標ベース)

「画像を抽出」ボタン(旧「fps指定で抽出」)を押すと、`ui/extraction_dialog.py`の
`ExtractionOptionsDialog`が開き、2つの抽出方式を選べる。

- **fps指定で抽出**: 従来通り、指定fpsで等間隔にフレームを抽出する
- **指標で抽出**: 直近保存フレームを基準に、`orb_iou_and_coverage()`(ORB
  Homography IoU / Feature Coverage)がTarget値を下回る直前のフレームを保存していく。
  アルゴリズムの詳細は`MainWindow._extract_by_metrics()`のdocstring参照。
  - Min/Max Frame Gapの範囲内に収まるよう、間隔をクランプする
    (常に条件を満たし続ける場合はMax Frame Gapで強制保存、常に満たさない場合は
    Min Frame Gapで強制保存)
  - 動作確認: 合成動画でMin/Max Frame Gap境界ケース(常時満たす/常時満たさない)の
    両方をテスト済み

## グラフのマウスオーバー表示

詳細ビュー上にマウスを乗せると、その時刻での表示中の各指標の生の値(正規化前)を
`QToolTip`で一覧表示する(`GraphWidget._on_detail_hover`)。クリックでのフレーム選択
とは独立した動作(クリック=選択確定、ホバー=値の確認)。

## 「指標で抽出」が常にMin Frame Gapになっていた件(根本原因判明・対応済み)

実機ログにより根本原因が判明した。

```
[extract_by_metrics] last_saved_frame=5 candidate_frame=11 gap=6 iou=0.9855(target=0.85) coverage=0.8750(target=0.95) ok=False
```

**グラフのFeature Coverageと、抽出ロジックのcoverageが別の特徴点検出器で計算されていた。**

- グラフの`feature_coverage`(`_recompute_edges_async`経由): SuperPoint+LightGlue(高密度・
  学習ベース、グリッドを均等に埋めやすい)でマッチングした結果から計算
- 旧・抽出ロジックの`coverage`: 高速化のため意図的にORB(疎ら、グリッドを均等に
  埋めにくい)でマッチングした結果から計算

同じ「Feature Coverage」という名前でも検出器が異なるためスケール感が違い、
ORBベースだと0.95まで届くことは稀だった。つまりTarget Feature Coverageの
デフォルト値(0.95)自体が、ORBベースの指標に対しては最初から厳しすぎる設定になっており、
常にMin Frame Gapで打ち切られていた。

なお、正規化(グラフ表示用に系列ごとの最大値を1にスケーリングする処理、
`graph_widget.py`の`_redraw()`内のみで行われる)が抽出ロジックに紛れ込んでいる、
という可能性は調査の結果否定された。抽出ロジックは常に生の値を比較しており、
正規化処理とは経路が完全に分離されている。

### 対応: Coverageの計算をグラフと同じ検出器に統一

`MainWindow._compute_extraction_metrics()`を新設し、`compute_edge_metrics()`
(グラフの値もこれを経由する)を直接呼ぶようにした。

- IoU: 常にORB(「Target ORB Homography IoU」の名前通り。グラフの
  `orb_homography_iou`も常にORBなので、これは元々一致していた)
- Coverage: SuperPoint+LightGlueが使えればそれを使う(グラフの`feature_coverage`と
  完全に同じ計算経路になる)。未インストール等で使えない場合はORBに自動フォールバック
  (初回のみ警告ダイアログを表示)

これにより、ダイアログのTarget Feature Coverageに、グラフで実際に見ている値と
同じ感覚の数値を入力できるようになった。

## 指標評価の高速化(適応的な幅探索)

1フレームずつ評価すると低速なため、以下の2段階の適応的な幅探索アルゴリズムに変更した
(`MainWindow._extract_by_metrics()`)。動画全体を舐めるO(フレーム数)ではなく、
1回の保存あたりおおむねO(log(max_gap))回の評価で済む。

- **Phase 1(指数拡大)**: floor(=直近保存フレーム+min_gap)から、直前2回分の移動幅の和
  (フィボナッチ的に成長する幅: 例 g, g, 2g, 3g, 5g, 8g, ...)だけ前進し続ける。
  条件を満たしている間はどんどん幅を広げて先へ進み、満たさなくなった時点で
  「満たす最後の位置」と「満たさない位置」の区間を特定する。
- **Phase 2(二分探索での絞り込み)**: Phase 1で特定した区間内を、幅を半分ずつ狭めながら
  評価し、「条件を満たす最後のフレーム」を正確に特定する。

二分探索を使っているのは、ご要望の「幅を狭めて複数フレーム戻す」を、無限ループの
リスクなく確実に収束させるための実装選択で、狙いは同じ(幅を調整しながら評価回数を
減らして境界フレームを探す)。

動作確認(400フレームの合成パン動画、min_gap=5, max_gap=150):
- 保存されたgapが`[48, 48, 48, 48, 48, 48, 49, 48, 14]`と、常にMin Frame Gapにならず
  適切な値に収束することを確認
- `get_frame()`呼び出しが84回(動画全体400フレームに対し、1フレームずつ評価する
  場合よりも大幅に削減)
- ログ上でも`floor→expand(幅5,10,20,35,60で失敗)→narrow(47,53,50,48,49で収束)`という
  想定通りのフェーズ遷移を確認

## グラフのツールチップが1秒で消える件(対応済み)

`QToolTip.showText()`で表示時間(`msecShowTime`)を明示指定していなかったため、
テキストの長さから自動算出される時間(短いテキストだと短くなる)で消えてしまっていた。
明示的に60秒を指定するよう修正した(`GraphWidget._on_detail_hover`)。マウスが
動いている間は呼び出しのたびに再表示されるため、実質的にマウスを離すかプロット外に
出るまで表示され続けるようになる。

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
