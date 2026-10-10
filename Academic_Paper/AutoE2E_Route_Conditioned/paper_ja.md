---
lang: 'ja'
title: "AutoE2E：地図・事後経路条件付き時系列 BEV プランナのオープンループ評価"
link:
  - rel: 'stylesheet'
    href: 'theme/paper.css'
---
<header>

# AutoE2E：地図・事後経路条件付き時系列 BEV プランナのオープンループ評価 #

## AutoE2E: Open-Loop Evaluation of a Map- and Oracle-Route-Conditioned Temporal BEV Planner{.title-en lang="en"} ##

## 著者{.author}

Ryota Yamada

<p class="affiliation">Amazon Web Services, Inc.<br>Autoware Foundation Robotaxi Working Group（AutoE2E），github.com/autowarefoundation/auto_e2e，Apache-2.0</p>

## 概要:{.abstract}

実行時に HD 地図と経路を消費する方策をカメラのみのベンチマークで測ると，方策が利用するよう設計されたモダリティが欠落する。本稿は，UniAD の planning 指向の思想に倣いつつ範囲を絞った時系列 BEV プランナ AutoE2E を，実行時のモダリティを揃えたうえで，事後的な oracle 経路を与える条件で評価する。経路コリドーと目的地は，記録されたシーン全体の将来軌跡から遡及的に構成される。凍結した ResNet-50 BEVFormer V2 T8 カメラ BEV に，個別符号化した 14 チャネル地図と 2 チャネル経路を，チャネル別ゲートと変形可能クロスアテンションで融合し，GRU が加速度・曲率を 64 ステップ予測する。外部 KITScenes プロトコルで実データの自車運動履歴は 4.0 s であり，6.4 s へ左ゼロ埋めされ，目標は 5.0 s まで有効である。KITScenes Val v3.5（117 シーン，11,035 サンプル）の決定的制御再生では，KITScenes Epoch 5 の報告平均が 2・3・5 s で最小（5 s ADE/FDE：1.9405/5.5645 m），Epoch 7 は 1 s で最小であった。内部検証では Epoch 7 が上限付き投影進捗プロキシを増加させる一方，6.4 s 誤差を増加させ快適率を低下させた。厳格な経路コリドー成功率は 0.061〜0.076，走行可能領域成功率は 0.414〜0.444 である。カメラのみの Test は別のシーン集団であり，地図・経路切除ではない。統計的有意差やナビゲーション入力の因果的利得は主張せず，同一シーンの Camera / Camera+Map / Camera+Map+Route 評価を今後必要とする。

## キーワード{.keyword}

エンドツーエンド自動運転，時系列 BEV，HD 地図条件付け，経路条件付け，オープンループ評価，KITScenes，nuPlan

## Abstract{.abstract lang="en"}

When a policy consumes an HD map and a route at runtime, a camera-only benchmark omits modalities the policy was designed to use. We evaluate AutoE2E, a UniAD-inspired but narrower temporal BEV planner, under a runtime-modality-matched but oracle-route condition: the route corridor and destination are reconstructed retrospectively from the logged whole-scene trajectory. A frozen ResNet-50 BEVFormer V2 T8 camera BEV is fused with separately encoded 14-channel map and 2-channel route rasters through a channel-wise gate and deformable cross-attention; a GRU predicts 64 acceleration/curvature steps. The external KITScenes protocol supplies 4.0 s of real egomotion history, left-zero-padded to 6.4 s, and valid targets through 5.0 s. On KITScenes Val v3.5 (117 scenes, 11,035 samples), deterministic replay of published controls gives KITScenes Epoch 5 the lowest reported means at 2, 3 and 5 s (5 s ADE/FDE: 1.9405/5.5645 m), while Epoch 7 is lowest at 1 s. Internal validation shows that Epoch 7 increases a censored projected-progress proxy while increasing 6.4 s error and lowering comfort; strict route-corridor success is 0.061–0.076 and drivable-area success 0.414–0.444. The camera-only KITScenes Test track is a different scene population and is not a map or route ablation. No statistical significance or causal navigation-input gain is claimed. A paired same-scene Camera / Camera+Map / Camera+Map+Route evaluation remains required.

</header>

## はじめに

エンドツーエンド（E2E）自動運転の研究は，nuScenes <a class="cite" href="#ref-nuscenes"></a> 上のカメラのみオープンループ評価を中心に発展してきた。UniAD <a class="cite" href="#ref-uniad"></a> と VAD <a class="cite" href="#ref-vad"></a> はこの枠組みで planning 指向の統合スタックを示し，BEVFormer <a class="cite" href="#ref-bevformer"></a> 系の時系列 BEV 表現がその共通基盤となった。しかし，実車に配備される方策の入力契約はカメラだけではないことが多い。本稿が扱う AutoE2E の配備契約は，カメラ画像に加えて HD 地図ラスタと選択経路ラスタを毎フレーム受け取ることを前提とする。この方策をカメラのみのベンチマークで評価すると，方策が消費するよう設計された情報が欠けた条件での数値を「本番性能」として読むことになる。逆に，地図と経路を与えた条件だけを報告すれば，地図が失われた状況での挙動が見えない。両者は別の問いに答える評価であり，混同すべきではない。

本稿の主張は控えめである。地図と経路が計画精度を因果的に改善したことを示す実験は，まだ行っていない。示すのは，（i）配備契約と一致する入力（Camera + HD Map + Route）を主ベンチマークとし，カメラのみの条件を別トラックとして保持する評価プロトコルと，（ii）その上で計測した 4 つの学習チェックポイントの結果，（iii）ソースコードとチェックポイント登録メタデータから検証した実装事実，（iv）内部検証で観測した短期精度・長期精度・経路進捗・走行可能領域・快適性のトレードオフ，そして（v）因果的主張に必要な同一シーン対照実験の仕様である。

<figure class="fig col-span-2" id="fig-arch">
<img src="figures/fig01_architecture.svg" alt="AutoE2E architecture">
<figcaption>評価対象の AutoE2E 構成（bevformer_v2_t8_split_navigation_v5）。8 時刻のそれぞれで 6 視点の 512×512 カメラを処理する（現在 1 時刻と履歴 7 時刻）。1024×1024 前方視点は現在時刻だけに加える。凍結した ResNet-50 + FPN と 6 層の単一時刻 BEVFormer V2 エンコーダを各時刻へ独立に適用し，その後 8 個の BEV を T8 畳み込み融合する。14 チャネル地図と 2 チャネル経路を別々に符号化するナビゲーション経路，ゲート付き加算，変形可能ナビゲーション融合，6.4 s の自車運動履歴，GRU 制御プランナ，ユニサイクル積分を示す。数値はすべて実装をインスタンス化して計測したパラメータ数（合計 79,906,522，学習対象 1,448,582）である。点線枠はチェックポイントに存在するが無効化された分岐を表す。</figcaption>
</figure>

本稿は次の 5 つの研究課題（RQ）に沿って構成する。RQ1：カメラ・HD 地図・経路が揃う実運用整合の KITScenes Val プロトコルで，経路条件付き方策はどの精度で自車軌跡を予測するか。RQ2：同じ KITScenes Val シーン上で，KITScenes 微調整チェックポイントは nuPlan 学習チェックポイントとどう比較されるか。RQ3：別集団であるカメラのみの KITScenes Test トラックで性能はどう変わるか。RQ4：学習エポック間で，短期精度・長期精度・経路進捗・走行可能領域遵守・快適性にどのトレードオフが現れるか。RQ5：現在の証拠から HD 地図と経路の効果について何が言え，何が言えないか。

貢献は以下の 4 点に限定する。第一に，凍結した BEVFormer V2 T8 カメラ BEV に，分離符号化した HD 地図・経路ラスタをゲート付き残差で結合する planning 指向モデルの，ソースコード検証に基づく記述と正確なパラメータ予算（<a class="figref" href="#fig-arch"></a>，<a class="tabref" href="#tab-params"></a>）。第二に，配備契約と一致する主ベンチマークとカメラのみの副トラックを分離し，モデルを再実行しない決定的オーバーレイ再生とチェックポイント内部検証を明確に区別する評価プロトコル（<a class="figref" href="#fig-protocol"></a>）。第三に，4 チェックポイントの 8 通りの外部評価と内部検証の結果（<a class="tabref" href="#tab-val-ade"></a>〜<a class="tabref" href="#tab-integrity"></a>）。第四に，地図・経路効果の因果的主張に必要な同一シーン対照実験の仕様と，主張ごとの証拠を列挙した監査表（付録）。

## 関連研究

### 統合型エンドツーエンド運転

UniAD <a class="cite" href="#ref-uniad"></a> は，検出・追跡・地図・動作予測・占有予測・計画をクエリ相互作用で結び，すべてのタスクを最終的な計画に寄与させる planning 指向設計を提示した。VAD <a class="cite" href="#ref-vad"></a> はシーンをベクトル表現に置き換えて計算量を抑え，TransFuser <a class="cite" href="#ref-transfuser"></a><a class="cite" href="#ref-transfuser-cvpr"></a> はカメラと LiDAR の特徴をトランスフォーマで融合して閉ループ CARLA で評価した。AutoE2E はこれらの完全なタスク連鎖を再現しない。共有 BEV 表現を計画に向けて構成するという思想を継承しつつ，時系列カメラ BEV，明示的なナビゲーション文脈，補助経路再構成，直接の自車計画に範囲を絞る。UniAD との比較は共通ベンチマーク上では行っておらず，優劣は主張しない。

### 時系列 BEV エンコーダ

BEVFormer <a class="cite" href="#ref-bevformer"></a> は学習可能 BEV クエリと，Deformable DETR <a class="cite" href="#ref-deformable"></a> に由来する変形可能アテンションで，多カメラ特徴を時空間的に BEV へ集約する。BEVFormer v2 <a class="cite" href="#ref-bevformerv2"></a> は透視図監督により現代的な画像バックボーンを BEV 認識へ適応させ，ResNet-50 版の公式チェックポイントと，複数フレーム BEV を畳み込みで融合する時系列構成を公開した。本稿の凍結カメラ経路は公式 R50 T8 チェックポイントを起点とし，BEV クエリと位置埋め込みを 300×200 格子へリサイズしたうえで，検出ヘッドを除いて読み込む。

### 地図・経路を条件とする計画

ChauffeurNet <a class="cite" href="#ref-chauffeurnet"></a> は道路地図と意図経路をラスタとして描画し模倣学習の入力とした。VectorNet <a class="cite" href="#ref-vectornet"></a> と LaneGCN <a class="cite" href="#ref-lanegcn"></a> は HD 地図をベクトルや車線グラフとして符号化し，地図構造が動作予測の精度を左右することを示した。KITScenes <a class="cite" href="#ref-kitscenes"></a> の E2E ベンチマークで評価された UniAD 系ベースラインは離散的な航法コマンド（左折・右折・直進）を条件とするが，本稿の経路は Lanelet2 <a class="cite" href="#ref-lanelet2"></a> 地図上の車線列コリドーと目的地マーカーというラスタである。経路条件付けが計画に与える効果を切り分けるには同一シーンでの対照実験が必要であり，その仕様を付録に示す。

### 評価プロトコル

nuScenes のオープンループ計画評価には，自車状態だけで高い数値が得られる近道の存在が繰り返し指摘されてきた <a class="cite" href="#ref-admlp"></a><a class="cite" href="#ref-egostatus"></a>。閉ループ評価の nuPlan <a class="cite" href="#ref-nuplan"></a>，非反応型シミュレーションの NAVSIM <a class="cite" href="#ref-navsim"></a>，規則ベース計画器が学習ベース計画器に匹敵するという報告 <a class="cite" href="#ref-pdm"></a>，条件付け信号に起因する近道学習のバイアス <a class="cite" href="#ref-hidden"></a> は，評価条件の定義自体が結論を左右することを示している。本稿はオープンループ評価にとどまるが，評価入力を配備契約と一致させること，モデルを再実行しない決定的再生と内部検証を区別すること，そして異なるシーン集団間の差を因果効果と読まないことを設計原則とする。

## 問題設定と要件

### 配備契約

配備される AutoE2E 方策は毎フレーム次を受け取る。6 視点のカメラ画像（現在フレームおよび 0.5 s 間隔の 7 履歴フレーム，各 512×512），現在フレームの前方カメラ 1024×1024，自車中心の HD 地図ラスタ（14 チャネル）と選択経路ラスタ（2 チャネル），10 Hz で 6.4 s 分の自車運動履歴。出力は 10 Hz，6.4 s 分の加速度と曲率である。したがって主評価は，これら全入力が利用可能な条件で行うべきである。一方，地図や経路が欠損する状況は運用上起こり得るため，カメラのみで動作した場合の挙動は別トラックとして計測する価値がある。両トラックは異なる問いに答える。

### 評価の分離原則

（1）主ベンチマークは Camera + HD Map + Route が揃う KITScenes Val v3.5 とし，厳密な公式 Map・Route 評価同一性を満たす 117 シーン，11,035 サンプルで数値を算出する。（2）KITScenes Test v1.0 は地図が公開されていない構成のため，カメラのみの頑健性・転移トラックとして全 206 シーン，23,690 サンプルで評価する。（3）Val と Test はシーン集団が異なるため，Test と Val の差を地図・経路の因果効果として解釈しない。（4）外部評価は公開された決定的な制御オーバーレイの再生であり，モデルを再実行せず，経路の反事実入力や入力勾配，経路再構成出力を含まない。（5）チェックポイント内部検証は別プロトコルであり，同じ順位表に混在させない。

<figure class="table col-span-2" id="tab-position">
<figcaption>代表的な E2E 運転システムとの位置づけ。◯：該当，—：該当しない，n/a：本稿の証拠からは判断しない。本表は主張の範囲を示すものであり，性能の優劣を示すものではない。</figcaption>
<table class="small">
<thead><tr><th>システム</th><th class="c">カメラのみ主評価</th><th class="c">HD 地図入力</th><th class="c">経路入力</th><th class="c">時系列 BEV</th><th class="c">検出・追跡・占有の統合</th><th class="c">閉ループ評価</th><th class="c">評価データ</th></tr></thead>
<tbody>
<tr><td>UniAD <a class="cite" href="#ref-uniad"></a></td><td class="c">◯</td><td class="c">—（オンライン地図推定）</td><td class="c">航法コマンド</td><td class="c">◯</td><td class="c">◯</td><td class="c">—</td><td class="c">nuScenes</td></tr>
<tr><td>VAD <a class="cite" href="#ref-vad"></a></td><td class="c">◯</td><td class="c">—（ベクトル地図推定）</td><td class="c">航法コマンド</td><td class="c">◯</td><td class="c">部分</td><td class="c">◯（CARLA）</td><td class="c">nuScenes / CARLA</td></tr>
<tr><td>TransFuser <a class="cite" href="#ref-transfuser"></a></td><td class="c">—（カメラ+LiDAR）</td><td class="c">—</td><td class="c">目標点</td><td class="c">—</td><td class="c">補助タスク</td><td class="c">◯（CARLA）</td><td class="c">CARLA</td></tr>
<tr><td>AutoE2E（本稿）</td><td class="c">—（副トラック）</td><td class="c">◯（14 ch ラスタ）</td><td class="c">◯（2 ch ラスタ）</td><td class="c">◯（T8, 3.5 s）</td><td class="c">—（BEV 分割ヘッドは無効）</td><td class="c">—</td><td class="c">KITScenes Val / Test，nuPlan 内部</td></tr>
</tbody>
</table>
</figure>

## 手法

本章の記述は，評価対象ブランチのソースコードを読み，同一構成でモデルをインスタンス化して確認した事実に基づく。数式中の記号は <a class="figref" href="#fig-arch"></a> に対応する。

### 時系列カメラ BEV

カメラ経路は公式 BEVFormer V2 R50 T8 チェックポイント <a class="cite" href="#ref-bevformerv2"></a> に基づく。ResNet-50 <a class="cite" href="#ref-resnet"></a> の最終 3 ステージから 4 段の 256 チャネル特徴ピラミッドを構成し，6 層の BEVFormer エンコーダが 300×200 個の学習済み BEV クエリ（256 次元，15.36 M パラメータ）を更新する（<a class="figref" href="#fig-encoder"></a>）。BEV 格子は自車座標で X が −60 m から 120 m，Y が −60 m から 60 m を覆い，セルは 0.6 m である（<a class="figref" href="#fig-geometry"></a>）。各層は，現在 BEV を複製した 2 キューに対する変形可能自己注意（8 ヘッド，各 4 点），4 つの柱高さ（z ∈ [−5, 3] m）の参照点をキャリブレーションされた投影演算子で画像へ写像する多スケール空間クロスアテンション（8 ヘッド，4 レベル，各 8 点），512 次元の FFN と 3 つの LayerNorm から成る。投影演算子はピンホールと魚眼（F-theta）を実装しており，本評価はキャリブレーション不要ではない。

<figure class="fig" id="fig-geometry">
<img src="figures/fig_geometry.svg" alt="BEV extents">
<figcaption>BEV の空間範囲。カメラ BEV 潜在（300×200，0.6 m）と，モデルに入力される地図・経路ラスタ（450×300，0.4 m）は同一の範囲（X：−60〜120 m，Y：−60〜60 m）を覆う。点線は KITScenes v3 で監査された 256×256，1.0 m の公開ジオメトリであり，本評価でモデルが受け取るラスタではない。</figcaption>
</figure>

T8 の時系列構成では，0.5 s 間隔の 7 履歴フレーム（t−3.5 s から t−0.5 s）と現在フレームを同じバックボーンとエンコーダに通し，履歴 BEV は勾配を切り離す。8 個の BEV を連結した 2048 チャネルを，512 中間チャネルの 3 ブロック ResNet 型融合器に通し，線形層で 256 チャネルへ射影して B_img を得る（式 <a class="eqref" href="#eq-temporal"></a>）。カメラ経路は 2 Hz で 3.5 s を見ており，これは後述の 6.4 s の自車運動履歴とは異なる時間幅である（<a class="figref" href="#fig-temporal"></a>）。

<div class="equation number" id="eq-temporal">

$$B_{\mathrm{img}} = T\left(B_{-7}, B_{-6}, \ldots, B_{-1}, B_{0}\right)$$

</div>

<figure class="fig col-span-2" id="fig-temporal">
<img src="figures/fig02_temporal_contract_v2.svg" alt="Temporal contract">
<figcaption>時間契約。カメラ分岐は 0.5 s 間隔の 8 フレーム（3.5 s），自車運動履歴は 10 Hz で 64 ステップ（6.4 s），出力は 10 Hz で 64 ステップ（6.4 s）である。KITScenes ベンチマークプロトコルでは履歴 40 ステップ（4.0 s，残りはゼロ埋め）と将来 50 ステップ（5.0 s）のみが実データであるため，外部再生の有効ホライズン被覆率は 50/64 = 78.125% となり，6.4 s の ADE/FDE は外部再生では得られない。</figcaption>
</figure>

<figure class="fig col-span-2" id="fig-encoder">
<img src="figures/fig03_encoder_layer_v2.svg" alt="Encoder layer">
<figcaption>BEVFormer V2 エンコーダ層（左，6 層，凍結）と前方カメラ残差分岐（右）。前方分岐は最終層のクロスアテンションの凍結コピーで 1024×1024 の現在前方画像から内容依存の差分のみを計算し，ゼロ初期化された 256 次元ゲートの tanh を乗じて現在 BEV に加算する。初期状態では厳密に恒等であり，凍結後もこのゲートのみが学習される。履歴フレームは 512×512 のみを用いる。</figcaption>
</figure>

### 非対称解像度の前方分岐

現在フレームの前方カメラは 1024×1024 でも処理される。通常の 512 px 前方視点は全視点エンコーダに残したまま，最終エンコーダ層のクロスアテンションを凍結コピーした分岐が高解像度前方特徴から内容依存の差分 δ を計算し，B_0 = B_enc + tanh(g) ⊙ δ として加算する。g は 256 次元でゼロ初期化されるため，学習開始時点で事前学習カメラ経路の出力は変化しない。凍結の対象外となるカメラ側パラメータはこの 256 個のみである。履歴フレームに 1024 px 画像は用いない。

### ナビゲーションラスタ

HD 地図と経路は自車中心の意味ラスタとして表現する。評価対象チェックポイントに入力されるラスタは 450×300 セル，0.4 m/px で，カメラ BEV と同一の範囲（X：−60〜120 m，Y：−60〜60 m）を覆う。地図ラスタの 14 チャネルと経路ラスタの 2 チャネルの意味を <a class="tabref" href="#tab-channels"></a> に，KITScenes Val v3.5 の実サンプルでの描画を <a class="figref" href="#fig-channels"></a> に示す。二値チャネルは 1，方向チャネルは (sin θ + 1)/2 と (cos θ + 1)/2，道路レベルは (clip(level, −8, 8) + 8)/16 を格納し，値域はすべて [0, 1] である。経路コリドーは車線中心線列を幅 3.5 m で描き（自車後方 10 m で切断），目的地マーカーは半径 2.0 m の円である。地図の有効性と経路の有効性はサンプル単位の明示的ゲートであり，無効なラスタはゼロに乗じられる。経路条件付けはチェックポイントまたは評価時に無効化でき，同一シーンでの対照実験を可能にする。

<figure class="table col-span-2" id="tab-channels">
<figcaption>地図ラスタ 14 チャネルと経路ラスタ 2 チャネルの定義（ネイティブ C++ ラスタライザとレーンレット適合器の実装から抽出）。右列は本稿の決定的走査で取得した KITScenes Val v3.5 の 30 サンプル（付録 A.5）における平均占有率と非空サンプル数であり，全データセットの統計ではない。</figcaption>
<table class="small">
<thead><tr><th>ch</th><th>名称</th><th>生成元プリミティブ（Lanelet2 <a class="cite" href="#ref-lanelet2"></a>）</th><th>値</th><th class="num">平均占有率 [%]</th><th class="num">非空 / 30</th></tr></thead>
<tbody>
<tr><td>0</td><td>走行可能領域</td><td>レーンレット多角形（横断歩道以外）</td><td>二値</td><td class="num">15.55</td><td class="num">30</td></tr>
<tr><td>1</td><td>車線境界</td><td>左右境界線（幅 1.0 m）</td><td>二値</td><td class="num">8.29</td><td class="num">30</td></tr>
<tr><td>2</td><td>車線中心線</td><td>中心線（幅 1.0 m）</td><td>二値</td><td class="num">5.52</td><td class="num">30</td></tr>
<tr><td>3</td><td>交差点</td><td>turn_direction 属性を持つレーンレット多角形</td><td>二値</td><td class="num">1.46</td><td class="num">28</td></tr>
<tr><td>4</td><td>横断歩道</td><td>subtype crosswalk の多角形</td><td>二値</td><td class="num">0.96</td><td class="num">24</td></tr>
<tr><td>5</td><td>停止線</td><td>停止線ポリライン（幅 1.0 m）</td><td>二値</td><td class="num">0.18</td><td class="num">27</td></tr>
<tr><td>6</td><td>静的信号機</td><td>規制要素 traffic_light の位置（半径 1.0 m）</td><td>二値</td><td class="num">0.07</td><td class="num">26</td></tr>
<tr><td>7</td><td>進行方向 sin</td><td>中心線の向き（幅 3.5 m）</td><td>(sin+1)/2</td><td class="num">14.72</td><td class="num">30</td></tr>
<tr><td>8</td><td>進行方向 cos</td><td>同上</td><td>(cos+1)/2</td><td class="num">14.72</td><td class="num">30</td></tr>
<tr><td>9</td><td>進行方向有効</td><td>同上</td><td>二値</td><td class="num">14.72</td><td class="num">30</td></tr>
<tr><td>10</td><td>既知地図領域</td><td>地図境界多角形</td><td>二値</td><td class="num">96.98</td><td class="num">30</td></tr>
<tr><td>11</td><td>道路レベル</td><td>layer / level 属性</td><td>(level+8)/16</td><td class="num">0.00</td><td class="num">0</td></tr>
<tr><td>12</td><td>道路レベル有効</td><td>同上</td><td>二値</td><td class="num">0.00</td><td class="num">0</td></tr>
<tr><td>13</td><td>重複レベル曖昧性</td><td>異なるレベルの上書き，または経路レベルとの不一致</td><td>二値</td><td class="num">0.00</td><td class="num">0</td></tr>
<tr class="group"><td>R0</td><td>選択経路コリドー</td><td>経路車線列の中心線（幅 3.5 m，後方 10 m で切断）</td><td>二値</td><td class="num">1.45</td><td class="num">28</td></tr>
<tr><td>R1</td><td>目的地マーカー</td><td>経路終点（半径 2.0 m）</td><td>二値</td><td class="num">0.05</td><td class="num">26</td></tr>
</tbody>
</table>
</figure>

<figure class="fig" id="fig-channels">
<img src="figures/fig_channels_real.svg" alt="Map and route channels of a real sample">
<figcaption>KITScenes Val v3.5 の 1 サンプルにおける地図 14 チャネルと経路 2 チャネル（450×300，0.4 m/px）。公開ダッシュボードのデータセットアーティファクトから取得した実データであり，赤は自車位置と記録された 5 s の軌跡である。サンプルの選択規則は付録 A.5 に示す。走査した 30 サンプルでは道路レベル系のチャネル 11〜13 はすべて空であった。</figcaption>
</figure>

KITScenes の経路は，シーン全体の自車走行軌跡を Lanelet2 地図に適合させて得た車線列であり，目的地はシーン終端の自車位置である（<a class="figref" href="#fig-route"></a>）。将来の自車点そのものは描画されないが，経路は事後的に決まる走行済み車線列，すなわち運転者が実際に辿った意図に等しい。この事実は経路条件付けの解釈に直接関わるため，7 章で改めて扱う。

<figure class="fig col-span-2" id="fig-route">
<img src="figures/fig14_route_provenance.svg" alt="Route provenance">
<figcaption>KITScenes における地図・経路ラスタの生成過程。Lanelet2 地図からプリミティブを抽出し，シーン全体の自車軌跡を HMM 風の遷移コストで車線列に適合させて経路とする。ラスタは 500 ms 間隔のアンカー姿勢で描画し，各サンプル姿勢へ SE(2) 変換で歪ませる。経路はライブの経路計画器の出力ではなく，事後的な車線レベルの走行意図である。</figcaption>
</figure>

### 分離符号化とゲート付き加算

静的地図と選択経路は別々の符号器を持つ（<a class="figref" href="#fig-navfusion"></a>）。地図符号器 E_map は 5×5 畳み込み（96 チャネル）の幹と，深さ方向畳み込みでストライド 2 と 4 に落とした文脈経路（192 チャネル）を 1×1 射影で 256 チャネルに揃えて加算する全畳み込み網で，入力を 300×200 へ双線形リサンプルする。経路符号器 E_route は 96 隠れチャネルの軽量畳み込みで，細い経路コリドーと目的地の特徴を粗いプーリングで失わないよう設計されている。経路の寄与は 256 次元のチャネル別ゲートで制御され，符号化地図に加算される（式 <a class="eqref" href="#eq-nav"></a>）。ゲート g はゼロ初期化され，学習開始時の重みは 0.5 である。

<div class="equation number" id="eq-nav">

$$B_{\mathrm{nav}} = E_{\mathrm{map}}(M) + \sigma(g) \odot E_{\mathrm{route}}(R)$$

</div>

<figure class="fig col-span-2" id="fig-navfusion">
<img src="figures/fig06_navigation_fusion.svg" alt="Navigation encoders and deformable fusion">
<figcaption>分離ナビゲーション符号化と変形可能ナビゲーション融合の内部構成。左：地図符号器 E_map（0.167 M）と経路符号器 E_route（0.078 M），チャネル別ゲート，加算。右：B_img の各セルをクエリとして B_nav 上の 8 点をサンプリングする変形可能クロスアテンション（8 ヘッド，256 チャネル，0.350 M）。出力射影と FFN の最終線形層はゼロ初期化され，学習開始時に融合は厳密な恒等写像である。ゲート後の経路寄与は経路再構成ヘッド（0.017 M）に供給される。</figcaption>
</figure>

### 変形可能ナビゲーション融合

ナビゲーション BEV は変形可能クロスアテンション <a class="cite" href="#ref-deformable"></a> でカメラ BEV に融合される（式 <a class="eqref" href="#eq-fuse"></a>）。画像 BEV の各セルがクエリとなり，自身の位置を参照点として 8 個のオフセットを予測し，B_nav を双線形サンプリングして 8 ヘッドのソフトマックス重みで集約する。密なクロスアテンションが 60,000 セルに対して O(N²) となるところを O(NK) に抑えるための構成である。出力射影と FFN の最終線形層はゼロ初期化されているため，融合は残差であり，学習開始時に事前学習カメラ経路の表現を変えない。これは，凍結された視覚 BEV 表現に新しいナビゲーション特徴を後付けする際の安定化機構である。

<div class="equation number" id="eq-fuse">

$$B_{\mathrm{fused}} = B_{\mathrm{img}} + F_{\mathrm{deform}}\left(B_{\mathrm{img}}, B_{\mathrm{nav}}\right)$$

</div>

### 自車運動履歴と GRU 制御プランナ

自車運動履歴 h_ego は 10 Hz，64 ステップの [速度，加速度，ヨーレート，曲率] を平坦化した 256 次元で，速度は 33 m/s，加速度は 8 m/s² で正規化される。KITScenes では姿勢列の有限差分から導出され，曲率はヨーレートを 0.1 m/s で下限化した速度で除したものである。モデル API は 896 次元の視覚履歴入力 h_vis を保持するが，評価対象チェックポイントは enable_world_model = false，temporal_memory_mode = no_memory であり，h_vis はゼロが入力される。JEPA 学習や学習済み世界モデル，6.4 s の学習済み視覚履歴符号器はこれらのチェックポイントには存在しない。

<figure class="fig col-span-2" id="fig-planner">
<img src="figures/fig07_gru_planner.svg" alt="GRU planner">
<figcaption>決定的 GRU 制御プランナ（0.835 M）とユニサイクル積分。初期状態は自車運動履歴と視覚履歴の線形射影の和である。各ステップで再帰状態（勾配を切り離す）と学習済み自車クエリの和からサンプリング位置を予測し，融合 BEV 上の 16 点を集約して GRU を更新し，加速度と曲率を出力する。この手順を 64 ステップ繰り返す。</figcaption>
</figure>

プランナは決定的 GRU である（<a class="figref" href="#fig-planner"></a>）。リポジトリにある Bezier や flow-matching プランナは評価対象チェックポイントでは用いていない。初期文脈 c_0 は式 <a class="eqref" href="#eq-ctx"></a> で与え，各将来ステップで融合 BEV 上の 16 点を変形可能に参照し（式 <a class="eqref" href="#eq-attn"></a>），GRU が状態を更新して加速度と曲率を出力する（式 <a class="eqref" href="#eq-gru"></a>，<a class="eqref" href="#eq-out"></a>）。これを 64 ステップ繰り返す。

<div class="equation number" id="eq-ctx">

$$c_0 = P_{\mathrm{ego}}(h_{\mathrm{ego}}) + P_{\mathrm{vis}}(h_{\mathrm{vis}})$$

</div>
<div class="equation number" id="eq-attn">

$$a_t = A_{\mathrm{deform}}\left(q_t, B_{\mathrm{fused}}\right), \quad q_t = \mathrm{sg}(h_{t-1}) + e$$

</div>
<div class="equation number" id="eq-gru">

$$h_t = \mathrm{GRU}\left(a_t, h_{t-1}\right)$$

</div>
<div class="equation number" id="eq-out">

$$u_t = W_u h_t = \left(a^{\mathrm{acc}}_t, \kappa_t\right)$$

</div>

### ユニサイクル積分

評価と学習損失は，制御列を半陰的ユニサイクルモデルで自車座標 XY へ変換する。時間刻み Δt = 0.1 s，初期速度は現在速度 v_0 とし，速度は非負に切り詰める（式 <a class="eqref" href="#eq-roll"></a>）。積分は float32 で実行される。速度以外の実装固有の切り詰め値は存在しない。

<div class="equation number" id="eq-roll">

$$v_t = \max\left(v_{t-1} + a^{\mathrm{acc}}_t \Delta t, 0\right),\;
\theta_t = \sum_{i \le t} v_i \kappa_i \Delta t,\;
x_t = \sum_{i \le t} v_i \cos\theta_i \Delta t,\;
y_t = \sum_{i \le t} v_i \sin\theta_i \Delta t$$

</div>

### 補助タスクと学習目的

経路再構成ヘッドはゲート後の経路寄与から 2 チャネルのロジットを 450×300 で復号し，コリドーには BCE と soft Dice の平均，目的地にはヒートマップ焦点損失（重み 0.25）を課す。軌跡損失は積分後の XY と記録軌跡の Smooth-L1（β = 1 m）を有効ステップで平均する。4 つの報告チェックポイントはいずれも軌跡重み 1.0，経路再構成重み 1.0，BEV 分割重み 0.0 で学習された。BEV 分割ヘッド（8 クラス）はチェックポイントに存在するが，報告チェックポイントは BEV 分割で共同最適化されていない。知覚分岐と世界モデル分岐も無効である。

### パラメータ予算

同一構成（6 視点）でモデルをインスタンス化して数えたパラメータを <a class="tabref" href="#tab-params"></a> と <a class="figref" href="#fig-params"></a> に示す。合計 79,906,522 のうち，カメラ BEV 経路が 78,293,196（97.98%）を占める。したがって総数ベースでは，このモデルはほぼ事前学習済み・凍結済みの視覚表現器である。主な内訳は T8 時系列融合 38.56%，ResNet-50 29.42%，BEV クエリ 19.22%，BEVFormer エンコーダ 6.18%，FPN 4.10% である。一方，軌跡学習段階で更新される 1,448,582（全体の 1.81%）は，GRU プランナ 835,380（学習対象の 57.67%），変形可能ナビゲーション融合 350,288（24.18%），地図・経路符号器 245,440（16.94%），経路再構成ヘッド 17,218（1.19%），前方残差ゲート 256（0.02%）に配分される。すなわち「79.91 M 全体を計画向けに学習した」のではなく，大きな凍結カメラ BEV 表現の上で 1.45 M のタスク固有部を学習した構成である。リポジトリのベンチマーク文書が 8 視点構成で記録する 79,907,034 との差 512 は，カメラ埋め込み 2 視点分に一致する。

<figure class="table" id="tab-params">
<figcaption>評価構成のパラメータ数（実装をインスタンス化して計測）。学習対象は軌跡学習段階で requires_grad が真のもの。</figcaption>
<table class="small">
<thead><tr><th>構成要素</th><th class="num">パラメータ</th><th class="c">状態</th></tr></thead>
<tbody>
<tr><td>ResNet-50 バックボーン</td><td class="num">23,508,032</td><td class="c">凍結</td></tr>
<tr><td>特徴ピラミッド（4 段）</td><td class="num">3,278,592</td><td class="c">凍結</td></tr>
<tr><td>BEV クエリ 300×200×256</td><td class="num">15,360,000</td><td class="c">凍結</td></tr>
<tr><td>行・列・レベル・カメラ埋め込み</td><td class="num">66,560</td><td class="c">凍結</td></tr>
<tr><td>疑似投影行列（キャリブレーション評価では未使用）</td><td class="num">12</td><td class="c">凍結</td></tr>
<tr><td>エンコーダ 6 層（1 層 823,488）</td><td class="num">4,940,928</td><td class="c">凍結</td></tr>
<tr><td>前方クロスアテンション（コピー）</td><td class="num">328,960</td><td class="c">凍結</td></tr>
<tr><td>前方残差ゲート</td><td class="num">256</td><td class="c">学習</td></tr>
<tr><td>T8 時系列融合</td><td class="num">30,809,856</td><td class="c">凍結</td></tr>
<tr><td>地図符号器 E_map</td><td class="num">167,200</td><td class="c">学習</td></tr>
<tr><td>経路符号器 E_route</td><td class="num">77,984</td><td class="c">学習</td></tr>
<tr><td>経路ゲート</td><td class="num">256</td><td class="c">学習</td></tr>
<tr><td>変形可能ナビゲーション融合</td><td class="num">350,288</td><td class="c">学習</td></tr>
<tr><td>GRU 制御プランナ</td><td class="num">835,380</td><td class="c">学習</td></tr>
<tr><td>経路再構成ヘッド</td><td class="num">17,218</td><td class="c">学習</td></tr>
<tr><td>BEV 分割ヘッド（重み 0.0）</td><td class="num">165,000</td><td class="c">凍結</td></tr>
<tr class="group"><td>合計 / 学習対象</td><td class="num">79,906,522 / 1,448,582</td><td class="c">—</td></tr>
</tbody>
</table>
</figure>

<figure class="fig col-span-2" id="fig-params">
<img src="figures/fig_param_decomposition_v2.svg" alt="Parameter budget">
<figcaption>評価対象 6 視点構成のパラメータ配分。（a）は全 79,906,522 パラメータの内訳で，凍結カメラ BEV 経路が 78.29 M（97.98%）を占める。（b）は学習対象 1,448,582 パラメータの内訳で，GRU プランナが 57.67%，変形可能ナビゲーション融合が 24.18%，地図・経路符号器と経路ゲートが 16.94% を占める。</figcaption>
</figure>

## データセットと評価プロトコル

### 学習データと系譜

4 つの軌跡チェックポイントはすべて 8 台の分散 GPU ワーカ，学習シード 149，凍結 BEVFormer カメラ分岐，軌跡重み 1.0，経路再構成重み 1.0，BEV 分割重み 0.0 で学習された（<a class="tabref" href="#tab-config"></a>，<a class="figref" href="#fig-protocol"></a>）。nuPlan <a class="cite" href="#ref-nuplan"></a> 段階のチェックポイント（Epoch 4，Epoch 5）は学習率 1e-4 で，内部検証は 1,024 サンプルである。KITScenes <a class="cite" href="#ref-kitscenes"></a> 微調整は，凍結された nuPlan Epoch 5 を親として学習率 3e-5 で行った。分割マニフェストは公式 KITScenes train 分割を出典として記録しており，利用可能な 533 シーンから空の 129 シーンを除いた 404 シーン，42,667 サンプルが対象である。シーン単位の 90% / 10% 分割により 364 シーン 38,847 サンプルを学習，40 シーン 3,820 サンプルを凍結内部検証に用いた（検証精度 BF16）。この 404 シーンは公式 train 分割に属するため，公式 Val 117 シーンおよび Test 206 シーンとはシーン ID が交差しない。KITScenes Epoch 5 はその時点で保持された最良チェックポイント，Epoch 7 は最良指定なしであるが，その登録フラグに用いた選択スコアの定義は本稿が利用できる証拠に記録されていない。登録メタデータは 4 チェックポイントすべてで eval_gate_pass = false を記録しており，本稿は品質ゲートの通過を主張しない（付録 A.3）。

<figure class="table col-span-2" id="tab-config">
<figcaption>報告チェックポイントの学習設定と識別子。SHA-256 は先頭 12 桁のみ示す。</figcaption>
<table class="small">
<thead><tr><th>チェックポイント</th><th>SHA-256 接頭辞</th><th>学習段階</th><th class="num">学習率</th><th>親</th><th class="num">内部検証サンプル</th><th>備考</th></tr></thead>
<tbody>
<tr><td>nuPlan Epoch 4</td><td>ed00e072471a</td><td>nuPlan 軌跡・経路（凍結カメラ BEV）</td><td class="num">1e-4</td><td>公式 BEVFormer V2 R50 T8</td><td class="num">1,024</td><td>登録上 nuPlan 最良</td></tr>
<tr><td>nuPlan Epoch 5</td><td>ca8b43d7a777</td><td>同上</td><td class="num">1e-4</td><td>同上</td><td class="num">1,024</td><td>KITScenes 微調整の親</td></tr>
<tr><td>KITScenes Epoch 5</td><td>120a21639d97</td><td>KITScenes 微調整</td><td class="num">3e-5</td><td>nuPlan Epoch 5（凍結）</td><td class="num">3,820</td><td>保持された最良</td></tr>
<tr><td>KITScenes Epoch 7</td><td>a1e6b1621018</td><td>KITScenes 微調整</td><td class="num">3e-5</td><td>nuPlan Epoch 5（凍結）</td><td class="num">3,820</td><td>最良指定なし</td></tr>
</tbody>
</table>
<p class="note" style="text-indent:0">共通：8 GPU ワーカ，seed 149，軌跡重み 1.0，経路再構成重み 1.0，BEV 分割重み 0.0，eval_gate_pass = false。</p>
</figure>

<figure class="fig col-span-2" id="fig-protocol">
<img src="figures/fig08_lineage_protocol.svg" alt="Lineage and protocols">
<figcaption>上：チェックポイント系譜。公式 BEVFormer V2 R50 T8 を起点に，nuPlan 軌跡段階（Epoch 4 / 5），凍結 Epoch 5 を親とする KITScenes 微調整（Epoch 5 / 7）。下：2 つの評価プロトコル。プロトコル A はモデルを実行するチェックポイント内部検証（6.4 s），プロトコル B は公開された決定的制御オーバーレイの再生（5 s まで）である。両者は同じ順位表に混在させない。Val と Test は異なるシーン集団である。</figcaption>
</figure>

### 評価データセット

主評価は KITScenes Val v3.5 の Camera + HD Map + oracle 事後経路条件である。公開ダッシュボードは 140 個のシャードアーティファクトと 13,525 サンプルを表示するが，数値評価は公式 Val 117 シーンのうち，より厳格な Map・Route 評価同一性を満たす 11,035 サンプルを用いる。利用できる報告には除外された 2,490 サンプルの理由別内訳がなく，この同一性フィルタによる選択バイアスを否定できない。11,035 サンプル集団の map_valid と route_valid の比率も再生報告には保存されていない。副評価は KITScenes Test v1.0 の全 206 シーン，23,690 サンプルである。マニフェストは Map と Route を利用不可とし，ローダは符号器の前で有効性ゲートを偽にしてゼロラスタを与える。ただし学習済み GroupNorm のアフィン項はゼロテンソルへ応答し得るため，この条件はナビゲーションモジュールを構造的に除いた「純粋なカメラのみ」より，ナビゲーション入力無効条件と呼ぶ方が正確である。KITScenes 論文の公式 E2E プロトコルは検証データから 200 個の 9 s 窓を抽出し，4 s の観測と最大 5 s の将来軌跡を用いる <a class="cite" href="#ref-kitscenes"></a>。本稿の再生は履歴 40・将来 50 ステップの時間契約を近似し，公式 Val/Test 同一性が認めた全サンプルへ適用する。Val と Test は異なるシーン集団であり，対になった切除実験として扱わない。

### プロトコル A：チェックポイント内部検証

学習ジョブと同じコードでモデルを実行し，6.4 s（64 ステップ）の ADE/FDE，経路再構成 IoU，および開ループの経路・走行可能領域・快適性指標を計算する。経路コリドー遵守率は，4.8 m × 2.0 m の自車フットプリント 4 隅がすべて 3.5 m 幅の経路コリドー内にあるステップの割合，成功率は全 64 ステップで遵守したサンプルの割合である。走行可能領域遵守率・成功率は走行可能領域マスクに対して同じ定義をとる。投影終点弧長プロキシは，予測終点を記録軌跡ポリラインに射影した弧長で，比率はサンプルごとの弧長比を平均する。この値は軌跡終端で 1 に上限化されるため，適切な前進とオーバーシュートを区別しない。快適性は nuPlan の閾値（縦加速度 −4.05〜2.40 m/s²，横加速度 4.89 m/s²，ヨーレート 0.95 rad/s，ヨー加速度 1.93 rad/s²，縦ジャーク 4.13 m/s³，ジャーク大きさ 8.37 m/s³）のいずれかを超えたサンプルを違反とし，快適率は 1 − 違反率である。経路指標は経路が有効な 2,911 サンプルで計算される。

### プロトコル B：外部の決定的オーバーレイ再生

外部評価は，各チェックポイントの公開制御オーバーレイ（サンプルごとの加速度・曲率 64 組と初期速度）を式 <a class="eqref" href="#eq-roll"></a> で再生し，記録軌跡と比較する。モデルは再実行しない。登録役割は Val オーバーレイが Camera・HD Map・Route を入力した推論で生成され，Test が camera_only_missing_map_route で生成されたことを記録するが，再生自体は当時の入力を独立に検証できない。ADE_h は h 秒までの有効ステップのユークリッド誤差をサンプル内で平均した後にサンプル平均し，FDE_h は h 秒時点の平均誤差である。横・縦方向誤差は有効な 0〜5 s 全ステップの |Δy| と |Δx| を平均するため，特定ホライズンや機構を識別しない。予測制御のいずれかが非有限なら非有限サンプルとする。再生アーティファクトは経路反事実・入力勾配・経路再構成出力を含まない。ベンチマーク窓では実データの自車運動履歴 40 ステップ（4.0 s）を 64 ステップ ABI へ左ゼロ埋めし，将来目標は 50 ステップ（5.0 s）のみ有効である。したがって 1・2・3・5 s を報告し，外部 6.4 s 指標は報告しない。一方，プロトコル A は学習用の窓構成から 64 個の実将来目標を用いるため，その KITScenes 6.4 s 結果は別のサンプル構成であり，プロトコル B と順位付けしない。完了した再生報告には運動学的参照ベースラインも対の不確実性区間もない。表中の太字と下線は数値順のみを示し，統計的有意差を意味しない。

## 結果

### 主評価：KITScenes Val（Camera + HD Map + Route）

<a class="tabref" href="#tab-val-ade"></a> と <a class="figref" href="#fig-val"></a> に，11,035 サンプルにおける 4 チェックポイントの ADE/FDE を示す。KITScenes Epoch 7 の報告平均は 1 s の ADE（0.1347 m）と FDE（0.2831 m）で最小であり，KITScenes Epoch 5 の報告平均は 2・3・5 s の ADE/FDE で最小である（5 s：ADE 1.9405 m，FDE 5.5645 m）。nuPlan の 2 チェックポイントは全ホライズンで KITScenes 微調整に劣り，5 s の FDE で 6.2〜6.4 m である。

<figure class="table col-span-2" id="tab-val-ade">
<figcaption>KITScenes Val v3.5，Camera + HD Map + Route，外部オーバーレイ再生（各モデル n = 11,035 サンプル）。単位 m，小さいほど良い。太字は同一指標・同一ホライズン内の最良，下線は 2 位。</figcaption>
<table>
<thead><tr><th>モデル</th><th class="num">ADE 1 s</th><th class="num">ADE 2 s</th><th class="num">ADE 3 s</th><th class="num">ADE 5 s</th><th class="num">FDE 1 s</th><th class="num">FDE 2 s</th><th class="num">FDE 3 s</th><th class="num">FDE 5 s</th></tr></thead>
<tbody>
<tr><td>KITScenes Epoch 5</td><td class="num second">0.1472</td><td class="num best">0.3729</td><td class="num best">0.7449</td><td class="num best">1.9405</td><td class="num second">0.2850</td><td class="num best">0.9239</td><td class="num best">2.0196</td><td class="num best">5.5645</td></tr>
<tr><td>KITScenes Epoch 7</td><td class="num best">0.1347</td><td class="num second">0.3886</td><td class="num second">0.8035</td><td class="num second">2.0952</td><td class="num best">0.2831</td><td class="num second">1.0101</td><td class="num second">2.2127</td><td class="num second">5.9509</td></tr>
<tr><td>nuPlan Epoch 5</td><td class="num">0.1951</td><td class="num">0.4723</td><td class="num">0.9147</td><td class="num">2.2951</td><td class="num">0.3779</td><td class="num">1.1331</td><td class="num">2.4226</td><td class="num">6.3885</td></tr>
<tr><td>nuPlan Epoch 4</td><td class="num">0.2030</td><td class="num">0.5018</td><td class="num">0.9503</td><td class="num">2.2919</td><td class="num">0.4066</td><td class="num">1.1935</td><td class="num">2.4508</td><td class="num">6.2272</td></tr>
</tbody>
</table>
</figure>

<figure class="fig col-span-2" id="fig-val">
<img src="figures/fig_results_val.svg" alt="Val ADE/FDE curves">
<figcaption>KITScenes Val v3.5（Camera + HD Map + Route）における ADE（左）と FDE（右）のホライズン別推移（n = 11,035）。実線は KITScenes 微調整，破線は nuPlan 学習チェックポイント。5 s の値を注記した。</figcaption>
</figure>

### 副評価：KITScenes Test（Camera only）

<a class="tabref" href="#tab-test-ade"></a> と <a class="figref" href="#fig-test"></a> に，23,690 サンプルのカメラのみトラックの結果を示す。KITScenes Epoch 5 の報告平均は 1 s（ADE 0.1463 m，FDE 0.3107 m）と 5 s（ADE 2.1042 m，FDE 5.9422 m）で最小，nuPlan Epoch 5 は 2 s と 3 s で最小である。KITScenes Epoch 7 は 2・3・5 s で 4 モデル中最大の誤差を示し，1 s では nuPlan Epoch 4 が最大である。Val と Test はシーン集団が異なるため，この表と <a class="tabref" href="#tab-val-ade"></a> を並べても地図・経路の効果は測れない。

<figure class="table col-span-2" id="tab-test-ade">
<figcaption>KITScenes Test v1.0，Camera only（地図・経路はデータセット構成上利用不可），外部オーバーレイ再生（各モデル n = 23,690）。単位 m。太字は最良，下線は 2 位。本トラックは地図の切除実験ではない。</figcaption>
<table>
<thead><tr><th>モデル</th><th class="num">ADE 1 s</th><th class="num">ADE 2 s</th><th class="num">ADE 3 s</th><th class="num">ADE 5 s</th><th class="num">FDE 1 s</th><th class="num">FDE 2 s</th><th class="num">FDE 3 s</th><th class="num">FDE 5 s</th></tr></thead>
<tbody>
<tr><td>KITScenes Epoch 5</td><td class="num best">0.1463</td><td class="num second">0.4121</td><td class="num second">0.8263</td><td class="num best">2.1042</td><td class="num best">0.3107</td><td class="num second">1.0453</td><td class="num second">2.2199</td><td class="num best">5.9422</td></tr>
<tr><td>KITScenes Epoch 7</td><td class="num">0.1680</td><td class="num">0.4923</td><td class="num">0.9831</td><td class="num">2.4699</td><td class="num">0.3754</td><td class="num">1.2512</td><td class="num">2.6242</td><td class="num">6.8964</td></tr>
<tr><td>nuPlan Epoch 5</td><td class="num second">0.1602</td><td class="num best">0.3979</td><td class="num best">0.7924</td><td class="num second">2.1120</td><td class="num second">0.3134</td><td class="num best">0.9727</td><td class="num best">2.1621</td><td class="num second">6.1751</td></tr>
<tr><td>nuPlan Epoch 4</td><td class="num">0.1916</td><td class="num">0.4817</td><td class="num">0.9181</td><td class="num">2.2840</td><td class="num">0.3899</td><td class="num">1.1514</td><td class="num">2.3895</td><td class="num">6.4266</td></tr>
</tbody>
</table>
</figure>

<figure class="fig col-span-2" id="fig-test">
<img src="figures/fig_results_test.svg" alt="Test ADE/FDE curves">
<figcaption>KITScenes Test v1.0（Camera only）における ADE と FDE（n = 23,690）。Val とは別のシーン集団であり，<a class="figref" href="#fig-val"></a> との差は地図・経路の因果効果を意味しない。</figcaption>
</figure>

### 横方向・縦方向誤差

<a class="tabref" href="#tab-latlon"></a> と <a class="figref" href="#fig-latlon"></a> に有効ステップ全体の平均絶対横方向誤差（|Δy|）と縦方向誤差（|Δx|）を示す。Val では KITScenes Epoch 5 が両方で最小（0.9844 m，1.4025 m）である。Test では横方向は nuPlan Epoch 5（1.0584 m），縦方向は KITScenes Epoch 5（1.4051 m）が最小である。すべてのモデルで有効な 0〜5 s 全ステップの平均縦方向誤差が平均横方向誤差を上回る。この集約値は特定のホライズンや原因を識別せず，対象シーンの移動量が横方向より進行方向に大きいことにも影響される。

<figure class="table col-span-2" id="tab-latlon">
<figcaption>有効ステップにおける平均絶対横方向誤差と縦方向誤差（単位 m，外部オーバーレイ再生）。Val：n = 11,035，Test：n = 23,690。太字は同一データセット・同一指標内の最良，下線は 2 位。</figcaption>
<table>
<thead><tr><th rowspan="2">モデル</th><th class="num" colspan="2">KITScenes Val v3.5（Camera + Map + Route）</th><th class="num" colspan="2">KITScenes Test v1.0（Camera only）</th></tr>
<tr><th class="num">横方向 |Δy|</th><th class="num">縦方向 |Δx|</th><th class="num">横方向 |Δy|</th><th class="num">縦方向 |Δx|</th></tr></thead>
<tbody>
<tr><td>KITScenes Epoch 5</td><td class="num best">0.9844</td><td class="num best">1.4025</td><td class="num second">1.2044</td><td class="num best">1.4051</td></tr>
<tr><td>KITScenes Epoch 7</td><td class="num second">1.0220</td><td class="num second">1.5659</td><td class="num">1.4730</td><td class="num">1.6598</td></tr>
<tr><td>nuPlan Epoch 5</td><td class="num">1.1911</td><td class="num">1.6633</td><td class="num best">1.0584</td><td class="num second">1.5496</td></tr>
<tr><td>nuPlan Epoch 4</td><td class="num">1.2082</td><td class="num">1.6448</td><td class="num">1.2167</td><td class="num">1.6287</td></tr>
</tbody>
</table>
</figure>

<figure class="fig col-span-2" id="fig-latlon">
<img src="figures/fig_results_latlon.svg" alt="Lateral and longitudinal error">
<figcaption>平均絶対横方向誤差（薄色）と縦方向誤差（斜線）。左：Val（Camera + Map + Route），右：Test（Camera only）。両パネルは別集団の評価である。</figcaption>
</figure>

### チェックポイント内部検証

<a class="tabref" href="#tab-internal"></a> は内部検証プロトコルの結果であり，外部再生とは別の順位表として扱う。nuPlan チェックポイントは nuPlan 内部検証（1,024 サンプル）で 6.4 s の ADE 約 1.26 m，FDE 約 3.85〜3.90 m，KITScenes チェックポイントは KITScenes 凍結検証（3,820 サンプル）で ADE 2.63〜2.75 m，FDE 7.56〜7.86 m である。異なるデータセット上の数値であるため，nuPlan と KITScenes の行同士も比較しない。経路再構成 IoU はいずれも 0.99 以上であるが，これは再構成ヘッドが経路由来の特徴を受け取っている以上，主に経路情報が表現内で保存されていることを検証する指標であり，経路に沿って走行したことを意味しない。

<figure class="table" id="tab-internal">
<figcaption>チェックポイント内部検証（プロトコル A，6.4 s，64 ステップ）。nuPlan 行は nuPlan 内部検証，KITScenes 行は KITScenes 凍結検証であり，行間の比較は同一データセット内に限る。</figcaption>
<table>
<thead><tr><th>モデル</th><th class="num">n</th><th class="num">ADE 6.4 s</th><th class="num">FDE 6.4 s</th><th class="num">経路再構成 IoU</th></tr></thead>
<tbody>
<tr><td>nuPlan Epoch 4</td><td class="num">1,024</td><td class="num best">1.2578</td><td class="num best">3.8450</td><td class="num">0.9912</td></tr>
<tr><td>nuPlan Epoch 5</td><td class="num">1,024</td><td class="num">1.2668</td><td class="num">3.8982</td><td class="num best">0.9980</td></tr>
<tr class="group"><td>KITScenes Epoch 5</td><td class="num">3,820</td><td class="num best">2.6326</td><td class="num best">7.5567</td><td class="num best">0.9991</td></tr>
<tr><td>KITScenes Epoch 7</td><td class="num">3,820</td><td class="num">2.7548</td><td class="num">7.8574</td><td class="num">0.9990</td></tr>
</tbody>
</table>
</figure>

KITScenes の 2 チェックポイントについては，<a class="tabref" href="#tab-internal-kit"></a> と <a class="figref" href="#fig-tradeoff"></a> に経路・走行可能領域・快適性の開ループ指標を示す。Epoch 7 は経路コリドー遵守率を（0.3959 → 0.4236），経路コリドー成功率（0.0608 → 0.0763），投影終点弧長プロキシ（44.7857 → 45.5856 m，比率 0.8725 → 0.8982），走行可能領域遵守率（0.7758 → 0.7988）へ増加させる一方，走行可能領域成功率（0.4442 → 0.4139）と快適率（0.7542 → 0.7466）を悪化させ，6.4 s の ADE/FDE も悪化する。快適性違反の内訳（付録 A.4）ではEpoch 7 はジャーク大きさ以外の違反率を低下させ，快適率の低下はジャーク大きさ違反の 0.2356 から 0.2442 への増加に対応し，縦・横加速度やヨーレートの違反は 2% 未満である。

<figure class="table" id="tab-internal-kit">
<figcaption>KITScenes 凍結内部検証（n = 3,820；経路系指標は経路有効な 2,911 サンプル）における開ループ指標。↑は大きいほど良い，↓は小さいほど良い。</figcaption>
<table>
<thead><tr><th>指標</th><th class="num">Epoch 5</th><th class="num">Epoch 7</th></tr></thead>
<tbody>
<tr><td>経路コリドー遵守率 ↑</td><td class="num">0.3959</td><td class="num best">0.4236</td></tr>
<tr><td>経路コリドー成功率 ↑</td><td class="num">0.0608</td><td class="num best">0.0763</td></tr>
<tr><td>投影終点弧長プロキシ [m] ↑</td><td class="num">44.7857</td><td class="num best">45.5856</td></tr>
<tr><td>投影終点弧長プロキシ比率 ↑</td><td class="num">0.8725</td><td class="num best">0.8982</td></tr>
<tr><td>走行可能領域遵守率 ↑</td><td class="num">0.7758</td><td class="num best">0.7988</td></tr>
<tr><td>走行可能領域成功率 ↑</td><td class="num best">0.4442</td><td class="num">0.4139</td></tr>
<tr><td>快適率 ↑</td><td class="num best">0.7542</td><td class="num">0.7466</td></tr>
<tr><td>快適性違反率 ↓</td><td class="num best">0.2458</td><td class="num">0.2534</td></tr>
<tr class="group"><td>ADE 6.4 s [m] ↓</td><td class="num best">2.6326</td><td class="num">2.7548</td></tr>
<tr><td>FDE 6.4 s [m] ↓</td><td class="num best">7.5567</td><td class="num">7.8574</td></tr>
</tbody>
</table>
</figure>

<figure class="fig col-span-2" id="fig-tradeoff">
<img src="figures/fig_tradeoff_ep5_ep7.svg" alt="Epoch 5 vs Epoch 7 trade-off">
<figcaption>KITScenes Epoch 5 から Epoch 7 への相対変化（内部検証）。緑は各指標の望ましい方向への変化，赤は望ましくない方向への変化。経路系プロキシは改善し，長期の軌跡誤差・走行可能領域成功率・快適率は悪化する。</figcaption>
</figure>

### 数値的健全性と被覆率

<a class="tabref" href="#tab-integrity"></a> に外部再生の被覆率を示す。Val では 11,035 サンプル × 64 ステップ = 706,240 ステップのうち 551,750，Test では 23,690 × 64 = 1,516,160 のうち 1,184,500 が有効で，被覆率はいずれも 78.125%（50/64）である。非有限予測は 8 通りの外部評価すべてで 0 であった。

<figure class="table" id="tab-integrity">
<figcaption>外部オーバーレイ再生の評価健全性（4 モデル共通）。</figcaption>
<table>
<thead><tr><th>データセット</th><th class="num">サンプル / モデル</th><th class="num">有効ステップ</th><th class="num">総ステップ</th><th class="num">被覆率</th><th class="num">非有限予測</th></tr></thead>
<tbody>
<tr><td>KITScenes Val v3.5</td><td class="num">11,035</td><td class="num">551,750</td><td class="num">706,240</td><td class="num">78.125%</td><td class="num">0</td></tr>
<tr><td>KITScenes Test v1.0</td><td class="num">23,690</td><td class="num">1,184,500</td><td class="num">1,516,160</td><td class="num">78.125%</td><td class="num">0</td></tr>
</tbody>
</table>
</figure>

### 定性結果

<a class="figref" href="#fig-qual"></a> は，公開された制御オーバーレイをサンプル自身の地図・経路ラスタ上に再生した 6 例である。選択は，シャードアーティファクトを名前順に並べ，先頭 30 シャードの中央インデックスのサンプルを走査し，（a）Epoch 5 の 5 s FDE が最小と（b）最大の例，（c）最初の左折，（d）最初の右折，（e）最初の交差点前直進，（f）最初の交差点なし直進という規則で行った（付録 A.5）。誤差による事後選別はこの規則の外では行っていない。例（a）では Epoch 5 が右折を 0.47 m の FDE で追従する一方で Epoch 7 は 5.29 m 外れ，例（d）の環状交差点では Epoch 7（3.90 m）が Epoch 5（6.50 m）を上回る。例（b）では 4 モデルとも記録軌跡より長く進み（FDE 9.9〜18.5 m），減速の予測に失敗している。これらは単一サンプルの観察であり，統計的結論ではない。

<figure class="fig col-span-2" id="fig-qual">
<img src="figures/fig_qualitative.svg" alt="Qualitative examples">
<figcaption>KITScenes Val v3.5 の定性例。上段は各サンプルのモデル入力前方タイル（中央クロップ），下段は地図・経路ラスタ上の記録軌跡（黒，5 s）と 4 チェックポイントの再生軌跡（先頭 50 ステップ）。パネル内に 5 s FDE を記す。選択規則は本文と付録 A.5 に従い，誤差による事後選別を含まない。</figcaption>
</figure>

## 考察

### Epoch 5 と Epoch 7

プロトコル B では，Epoch 5 から Epoch 7 への追加学習が 1 s 誤差を減少させ（ADE 0.1472 → 0.1347 m），2 s 以降の精度を悪化させた（5 s ADE 1.9405 → 2.0952 m，FDE 5.5645 → 5.9509 m）。プロトコル A では同じ追加学習が経路コリドー遵守率と上限付き投影終点弧長プロキシを増加させる一方，6.4 s 誤差を増加させ快適率を低下させた。これらは異なるプロトコル・サンプル集団での直接観測であり，共同順位ではない。このチェックポイント対では，単一の軌跡指標または経路指標で選ぶと別の報告指標が後退する。チェックポイント選択は多目的であるべきであり，登録簿は定義未記録の選択スコアで Epoch 5 を最良とするが，Epoch 7 はプロトコル B の 1 s 平均が小さく，プロトコル A の上限付き投影終点弧長プロキシが大きい。統計的な差は主張しない。

### データセット間転移

同じ KITScenes Val シーン上で，KITScenes 微調整は nuPlan 学習チェックポイントを全ホライズンで上回った（RQ2）。nuPlan Epoch 4 と 5 の差は小さく，5 s の FDE では Epoch 4 の方がわずかに良い。カメラのみの Test では順位が変わり，nuPlan Epoch 5 の報告平均が 2 s と 3 s で最小となり，KITScenes Epoch 7 は Test 表内の 2・3・5 s で最大となる。KITScenes 微調整は地図と経路が与えられる条件で学習されており，それらが欠けた条件で相対的に脆くなる可能性は考えられるが，Test と Val は異なるシーン・地理・運動分布・目標同一性を含むため，この差を地図・経路の欠落の効果として定量化することはできない（RQ3）。

### 経路の由来と経路再構成

KITScenes の経路は，シーン全体の走行軌跡を Lanelet2 に適合させた車線列であり，目的地はシーン終端である（<a class="figref" href="#fig-route"></a>）。この経路は運転者が実際に選んだ車線レベルの意図に等しく，配備時にナビゲーションシステムから与えられる経路より情報量が多い場合がある。したがって主評価の数値は「記録された将来軌跡から事後構成した oracle 経路が与えられた場合」の性能として読むべきであり，経路誤りや経路再計画への頑健性は含まない。経路再構成 IoU が 0.999 に達するのは，再構成ヘッドがゲート後の経路寄与から経路ラスタを復元できる，すなわちゲートが経路情報を潰していないことの検証であって，方策が経路に沿って走行したことの証明ではない。振る舞いとしての経路遵守は経路コリドー遵守率・成功率が測っており，その値（遵守率 0.40〜0.42，成功率 0.06〜0.08）は 2.0 m 幅のフットプリントが 3.5 m 幅のコリドーに全ステップ収まるという厳しい定義の下で低い。

### 地図と経路の効果について言えること

本稿の証拠から，HD 地図と経路が計画精度を因果的に改善したとは言えない（RQ5）。理由は 5 つある。Val と Test は異なるシーンである。Test はデータセット構成上，地図と経路を持たない。現在の報告には，Val と同一シーンでのカメラのみベースラインが存在しない。外部オーバーレイ再生は経路の反事実入力を保存していない。シーン単位の対誤差とブートストラップ信頼区間はまだ計算されていない。

一方，設計上の仮説として次を提示する。HD 地図は車線トポロジー・交差点・停止線・走行可能境界の幾何学的曖昧さを減らすはずである。経路は，トポロジー的に妥当な複数の分岐のうち自車が進む分岐を一意に定めるはずである。地図と経路を独立に符号化することで，疎な選択経路が密な静的地図特徴に埋没することを防ぐはずである。残差かつゲート付きの融合は，事前学習された視覚 BEV 表現を壊さずにナビゲーション情報を加えるはずである。これらは対照実験が得られるまで仮説にとどまる。

### UniAD との関係

UniAD と AutoE2E はいずれも BEV 中心の表現と planning 指向の特徴共有を採る。UniAD は知覚・追跡・地図・動作予測・占有・計画をタスク相互作用で統合するが，AutoE2E は時系列カメラ BEV，明示的なナビゲーション文脈，補助経路再構成，直接の自車計画に範囲を絞り，外部から与えられる HD 地図と経路を配備入力の第一級市民として扱う。本稿の主張は，評価入力を配備方策の契約と一致させるべきだという点にあり，UniAD に対する優劣は共通ベンチマークがない以上主張しない。KITScenes 論文が報告する UniAD のゼロショット数値（200 サンプル，3 s）とも，サンプル集合・ホライズン・入力条件が異なるため比較しない。

## 限界

本稿の評価は開ループのみであり，閉ループの安全性・介入指標は含まない。同一シーンでの地図・経路の対照実験は行っていない。外部再生は 5 s までに限られ，6.4 s の出力のうち 22% は評価されていない。Val と Test は異なる集団であり，両者の差に因果的意味はない。カメラ符号器は凍結されており，nuScenes で学習された表現が KITScenes の 6 台のカメラ配置と 0.6 m 格子へどの程度適合しているかは別途評価していない（BEV 分割ヘッドは無効化されており，占有 IoU は計算していない）。UniAD の完全なタスクスタックは含まない。信頼区間はシーン単位データが未集計のため付けていない。 定速・自車状態のみ・記録軌跡などの参照ベースラインも含まないため，ADE/FDE の絶対値はタスク充足性ではなくチェックポイント間比較として読む。太字・下線は観測平均の数値順だけを示す。チェックポイントの品質ゲートは登録上すべて未通過であり，ゲート定義の詳細は本稿の証拠に含まれない（付録 A.3）。学習データは nuPlan とドイツ 3 都市の KITScenes であり，地理・季節・地図の分布シフトが存在し得る。経路は事後的に決まる走行済み車線列であるため，経路誤りへの頑健性は測られていない。KITScenes ベンチマークプロトコルでは自車運動履歴の 6.4 s のうち 4.0 s のみが実データであり，学習時（6.4 s）との分布差が残る。定性図の 6 例と <a class="tabref" href="#tab-channels"></a> の占有率は 30 サンプルの走査に基づく記述的観察であり，母集団の統計ではない。

## 結論

AutoE2E は，凍結した時系列カメラ BEV に分離符号化した HD 地図と oracle 事後経路をゲート付き残差で結合し，GRU で制御列を生成する。本稿はカメラ・地図・経路のモダリティを揃えつつ，実データの自車運動履歴は 4.0 s を 6.4 s へゼロ埋めし，出力は 5.0 s までを評価する条件で大規模に計測した。Camera + HD Map + Route の KITScenes Val プロトコルでは KITScenes Epoch 5 のプロトコル B 報告平均は 2・3・5 s で最小，Epoch 7 は 1 s で最小である。別のプロトコル A では Epoch 7 の上限付き投影終点弧長プロキシが大きい一方，6.4 s 誤差が大きく集約快適率が低い。カメラのみの Test は頑健性解析に有用だが，異なるシーンを含むため，ナビゲーション入力の因果的寄与を定量化できない。その主張には，同一シーンで対にした Camera / Camera + Map / Camera + Map + Route の評価が必要であり，本稿はその仕様を付録に示した。実運用整合の評価は方策が必要とするナビゲーション入力を保持すべきであり，カメラのみの評価はそれを補完する頑健性トラックとして位置づけるべきである。

## 謝辞{.acknowledgement}

Autoware Foundation Robotaxi Working Group の貢献者とコミュニティに感謝する。


## 参考文献{.reference}

<div class="reference" id="ref-uniad" lang="en">Y. Hu, J. Yang, L. Chen, K. Li, C. Sima, X. Zhu, S. Chai, S. Du, T. Lin, W. Wang, L. Lu, X. Jia, Q. Liu, J. Dai, Y. Qiao, and H. Li. Planning-Oriented Autonomous Driving. In <i>Proc. IEEE/CVF CVPR</i>, pp. 17853–17862, 2023. arXiv:2212.10156.</div>
<div class="reference" id="ref-bevformer" lang="en">Z. Li, W. Wang, H. Li, E. Xie, C. Sima, T. Lu, Y. Qiao, and J. Dai. BEVFormer: Learning Bird's-Eye-View Representation from Multi-camera Images via Spatiotemporal Transformers. In <i>Computer Vision – ECCV 2022</i>, LNCS 13669, pp. 1–18, Springer, 2022. doi:10.1007/978-3-031-20077-9_1.</div>
<div class="reference" id="ref-bevformerv2" lang="en">C. Yang, Y. Chen, H. Tian, C. Tao, X. Zhu, Z. Zhang, G. Huang, H. Li, Y. Qiao, L. Lu, J. Zhou, and J. Dai. BEVFormer v2: Adapting Modern Image Backbones to Bird's-Eye-View Recognition via Perspective Supervision. In <i>Proc. IEEE/CVF CVPR</i>, pp. 17830–17839, 2023. arXiv:2211.10439.</div>
<div class="reference" id="ref-deformable" lang="en">X. Zhu, W. Su, L. Lu, B. Li, X. Wang, and J. Dai. Deformable DETR: Deformable Transformers for End-to-End Object Detection. In <i>Proc. ICLR</i>, 2021. arXiv:2010.04159.</div>
<div class="reference" id="ref-vad" lang="en">B. Jiang, S. Chen, Q. Xu, B. Liao, J. Chen, H. Zhou, Q. Zhang, W. Liu, C. Huang, and X. Wang. VAD: Vectorized Scene Representation for Efficient Autonomous Driving. In <i>Proc. IEEE/CVF ICCV</i>, pp. 8340–8350, 2023. arXiv:2303.12077.</div>
<div class="reference" id="ref-transfuser" lang="en">K. Chitta, A. Prakash, B. Jaeger, Z. Yu, K. Renz, and A. Geiger. TransFuser: Imitation with Transformer-Based Sensor Fusion for Autonomous Driving. <i>IEEE Trans. Pattern Analysis and Machine Intelligence</i>, 45(11):12878–12895, 2023. doi:10.1109/TPAMI.2022.3200245.</div>
<div class="reference" id="ref-transfuser-cvpr" lang="en">A. Prakash, K. Chitta, and A. Geiger. Multi-Modal Fusion Transformer for End-to-End Autonomous Driving. In <i>Proc. IEEE/CVF CVPR</i>, pp. 7077–7087, 2021. arXiv:2104.09224.</div>
<div class="reference" id="ref-nuplan" lang="en">H. Caesar, J. Kabzan, K. S. Tan, W. K. Fong, E. Wolff, A. Lang, L. Fletcher, O. Beijbom, and S. Omari. nuPlan: A closed-loop ML-based planning benchmark for autonomous vehicles. <i>CVPR 2021 Workshop on Autonomous Driving: Perception, Prediction and Planning (ADP3)</i>, 2021. arXiv:2106.11810.</div>
<div class="reference" id="ref-nuscenes" lang="en">H. Caesar, V. Bankiti, A. H. Lang, S. Vora, V. E. Liong, Q. Xu, A. Krishnan, Y. Pan, G. Baldan, and O. Beijbom. nuScenes: A Multimodal Dataset for Autonomous Driving. In <i>Proc. IEEE/CVF CVPR</i>, pp. 11621–11631, 2020.</div>
<div class="reference" id="ref-kitscenes" lang="en">R. Schwarzkopf, F. Immel, A. Blumberg, J. Merkert, N. Rack, K. Wang, F. Konstantinidis, J. Truetsch, C. Fernandez, A. Bätz, K. Rösch, M. Steiner, W. Poh, Y. Shen, R. Wagner, F. Hauser, D. Strutz, J. Villa, G. Stepanov, H. Caesar, Ö. Ş. Taş, F. Bieder, J.-H. Pauls, and C. Stiller. The Road Ahead in Autonomous Driving: The KITScenes Multimodal Dataset. arXiv:2606.02956, 2026.</div>
<div class="reference" id="ref-lanelet2" lang="en">F. Poggenhans, J.-H. Pauls, J. Janosovits, S. Orf, M. Naumann, F. Kuhnt, and M. Mayr. Lanelet2: A High-Definition Map Framework for the Future of Automated Driving. In <i>Proc. 21st IEEE ITSC</i>, pp. 1672–1679, 2018. doi:10.1109/ITSC.2018.8569929.</div>
<div class="reference" id="ref-chauffeurnet" lang="en">M. Bansal, A. Krizhevsky, and A. Ogale. ChauffeurNet: Learning to Drive by Imitating the Best and Synthesizing the Worst. In <i>Proc. Robotics: Science and Systems (RSS)</i>, 2019. doi:10.15607/RSS.2019.XV.031.</div>
<div class="reference" id="ref-vectornet" lang="en">J. Gao, C. Sun, H. Zhao, Y. Shen, D. Anguelov, C. Li, and C. Schmid. VectorNet: Encoding HD Maps and Agent Dynamics From Vectorized Representation. In <i>Proc. IEEE/CVF CVPR</i>, pp. 11525–11533, 2020.</div>
<div class="reference" id="ref-lanegcn" lang="en">M. Liang, B. Yang, R. Hu, Y. Chen, R. Liao, S. Feng, and R. Urtasun. Learning Lane Graph Representations for Motion Forecasting. In <i>Computer Vision – ECCV 2020</i>, LNCS 12347, pp. 541–556, Springer, 2020. doi:10.1007/978-3-030-58536-5_32.</div>
<div class="reference" id="ref-admlp" lang="en">J.-T. Zhai, Z. Feng, J. Du, Y. Mao, J.-J. Liu, Z. Tan, Y. Zhang, X. Ye, and J. Wang. Rethinking the Open-Loop Evaluation of End-to-End Autonomous Driving in nuScenes. arXiv:2305.10430, 2023.</div>
<div class="reference" id="ref-egostatus" lang="en">Z. Li, Z. Yu, S. Lan, J. Li, J. Kautz, T. Lu, and J. M. Alvarez. Is Ego Status All You Need for Open-Loop End-to-End Autonomous Driving? In <i>Proc. IEEE/CVF CVPR</i>, pp. 14864–14873, 2024. arXiv:2312.03031.</div>
<div class="reference" id="ref-pdm" lang="en">D. Dauner, M. Hallgarten, A. Geiger, and K. Chitta. Parting with Misconceptions about Learning-based Vehicle Motion Planning. In <i>Proc. 7th Conference on Robot Learning (CoRL)</i>, PMLR 229, pp. 1268–1281, 2023.</div>
<div class="reference" id="ref-navsim" lang="en">D. Dauner, M. Hallgarten, T. Li, X. Weng, Z. Huang, Z. Yang, H. Li, I. Gilitschenski, B. Ivanovic, M. Pavone, A. Geiger, and K. Chitta. NAVSIM: Data-Driven Non-Reactive Autonomous Vehicle Simulation and Benchmarking. In <i>Advances in Neural Information Processing Systems 37 (NeurIPS 2024), Datasets and Benchmarks Track</i>, 2024. arXiv:2406.15349.</div>
<div class="reference" id="ref-hidden" lang="en">B. Jaeger, K. Chitta, and A. Geiger. Hidden Biases of End-to-End Driving Models. In <i>Proc. IEEE/CVF ICCV</i>, pp. 8240–8249, 2023. arXiv:2306.07957.</div>
<div class="reference" id="ref-resnet" lang="en">K. He, X. Zhang, S. Ren, and J. Sun. Deep Residual Learning for Image Recognition. In <i>Proc. IEEE CVPR</i>, pp. 770–778, 2016.</div>

## 付録{.appendix}

### 再現性

評価対象のモデル構成識別子は bevformer_v2_t8_split_navigation_v5 である。カメラ経路の初期値は公式 BEVFormer V2 R50 T8 チェックポイント（SHA-256 接頭辞 5585bc4d3ff8，重みライセンス NOASSERTION，学習データは nuScenes の CC-BY-NC-SA-4.0）であり，BEV クエリと行・列位置埋め込みを 300×200 へリサイズし，カメラ埋め込みを視点数に合わせ，検出・透視ヘッドを省いて読み込む。本稿のパラメータ数はこの構成を 6 視点でインスタンス化して数えた（<a class="tabref" href="#tab-params"></a>）。ラスタライザ・評価器・積分器のバージョン識別子は，navigation_rasterizer_v1，reactive_open_loop_metrics_v1，semi_implicit_unicycle_v1 である。KITScenes の出典は Hugging Face の KIT-MRT/KITScenes-Multimodal（データ改訂 6fde0034…，SDK 改訂 7765cdec…）である。外部評価インデックスと Val・Test マニフェストの内容ハッシュ接頭辞は be1cec5f77a7，4a12ede73949，b5dbe72c8d2c である。内部検証の各評価報告書はモデル登録簿にハッシュ付きで記録されており，本稿の内部検証数値は公開実験登録の記録と一致することを確認した。

### 評価識別子

外部評価は KITScenes Val v3.5（役割 official_val_camera_map_route，入力トラック camera_map_route，117 シーン，11,035 サンプル）と KITScenes Test v1.0（役割 official_test_camera_only_missing_map_route，入力トラック camera_only_missing_map_route，206 シーン，23,690 サンプル）である。推論精度方針は CUDA BF16 自動混合精度である。Val のライブ評価方針は事前計算済み画像 BEV を再利用する経路ゼロ化パスを宣言し，公開コードは集約値 route_zero_sample_count と route_zero_trajectory_delta_m を要求するため，生成時に集約的な経路ゼロ化パスは実行されたと判断できる。しかし外部再生アーティファクトにはサンプル単位の反事実制御・誤差がなく，集約差分も本稿が受け取った結果パケットに含まれない。このため値は報告せず，シーン単位差分と信頼区間を備えた A/B/C の 3 条件対照実験ともみなさない。Test は該当なしである。内部検証プロトコルは reactive_trajectory_route_validation_6p4s_v1（nuPlan）と kitscenes_internal_trajectory_route_validation_6p4s_v1（KITScenes）である。

### 品質ゲート

登録メタデータの eval_gate_pass は 4 チェックポイントすべてで false である。リポジトリの旧ワークフローは 6.4 s ADE < 2.0 m かつ FDE < 4.0 m という条件を定義する。報告した nuPlan 2 チェックポイントはこの 2 数値を満たすにもかかわらず登録値が false であるため，当該フィールドは旧条件だけから計算されたものではなく，本稿では解釈不能として扱う。他のゲート条件は推測しない。

### 快適性違反の内訳

<figure class="table" id="tab-comfort">
<figcaption>KITScenes 凍結内部検証（n = 3,820）における快適性違反率の内訳（公開実験登録の記録値）。違反はサンプル単位で，いずれかの成分が nuPlan 閾値を超えた場合に計上される。</figcaption>
<table>
<thead><tr><th>成分</th><th class="num">閾値</th><th class="num">Epoch 5</th><th class="num">Epoch 7</th></tr></thead>
<tbody>
<tr><td>縦加速度</td><td class="num">−4.05 / 2.40 m/s²</td><td class="num">0.0042</td><td class="num">0.0034</td></tr>
<tr><td>横加速度（v²κ）</td><td class="num">4.89 m/s²</td><td class="num">0.0147</td><td class="num">0.0126</td></tr>
<tr><td>ヨーレート（vκ）</td><td class="num">0.95 rad/s</td><td class="num">0.0089</td><td class="num">0.0079</td></tr>
<tr><td>ヨー加速度</td><td class="num">1.93 rad/s²</td><td class="num">0.1579</td><td class="num">0.1401</td></tr>
<tr><td>縦ジャーク</td><td class="num">4.13 m/s³</td><td class="num">0.0000</td><td class="num">0.0000</td></tr>
<tr><td>ジャーク大きさ</td><td class="num">8.37 m/s³</td><td class="num">0.2356</td><td class="num">0.2442</td></tr>
<tr class="group"><td>いずれかの違反</td><td class="num">—</td><td class="num">0.2458</td><td class="num">0.2534</td></tr>
</tbody>
</table>
</figure>

### 定性図と占有率統計の選択規則

公開ダッシュボードが列挙する KITScenes Val v3.5 の 140 シャードアーティファクトを名前の辞書順に並べ，先頭 30 シャードについてインデックスの中央位置（要素数の半分の整数部）のサンプルを 1 つずつ取得した。各サンプルについて地図・経路ラスタ，記録軌跡，ナビゲーションメタデータ，前方カメラタイル，4 チェックポイントの公開制御オーバーレイ（スキーマ v5，シード数 1）を取得し，式 <a class="eqref" href="#eq-roll"></a> で再生した。30 サンプルの内訳は直進 22，右折 4，左折 2，不明 2，経路有効 28，地図有効 30，目的地可視 26，有効将来ステップはすべて 50 である。<a class="figref" href="#fig-channels"></a> は非空チャネル数が最大のサンプル，<a class="figref" href="#fig-qual"></a> の 6 例は本文の規則で選び，誤差による追加の選別は行っていない。走査の Epoch 5 の 5 s FDE は 0.47〜14.25 m に分布した。この 30 サンプルは母集団の代表標本として設計されたものではない。

### ラスタジオメトリに関する注記

KITScenes v3 で監査された公開ナビゲーションジオメトリは 256×256 セル，1.0 m/px（X：−85.5〜170.5 m，Y：−128〜128 m）であり，6.4 s の終点の 99.79% を覆う。一方，評価対象チェックポイントが受け取る KITScenes Val v3.5 のラスタは 450×300 セル，0.4 m/px（X：−60〜120 m，Y：−60〜60 m）であることを，公開データセットアーティファクトのナビゲーションメタデータ（geometry_id = autoe2e-bev-450x300-0p4m-v1）と配列形状（14×450×300，2×450×300）で確認した。ナビゲーション符号器はこれを 300×200 の潜在格子へ双線形リサンプルするため，カメラ BEV との範囲は一致する。BEV 分割の評価では 1.0 m ジオメトリのラベルを 0.4 m 格子へ最近傍リサンプルするが，本稿では BEV 分割の数値を報告しない。

### 主張監査

<figure class="table col-span-2" id="tab-claims">
<figcaption>主要な主張とその証拠の種別。種別：実装＝ソースコードとインスタンス化，登録＝チェックポイント登録メタデータと公開実験登録，計測＝完了した評価の数値，解釈＝計測から支持される解釈，仮説＝未検証。</figcaption>
<table class="small">
<thead><tr><th>主張</th><th>種別</th><th>証拠</th></tr></thead>
<tbody>
<tr><td>カメラ経路は公式 BEVFormer V2 R50 T8 に基づき，軌跡学習段階で凍結される</td><td>実装・登録</td><td>初期化コードと freeze 契約；学習設定 freeze_bevformer = true</td></tr>
<tr><td>BEV 格子 300×200，X −60〜120 m，Y −60〜60 m，0.6 m セル</td><td>実装</td><td>固定モデル構成；インスタンス化で確認</td></tr>
<tr><td>7 履歴フレーム＋現在フレーム，0.5 s 間隔，3.5 s；履歴は 512 px のみ</td><td>実装</td><td>T8 契約定数と履歴符号化コード</td></tr>
<tr><td>1024 px 前方分岐は現在フレームのみ追加，512 px 前方視点は全視点符号器に残る</td><td>実装</td><td>前方残差分岐コード；ゲート zero-init</td></tr>
<tr><td>地図 14 ch，経路 2 ch，コリドー 3.5 m，目的地半径 2.0 m</td><td>実装</td><td>チャネル列挙とラスタライザ；実データ配列形状</td></tr>
<tr><td>評価対象の地図・経路ラスタは 450×300，0.4 m/px</td><td>登録・実装</td><td>Val v3.5 サンプルの geometry_id と配列形状；データ契約検証コード</td></tr>
<tr><td>変形可能融合は 8 点・8 ヘッド・256 ch，初期状態で恒等</td><td>実装</td><td>融合モジュールの zero-init</td></tr>
<tr><td>GRU プランナ，16 サンプル点，64 ステップ，加速度・曲率出力</td><td>実装</td><td>プランナ構成 num_points = 16</td></tr>
<tr><td>全 79,906,522 パラメータ，学習対象 1,448,582</td><td>実装</td><td>本稿のインスタンス化計数；リポジトリ記録 79,907,034（8 視点）と整合</td></tr>
<tr><td>enable_world_model = false，temporal_memory_mode = no_memory；h_vis はゼロ</td><td>実装・登録</td><td>固定構成；KITScenes サンプル生成コード</td></tr>
<tr><td>4 チェックポイントの学習設定（8 ワーカ，seed 149，重み 1.0/1.0/0.0，LR）</td><td>登録</td><td>チェックポイント登録メタデータ；公開実験登録の学習率記録</td></tr>
<tr><td>eval_gate_pass = false（全 4 チェックポイント）</td><td>登録</td><td>登録メタデータ；ゲート定義は未確認</td></tr>
<tr><td>Val で Epoch 7 は 1 s 最良，Epoch 5 は 2/3/5 s と横・縦誤差で最良</td><td>計測</td><td><a class="tabref" href="#tab-val-ade"></a>，<a class="tabref" href="#tab-latlon"></a></td></tr>
<tr><td>Test で Epoch 5 は 1 s・5 s 最良，nuPlan Epoch 5 は 2 s・3 s 最良</td><td>計測</td><td><a class="tabref" href="#tab-test-ade"></a></td></tr>
<tr><td>非有限予測 0，被覆率 78.125%</td><td>計測</td><td><a class="tabref" href="#tab-integrity"></a></td></tr>
<tr><td>Epoch 7 は経路系プロキシを改善し，6.4 s 誤差・走行可能成功率・快適率を悪化</td><td>計測</td><td><a class="tabref" href="#tab-internal-kit"></a>；公開実験登録の値と一致</td></tr>
<tr><td>追加学習は直近精度を改善し中・長期精度を悪化させる</td><td>解釈</td><td>Val の 1 s と 2〜5 s の順位反転</td></tr>
<tr><td>チェックポイント選択は多目的であるべき</td><td>解釈</td><td>Epoch 5/7 の指標間トレードオフ</td></tr>
<tr><td>経路再構成 IoU は情報保存の検証であり経路遵守ではない</td><td>解釈</td><td>ヘッドの入力がゲート後経路寄与であること；遵守率との乖離</td></tr>
<tr><td>Test の結果は地図が不要であることを示さない</td><td>解釈</td><td>Val/Test のシーン集団の相違</td></tr>
<tr><td>HD 地図・経路は幾何的曖昧さと分岐を解消するはずである</td><td>仮説</td><td>対照実験未実施</td></tr>
<tr><td>分離符号化と残差ゲート融合の利点</td><td>仮説</td><td>対照実験未実施</td></tr>
<tr><td>地図・経路が計画精度を因果的に改善した</td><td>主張しない</td><td>同一シーン対照なし；反事実未保存；信頼区間未計算</td></tr>
</tbody>
</table>
</figure>

### 因果的主張に必要な追加実験

<figure class="fig col-span-2" id="fig-ablation">
<img src="figures/fig15_matched_ablation.svg" alt="Matched ablation design">
<figcaption>地図・経路の効果を確立するために必要な同一シーン対照実験の設計。同一の 11,035 KITScenes Val サンプルと同一の凍結チェックポイントで，事前計算した同一のカメラ BEV を再利用し，ナビゲーション入力のみを A（カメラのみ），B（カメラ＋HD 地図），C（カメラ＋HD 地図＋経路）と切り替える。本稿はこの実験の結果を持たない。</figcaption>
</figure>

同一の 11,035 KITScenes Val サンプルと同一の凍結チェックポイントを，A：カメラのみ，B：カメラ＋HD 地図，C：カメラ＋HD 地図＋経路の 3 条件で対にして評価する。3 条件で同一の事前計算カメラ BEV を再利用し，ナビゲーション入力のみを変える。切り替えには明示的な map_valid，route_valid，enable_route_conditioning ゲートを用い，確率的な拡張は再実行せず，プランナ設定は固定する。報告項目は，1・2・3・5 s の ADE/FDE，平均絶対横方向・縦方向誤差，経路コリドー遵守率・成功率，投影終点弧長プロキシ，走行可能領域遵守率・成功率，快適率と各違反成分，失敗率と非有限予測，可能なら推論遅延とメモリである。シーン単位の対ブートストラップ信頼区間と，シーンごとの差分の分布を報告する。追加の対照として，別シーンの経路を与えるシャッフル経路，目的地マーカー除去，経路コリドー除去，地図のみの破損・ドロップアウト，経路外・曖昧交差点サブセット，有効地図・無効地図の層別を行う。これらの結果は本稿に含まれず，予測もしない。
