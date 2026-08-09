# SRA/DTS/AS: ManiFlow向けSRA/SiT-SRA_DTS_AS方式の自己蒸留

このドキュメントは、`ManiFlowTransformerImagePolicy`（`Flow_Policy/scripts/train_eval_libero.sh` が使う、DiTXベースのflow-matchingポリシー）が、既に実装済みだった素朴なSRAベースライン（`use_sra`、以前のコミットで追加）だけでなく、`SRA/SiT-SRA_DTS_AS`（`SRA/SiT-SRA_DTS_AS/loss.py` + `model.py`）と同じself-flow／トークンマスキングによる自己蒸留アルゴリズムを実行できるようにするために行った、学習アルゴリズム側の変更をまとめたものです。

「DTS/AS」= **D**ual-**T**ime **S**cheduling（デュアルタイムスケジューリング）+ **A**ttention **S**eparation（アテンション分離）の略で、`SiT-SRA_DTS_AS` が素のSRAに追加している2つの仕組みです。

## 1. SRA/SiT-SRA_DTS_AS が行っていること（画像ドメイン）

`SRALoss.__call__`（`SRA/SiT-SRA_DTS_AS/loss.py`）:

1. 各画像のパッチトークンを、サンプルごとにランダムな割り当てでグループに分割する（`mask_ratio`、例えば `0.25` → グループ `[0.25, 0.75]`）。
2. **デュアルタイムスケジューリング（DTS）**: 画像全体で共有する単一のタイムステップの代わりに、グループごとに独立した拡散タイムステップをサンプリングする。これにより、1枚の学習画像が複数のノイズレベルのパッチワークになる。
3. 各パッチについて、そのパッチが属するグループ自身のノイズ付き版を採用することで、1つの混合入力を作る。この混合された複数ノイズレベルの画像こそが、（補助的な分岐ではなく）*主となる* denoising loss の計算対象になる。
4. **アテンション分離（AS）**: 異なるグループのパッチ間のself-attentionを（`model.py` の `GroupSeparatedAttention` によって）任意で遮断する。これにより、あるトークンは実質的に自分自身のグループのノイズレベルだけを「見る」ことになる。
5. teacher/EMA分岐が、選択された「よりクリーンな」タイムステップ（`teacher_t`: `self_flow` / `same` / `sra`）で比較用の表現（`xr_t`）を生成する。これは任意でマスクされることもある（`teacher_mask`）。studentとteacherの中間ブロック出力の間の自己整合損失（`Simpleloss`、例えばsmooth-L1やコサイン）が、メインのdenoising lossに加算される。
6. `full_sample_prob` は、正則化として、2グループの混合サンプルを通常の単一ノイズレベルのサンプルにたまに置き換える。

## 2. ManiFlowの行動系列ドメインへの対応付け

ManiFlowには空間的なパッチが存在しないため、「トークンの系列」に相当するのは行動ホライズン（action-horizon）の次元 `T`（`DiTX` の `horizon`、LIBEROでは例えば10ステップ）です。上記の仕組みはすべて、この軸に対して再実装されています。

| SiT-SRA_DTS_AS（画像パッチ）                        | ManiFlow（行動ホライズンのトークン）                          |
|----------------------------------------------------|------------------------------------------------------------|
| パッチトークンのグルーピング（`mask_ratio`）           | `T` 個の行動ホライズン位置に対するトークンごとのグループID          |
| デュアルタイムスケジューリング                          | グループごとに独立したflowタイムステップ `t`                     |
| `GroupSeparatedAttention`（グループ間attentionの遮断） | `nn.MultiheadAttention` に渡すbooleanの `attn_mask`           |
| teacher_t ∈ {self_flow, same, sra}                  | 同じ3モード（下記の符号反転の注意点を参照）                        |
| `teacher_mask`, `full_sample_prob`                  | 同じ意味・同じ名前（`sra_teacher_mask`, `sra_full_sample_prob`） |

**方向の規約が異なるため、その差を吸収している。** SiTの順方向パスはクリーン（`t=0`）→ノイズ（`t=1`）で進むのに対し、ManiFlowのflowはノイズ（`t=0`）→クリーン/ターゲット（`t=1`）で進む（`linear_interpolate` を参照）。そのため、SiTが「よりクリーンな」ビューを得るために *より小さい t の方向* に動かす箇所（`sra` オフセットは区間を減算し、`self_flow` teacherはグループのタイムステップの `min` を取る）では、ManiFlow版は代わりに *より大きい t の方向* に動かす（`+delta` を1でクランプ、グループタイムステップの `max` を取る）。これは今回の変更以前から素のSRAコードで既に使われていた規約であり、DTS/ASもそれを一貫して踏襲している。

## 3. 新規追加したファイル・関数

### `maniflow/model/diffusion/sra_mask.py`（新規）

`loss.py` の `_normalize_mask_ratios`、`_compute_group_counts`、`_build_group_ids_from_counts` を移植した、純粋なテンソル演算のヘルパー群。空間方向のブロードキャスト用の `(B,T,1,1)` から、時間方向のブロードキャスト用の `(B,T)` に一般化している。

- `normalize_mask_ratios(mask_ratio)` - スカラーまたはリストを受け取り、合計が1になる比率のリストを返す。単一の値が `<= 0` または `>= 1`（デフォルトの `1.0` を含む）の場合は `[1.0]` に正規化される、つまり**マスキング無効**（グループ1つ）を意味する。
- `build_group_mask(mask_ratio, batch_size, seq_len, device, full_sample_prob)` - `(B, T)` のlong型グループIDテンソルを構築する。`full_sample_prob` によるオーバーライド（一部のバッチ行全体をグループ1に強制する）も含む。
- `mix_group_values(group_values, group_ids)` - 汎用的なトークン単位のgather処理。`(B,T)` テンソル（混合タイムステップ）にも `(B,T,D)` テンソル（混合された軌道）にも使える。
- `build_attention_separation_mask(group_ids, num_heads)` - `nn.MultiheadAttention` が期待する `(B*num_heads, T, T)` のbooleanマスクを構築する（`True` = 遮断。SiTの `GroupSeparatedAttention` の `True` = 許可、という規約とは逆になっている点に注意）。

### `maniflow/model/diffusion/ditx.py` / `ditx_block.py`

- `DiTX.forward` は `timestep` / `target_t` を `(B,)` **または `(B,T)`** として受け取れるようになった。`(B,T)` テンソルを渡すと、行動ホライズンの各位置がそれぞれ独自の値を持てる。これがDTSで混合されたタイムステップ系列を入力する仕組みである。トークンごとの埋め込みは `(B*T,)` にフラット化してから埋め込み、reshapeして戻すことで計算する。
- トークンごとのtime/target_t埋め込み（`time_c`）は、常に `(B, T, n_emb)` になった（以前は `(B, n_emb)` で、各ブロック内部で `.unsqueeze(1)` により遅延的にブロードキャストしていた）。それに合わせて `DiTXBlock.modulate()` とgate部分の乗算は、単純な要素ごとの演算に簡略化した。これは通常の（マスクなしの）学習に対しては**挙動を一切変えない純粋なリファクタリング**であり、変更前のモデルと（同じseed・同じforward入力で `max abs diff == 0.0`）ビット単位で一致することを確認済み。
- `DiTX.__init__` に `attention_separation: bool` を追加した。`DiTX.forward` には `group_ids: (B,T)`（任意）を追加し、両方が設定されている場合は、（`build_attention_separation_mask` で構築した）グループ間self-attentionマスクが各 `DiTXBlock` に渡される。vis/lang contextへのcross-attentionには影響しない（パッチのself-attentionのみを分離するSiT-SRA_DTS_ASと一致）。`attention_separation=False` のときの `group_ids` は何もしない（no-op）――`sample`/`timestep`/`target_t` の実際のトークン混合は `DiTX` の内部ではなく、（後述するように）その前段のポリシー側で行われる。
- `DiTXBlock.forward` に既にあった `attn_mask` パラメータ（以前は受け取るが常に `None` で呼ばれていた）が、実際に配線されて機能するようになった。

### `maniflow/policy/maniflow_image_policy.py`

新しいコンストラクタ引数（すべて `sra_` プレフィックス、デフォルト値は従来の素のSRA挙動を厳密に再現する）:

| パラメータ | デフォルト | 意味 |
|---|---|---|
| `sra_mask_ratio` | `1.0` | マスク比率。`1.0` → グループ1つ（マスキングoff） |
| `sra_dual_time_scheduling` | `False` | グループごとに独立したタイムステップにするか、1つの共有タイムステップにするか |
| `sra_attention_separation` | `False` | グループ間self-attentionを遮断するか |
| `sra_teacher_mask` | `False` | teacherがstudentと同じグループ混合入力を見るか |
| `sra_full_sample_prob` | `0.0` | マスクされたサンプルを単一グループに戻す（潰す）確率 |
| `sra_teacher_t` | `"sra"` | `"self_flow"` \| `"same"` \| `"sra"` |
| `sra_loss_type` | `"sml1"` | `"sml1"` \| `"l2"` \| `"l1"` \| `"cos"`（SiTの `Simpleloss` の選択肢） |

新しいメソッド:

- `_sample_sra_group_timesteps` - グループごとに1回 `sample_t(...)` を呼ぶ（DTS）か、DTSがoffの場合は全グループで同じ値を使い回す。
- `_build_sra_dts_mix` - グループID、グループごとの `x_t`/タイムステップ/`target_t` を構築し、それらを1つの `(B,T,...)` 系列に混合する（`_build_sra_dts_mix` は `SRALoss.__call__` の「maskが `None` でない」分岐のManiFlow版に相当する）。
- `_compute_sra_teacher_repr` - teacher/EMAのforward計算。`teacher_t` の3モード×`teacher_mask` のon/offの全組み合わせをカバーし、`loss.py` の分岐（`teacher_mask=False` の場合は `teacher_t="self_flow"` のみサポートし、それ以外の組み合わせは `NotImplementedError` を送出するという制限も含めて）と一致させている。これは `SRA/SiT-SRA_DTS_AS` そのままの挙動である。
- `_sra_align_loss` - `Simpleloss` の移植（sml1/l2/l1/cos）。

`compute_loss` は
`sra_active_mask = self.use_sra and self.sra_num_mask_groups > 1`
で分岐するようになった:

- **`sra_active_mask=True`**: *主となる* flow-matchingのforward計算自体が、DTSで混合された複数ノイズレベルの系列を使う（SiT-SRA_DTS_ASと同様、マスキングがalignment lossだけでなくメインの学習信号そのものを変える）。flow-matchingのターゲット `v = x_1 - x_0` はManiFlowの線形パス上でタイムステップに依存しないため、グループごとの混合は不要。
- **`sra_active_mask=False`**（デフォルト）: 元の素のSRAのコードパスをそのまま（single timestep、group idsなし、attention separationなし）実行する。唯一の一般化は、元々ハードコードされていた `"sra"` teacherオフセットの代わりに `sra_teacher_t="self_flow"` も選べるようにした点のみ。

## 4. 有効化方法

`use_sra` は引き続きマスタースイッチである（デフォルトはどの設定でも `false` のままで、既存configの挙動は変わらない）。`maniflow_image_timm_policy_libero.yaml` には、SiT-SRA_DTS_ASが推奨するDTS/ASのデフォルト値を設定済み（依然として `use_sra` の配下）:

```yaml
use_sra: false            # trueにすると有効化
sra_mask_ratio: 0.25       # -> グループ [0.25, 0.75]
sra_dual_time_scheduling: true
sra_attention_separation: true
sra_teacher_mask: true
sra_full_sample_prob: 0.0
sra_teacher_t: "self_flow"
sra_loss_type: "sml1"
```

実行例:

```bash
bash scripts/train_eval_libero.sh 0 42 sra_dts_as_run train libero_spatial policy.use_sra=true
```

## 5. 検証内容

- `DiTX` 単体のforwardテスト: 通常のスカラータイムステップ経路、attention separationなしのトークンごと（DTS）タイムステップ、attention separationあり＋`group_ids` の各パターンを確認（出力shape、グループ比率の妥当性、`mix_group_values`/`normalize_mask_ratios` の正しさをチェック）。
- `ManiFlowTransformerImagePolicy.compute_loss` をエンドツーエンドで実行（lossが有限かつ勾配が非ゼロであることを確認）: no-SRA、素のSRA（`sra`/`self_flow` teacher、マスキングなし）、および `sra_teacher_t ∈ {self_flow, same, sra}` × `sra_teacher_mask ∈ {True, False}` × `sra_full_sample_prob > 0` × `sra_loss_type="cos"` のDTS/AS全組み合わせ。ドキュメント通り、`teacher_mask=False` かつ `self_flow` 以外の組み合わせでは `NotImplementedError` が送出されることも確認済み。
- `DiTX` のトークンごとtime埋め込みへのリファクタリングが、通常の（マスクなしの）学習に対して変更前の実装と**ビット単位で一致**することを確認（同一seed・同一入力で `max abs diff == 0.0`）。
