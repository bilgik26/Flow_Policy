# wandb ログ指標一覧（RoboCasa 学習）

学習スクリプト: `maniflow/workspace/train_maniflow_robocasa_workspace.py`

wandb には 1 ステップごとの batch ログ（学習中）と、エポック終了時のまとめログの 2 タイミングで記録される。  
数値メトリクスは合計 18 種類。rollout 動画（`sim_video_eval_N`）と `task_name`（文字列）は別途ログされる。

---

## 1. 学習ステップごとに記録される値（毎 batch）

| キー | 内容 | ソース |
|------|------|--------|
| `train_loss` | バッチの総損失（= `bc_loss` と同値）（= loss_flow + loss_ct + loss_sra） | `raw_loss.item()` |
| `global_step` | 全学習ステップ通算カウンタ | `self.global_step` |
| `epoch` | 現在のエポック番号 | `self.epoch` |
| `lr` | 現在の学習率 | `lr_scheduler.get_last_lr()[0]` |
| `loss_flow` | Flow Matching 損失（常に計算） | `F.mse_loss(v_flow_pred, v_flow_target)` |
| `loss_ct` | Consistency Training 損失（`use_consistency=False` のとき 0.0） | `F.mse_loss(v_ct_pred, v_ct_target)` |
| `loss_sra` | SRA 損失（`use_sra=False` のとき 0.0） | `F.smooth_l1_loss(student_repr, teacher_repr)` |
| `v_flow_pred_magnitude` | flow 予測ベクトルの RMS(Root Mean Square)（予測スケールの監視用） | `sqrt(mean(v_flow_pred²))` |
| `v_ct_pred_magnitude` | consistency 予測ベクトルの RMS（`use_consistency=False` のとき 0.0） | `sqrt(mean(v_ct_pred²))` |
| `bc_loss` | 総損失（`train_loss` と同値） | `loss.item()` |

---

## 2. エポック終了時に記録される値

| キー | 内容 | 記録タイミング |
|------|------|----------------|
| `train_loss` | エポック全 batch の損失平均（ステップ値を上書き） | 毎エポック |
| `n_skipped_batches` | NaN/Inf でスキップした batch 数 | 毎エポック |
| `val_loss` | 検証データの損失平均 | `val_every=50` エポックごと |
| `n_val_nan_batches` | 検証中に NaN になった batch 数 | NaN 発生時のみ |
| `train_action_mse_error` | 学習 batch に対する行動予測 MSE（過学習の監視用） | `sample_every=5` エポックごと |
| `test_mean_score` | rollout 成功率（rollout 非実行時は `-train_loss` で代替） | 毎エポック |

---

## 3. Rollout 時に追加記録される値（`rollout_every=500` エポックごと）

| キー | 内容 |
|------|------|
| `mean_success_rates` | 評価エピソード（20 件）の平均成功率 |
| `SR_test_L3` | これまでの rollout 成功率の上位 3 回の平均 |
| `SR_test_L5` | これまでの rollout 成功率の上位 5 回の平均 |

> `SR_test_L3`/`SR_test_L5` は学習後半の安定した性能を測るための指標。  
> rollout の各エピソード動画も `sim_video_eval_0` ～ `sim_video_eval_19` として wandb にアップロードされる。

---

## 現在の学習設定（1task / flow-only）

- `use_consistency=False` → `loss_ct`, `v_ct_pred_magnitude` は常に 0.0
- `use_sra=False` → `loss_sra` は常に 0.0
- rollout エポック: 0, 500（501 エポック中 2 回）
