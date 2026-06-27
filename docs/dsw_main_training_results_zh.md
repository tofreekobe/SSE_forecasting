# DSW 全量主模型训练结果

更新时间：2026-06-28

## 数据口径

- 原始数据：6000 个慢滑移事件，`79,673,958,000` bytes
  (`74.202 GiB`)。
- 训练输入：已逐事件审计通过的压缩分片包
  `/mnt/workspace/hf_dataset_package_verified_b1f13c4`。
- 审计结论：6000/6000 事件 raw-vs-package 精确比对通过，slip/GNSS 最大差异为 0。

## DSW 环境

- 实例：PAI-DSW `final_sse`
- GPU：NVIDIA A10（`nvidia-smi` 当前实测显存约 23 GiB）
- 代码包：`/mnt/workspace/sse_codex_b1f13c4`
- 输出目录：
  `/mnt/workspace/sse_outputs/experiment_matrix_b1f13c4_full`

## 主模型设置

- 模型：`segmented_residual`
- 输入：`history_slip + history_gnss`
- 目标：50-step `future_slip`
- `forecast_start=60`
- `forecast_horizon=50`
- `epochs=50`
- `batch_size=16`
- `hidden_channels=64`
- `m0_loss_weight=0.005`
- AMP：开启

## 结果表

| Split | Test h50 RMSE | Persistence h50 RMSE | RMSE improvement | Test R2 | Model M0 rel abs | Persistence M0 rel abs | M0 change | Gate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| random | 0.00142173 | 0.0592355 | 97.60% | 0.999408 | 0.0155652 | 0.948833 | -98.36% | PASS |
| blocked | 0.00152041 | 0.0608568 | 97.50% | 0.999359 | 0.0166794 | 0.948833 | -98.24% | PASS |

## 已回收消融矩阵进度

| Run | Split | Model | Input | M0 loss | Test h50 RMSE | Persistence | Gain | R2 | M0 rel abs | Gate |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| main_residual_full | random | segmented_residual | full | 0.005 | 0.00142173 | 0.0592355 | 97.60% | 0.999408 | 0.0155652 | PASS |
| main_residual_full | blocked | segmented_residual | full | 0.005 | 0.00152041 | 0.0608568 | 97.50% | 0.999359 | 0.0166794 | PASS |
| ablate_gnss_only | random | segmented_residual | gnss_only | 0.005 | 0.06016635 | 0.0592355 | -1.57% | -0.060367 | 1.000000 | FAIL |
| ablate_no_gnss | random | segmented_residual | no_gnss | 0.005 | 0.00235053 | 0.0592355 | 96.03% | 0.998382 | 0.0277736 | PASS |
| ablate_no_gnss | blocked | segmented_residual | no_gnss | 0.005 | 0.00192682 | 0.0608568 | 96.83% | 0.998970 | 0.0173103 | PASS |
| model_plain_full | random | plain | full | 0.005 | 0.00431510 | 0.0592355 | 92.72% | 0.994546 | 0.0136518 | PASS |
| model_plain_full | blocked | plain | full | 0.005 | 0.00575492 | 0.0608568 | 90.54% | 0.990813 | 0.0281596 | PASS |
| model_segmented_full | random | segmented | full | 0.005 | 0.00473668 | 0.0592355 | 92.00% | 0.993428 | 0.0238614 | PASS |
| model_segmented_full | blocked | segmented | full | 0.005 | 0.00506467 | 0.0608568 | 91.68% | 0.992884 | 0.0323756 | PASS |

初步架构消融结论：`plain` 与 `segmented` 在已完成 split 上都明显超过 persistence，
但 h50 RMSE 约为 `segmented_residual` 的 3-4 倍。`plain/random` 的 RMSE 和 M0 误差略优于
`segmented/random`，但 `segmented/blocked` 的 RMSE 优于 `plain/blocked`。因此，几何拆分的
收益主要体现在 blocked 泛化侧，不能简单宣称它在所有指标上都更优。最稳妥的结论是：
residual 预测头和归一化/残差结构是高精度 future slip forecasting 的主要增益来源，
两子断层拆分则改善了 blocked split 的结构泛化。

初步输入消融结论：`no_gnss` 在 random 与 blocked 上分别达到 `0.00235053` 和
`0.00192682` 的 h50 RMSE，明显超过 persistence，也优于 plain/segmented 架构消融。
这说明在当前任务设定中 history slip 是最强信息源；不过 full input 主模型在 random 和
blocked 上分别进一步降至 `0.00142173` 和 `0.00152041`，说明 GNSS history 提供了稳定
边际增益，尤其体现在更低的滑移场 RMSE 和更稳的 M0 error 上。

`gnss_only/random` 是当前矩阵中的第一个明确失败项：h50 RMSE 为 `0.06016635`，
略差于 persistence，R2 为负，M0 relative absolute error 为 1。这说明仅凭三站 GNSS
历史无法在当前模型设定下恢复未来 slip field；本文主模型成功依赖 history slip 状态，
真实业务应用中需要由反演或数据同化系统提供该状态，不能声称 GNSS-only forecasting 已解决。

## 结论

主模型在 DSW 上通过 random 与 blocked 两个全量 split 的 h50 publication gate。
这确认了此前 `GO_WITH_CHANGES` 的判断：原始方案不能原样继续，但修正后的
data contract、`log1p` slip target、全局 GNSS 归一化、两子断层
`segmented_residual` 结构是可行的。

完整消融矩阵仍在后台继续运行，后续应补充：

- `gnss_only` blocked split
- `last_slip_only` random/blocked split
- `m0_loss_weight=0.005` vs `0`
