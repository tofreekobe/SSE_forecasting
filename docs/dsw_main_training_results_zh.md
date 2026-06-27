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
| model_segmented_full | random | segmented | full | 0.005 | 0.00473668 | 0.0592355 | 92.00% | 0.993428 | 0.0238614 | PASS |
| model_segmented_full | blocked | segmented | full | 0.005 | 0.00506467 | 0.0608568 | 91.68% | 0.992884 | 0.0323756 | PASS |

初步架构消融结论：`segmented` 在 random 与 blocked split 上仍然明显超过 persistence，
但 RMSE 约为 `segmented_residual` 的 3.3 倍，M0 误差也更高。这说明仅仅拆分两子断层
还不够，residual 预测头和归一化/残差结构对高精度 future slip forecasting 有实质贡献。

## 结论

主模型在 DSW 上通过 random 与 blocked 两个全量 split 的 h50 publication gate。
这确认了此前 `GO_WITH_CHANGES` 的判断：原始方案不能原样继续，但修正后的
data contract、`log1p` slip target、全局 GNSS 归一化、两子断层
`segmented_residual` 结构是可行的。

完整消融矩阵仍在后台继续运行，后续应补充：

- `plain` random/blocked split
- `full` vs `no_gnss` vs `gnss_only` vs `last_slip_only`
- `m0_loss_weight=0.005` vs `0`
