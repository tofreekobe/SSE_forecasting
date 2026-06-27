# SSE 项目论文重写计划

更新时间：2026-06-28

## 一句话研究目标

在已知两段不连续断层几何和稀疏 GNSS 观测条件下，建立一个可复现实验闭环：

```text
history_gnss + history_slip -> 50-step future_slip
```

所有训练目标使用 `log1p(slip / slip_scale)`，所有报告指标反变换回物理 slip。

## 必须保留的主张

- 使用 6000 个合成慢滑移事件的全量 catalog。
- 原始数据规模为 `74.202 GiB`，训练使用经过逐事件审计的压缩分片表示 `2.838 GiB`。
- 主模型为两子断层几何感知的 `segmented_residual`。
- 评估必须包含 random split 和 blocked split。
- baseline 必须包含 zero、mean、persistence，并报告 `h=1/5/10/30/50`。
- 论文主指标以 `h50` future slip forecasting 为核心，同时报告 M0 相对绝对误差。

## 必须删除或降级的旧稿主张

- 不再声称已经实现真实地震预测或业务预警。
- 不再声称硬约束 RSF / Okada / 物理反演，除非后续代码中真实实现并验证。
- 不再把只训练反演或重构任务描述成 forecasting 成功。
- 不再使用 mock 图表、硬编码消融结果或未复现的旧指标。
- 不再把 3030 个 slip 点强行视为连续 `15x202` 图像；两个子断层应分别建模。

## 新论文结构建议

1. Introduction
   - 慢滑移事件监测与 GNSS 深度学习的背景。
   - 现有工作多集中于检测、去噪、预警或静态/快速反演。
   - 本文聚焦合成 SSE 的未来 slip-field 多步预测。

2. Related Work
   - SSE detection from GNSS。
   - GNSS denoising and graph/time-series learning。
   - Geodetic source inversion。
   - General time-series forecasting backbones。

3. Data and Problem Formulation
   - 6000 事件全量 catalog。
   - slip/GNSS 张量契约。
   - `history_steps=60`，`forecast_horizon=50`。
   - random/blocked split。
   - 数据规模口径：raw 74.202 GiB，audited package 2.838 GiB。

4. Method
   - slip `log1p` target transform。
   - training-set global GNSS normalization。
   - `segmented_residual` 两子断层结构。
   - M0 auxiliary loss。
   - inference and inverse transform。

5. Experiments
   - small overfit sanity check。
   - full random / blocked training。
   - model architecture ablation: plain vs segmented vs segmented residual。
   - input ablation: full vs no GNSS vs GNSS-only vs last-slip-only。
   - M0 loss ablation。

6. Results
   - h50 RMSE / R2 / M0 error。
   - baseline comparison。
   - example future-slip maps and M0 curves。
   - reliability discussion: why high scores are plausible in synthetic/history-slip setting。

7. Limitations
   - 合成数据，不等同真实地震预测。
   - history slip 可用性在真实业务中需要由反演或同化系统提供。
   - 仅三站 GNSS、固定断层几何，泛化仍需更多区域与噪声测试。

8. Conclusion
   - 总结合成 SSE future slip forecasting 闭环。
   - 给出下一步：真实 GNSS 适配、反演辅助、噪声鲁棒性和跨事件族验证。

## 当前结果状态

- 本地 5070 Ti 已完成基于全量审计压缩包的 random/blocked full training。
- DSW 上正在跑更完整的 baseline/model/input/M0 消融矩阵。
- 最终论文结果表以 DSW 矩阵回收后的 `experiment_matrix_summary.*` 为准。

