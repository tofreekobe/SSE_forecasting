# 最终版顶会论文大纲（SSE Future-Slip Forecasting）

更新时间：2026-06-28

## 推荐题目

**Geometry-Aware Multi-Step Forecasting of Synthetic Slow Slip Evolution from Sparse Geodetic Histories**

中文工作题目：

**面向慢滑移演化的几何感知多步断层滑移预测：基于稀疏 GNSS 与历史滑移的全量合成事件实验**

## 论文定位

本文不主张真实地震预测，也不主张已经形成业务级预警系统。论文应定位为：

> 在已知断层几何与全量合成慢滑移事件 catalog 上，验证一个可复现的
> `history_gnss + history_slip -> future_slip` 多步断层滑移场预测闭环，并用
> 物理量指标、随机划分和事件块划分共同评估其有效性。

## 摘要草稿

Slow slip events (SSEs) represent transient aseismic fault slip that can alter
regional stress conditions and provide important context for earthquake hazard
assessment. Existing deep-learning studies mainly focus on detecting SSE
occurrence from GNSS time series or denoising geodetic observations, while
direct multi-step forecasting of the underlying fault slip field remains less
explored. We formulate a synthetic SSE forecasting task in which sparse GNSS
histories and historical fault slip are used to predict the next 50 steps of
slip evolution on two disconnected subfaults. To avoid physically inconsistent
training targets, slip is modeled with a nonnegative-compatible log transform
and all metrics are reported after inverse transformation to physical slip. We
introduce a geometry-aware segmented residual network that separately models the
two subfaults and fuses GNSS history through shared temporal features. On the
full 6000-event synthetic catalog, the model substantially outperforms zero,
mean, and persistence baselines under both random and blocked event splits,
achieving h50 test RMSE improvements above 97% while reducing moment-proxy error.
These results demonstrate that future slip-field forecasting is learnable in
this controlled synthetic setting and provide a reproducible baseline for future
GNSS-only inversion, uncertainty quantification, and real-data adaptation.

## 核心贡献

1. **任务重定义**：将旧方案中混杂的反演、重构、预测目标收敛为明确的
   50-step future slip forecasting。
2. **数据契约**：建立 `history_gnss/history_slip/future_slip/future_gnss/metadata`
   契约，并完成 6000 事件、74.202 GiB 原始数据与 2.838 GiB 压缩训练包的逐事件审计。
3. **目标修正**：废弃 `z-score slip + softplus` 的冲突组合，采用
   `log1p(slip / slip_scale)` 训练并反变换回物理 slip 评估。
4. **几何感知模型**：提出 `segmented_residual`，分别建模 `15x166` 与 `15x36`
   两个不连续子断层，避免普通 `15x202` 图像卷积造成的伪邻接。
5. **全量实验证据**：在 PAI-DSW A10 上完成 random 与 blocked 全量主模型训练，
   两个 split 均通过 h50 publication gate。
6. **可复现演示**：提供静态 demo 页面，展示预测曲线、最终 slip 图和 persistence 对比。

## 章节结构

### 1. Introduction

重点回答：

- 慢滑移事件为什么重要：它们不直接等同于破坏性地震，但会反映和改变断层应力状态。
- GNSS 为什么重要：提供连续地表形变观测，是 SSE 检测、反演和灾害背景分析的重要数据源。
- 现有深度学习研究的边界：多数做检测、去噪或地震预警特征学习，较少直接预测未来断层 slip field。
- 本文边界：合成数据、固定几何、稀疏三站 GNSS 与历史 slip；不宣称业务级预警。

建议写法：

> Rather than treating warning as a direct output, we study a preceding modeling
> layer: whether future slip evolution on a known fault geometry can be learned
> from geodetic and slip histories under controlled synthetic conditions.

### 2. Related Work

建议分四类：

1. **GNSS-based SSE detection**
   - Multi-station CNN/Transformer SSE detection。
   - vbICA + BiLSTM + attention SSE detection。
2. **Geodetic denoising and graph learning**
   - SSEdenoiser / STGNN 类方法。
   - 说明其目标是 denoising，不是 future slip forecasting。
3. **Geodetic inversion and source modeling**
   - GNSS-to-slip inversion、coseismic slip inversion、short-term SSE slip area estimation。
   - 强调本文暂不把 proxy inversion 当作主贡献。
4. **General time-series forecasting**
   - TimesNet、PatchTST、iTransformer、TimesFM、Chronos。
   - 说明通用序列 backbone 可作后续 baseline，但本文优先利用断层几何结构。

### 3. Data and Problem Formulation

必须写清：

- 原始数据：6000 events，74.202 GiB。
- 每事件 shape：`T=273`，`cols=3040`。
- slip：3030 维，拆成 `15x166` 与 `15x36`。
- GNSS：9 维，代表三站三分量或等价稀疏测点特征。
- 默认任务：

```text
history_steps = 60
forecast_horizon = 50
input = history_slip[0:60] + history_gnss[0:60]
target = future_slip[60:110]
```

- 数据划分：
  - random split：检验同分布泛化。
  - blocked split：检验按事件编号/生成顺序分块后的分布漂移鲁棒性。
- 数据规模表述：
  - 原始 catalog 是 74.202 GiB。
  - 训练包是全量、无损、审计过的压缩表示，不是子集。

### 4. Method

建议小节：

1. **Slip target transform**
   - `encoded = log1p(slip / slip_scale)`
   - `slip = expm1(encoded) * slip_scale`
   - 解释避免负 slip target 与非负输出冲突。
2. **GNSS normalization**
   - 仅用训练集全局统计量。
   - 禁止每事件 scaler，以免破坏物理尺度。
3. **Segmented residual architecture**
   - 两个子断层分支。
   - GNSS temporal encoder。
   - 从 last observed slip 出发预测 residual evolution。
4. **Loss**
   - encoded MSE。
   - M0 auxiliary loss：约束总 slip/moment-proxy。
5. **Inference**
   - 输出先在 encoded space，报告前全部 inverse 到 physical slip。

### 5. Experiments

已完成主实验：

| Experiment | Status | Purpose |
| --- | --- | --- |
| small overfit | done | 检查 data contract、loss、inverse transform、M0 指标闭环 |
| full random main | done | 同分布泛化主结果 |
| full blocked main | done | 分块泛化主结果 |

正在跑/待补齐：

| Experiment | Model/Input | Purpose |
| --- | --- | --- |
| segmented vs segmented_residual | architecture ablation | 验证残差与结构归纳偏置 |
| plain `15x202` vs segmented | geometry ablation | 验证避免两子断层伪邻接的收益 |
| no_gnss | input ablation | 衡量 GNSS history 的增益 |
| gnss_only | input ablation | 估计无 history slip 时任务难度 |
| last_slip_only | input ablation | 衡量完整历史 slip 是否必要 |
| no_m0_loss | loss ablation | 验证 M0 auxiliary loss 对物理指标的影响 |

### 6. Results

已完成 DSW 主结果：

| Split | Model h50 RMSE | Persistence h50 RMSE | Improvement | R2 | Gate |
| --- | ---: | ---: | ---: | ---: | --- |
| random | 0.00142173 | 0.0592355 | 97.60% | 0.999408 | PASS |
| blocked | 0.00152041 | 0.0608568 | 97.50% | 0.999359 | PASS |

结果分析要点：

- persistence 的 R2 为负，说明 50 步外简单保持最后一帧不能解释未来活跃 slip 变化。
- 主模型显著降低 RMSE 与 M0 误差，说明模型不仅拟合局部 slip，也学习到总 slip 增长趋势。
- random 与 blocked 都通过，初步说明模型不是只记住随机事件局部统计。
- 高分需要谨慎解释：输入包含 history slip，因此这是受控合成 forecasting，不是 GNSS-only 真实反演。

### 7. Hazard and Warning Relevance

这一节要写得有意义但克制：

- SSE 预测对地震预警/预报的意义不是直接“预测地震”，而是提供断层慢滑移状态和应力演化的先验信息。
- 更准确的 future slip field 可能服务于：
  - 区域应力状态监测；
  - 后续地震危险性背景评估；
  - 多源地质灾害链条分析中的触发条件估计；
  - GNSS 业务系统中的事件同化、反演初始化和异常筛查。
- 当前模型尚不能用于：
  - 发布真实预警；
  - 预测破坏性地震发生时间；
  - 替代物理反演或专家判读。

建议表述：

> The practical value of this work lies in state estimation and scenario
> analysis rather than direct alarm issuance.

### 8. Limitations

必须诚实列出：

- 合成数据与真实 GNSS 噪声/缺测/站点变化仍有距离。
- history slip 在真实系统中通常来自反演/同化，不能默认直接观测。
- 当前反演 demo 是 proxy，不是论文主模型。
- 尚缺跨断层、跨区域、跨生成参数族泛化。
- 消融矩阵尚需完整回收后才能支撑最终 ablation claim。

### 9. Conclusion

结论应落在：

- 原失败方案的关键问题已被识别并修正。
- 新闭环在全量 6000 事件上通过 random/blocked gate。
- geometry-aware future slip forecasting 是可行的。
- 下一步是 GNSS-only slip estimation、真实噪声适配、不确定性和跨区域迁移。

## 建议图表清单

| Figure/Table | 内容 | 状态 |
| --- | --- | --- |
| Figure 1 | 两子断层几何 + 三站 GNSS + 任务窗口 | 可用现有 paper 图片/MATLAB 图重绘 |
| Figure 2 | Data contract 与模型流程图 | 待绘 |
| Figure 3 | `segmented_residual` 架构图 | 待绘 |
| Figure 4 | random/blocked 训练曲线 | 已可由脚本生成 |
| Figure 5 | 事件级 future slip map 与 M0 curve | 已可由 demo 生成 |
| Table 1 | 数据集统计与审计结果 | 已有数据 |
| Table 2 | baseline vs main model | 已有主结果 |
| Table 3 | architecture ablation | 等 DSW 矩阵 |
| Table 4 | input/loss ablation | 等 DSW 矩阵 |

## 最终交付前检查清单

- [ ] DSW 完整消融矩阵全部回收并写入结果表。
- [ ] 生成 final paper figures。
- [ ] 将大纲扩写为完整中文/英文论文草稿。
- [ ] 如需要 `.docx`，按 Documents skill 渲染 PNG 检查。
- [ ] 用最终 checkpoint 重新生成 demo 页面。
- [ ] 配置 GitHub remote 并推送代码。

