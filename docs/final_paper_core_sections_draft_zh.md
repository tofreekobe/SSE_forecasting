# 最终论文核心章节草稿（数据说明以外）

更新时间：2026-06-28

> 说明：本文档用于替换旧论文草稿中除数据说明以外的核心章节。
> 数据集细节可沿用旧稿并结合 `docs/full_dataset_package_audit.md` 修订口径；
> 这里重点重写研究背景、相关工作、方法、实验、结果分析、研究意义与局限。

## 摘要

慢滑移事件（slow slip events, SSEs）是不以强震动形式释放能量的瞬态断层滑移过程，
其时空演化能够反映板块边界或活动断层的应力调整状态，对区域地震危险性分析、GNSS
形变监测和灾害情景研判具有重要意义。已有深度学习研究主要集中于从 GNSS 时间序列中
检测 SSE 是否发生、对测地时间序列进行去噪，或在地震预警中学习空间测地观测的异常表征；
相比之下，直接面向断层滑移场未来演化的多步预测研究仍较少。

本文将早期混合反演与预测目标的原型系统重构为一个明确的 future slip forecasting
任务：给定稀疏 GNSS 历史观测和历史断层滑移，预测未来 50 个时间步的断层滑移场。
为避免旧方案中 `z-score slip target` 与非负输出约束之间的物理冲突，本文采用
`log1p(slip / slip_scale)` 作为训练目标，并在所有评价中反变换回物理滑移量。
针对研究区域中两个不连续子断层，本文提出几何感知的 segmented residual 网络，
分别建模 `15x166` 与 `15x36` 两个子断层并融合 GNSS 历史特征，从而避免普通
`15x202` 卷积表示引入伪空间邻接。

在 6000 个合成慢滑移事件的全量 catalog 上，本文完成了 raw 数据与压缩训练包的逐事件
审计，并在 random split 与 blocked split 上进行全量训练评估。PAI-DSW A10 实验显示，
主模型在 h50 test RMSE 上相对 persistence baseline 分别提升 97.60% 与 97.50%，
同时显著降低总滑移矩代理量 M0 的相对误差。结果表明，在该受控合成场景下，未来断层
滑移场多步预测是可学习的；该闭环为后续 GNSS-only 反演、真实噪声适配、不确定性量化
和地震危险性背景分析提供了可复现基础。

## 1. 引言

慢滑移事件是介于稳定蠕滑与快速地震破裂之间的重要断层运动形式。它们通常不会产生强烈
地震波辐射，却能在数天到数月尺度上造成可观的地表形变，并改变相邻断层区域的应力状态。
因此，SSE 既是理解断层力学的重要窗口，也是地震危险性背景分析中不可忽视的慢变量。
随着 GNSS 连续观测网络的发展，研究者可以在更高时间分辨率下追踪地表形变异常，并通过
物理反演或机器学习方法推断潜在断层滑移过程。

近年来，深度学习已经被用于 SSE 检测、测地时间序列去噪以及地震预警特征学习。例如，
多站 GNSS 的 CNN/Transformer 检测框架证明了合成事件生成器与深度时空模型结合的可行性；
图神经网络与循环/Transformer 结构也被用于从噪声测地序列中提取 SSE 相关信号。这些工作
共同说明，GNSS 时间序列中确实存在可被神经网络学习的慢滑移信号。然而，检测“事件是否
发生”和预测“断层滑移场将如何继续演化”是两个不同层级的问题。前者主要输出事件标签或
时间窗口，后者则要求模型在物理空间中预测未来滑移分布，并接受物理单位指标的检验。

本研究关注后一个问题。我们不把模型输出直接解释为真实地震预警或破坏性地震预测，而是
研究一个更基础也更可检验的建模层：在已知断层几何和受控合成数据条件下，历史 GNSS 与
历史滑移是否足以预测未来 50 步断层滑移演化。这个问题具有双重意义。首先，它可以检验
历史滑移场中是否包含可学习的未来演化信息；其次，它为真实业务中可能出现的“反演/同化
得到当前滑移状态，再预测短期演化”的工作流提供算法原型。

旧项目失败的主要原因并不只是训练轮数或模型容量不足，而是任务定义和物理约束之间存在
不一致：将非负滑移逐维 z-score 后再使用 softplus 非负输出，会让模型在训练空间与物理
空间之间产生冲突；每事件 GNSS RobustScaler 也会消除事件间真实幅值差异，不利于学习
测地观测与断层滑移之间的物理耦合。本文因此重新定义数据契约、目标变换、模型结构和
评价指标，将反演和 GNSS 重构降级为辅助证据，把 future slip forecasting 作为唯一主任务。

本文贡献如下：

1. 提出一个面向合成 SSE 的可复现 future slip forecasting 闭环，包含全量数据审计、
   baseline、small overfit、full training、物理指标和 demo 报告。
2. 采用与非负滑移相容的 `log1p` 目标变换，并用训练集全局 GNSS 统计量归一化，修正旧方案
   中的尺度与输出冲突。
3. 提出两子断层几何感知的 segmented residual 网络，避免将不连续断层强行拼接为连续图像。
4. 在 6000 事件全量合成 catalog 上完成 random 与 blocked 两种 split 的主模型实验，
   证明模型在 h50 future slip 预测中显著超过 persistence baseline。

## 2. 相关工作

### 2.1 基于 GNSS 的慢滑移事件检测

现有 SSE 深度学习研究中，检测任务最为成熟。多站 GNSS 检测方法通常利用合成 SSE 生成器
构造训练样本，再将真实或模拟 GNSS 序列输入 CNN、Transformer 或循环网络，输出事件发生
概率、时间窗口或事件标签。这类方法的优势是任务定义清晰、可与人工 catalog 或 tremor
活动对齐，但其输出并不直接给出未来断层滑移分布。

另一类研究将独立成分分析、BiLSTM 和注意力机制结合，用于从 GNSS 时间序列中发现慢滑移
事件。这些方法进一步说明了深度序列模型能够从测地观测中提取 SSE 特征，但仍主要回答
“是否存在事件”和“事件何时发生”，而不是预测断层源区未来如何滑移。

### 2.2 测地时间序列去噪与图学习

GNSS 原始序列包含仪器噪声、环境负载、季节项、站点异常和非构造信号。SSEdenoiser 等图
时空网络方法将测站网络结构与时间依赖结合，学习从噪声序列中恢复 SSE 相关形变。该方向
与本文高度相关，因为高质量 GNSS 表征是后续反演和预测的基础；但去噪模型的目标仍是
观测空间中的信号恢复，而不是断层滑移场的未来演化预测。

### 2.3 测地反演与断层源建模

传统 GNSS-to-slip 反演通常依赖弹性半空间或分层介质中的位错 Green's functions，并通过
正则化优化求解断层滑移分布。近年来也有神经网络用于快速估计同震滑移或短期 SSE 滑移
区域。这些研究证明了用机器学习近似测地反演的可行性。然而，本文当前并不声称已经实现
专业级 GNSS-only 反演；本文使用的反演脚本只是 ridge-regression proxy，用于展示未来
系统中可能存在的 GNSS-to-slip 数据流。

### 2.4 通用时间序列预测模型

TimesNet、PatchTST、iTransformer、TimesFM 和 Chronos 等通用时序模型推动了长序列预测
的发展。这些模型强调周期结构、patch 表示、变量维度重排或大规模预训练。它们为后续
baseline 提供有价值参考，但本文的核心结构先验来自断层几何：3030 维 slip 并不是无结构
多变量序列，而是两个已知形状的不连续子断层。因此，本文优先采用 geometry-aware 模型，
再将通用时序 backbone 作为后续扩展方向。

## 3. 问题定义（数据说明占位）

详细数据来源、物理模拟参数和断层几何说明可沿用旧稿数据章节，并按以下口径修订：

- 原始 catalog 为 6000 个事件，约 74.202 GiB。
- 压缩训练包为全量无损表示，约 2.838 GiB，不是抽样子集。
- 每事件包含 273 个时间步；slip 为 3030 维，GNSS 为 9 维。
- 默认窗口为 `history_steps=60`，`forecast_horizon=50`。

形式化任务为：

```text
X = {history_slip[0:60], history_gnss[0:60]}
Y = future_slip[60:110]
```

模型需要输出未来 50 步的 3030 维 slip field。评价时，所有预测均反变换为物理 slip 后
计算 RMSE、R2、M0 相对误差和 baseline 改善率。

## 4. 方法

### 4.1 数据契约

数据集输出被显式拆分为：

- `history_gnss`
- `history_slip`
- `future_slip`
- `future_gnss`
- `metadata`

这种契约避免了旧代码中反演、重构、预测目标混杂的问题。训练主路径只优化
`history_gnss + history_slip -> future_slip`，其中 `future_gnss` 保留给后续辅助任务或
一致性检查。

### 4.2 滑移目标变换

断层滑移在物理上非负。旧方案将 slip 逐维 z-score 后再使用 softplus 输出，会导致训练
目标中大量负值与非负输出空间冲突。本文改用：

```text
encoded_slip = log1p(slip / slip_scale)
physical_slip = expm1(encoded_slip) * slip_scale
```

其中 `slip_scale` 由训练集活跃滑移分位数估计。该变换保留非负性，压缩长尾幅值，并使
训练损失与物理空间评价之间保持可逆关系。

### 4.3 GNSS 全局归一化

本文仅使用训练集全局均值与标准差归一化 GNSS。这样既避免测试集泄漏，也保留事件间幅值
差异。相比之下，每事件 RobustScaler 会把每个事件内部重新定标，削弱 GNSS 幅值与 slip
规模之间的物理对应关系。

### 4.4 几何感知 segmented residual 网络

3030 个 slip 点由两个不连续子断层组成：

- segment 1: `15x166`
- segment 2: `15x36`

普通 `15x202` 卷积会把两个子断层边界处的网格点视为相邻，产生不真实的信息传播。本文
模型将两段 slip 分别 reshape 并送入独立卷积分支，同时用 GNSS temporal encoder 提取
历史观测特征并广播到两个子断层分支。输出采用 residual 形式，从最后一个历史 slip 状态
出发预测未来增量，从而把 persistence 作为自然起点。

### 4.5 损失函数与物理指标

训练损失包括 encoded MSE 与可选 M0 auxiliary loss。M0 在本文中作为总滑移矩代理量：

```text
M0_proxy(t) = sum_i slip_i(t)
```

M0 loss 用于约束模型不仅在像素级匹配 slip field，也能匹配整体事件释放趋势。最终报告
包括：

- physical RMSE；
- physical R2；
- M0 relative absolute error；
- h1/h5/h10/h30/h50 baseline 表；
- 相对 persistence 的 RMSE 改善率。

## 5. 实验设计

### 5.1 Baselines

本文使用三类物理空间 baseline：

1. zero：未来 slip 全为 0；
2. mean：训练集未来窗口均值；
3. persistence：未来保持最后一个历史 slip 状态。

persistence 是最关键 baseline，因为它代表“慢滑移短期不再变化”的强基线假设。

### 5.2 Small Overfit

small overfit 在 16-32 个事件上验证训练闭环。如果模型不能在小样本上明显超过 persistence，
则说明 data contract、loss、inverse transform 或模型实现存在问题。当前 small overfit
已通过，为 full train 提供了基本可信度。

### 5.3 Full Training

full training 使用全量 6000 事件，并报告：

- random split：检验同分布泛化；
- blocked split：检验事件编号/生成顺序分块后的稳健性。

publication gate 设定为：

- random h50 RMSE 至少比 persistence 改善 5%；
- blocked h50 RMSE 至少比 persistence 改善 2%；
- M0 error 不得比 persistence 恶化超过 10%。

### 5.4 消融实验

完整消融矩阵包括：

- `plain`：把两子断层拼成 `15x202`，检验伪邻接影响；
- `segmented`：两子断层独立卷积但无 residual 强化；
- `segmented_residual`：默认主模型；
- `no_gnss`：去除 GNSS history；
- `gnss_only`：去除 history slip；
- `last_slip_only`：只保留最后一帧 history slip；
- `no_m0_loss`：移除 M0 auxiliary loss。

这些实验用于回答三个问题：

1. 几何结构是否必要？
2. GNSS 历史在已有 history slip 时是否仍提供增益？
3. M0 loss 是否改善物理一致性？

## 6. 当前结果与分析

PAI-DSW A10 上已完成主模型 random 与 blocked split：

| Split | Test h50 RMSE | Persistence h50 RMSE | Improvement | Test R2 | M0 rel abs | Gate |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| random | 0.00142173 | 0.0592355 | 97.60% | 0.999408 | 0.0155652 | PASS |
| blocked | 0.00152041 | 0.0608568 | 97.50% | 0.999359 | 0.0166794 | PASS |

这些结果说明，在合成 catalog 和 history slip 可用的设定下，未来 50 步 slip field 具有很强
可学习性。persistence baseline 在 h50 上表现较差，说明未来活跃滑移并不能简单由最后一帧
外推；模型显著降低 RMSE 与 M0 error，表明其学习到局部 slip 分布和整体释放趋势。

消融实验进一步揭示了任务中的信息来源。移除 GNSS 后，模型在 random 与 blocked split 上
仍能保持较好性能，说明 history slip 是当前 forecasting 任务的主导信息源；但 full input
仍稳定优于 no-GNSS，表明 GNSS history 提供了额外边际约束。相反，GNSS-only 在
random 与 blocked split 上均未能超过 persistence，h50 RMSE 与 zero-level baseline
接近，R2 为负。这一负结果非常重要：它说明在当前三站稀疏 GNSS 和模型设定下，直接从 GNSS 历史预测未来完整 slip
field 仍不可行，真实业务应用必须依赖反演、同化或其他状态估计方法先获得 history slip。

`last_slip_only` 消融在两个 split 上仍显著超过 persistence，说明最后一帧滑移状态本身
已经包含强短期外推信息；但其 RMSE 和 M0 error 均弱于 full input 主模型，尤其 blocked
split 上退化更明显。因此，本文不把最后一帧 shortcut 视为最终方案，而是将完整 history
slip 与 GNSS history 作为建模短期演化和物理一致性的必要信息。

移除 M0 auxiliary loss 后，random split 的 h50 RMSE 可进一步降至 `0.00112126`，但
M0 relative absolute error 升至 `0.0594853`；blocked split 的 h50 RMSE 为 `0.00278276`，
M0 error 升至 `0.107937`。这表明 M0 loss 的主要价值是约束整体滑移矩代理量，而不是
机械地降低像素级 RMSE。考虑论文目标包含物理一致性，默认主模型仍保留
`m0_loss_weight=0.005`。

需要强调的是，结果很高并不意味着模型可以直接用于真实地震预测。当前输入包含 history slip，
这在真实业务中通常需要由反演或数据同化系统提供。因此，本文结果更准确的解释是：

> 在受控合成场景中，如果当前滑移状态和稀疏 GNSS 历史可用，则 geometry-aware 网络能够
> 有效预测短期慢滑移演化。

完整消融矩阵已在 DSW 上完成。最终论文应将 architecture、input 和 loss 三类消融作为
结果章节的核心证据：architecture 消融说明 residual 结构是主要增益来源，input 消融说明
history slip 是主导信息源且 GNSS 提供边际约束，loss 消融说明 M0 项提升物理一致性。

## 7. 研究意义与灾害应用边界

慢滑移预测对地震预警预报的意义主要体现在状态估计和情景分析，而不是直接发布警报。破坏性
地震的发生涉及复杂的多尺度断层物理过程，不能由单一合成数据模型直接判定。然而，SSE 的
时空演化能够提供断层应力调整、板块耦合变化和潜在触发条件的重要背景信息。

本文方法未来可能服务于以下方向：

1. **区域应力状态监测**：预测未来 slip evolution 可作为断层慢变量状态估计的一部分。
2. **地震危险性背景分析**：SSE 演化可辅助判断某些区域应力加载或释放是否异常。
3. **次生灾害情景研判**：在地震、海啸、滑坡等链式灾害分析中，断层滑移状态可作为场景输入。
4. **GNSS 业务系统辅助**：模型可用于反演初始化、异常窗口筛查、情景库快速生成。
5. **物理模型耦合**：未来可与摩擦定律、弹性位错模型或数据同化系统结合。

但本文当前不能用于：

- 真实地震发生时间预测；
- 业务级地震预警发布；
- 替代专业断层反演；
- 对未验证区域或真实噪声环境直接泛化。

## 8. 局限与未来工作

本文主要局限包括：

1. 数据为合成 catalog，尚未证明真实 GNSS 环境中的鲁棒性。
2. history slip 在真实应用中不是直接观测量，需要由反演或同化系统产生。
3. 当前 GNSS 站点数量少，尚未研究站点缺失、噪声增强和台站几何变化。
4. 当前反演 demo 是 proxy，不是主模型。
5. 当前消融矩阵仍局限于合成 catalog 内部的 random/blocked split，尚未覆盖跨区域和跨物理参数泛化。
6. 尚未加入不确定性量化，无法给出预测置信区间。

未来工作应包括：

- 训练 GNSS-only 或 GNSS-dominant slip estimation 模块；
- 加入真实噪声、缺测和台站扰动；
- 设计跨事件族、跨参数、跨区域 blocked split；
- 将通用时序 backbone 作为外部 baseline；
- 引入不确定性估计和物理一致性正则；
- 与真实 GNSS catalog 和地震活动数据进行后验对比。

## 9. 结论

本文从失败的慢滑移深度学习原型出发，识别并修正了任务定义、目标变换、归一化和断层几何
表示中的关键问题。通过新的 data contract、`log1p` slip target、全局 GNSS normalizer 和
两子断层 `segmented_residual` 模型，项目形成了可复现的 future slip forecasting 闭环。
全量 6000 事件实验表明，主模型在 random 与 blocked split 上均显著超过 persistence
baseline，并通过预设 publication gate。

这些结果支持本文的核心结论：在受控合成慢滑移场景中，几何感知深度模型可以有效学习未来
断层滑移演化。该工作为慢滑移状态估计、GNSS 辅助反演、地震危险性背景分析和灾害情景建模
提供了基础，但仍需真实数据验证和更严格的跨区域泛化研究后，才能向业务应用推进。

## 参考文献占位

最终稿建议至少覆盖：

- Multi-station GNSS SSE detection with CNN/Transformer。
- Cascadia GNSS SSE detection with vbICA + BiLSTM + attention。
- SSEdenoiser / spatiotemporal graph neural network。
- Geodetic earthquake alerting with RSIT。
- GNSS-to-slip inversion / finite fault inversion。
- TimesNet、PatchTST、iTransformer、TimesFM、Chronos。
- 慢滑移事件与断层应力、地震危险性关系的地球物理基础论文。
