# 参考文献调研矩阵（面向最终论文）

更新时间：2026-06-28

本文档用于支撑最终论文的 Related Work、研究意义与局限性章节。调研结论是：现有研究已经较充分地覆盖了
GNSS 慢滑移检测、测地序列去噪、地震预警特征学习与通用时序预测，但直接面向“两子断层几何上的
future slip-field forecasting”的工作仍相对少。因此，本文的贡献应落在受控合成场景下的几何感知
多步断层滑移预测，而不是泛化成真实地震预测或业务预警。

## 1. SSE / GNSS 深度学习

| 方向 | 代表工作 | 已有贡献 | 与本文关系 | 本文不能借用的主张 |
| --- | --- | --- | --- | --- |
| 多站 GNSS 慢滑移检测 | [Multi-station deep learning on geodetic time series detects slow slip events in Cascadia](https://www.nature.com/articles/s43247-023-01107-7) | 用逼真的合成训练集和 CNN/attention 类模型从 raw GNSS 中检测 Cascadia SSE。 | 支撑“合成测地数据 + 深度时空模型”用于 SSE 分析的合理性。 | 该工作输出事件检测，不是 future slip-field forecasting。 |
| GNSS SSE 检测 | [Detecting slow slip events in the Cascadia subduction zone from GNSS time series using deep learning](https://doi.org/10.1007/s10291-024-01701-y) | 使用 vbICA 提升信噪比，并用 BiLSTM 与注意力机制识别 SSE。 | 支撑“深度序列模型能从 GNSS 中提取 SSE 信号”。 | 该工作仍是检测任务，不输出未来断层滑移场。 |
| GNSS 去噪 | [Denoising of Geodetic Time Series Using Spatiotemporal Graph Neural Networks](https://arxiv.org/abs/2405.03320) | SSEdenoiser 用图循环网络和时空 Transformer 从多站 GNSS 中恢复 SSE 相关位移。 | 支撑“GNSS 网络结构和时序结构都重要”。 | 去噪输出是观测空间位移，不是 fault slip forecasting。 |

## 2. 测地反演与断层源建模

| 方向 | 代表工作 | 已有贡献 | 与本文关系 | 本文边界 |
| --- | --- | --- | --- | --- |
| 传统测地滑移反演 | [A simple method for improving the resolution of geodetic slip inversion](https://academic.oup.com/gji/article/241/3/1781/8112868) | 分析并改善 geodetic slip inversion 的深部分辨率问题。 | 支撑“slip inversion 是一个物理约束强、分辨率受限的问题”。 | 本文当前没有实现专业 Green's function 反演。 |
| Bayesian / regularized inversion | [slipBERI example in aseismic fault slip study](https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2021JB022621) | 用 GNSS/InSAR 和 Bayesian 方法求解 fault slip。 | 支撑后续可把本文预测结果接入物理反演或同化。 | 当前 `demo_inversion_proxy.py` 只是 ridge proxy。 |
| 有限断层滑移建模 | [Coseismic slip distribution of the 2024 Noto Peninsula earthquake](https://link.springer.com/article/10.1186/s40623-025-02154-4) | 结合 GNSS 和 SAR 构建滑移模型。 | 说明真实 slip modeling 通常依赖多源观测与物理模型。 | 本文合成 SSE 预测不能替代真实地震滑移反演。 |

## 3. 地震预警、预报与灾害意义

| 方向 | 代表工作 | 已有贡献 | 本文可借鉴点 | 必须克制的表述 |
| --- | --- | --- | --- | --- |
| 测地地震预警 | [Earthquake alerting based on spatial geodetic data by spatiotemporal information transformation learning](https://pmc.ncbi.nlm.nih.gov/articles/PMC10500272/) | RSIT 将高维 GNSS 观测转化为低维动态表征，用于地震 alerting。 | 支撑“空间测地数据可用于快速地震状态识别”。 | 本文不是实时地震 alerting 系统。 |
| GNSS 快速震源刻画 | [GFAST/G-larmS/BEEFORES overview snippet](https://scholarsbank.uoregon.edu/server/api/core/bitstreams/deac45e3-f7a0-4966-a9a9-e7c40d21aee4/content) | GNSS 可用于快速估计震级和滑移分布，避免强震饱和问题。 | 支撑测地数据在灾害响应中的价值。 | 本文处理的是合成 SSE 演化，不是快速同震震源反演。 |
| 慢滑移与断层状态 | [Rupture continuity through intermittent pauses in Cascadia slow slip events](https://www.usgs.gov/publications/rupture-continuity-through-intermittent-pauses-cascadia-slow-slip-events) | 说明 SSE 传播在宏观上可平滑迁移，细节上可间歇、复杂。 | 支撑 future evolution modeling 的科学意义。 | 本文不能直接推出破坏性地震发生概率。 |

## 4. 通用时序预测模型

| 模型 | 代表来源 | 核心思想 | 可作为本文后续 baseline 的原因 | 当前不作为主方法的原因 |
| --- | --- | --- | --- | --- |
| TimesNet | [TimesNet arXiv](https://arxiv.org/abs/2210.02186) / [TimesNet GitHub](https://github.com/thuml/TimesNet) | 将 1D 时间序列转换为多周期 2D 表征，用 2D kernel 捕捉周期内和周期间变化。 | 可比较“通用 2D 时间结构”与“断层几何 2D 结构”。 | 本文 slip 结构首先由断层几何决定，不是纯周期图像化问题。 |
| PatchTST | [PatchTST arXiv](https://arxiv.org/abs/2211.14730) / [PatchTST GitHub](https://github.com/yuqinie98/PatchTST) | 将时间序列切成 patch token，并采用 channel-independent Transformer。 | 适合后续测试 GNSS-only 或多变量 slip 序列预测。 | 需要额外适配 3030 维空间结构和物理指标。 |
| iTransformer | [iTransformer OpenReview](https://openreview.net/forum?id=JePfAI8fah) / [iTransformer GitHub](https://github.com/thuml/iTransformer) | 反转 Transformer 维度，把变量作为 token 建模跨变量关系。 | 可作为大规模多变量 forecasting backbone 对照。 | 直接把 3030 slip 点当变量会忽视两子断层几何。 |
| Chronos | [Chronos GitHub](https://github.com/amazon-science/chronos-forecasting) / [Chronos arXiv](https://arxiv.org/abs/2403.07815) | 把时间序列缩放量化为 token，用语言模型框架做概率预测。 | 可作为 zero-shot / foundation model 背景引用。 | 当前任务是结构化 slip field，不是独立单变量概率预测。 |
| TimesFM | [TimesFM GitHub](https://github.com/google-research/timesfm) | 大规模预训练时间序列基础模型。 | 可在论文中作为通用 forecasting 发展趋势。 | 不直接处理断层几何和物理 slip map。 |
| Time-Series-Library | [Time-Series-Library GitHub](https://github.com/thuml/Time-Series-Library) | 集成多种先进时序模型。 | 后续可快速接入 TimesNet、PatchTST、iTransformer baseline。 | 当前阶段优先完成本项目数据契约下的几何模型和消融。 |

## 5. 本文相对已有工作的研究缺口

综合上述工作，本文最稳妥的研究缺口表述为：

> Existing deep-learning studies on slow slip events mainly address event detection,
> geodetic denoising, or alert-oriented representation learning. Direct multi-step
> forecasting of the future fault slip field under a known two-subfault geometry,
> evaluated in physical slip units against persistence and blocked-split baselines,
> remains less explored.

中文表述：

> 现有慢滑移深度学习研究主要集中于事件检测、测地序列去噪或预警特征学习；在已知两子断层几何下，
> 直接预测未来断层滑移场并以物理量指标进行 random/blocked 双划分评估的工作仍相对不足。

## 6. 论文写作建议

1. Related Work 不要写成“别人都没做深度学习”，而要写成“别人主要做检测/去噪/预警，我们做 future slip-field forecasting”。
2. 研究意义可以连接地震危险性背景分析、慢变量状态估计、灾害情景库生成，但不能宣称真实地震预测。
3. 反演相关工作应作为未来扩展背景，当前 proxy inversion 只能放在 demo 或 appendix。
4. 通用时序模型应作为对照与未来 baseline，而不是削弱本文几何感知模型的主线。
5. 最终结果表必须同时报告 persistence、mean、zero baseline，尤其是 h50 persistence。

