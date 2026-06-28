# 顶刊审稿人视角缺口审计

日期：2026-06-28

本文档从顶刊审稿人视角评估当前 SSE 论文初稿距离高水平投稿仍需补齐的内容。结论：当前项目已经形成可信的合成数据 forecasting 闭环和强实验结果，但若目标是顶级地学或机器学习交叉期刊，仍应把“合成场景下的可学性证明”与“真实业务可用性”严格分开。

## 总体判断

当前稿件可以支撑一篇方法与数据闭环论文的初稿，但还不足以直接投顶刊。最强证据是 6000 事件全量训练、random/blocked 双 split、14 项消融、全量数据审计和 GNSS-only 失败边界。最大短板是真实数据泛化、history slip 的可获得性、外部强 baseline、统计显著性和不确定性量化。

## Major Revision 级问题

| 审稿风险 | 为什么会被质疑 | 当前证据 | 必做补充 |
| --- | --- | --- | --- |
| 合成数据外推性不足 | 顶刊审稿人会要求证明模型不是只学会生成器规律 | 6000 synthetic catalog 和 blocked split | 增加跨生成参数、跨事件族、噪声增强、站点缺失 split |
| history slip 可获得性 | 真实 GNSS 不能直接观测完整 slip field | 本文主任务依赖 history_slip | 增加 GNSS-to-slip/state-estimation 前端或把任务明确限定为 assimilation 后预测 |
| persistence baseline 被过度击败 | 97% 提升很强，审稿人会担心任务过易、泄漏或 baseline 不够强 | raw-package 审计和 split 结果 | 加入 last-frame residual oracle、linear AR、ConvLSTM/U-Net、PatchTST/iTransformer 对照 |
| GNSS-only 失败削弱应用叙事 | 如果 GNSS-only 失败，真实业务入口必须解释清楚 | 两个 split 均 FAIL | 把 GNSS-only 改为未来工作；主张改为“given estimated slip state” |
| 缺少不确定性 | 地学预测任务需要置信区间和失败样本分析 | 当前只有点预测指标 | 增加 ensemble、MC dropout 或 quantile head，并报告 calibration |
| 物理约束仍偏弱 | M0 proxy 是 sum slip，不是真正 seismic moment | M0 auxiliary loss 有效 | 加入面积/刚度权重的 moment proxy，或说明当前仅为 proxy |
| 文献综述还偏薄 | 顶刊需要更完整的 GNSS inversion、SSE physics、forecasting baseline 对话 | 已有 10 篇核心来源 | 扩到 30-50 篇，区分 detection、denoising、inversion、data assimilation、forecasting |

## 建议的下一轮实验矩阵

| 优先级 | 实验 | 目标证据 | 投稿前必要性 |
| --- | --- | --- | --- |
| P0 | 跨生成参数 blocked split | 证明不是记忆同一生成族 | 顶刊必做 |
| P0 | 噪声/缺测 GNSS stress test | 证明 GNSS 辅助项稳定 | 顶刊必做 |
| P0 | 从反演 proxy 产生 history_slip 再 forecasting | 闭合真实业务入口 | 顶刊必做 |
| P1 | ConvLSTM / U-Net residual baseline | 防止审稿人认为 baseline 太弱 | 强烈建议 |
| P1 | PatchTST / iTransformer 多变量 baseline | 对齐时序预测社区 | 强烈建议 |
| P1 | Uncertainty ensemble | 给出置信区间与失败样本 | 强烈建议 |
| P2 | 更多真实 SSE case 后验对比 | 提升地学可信度 | 有真实数据时必做 |

## 写作层面的主要修改

1. 标题和摘要必须出现 `synthetic` 或 `controlled synthetic catalog`，避免被解读为真实地震预测。
2. 主张应写成“estimated/history slip state conditioned forecasting”，不要写成“from sparse GNSS directly forecasts fault slip”。
3. GNSS-only 失败不是负面结果，而是重要边界：它说明真实应用需要反演或同化前端。
4. 结果段需要同时报告 mean/zero/persistence 全 horizon 表，而不是只突出 h50。
5. Discussion 需要增加 failure cases、active slip mask、空间误差分布和 blocked split 的物理解释。
6. 参考文献应补齐 geodetic inversion、slow-slip physics、data assimilation 和 uncertainty forecasting。

## 当前可投稿定位

更稳妥的定位是：

> A reproducible synthetic benchmark and geometry-aware residual forecasting baseline for future slow-slip field evolution conditioned on estimated slip histories.

不建议当前直接定位为：

> A GNSS-only operational slow-slip or earthquake prediction system.

## 结论

当前成果已经足以形成完整初稿和内部预审版本；距离顶级论文最关键的下一步不是继续堆模型，而是补强泛化、真实入口、外部 baseline 和不确定性证据。
