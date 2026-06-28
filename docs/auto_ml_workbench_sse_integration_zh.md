# auto_ml 科研工作台与 SSE 项目联动记录

日期：2026-06-28

## 当前状态

已将 SSE 项目整理为 `auto_ml` 可继续处理的研究工作台项目：

```text
C:\Users\Administrator\Desktop\王一帆项目文件\mm\auto_ml\projects\sse_slow_slip_forecasting
```

该项目包含研究背景、文献综述材料、证据表、局限性分析、中英文论文草稿、图表、结果表和 DeepSeek 审阅材料。导入边界是可发布研究材料；不导入 `data/`、`hf_dataset_package/`、DSW 输出目录或任何 HF/OpenAI/DeepSeek token。

## 正式导入能力

本轮已把此前的临时搬运脚本升级为 `auto_ml` 后端正式入口：

```http
POST /api/projects/import-external
```

入口接受外部成熟项目根目录和若干相对路径，创建或更新工作台项目，并完成：

- 复制 `docs/submission_draft_zh.md` 到 `writing/manuscript.md`。
- 复制 `docs/submission_draft_en.md` 到 `writing/manuscript_en.md`。
- 复制 `paper_figures_full/` 下的论文图件到 `writing/figures/`。
- 复制结果表到 `writing/tables/`。
- 复制 DeepSeek 审阅与审稿差距报告到 `reviews/`。
- 复制诊断和研究定位材料到 `analysis/` 与 `knowledge/`。
- 将导入文件登记为 `external_import:*` source assets，可靠性标记为 `local_verified`。
- 从 Markdown 标题解析出 `abstract`、`introduction`、`literature_review`、`methods`、`results`、`discussion`、`conclusion` 等研文章节。

真实 SSE 导入验证结果：

```text
project_id: sse_slow_slip_forecasting
imported_files: 13
job_status: succeeded
sources_count: 13
workflow: topic/knowledge/plan/writing completed; experiment/analysis not_started
manuscript_sections: title, abstract, introduction, literature_review, methods, results, discussion, conclusion
```

## DeepSeek 模型网关状态

当前没有使用 OpenAI API；按项目决策，暂时使用 `auto_ml` 中已经绑定并设为默认的 DeepSeek provider。

本轮验证：

- DeepSeek provider 为默认启用 provider。
- 小额度探针在 `max_output_tokens=64` 时返回空内容，已确认属于输出额度过小导致的 `empty_response`。
- 将输出上限调到 `512` 后，DeepSeek 成功返回正文：`DeepSeek route is ready.`
- 已通过正式作业链路运行 `generate-keywords`，生成 5 个中文关键词和 5 个英文关键词。

本次英文关键词包括：

```text
slow slip event
fault slip forecasting
GNSS time series
geometry-aware baseline
crustal deformation
```

## 后续使用建议

在 DeepSeek 路径下，可继续对 `sse_slow_slip_forecasting` 执行：

1. 多来源论文检索与筛选。
2. 文献综述生成与人工核对。
3. `manuscript-assist` 逐节润色，但建议先复制章节或使用审阅模式，避免直接覆盖正式草稿。
4. `export-publication` 或 `export-docx` 输出投稿版文档。
5. 在 full-data baseline 和模型实验完成后，把新指标与图件再次通过 `import-external` 同步进工作台。

## 仍需注意

`auto_ml` 已可作为论文整理与审阅工作台使用，但它不能替代关键实验证据。当前论文能否进入高水平投稿，仍取决于：

- 真实 6000 事件全量数据诊断与 baseline 是否完成。
- h50 future slip forecasting 是否稳定超过 persistence baseline。
- blocked split 是否仍有正收益。
- M0 error、active-region metrics 和物理反变换报告是否闭环。
