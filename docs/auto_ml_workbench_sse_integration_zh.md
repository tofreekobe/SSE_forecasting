# auto_ml 科研工作台 SSE 联动记录

日期：2026-06-28

## 当前状态

已将 SSE 项目整理为 auto_ml 可继续处理的研究工作台项目：

```text
C:\Users\Administrator\Desktop\王一帆项目文件\mm\auto_ml\projects\sse_slow_slip_forecasting
```

该项目包包含研究背景、文献综述源、证据表、局限性分析、中英文论文草稿、参考文献和结果表。它不包含 `data/`、`hf_dataset_package/`、DSW 输出目录或任何 HF/OpenAI token。

## 模型调用状态

当前 shell 环境未检测到 `OPENAI_API_KEY`，因此本轮不使用 OpenAI API。用户已确认可暂时使用平台中已绑定的 DeepSeek API；auto_ml 的 `ModelGateway` 中 DeepSeek provider 已配置为默认 provider，并已通过 live 连接测试。

本轮已通过 DeepSeek 生成两份辅助审阅材料：

```text
docs/deepseek_literature_and_positioning_review_zh.md
docs/deepseek_manuscript_polish_review_zh.md
```

并同步到：

```text
C:\Users\Administrator\Desktop\王一帆项目文件\mm\auto_ml\projects\sse_slow_slip_forecasting\reviews
```

后续如果需要切回 OpenAI，可配置 `OPENAI_API_KEY` 后再执行：

```powershell
cd "C:\Users\Administrator\Desktop\王一帆项目文件\mm\auto_ml"
$env:OPENAI_API_KEY="<your key>"
python -m pytest tests/test_v21_fixes.py::test_review_without_available_model_prompts_configuration -q
```

当前 DeepSeek 路径下，可在 Web UI 或 API 中对 `sse_slow_slip_forecasting` 继续执行：

1. 关键词生成；
2. 论文检索；
3. 文献综述生成；
4. manuscript-assist 逐节润色；
5. publication export。

## 反向升级建议

本轮发现 auto_ml 对“外部成熟项目导入”的支持仍不够显式。建议下一步增加一个正式导入入口：

- 输入：外部项目根目录、manuscript.md、figures 目录、result tables、references.bib；
- 输出：标准 `projects/<id>/knowledge|analysis|writing` 目录；
- 验证：导入后 `workspace`、`yanwen`、`export-publication` 均可识别已有章节，而不是强制从空白项目开始。

本轮已经用 SSE 项目包模拟了该目标结构，后续可以把生成脚本内化为 auto_ml 的正式功能。
