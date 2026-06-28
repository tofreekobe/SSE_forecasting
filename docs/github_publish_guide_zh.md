# GitHub 发布说明

本项目已经整理为“代码、测试、论文文档、实验摘要”可上 GitHub 的形态。原始数据、HF 数据包、训练输出和本地论文资料不进入 GitHub。

## 当前状态

- 本地 Git 分支：`main`
- 最近关键提交：
  - `338db59 Add one-command SSE demo launcher`
  - `bc5be15 Add final SSE paper manuscript`
  - `8fa4f53 Record complete DSW ablation matrix`
  - `fba4a3c Document full dataset recheck`
- 当前本地仓库尚未配置 GitHub remote。
- Codex GitHub 连接器可以访问账号仓库列表，但当前未发现专用 SSE 仓库；不要推送到旧的无关仓库。

## 发布前检查

先运行：

```powershell
.\.venv-cu128\Scripts\python.exe scripts\check_release_ready.py
```

它会检查：

- 是否有未提交变更；
- 必需的代码、测试、论文和文档是否存在；
- `data/`、`hf_dataset_package/`、`paper/`、`demo_pages/`、`dsw_results/` 等私有或生成目录是否被 Git 忽略；
- 是否有真实 HF token 形态的密钥误入受管文本文件；
- 最终论文、结果表和演示文档中是否包含关键证据。

如果只是想在仍有本地未提交修改时预览检查结果：

```powershell
.\.venv-cu128\Scripts\python.exe scripts\check_release_ready.py --allow-dirty --json
```

## 推荐发布流程

1. 在 GitHub 上新建一个专用仓库，建议名称：

```text
tofreekobe/sse-slow-slip-forecasting
```

建议先设为 private，因为文档中包含研究方向、实验设计和数据说明；后续投稿或开源前再决定是否公开。

2. 回到本地项目目录，先做 dry-run：

```powershell
.\scripts\publish_to_github.ps1 `
  -RepositoryUrl https://github.com/tofreekobe/sse-slow-slip-forecasting.git `
  -DryRun
```

如果目标仓库是刚创建的空仓库，且 dry-run 的远端探测因空仓库或凭据问题失败，可以先确认仓库 URL 正确，再加 `-SkipRemoteProbe` 只检查本地发布状态：

```powershell
.\scripts\publish_to_github.ps1 `
  -RepositoryUrl https://github.com/tofreekobe/sse-slow-slip-forecasting.git `
  -DryRun `
  -SkipRemoteProbe
```

3. 确认无误后正式发布：

```powershell
.\scripts\publish_to_github.ps1 `
  -RepositoryUrl https://github.com/tofreekobe/sse-slow-slip-forecasting.git
```

脚本会自动运行 `scripts\check_release_ready.py`，必要时添加 `origin`，然后执行 `git push -u origin main`。

也可以手动配置 remote：

```powershell
git remote add origin https://github.com/tofreekobe/sse-slow-slip-forecasting.git
git remote -v
```

4. 运行发布检查：

```powershell
.\.venv-cu128\Scripts\python.exe scripts\check_release_ready.py
```

5. 推送：

```powershell
git push -u origin main
```

## 不应上传的内容

这些内容应继续只保留在本机、HF 私有数据集或 DSW 环境中：

- `data/`：74.202 GiB 原始 6000 事件目录；
- `hf_dataset_package/`：2.838 GiB 压缩训练包；
- `paper/`：含早期参考论文全文和旧稿，可能涉及版权；
- `forecast_training*/`、`small_overfit*/`、`checkpoints/`、`logs/`；
- `demo_pages/`：可由 `scripts/serve_demo_gui.py` 重新生成；
- `dsw_results/`、`diagnostics_full_local/`：本地实验同步与诊断输出。

## 发布后建议

发布成功后，在 GitHub README 中保持以下入口：

- `docs/final_paper_manuscript_zh.md`：最终论文中文母稿；
- `docs/paper_result_tables_current.md`：全量 DSW 实验结果表；
- `docs/full_dataset_package_audit.md`：6000 事件、74.202 GiB 原始数据与压缩包一致性摘要；
- `docs/model_demo_usage.md`：预测与反演 proxy 演示；
- `scripts/check_release_ready.py`：发布门控；
- `scripts/serve_demo_gui.py`：本地演示 GUI。
