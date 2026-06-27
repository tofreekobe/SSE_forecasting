# SSE 慢滑移预测项目完整中文使用文档

更新时间：2026-06-28

## 1. 项目目标

本项目将早期慢滑移深度学习原型重构为一个可复现训练闭环：

```text
data contract -> diagnostics -> baselines -> small overfit -> full train -> inference/report/demo
```

当前主任务固定为：

```text
history_gnss + history_slip -> 50-step future_slip
```

反演和 GNSS 重构只作为辅助证据或演示流程，不作为当前论文主贡献。

## 2. 数据口径

真实全量数据：

- 事件数：6000
- 原始目录：`data/`
- 原始大小：`79,673,958,000` bytes，约 `74.202 GiB`
- 每事件结构：`T=273`，`cols=3040`
- slip 维度：3030
- GNSS 维度：9

训练包：

- 本地目录：`hf_dataset_package/`
- DSW 目录：`/mnt/workspace/hf_dataset_package_verified_b1f13c4`
- 压缩后大小：约 `2.838 GiB`
- 它是全量无损压缩表示，不是抽样子集。

全量审计结论见：

- `docs/full_dataset_package_audit.md`
- `diagnostics_full_local/full_dataset_package_audit_all.json`
- `diagnostics_full_local/full_dataset_package_audit_all.md`

## 3. 环境安装

### 本机 5070 Ti

本机使用支持 RTX 5070 Ti / `sm_120` 的 PyTorch：

```powershell
.\.venv-cu128\Scripts\python.exe -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

预期环境已记录在：

- `docs/local_cuda_5070ti.md`
- `docs/local_5070ti_training_results.md`

### PAI-DSW

DSW 通过反向 SSH 隧道访问：

```powershell
ssh aliyun-dsw-final-sse-via-124
```

当前实测环境：

- GPU：NVIDIA A10
- Python：3.12.13
- PyTorch：2.10.0+cu128
- `torch.cuda.is_available()`：True

DSW 使用说明见：

- `docs/pai_training.md`
- `docs/dsw_main_training_results_zh.md`

## 4. 基础验证

运行核心测试：

```powershell
.\.venv-cu128\Scripts\python.exe -m pytest tests\test_forecast_contract.py -q
```

语法检查：

```powershell
.\.venv-cu128\Scripts\python.exe -m py_compile `
  scripts\train_forecast_model.py `
  scripts\collect_experiment_matrix.py `
  scripts\build_forecast_demo_page.py `
  scripts\audit_full_dataset_package.py
```

## 5. 全量数据包审计

如需重新验证压缩训练包是否对应 raw data：

```powershell
.\.venv-cu128\Scripts\python.exe scripts\audit_full_dataset_package.py `
  --data-dir data `
  --package-dir hf_dataset_package `
  --mode all `
  --output-json diagnostics_full_local\full_dataset_package_audit_all.json `
  --output-md diagnostics_full_local\full_dataset_package_audit_all.md
```

审计必须满足：

- raw event count = 6000
- manifest event count = 6000
- event ID 连续 `1..6000`
- raw-vs-package failed = 0
- slip/GNSS 最大差异 = 0

## 6. Small Overfit

small overfit 用于确认 data contract、loss、inverse transform、M0 指标闭环：

```powershell
.\.venv-cu128\Scripts\python.exe scripts\run_small_overfit.py `
  --package-dir hf_dataset_package `
  --output-dir small_overfit_results `
  --max-events 16 `
  --forecast-start 60 `
  --forecast-horizon 50 `
  --model-type segmented_residual `
  --device cuda
```

只有 small overfit 明显超过 persistence，才应进入 full train。

## 7. 单次正式训练

random split：

```powershell
.\.venv-cu128\Scripts\python.exe scripts\train_forecast_model.py `
  --package-dir hf_dataset_package `
  --output-dir forecast_training_results `
  --protocol random `
  --forecast-start 60 `
  --forecast-horizon 50 `
  --epochs 50 `
  --batch-size 16 `
  --hidden-channels 64 `
  --model-type segmented_residual `
  --input-mode full `
  --device cuda `
  --lr 0.0007 `
  --m0-loss-weight 0.005 `
  --train-eval-max-batches 32 `
  --amp `
  --tensorboard-dir off
```

blocked split 只需把 `--protocol random` 改为：

```powershell
--protocol blocked
```

输出目录包含：

- `metrics.json`
- `training_history.csv`
- `forecast_training_report.md`
- `forecast_contract_stats.json`
- `split_event_ids.json`
- `model.pt`

## 8. DSW 完整实验矩阵

DSW 上完整矩阵脚本：

```bash
PROJECT_DIR=/mnt/workspace/sse_codex_b1f13c4 \
PACKAGE_DIR=/mnt/workspace/hf_dataset_package_verified_b1f13c4 \
OUTPUT_ROOT=/mnt/workspace/sse_outputs/experiment_matrix_b1f13c4_full \
MATRIX_PROFILE=full \
EPOCHS_MAIN=50 \
EPOCHS_COMPARE=30 \
BATCH_SIZE=16 \
NUM_WORKERS=2 \
HIDDEN_CHANNELS=64 \
LEARNING_RATE=0.0007 \
TRAIN_EVAL_MAX_BATCHES=32 \
bash scripts/run_dsw_experiment_matrix.sh
```

后台检查：

```powershell
ssh aliyun-dsw-final-sse-via-124 'root=/mnt/workspace/sse_outputs/experiment_matrix_b1f13c4_full; pid=$(cat $root/pid.txt); ps -p $pid -o pid,etime,cmd; ps --ppid $pid -o pid,ppid,etime,pcpu,pmem,cmd; find $root -maxdepth 4 -name metrics.json | sort; nvidia-smi'
```

已完成主模型结果：

| Split | Test h50 RMSE | Persistence h50 RMSE | Improvement | Gate |
| --- | ---: | ---: | ---: | --- |
| random | 0.00142173 | 0.0592355 | 97.60% | PASS |
| blocked | 0.00152041 | 0.0608568 | 97.50% | PASS |

完整消融矩阵包括：

- `segmented_residual`
- `segmented`
- `plain`
- `no_gnss`
- `gnss_only`
- `last_slip_only`
- `no_m0_loss`

## 9. 结果汇总

收集一个矩阵目录下所有 `metrics.json`：

```powershell
.\.venv-cu128\Scripts\python.exe scripts\collect_experiment_matrix.py `
  --matrix-dir dsw_results\experiment_matrix_b1f13c4_full
```

DSW 上可直接运行：

```bash
python scripts/collect_experiment_matrix.py \
  --matrix-dir /mnt/workspace/sse_outputs/experiment_matrix_b1f13c4_full
```

输出：

- `experiment_matrix_summary.csv`
- `experiment_matrix_summary.md`

## 10. Demo 页面

用训练好的 checkpoint 生成静态 HTML demo：

```powershell
.\.venv-cu128\Scripts\python.exe scripts\build_forecast_demo_page.py `
  --run-dir forecast_training_5070ti_full_streaming\random `
  --package-dir hf_dataset_package `
  --output-dir demo_pages\forecast_random_full `
  --split test `
  --max-events 3 `
  --device cuda
```

打开：

```text
demo_pages\forecast_random_full\index.html
```

页面展示：

- h50 RMSE、R2、M0 指标；
- data contract；
- 训练曲线；
- 事件级 true/model/persistence M0 曲线；
- 最终 slip map 与误差图。

## 11. 反演演示边界

当前可用反演脚本是 ridge-regression proxy：

```powershell
.\.venv-cu128\Scripts\python.exe scripts\demo_inversion_proxy.py `
  --run-dir forecast_training_5070ti_full_streaming\random `
  --package-dir hf_dataset_package `
  --split test `
  --index 0 `
  --max-train-events 1200
```

它只能说明 GNSS-to-slip 演示流程，不应在论文中声称为专业反演模型。

## 12. 论文材料

当前论文相关材料：

- `research_notes/pre_refactor_literature_review.md`
- `research_notes/web_literature_update_2026-06-28.md`
- `research_notes/current_research_objective.md`
- `docs/final_paper_rewrite_plan_zh.md`
- `docs/final_conference_paper_outline_zh.md`
- `docs/dsw_main_training_results_zh.md`

最终论文应强调：

- 慢滑移演化状态估计；
- 合成 SSE future slip forecasting；
- 两子断层几何结构；
- 物理单位评估；
- random + blocked 双 split；
- 不直接声称真实地震预测或业务预警。

## 13. GitHub 发布

仓库应只跟踪：

- `src/`
- `scripts/`
- `tests/`
- `docs/`
- `research_notes/` 中不含版权全文的笔记；
- `README.md`
- `requirements-pai.txt`
- `requirements-diagnostics.txt`

不应上传：

- `data/`
- `hf_dataset_package/`
- `paper/`
- `forecast_training*/`
- `small_overfit*/`
- `checkpoints/`
- `*.pt`
- `*.pth`
- `demo_pages/`
- `dsw_results/`

当前还需要配置目标 GitHub remote，例如：

```powershell
git remote add origin https://github.com/tofreekobe/sse-slow-slip-forecasting.git
git push -u origin main
```

只有确认目标仓库后才能推送，避免误推到旧项目。

