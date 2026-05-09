# -*- coding: utf-8 -*-
"""
baseline_timesfm.py — TimeSFM Zero-Shot Forecasting Baseline
============================================================
加载 Google TimesFM-1.0-200m，独立预测 9 维 GNSS 序列。
用于提供 Forecasting 任务的强大非物理对比基线。
"""
import os
import sys
import yaml
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.dataset.data_engine import SSEDataset

def run_timesfm_baseline(config_path="configs/default_config.yaml", event_idx=0, horizon=30):
    """提取一个事件的 9 维 GNSS 数据，利用 TimeSFM 进行多通道零样本预测"""
    try:
        import timesfm
        import torch
    except ImportError:
        print("❌ Error: 必须先安装 timesfm: pip install timesfm")
        return

    config_full_path = os.path.join(PROJECT_ROOT, config_path)
    with open(config_full_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    data_dir = config["data"]["local_dir"]
    try:
        dataset = SSEDataset(data_dir=data_dir, config=config, normalize=True)
    except Exception as e:
        print(f"数据加载失败: {e}")
        return

    if len(dataset) <= event_idx:
        print(f"事件索引 {event_idx} 越界")
        return

    x_gnss, _ = dataset[event_idx]
    x_gnss_np = x_gnss.numpy()  # [T, 9]

    # 将前 T-horizon 作为 context，后 horizon 作为 Ground Truth
    T = x_gnss_np.shape[0]
    if T <= horizon:
        print("时间步不足以进行预测")
        return
        
    context_len = T - horizon
    inputs = []
    
    # 构造 inputs: 每一维一个 1D array
    for ch in range(9):
        inputs.append(x_gnss_np[:context_len, ch])

    device_backend = "gpu" if torch.cuda.is_available() else "cpu"
    print(f"🚀 初始化 Google TimesFM (Backend: {device_backend})...")
    
    tfm = timesfm.TimesFm(
        context_len=context_len,
        horizon_len=horizon,
        input_patch_len=32,
        output_patch_len=128,
        num_layers=20,
        model_dims=1280,
        backend="cpu", # Force CPU for compatibility unless explicitly configured for GPU compiling
    )
    
    # Load Weights
    print("⏳ 下载/加载模型权重 (google/timesfm-1.0-200m)...")
    try:
        tfm.load_from_checkpoint(repo_id="google/timesfm-1.0-200m")
    except Exception as e:
        print(f"❌ 加载模型权重失败: {e}")
        return

    print(f"🔮 预测未来 {horizon} 步...")
    forecast_output = tfm.forecast(
        inputs, 
        freq=[0]*9,  # 0 indicates no specific frequency mapping
    )
    
    point_forecasts = forecast_output.point_forecast  # [9, horizon]

    # 绘图展示
    fig, axes = plt.subplots(3, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    labels = ["S1-E", "S1-N", "S1-U", "S2-E", "S2-N", "S2-U", "S3-E", "S3-N", "S3-U"]
    
    total_mse = 0.0
    for ch in range(9):
        true_future = x_gnss_np[context_len:, ch]
        pred_future = point_forecasts[ch]
        total_mse += np.mean((true_future - pred_future)**2)
        
        ax = axes[ch]
        ax.plot(range(context_len), x_gnss_np[:context_len, ch], label="Context", color="black", alpha=0.6)
        ax.plot(range(context_len, T), true_future, label="True Future", color="blue")
        ax.plot(range(context_len, T), pred_future, label="TimeSFM", color="red", linestyle="--")
        
        ax.set_title(f"{labels[ch]}")
        if ch == 0:
            ax.legend()
            
    avg_rmse = np.sqrt(total_mse / 9)
    plt.suptitle(f"TimeSFM Zero-Shot Baseline (Event {event_idx}) | RMSE: {avg_rmse:.4f}")
    plt.tight_layout()
    
    out_img = os.path.join(PROJECT_ROOT, "timesfm_baseline.png")
    plt.savefig(out_img)
    print(f"✅ 基线测试完成！可视化结果已保存至: {out_img}")
    print(f"📊 平均 GNSS RMSE: {avg_rmse:.4f}")

if __name__ == "__main__":
    run_timesfm_baseline()
