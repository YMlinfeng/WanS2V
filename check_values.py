import torch
from safetensors.torch import load_file
import numpy as np

# ================= 配置 =================
# 你的权重
MY_CKPT = "/m2v_intern/mengzijie/DiffSynth-Studio/models/train/Wan2.2-S2V-14B_t8/initial.safetensors"
# 官方权重 (如果你没有合并成一个文件，这里需要加载所有分卷并合并 dict，或者只加载第一卷测一部分)
# 为了方便，这里我们假设你有一个办法加载官方完整权重，或者我们只对比第一卷里的重合部分
OFFICIAL_CKPT_1 = "/m2v_intern/mengzijie/DiffSynth-Studio/models/Wan-AI/Wan2.2-S2V-14B/diffusion_pytorch_model-00001-of-00004.safetensors"
# 如果你想对比全量，可以在这里把4个文件都 load 进来 update 到一个 dict
OFFICIAL_CKPT_PATHS = [
    f"/m2v_intern/mengzijie/DiffSynth-Studio/models/Wan-AI/Wan2.2-S2V-14B/diffusion_pytorch_model-0000{i}-of-00004.safetensors" 
    for i in range(1, 5)
]
# =======================================

print("⏳ 正在加载你的 checkpoint...")
my_state_dict = load_file(MY_CKPT, device="cpu")

print("⏳ 正在加载官方 checkpoint (全量)...")
off_state_dict = {}
for p in OFFICIAL_CKPT_PATHS:
    print(f"  -> Loading {p}...")
    off_state_dict.update(load_file(p, device="cpu"))

print("⚔️ 开始全量数值对比...")
diff_keys = []
param_count = 0

for key in my_state_dict:
    if key not in off_state_dict:
        continue # 忽略不匹配的 Key（前面 check_keys 已经排除了）
    
    v_my = my_state_dict[key].float()
    v_off = off_state_dict[key].float()
    
    # 简单的 MSE 检查
    diff = (v_my - v_off).abs().max().item()
    
    if diff > 1e-3: # 容忍度设为 0.001
        print(f"❌ 发现差异! Key: {key}")
        print(f"   Max Diff: {diff:.6f}")
        print(f"   My  Mean: {v_my.mean():.6f}, Std: {v_my.std():.6f}")
        print(f"   Off Mean: {v_off.mean():.6f}, Std: {v_off.std():.6f}")
        diff_keys.append(key)
        
        # 只要发现前5个错误就停止打印，避免刷屏
        if len(diff_keys) >= 5:
            print("... (更多差异省略)")
            break
    
    param_count += 1
    if param_count % 100 == 0:
        print(f"已扫描 {param_count} 个参数...")

print("=" * 40)
if len(diff_keys) > 0:
    print(f"🔴 致命发现：共有 {len(diff_keys)} (或更多) 个参数数值不一致！")
    print("这解释了为什么会有噪声：部分权重被重置了！")
else:
    print("🟢 全量对比通过：所有参数数值完全一致。")
    print("如果这样还是噪声，那只能是 PyTorch 运行时的玄学了（如 CUDA 核心计算错误）。")