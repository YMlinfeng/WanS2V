import os
import torch
from safetensors.torch import safe_open

# ================= 配置区域 =================

# 1. 你的训练保存的 Checkpoint 路径 (出问题的那个)
MY_CKPT_PATH = "/m2v_intern/mengzijie/DiffSynth-Studio/models/train/initial.safetensors"

# 2. 官方 Checkpoint 路径 (列表，因为它通常是分卷的)
# 如果官方是单个文件，列表里写一个路径即可
OFFICIAL_CKPT_PATHS = [
    f"/m2v_intern/mengzijie/DiffSynth-Studio/models/Wan-AI/Wan2.2-S2V-14B/diffusion_pytorch_model-0000{i}-of-00004.safetensors" 
    for i in range(1, 5)
]

# 3. 输出对比结果的文件名
OUTPUT_FILE = "keys_comparison_result.txt"

# ===========================================

def load_keys_from_safetensors(file_paths):
    all_keys = set()
    if isinstance(file_paths, str):
        file_paths = [file_paths]
        
    print(f"正在读取: {file_paths} ...")
    for path in file_paths:
        if not os.path.exists(path):
            print(f"❌ 文件不存在: {path}")
            continue
            
        try:
            # 尝试作为 safetensors 读取
            with safe_open(path, framework="pt") as f:
                keys = f.keys()
                all_keys.update(keys)
        except Exception as e:
            print(f"⚠️ 无法作为 safetensors 读取 {path}, 尝试 torch.load...")
            try:
                # 尝试作为 pytorch bin 读取
                state_dict = torch.load(path, map_location="cpu")
                all_keys.update(state_dict.keys())
            except Exception as e2:
                print(f"❌ 读取失败: {e2}")
    
    return sorted(list(all_keys))

def analyze_prefix_diff(official_keys, my_keys):
    """分析是否存在常见的前缀差异（如 module.）"""
    if not official_keys or not my_keys:
        return "无法分析（Keys为空）"
    
    off_k = official_keys[0]
    my_k = my_keys[0]
    
    msg = []
    if my_k.startswith("module.") and not off_k.startswith("module."):
        msg.append("⚠️ 警告: 你的 Key 包含 'module.' 前缀，而官方没有！这通常是 DDP 保存导致的。")
        msg.append("👉 解决方法: 在保存时遍历 state_dict，把 key.replace('module.', '') 去掉。")
    elif not my_k.startswith("module.") and off_k.startswith("module."):
        msg.append("⚠️ 警告: 官方 Key 包含 'module.' 前缀，而你的没有。")
    else:
        msg.append("✅ 前缀看起来一致（或都无特殊前缀）。")
        
    return "\n".join(msg)

def main():
    print(">>> 开始提取 Key...")
    
    # 获取 keys
    my_keys = load_keys_from_safetensors(MY_CKPT_PATH)
    official_keys = load_keys_from_safetensors(OFFICIAL_CKPT_PATHS)
    
    print(f"我的 Checkpoint Key 数量: {len(my_keys)}")
    print(f"官方 Checkpoint Key 数量: {len(official_keys)}")
    
    # 集合运算
    set_my = set(my_keys)
    set_off = set(official_keys)
    
    common = set_my & set_off
    only_in_my = set_my - set_off
    only_in_off = set_off - set_my
    
    # 写入文件
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write("="*60 + "\n")
        f.write("KEY 对比诊断报告\n")
        f.write("="*60 + "\n\n")
        
        # 1. 快速诊断
        f.write("【1. 前缀快速诊断】\n")
        if len(my_keys) > 0 and len(official_keys) > 0:
            f.write(f"我的第一个 Key:   {my_keys[0]}\n")
            f.write(f"官方第一个 Key:   {official_keys[0]}\n")
            f.write("-" * 30 + "\n")
            f.write(analyze_prefix_diff(official_keys, my_keys) + "\n")
        else:
            f.write("无法诊断（文件可能为空）\n")
        f.write("\n")

        # 2. 统计信息
        f.write("【2. 统计信息】\n")
        f.write(f"我的 Key 总数: {len(my_keys)}\n")
        f.write(f"官方 Key 总数: {len(official_keys)}\n")
        f.write(f"完全匹配的 Key 数量: {len(common)}\n")
        f.write(f"仅在我的文件中: {len(only_in_my)}\n")
        f.write(f"仅在官方文件中: {len(only_in_off)}\n\n")
        
        # 3. 仅在我的文件中 (Expected Unexpected)
        f.write("="*60 + "\n")
        f.write(f"【仅在我的文件中 (Unexpected Keys)】 (Top 50 of {len(only_in_my)})\n")
        f.write("说明: 如果这里全是带 'module.' 前缀的 key，说明就是前缀问题。\n")
        f.write("="*60 + "\n")
        for k in sorted(list(only_in_my))[:50]:
            f.write(f"  {k}\n")
        if len(only_in_my) > 50: f.write("  ... (更多省略)\n")
            
        # 4. 仅在官方文件中 (Missing)
        f.write("\n" + "="*60 + "\n")
        f.write(f"【仅在官方文件中 (Missing Keys)】 (Top 50 of {len(only_in_off)})\n")
        f.write("说明: 如果这里全是对应的不带 'module.' 的 key，再次确认为前缀问题。\n")
        f.write("="*60 + "\n")
        for k in sorted(list(only_in_off))[:50]:
            f.write(f"  {k}\n")
        if len(only_in_off) > 50: f.write("  ... (更多省略)\n")

        # 5. 全部 Key 列表 (我的)
        f.write("\n" + "="*60 + "\n")
        f.write("【附录：我的全部 Key】\n")
        f.write("="*60 + "\n")
        for k in my_keys:
            f.write(f"{k}\n")

    print(f"\n✅ 对比完成！结果已保存至: {os.path.abspath(OUTPUT_FILE)}")
    print("请打开该文本文件，重点查看【1. 前缀快速诊断】和【仅在我的文件中】部分。")

if __name__ == "__main__":
    main()