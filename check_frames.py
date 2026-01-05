import pandas as pd
import cv2
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import os

csv_path = "/m2v_intern/mengzijie/DiffSynth-Studio/720_filter_emo_40w_30fps_split_121f_with_md5_vae_caption_te_0925_40w_fix_pose.csv"
output_path = "/m2v_intern/mengzijie/DiffSynth-Studio/fix_filtered.csv"
min_frames = 81

def get_frame_count(video_path):
    """获取视频真实帧数"""
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return -1
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        return frame_count
    except Exception as e:
        return -1

def process_row(idx, row):
    """处理单行，返回 (idx, 真实帧数)"""
    video_path = row['video_path']
    real_frames = get_frame_count(video_path)
    return idx, real_frames

# 读取 CSV
print("读取 CSV...")
df = pd.read_csv(csv_path)
original_count = len(df)
print(f"原始样本数: {original_count}")

# 多线程统计帧数
print("统计真实帧数（多线程）...")
real_frame_counts = {}

with ThreadPoolExecutor(max_workers=64) as executor:
    futures = {
        executor.submit(process_row, idx, row): idx 
        for idx, row in df.iterrows()
    }
    
    for future in tqdm(as_completed(futures), total=len(futures)):
        idx, frame_count = future.result()
        real_frame_counts[idx] = frame_count

# 添加真实帧数列
df['real_frame_count'] = df.index.map(real_frame_counts)

# 统计信息
print("\n" + "="*50)
print("统计信息：")
print(f"  原始样本数: {original_count}")
print(f"  无法读取的视频: {(df['real_frame_count'] == -1).sum()}")
print(f"  帧数 < {min_frames} 的视频: {((df['real_frame_count'] < min_frames) & (df['real_frame_count'] >= 0)).sum()}")
print(f"  帧数 >= {min_frames} 的视频: {(df['real_frame_count'] >= min_frames).sum()}")

# 对比原始 video_length 字段
if 'video_length' in df.columns:
    mismatch = df[df['video_length'] != df['real_frame_count']]
    print(f"\n  video_length 字段与真实帧数不符的: {len(mismatch)}")
    if len(mismatch) > 0:
        print("  不符样例（前10个）：")
        for _, row in mismatch.head(10).iterrows():
            print(f"    {row['video_path']}: csv={row['video_length']}, 真实={row['real_frame_count']}")

# 过滤
df_filtered = df[df['real_frame_count'] >= min_frames].copy()
print(f"\n过滤后样本数: {len(df_filtered)}")
print(f"过滤掉: {original_count - len(df_filtered)}")

# 更新 video_length 为真实值（可选）
df_filtered['video_length'] = df_filtered['real_frame_count']
df_filtered = df_filtered.drop(columns=['real_frame_count'])

# 保存
df_filtered.to_csv(output_path, index=False)
print(f"\n已保存到: {output_path}")