import pandas as pd
import os
from decord import VideoReader
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

def get_frame_count(video_path):
    """只读取视频元信息获取帧数，不读取实际帧内容"""
    try:
        if not os.path.exists(video_path):
            return -1  # 文件不存在
        vr = VideoReader(video_path)
        frame_count = len(vr)
        del vr
        return frame_count
    except Exception as e:
        return -2  # 读取失败

def main():
    # ============ 配置 ============
    input_csv = '/ytech_m2v2_hdd/liujiwen/audio_v3/m2v-diffusers/id_data_480_720_1080_with_pose/720_filter_emo_40w_30fps_split_121f_with_md5_vae_caption_te_0925_40w_fix_pose.csv'
    output_csv = 'emo_ge81f_verified.csv'
    min_frames = 81
    num_workers = 64  # 根据你的机器调整，IO密集型可以开大一点
    # ==============================

    # 读取 CSV
    print("读取 CSV...")
    df = pd.read_csv(input_csv)
    total = len(df)
    print(f"总样本数: {total:,}")

    # 获取所有视频路径
    video_paths = df['video_path'].tolist()

    # 先测试几个样本估算时间
    print("\n测试采样估算时间...")
    test_count = min(100, total)
    start_time = time.time()
    for i in tqdm(range(test_count), desc="采样测试"):
        get_frame_count(video_paths[i])
    elapsed = time.time() - start_time
    per_sample = elapsed / test_count
    
    # 估算总时间（考虑多线程加速）
    estimated_single = per_sample * total
    estimated_parallel = estimated_single / num_workers
    print(f"\n单样本耗时: {per_sample*1000:.1f} ms")
    print(f"预估总时间 (单线程): {estimated_single/60:.1f} 分钟")
    print(f"预估总时间 ({num_workers}线程): {estimated_parallel/60:.1f} 分钟")

    # 多线程检查帧数
    print(f"\n开始检查所有视频帧数 (workers={num_workers})...")
    frame_counts = [None] * total

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        # 提交所有任务
        futures = {executor.submit(get_frame_count, path): i for i, path in enumerate(video_paths)}

        # 收集结果
        for future in tqdm(as_completed(futures), total=total, desc="检查帧数", 
                          unit="video", smoothing=0.1):
            idx = futures[future]
            frame_counts[idx] = future.result()

    # 添加实际帧数列
    df['actual_frame_count'] = frame_counts

    # 统计
    not_exist = (df['actual_frame_count'] == -1).sum()
    read_error = (df['actual_frame_count'] == -2).sum()
    too_short = ((df['actual_frame_count'] >= 0) & (df['actual_frame_count'] < min_frames)).sum()
    valid_count = (df['actual_frame_count'] >= min_frames).sum()

    print(f"\n{'='*50}")
    print(f"统计结果:")
    print(f"  ✓ 有效样本 (>= {min_frames} 帧): {valid_count:,} ({valid_count/total*100:.2f}%)")
    print(f"  ✗ 帧数不足 (< {min_frames} 帧):  {too_short:,}")
    print(f"  ✗ 文件不存在:                    {not_exist:,}")
    print(f"  ✗ 读取失败:                      {read_error:,}")
    print(f"{'='*50}")

    # 对比 video_length 字段和实际帧数
    valid_rows = df['actual_frame_count'] >= 0
    mismatch = (df.loc[valid_rows, 'video_length'] != df.loc[valid_rows, 'actual_frame_count']).sum()
    if mismatch > 0:
        print(f"\n⚠️  警告: {mismatch:,} 个样本的 video_length 与实际帧数不符!")
    else:
        print(f"\n✓ video_length 字段与实际帧数完全一致")

    # 过滤并保存
    df_filtered = df[df['actual_frame_count'] >= min_frames]
    df_filtered.to_csv(output_csv, index=False)
    print(f"\n已保存到: {output_csv}")
    print(f"保存样本数: {len(df_filtered):,}")

if __name__ == '__main__':
    main()