import os, torch
from tqdm import tqdm
from accelerate import Accelerator
from .training_module import DiffusionTrainingModule
from .logger import ModelLogger
import torch
from tqdm import tqdm
import time
import time
from contextlib import contextmanager
from collections import defaultdict

import time
from collections import defaultdict
from contextlib import contextmanager
import statistics


import os
import torch
from datetime import datetime

class StepTimer:
    def __init__(self, log_file="training_perf.log"):
        self.times = defaultdict(list)
        self.step_keys = []
        self.log_file = log_file

    @contextmanager
    def time_step(self, name):
        if name not in self.step_keys:
            self.step_keys.append(name)
        # 确保 GPU 同步，否则时间统计不准
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        start = time.perf_counter()
        yield
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        self.times[name].append(elapsed)

    def record(self, name, elapsed):
        if name not in self.step_keys:
            self.step_keys.append(name)
        self.times[name].append(elapsed)

    def print_summary(self, accelerator):
        # 仅在主进程中执行打印和写文件
        if not accelerator.is_main_process:
            return

        if not self.step_keys:
            return
            
        # 以记录最多的 key 为准（通常是 data_loading）
        num_steps = max(len(self.times[k]) for k in self.step_keys)
        
        output = []
        output.append("\n" + "="*120)
        output.append(f"{'Step 耗时详情 (单位: ms) - ' + datetime.now().strftime('%Y-%m-%d %H:%M:%S'):^120s}")
        output.append("="*120)
        
        headers = ["Step"] + self.step_keys + ["Total"]
        col_width = 15 
        header_str = "".join([f"{h:>{col_width}s}" for h in headers])
        output.append(header_str)
        output.append("-" * len(header_str))

        for i in range(num_steps):
            row_vals = []
            step_total = 0.0
            for key in self.step_keys:
                # 健壮性处理：如果某项没有记录（比如梯度累积跳过了），记为 0
                if i < len(self.times[key]):
                    val = self.times[key][i] * 1000
                else:
                    val = 0.0
                step_total += val
                row_vals.append(f"{val:{col_width}.2f}")
            
            row_str = f"{i:>{col_width}d}" + "".join(row_vals) + f"{step_total:{col_width}.2f}"
            output.append(row_str)

        # 统计摘要
        output.append("\n" + "="*120)
        output.append(f"{'统计摘要 (平均值)':^120s}")
        output.append("="*120)
        total_time_all_steps = sum(sum(v) for v in self.times.values())
        
        for name in self.step_keys:
            values = self.times[name]
            if values:
                avg = statistics.mean(values) * 1000
                total = sum(values)
                ratio = (total / total_time_all_steps) * 100 if total_time_all_steps > 0 else 0
                output.append(f"{name:<25s} | 平均: {avg:10.2f} ms | 总计: {total:10.2f} s | 占比: {ratio:7.1f}%")
        
        final_log = "\n".join(output)
        
        # 1. 打印到终端
        print(final_log)
        
        # 2. 保存到文件
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(final_log + "\n")

def diagnose_default_training_status(model):
    """
    诊断模型当前的默认训练状态（在人工修改 requires_grad 之前）
    """
    print("\n" + "="*50)
    print("🕵️ [诊断模式] 检查模型默认训练状态...")
    print("="*50)
    
    trainable_params = []
    frozen_params = []
    
    trainable_numel = 0
    frozen_numel = 0
    
    for name, param in model.named_parameters():
        if param.requires_grad:
            trainable_params.append(name)
            trainable_numel += param.numel()
        else:
            frozen_params.append(name)
            frozen_numel += param.numel()
            
    # 统计数据
    total_layers = len(trainable_params) + len(frozen_params)
    total_params = trainable_numel + frozen_numel
    
    print(f"📊 统计结果:")
    print(f"   - 总层数 (Keys): {total_layers}")
    print(f"   - 总参数量 (Elements): {total_params / 1e9:.2f} B (十亿)")
    print(f"   -------------------------------------------")
    print(f"   🔓 可训练层数 (Trainable): {len(trainable_params)}")
    print(f"      - 参数量: {trainable_numel / 1e9:.2f} B")
    print(f"      - 占比: {trainable_numel / total_params * 100:.2f}%")
    print(f"   🔒 不可训练层数 (Frozen): {len(frozen_params)}")
    print(f"      - 参数量: {frozen_numel / 1e9:.2f} B")
    print(f"   -------------------------------------------")
    
    # 打印具体名字（为了防止刷屏，每种只打印前5个和后5个）
    if len(trainable_params) > 0:
        print(f"\n📝 可训练参数示例 (Top 5):")
        for p in trainable_params[:10]:
            print(f"   - [√] {p}")
        if len(trainable_params) > 10: print("   ... (中间省略) ...")
        # 打印最后几个，看看音频部分在不在
        for p in trainable_params[-10:]:
            print(f"   - [√] {p}")
            
    if len(frozen_params) > 0:
        print(f"\n🧊 不可训练参数示例 (Top 5):")
        for p in frozen_params[:10]:
            print(f"   - [x] {p}")
            
    print("="*50 + "\n")


def prepare_model_and_optimizer_groups(model, base_lr=1e-5, target_lr=1e-4):
    print("\n" + "="*50)
    print("🛠️  正在配置模型参数、初始化及学习率分组...")
    print("="*50)

    # 1. 定义高学习率（且需要置零）的目标模块前缀
    target_prefixes = (
        "audio_injector", 
        # "trainable_cond_mask", 
        # "frame_packer"
    )
    
    # 2. 容器初始化
    high_lr_params = []
    low_lr_params = []
    
    # 统计用变量
    stats = {
        "high_lr_count": 0,    # 高学习率参数个数
        "low_lr_count": 0,     # 低学习率参数个数 (Backbone中原本可训练的)
        "frozen_skipped": 0,   # 被跳过的冻结参数 (如 TextEncoder)
        "zero_value_count": 0, # 实际值为0的参数个数
        "total_params": 0
    }

    # 3. 遍历模型所有参数
    for name, param in model.named_parameters():
        stats["total_params"] += 1
        
        # 判断是否属于目标模块 (Audio/Mask/Packer)
        is_target_module = any(prefix in name for prefix in target_prefixes)
        
        if is_target_module:
            # ============================================
            # A. 目标模块：强制训练 + 强制置零 + 高学习率
            # ============================================
            param.requires_grad = True # 确保开启
            
            # 执行全量置零 (恢复你之前的逻辑)
            # with torch.no_grad():
            #     param.zero_()
            
            high_lr_params.append(param)
            stats["high_lr_count"] += 1
            
            # 验证置零
            if param.sum() == 0:
                stats["zero_value_count"] += 1
                
        else:
            # ============================================
            # B. 非目标模块：尊重原状态 (只收录本来就开了梯度的)
            # ============================================
            if param.requires_grad:
                # 原本就是可训练的 (比如 Backbone 的 Attention) -> 低学习率
                low_lr_params.append(param)
                stats["low_lr_count"] += 1
            else:
                # 原本就是冻结的 (比如 Text Encoder) -> 跳过，不进优化器
                stats["frozen_skipped"] += 1

    # 4. 打印详细统计报告
    print(f"\n📊 参数统计报告:")
    print(f"   -------------------------------------------")
    print(f"   [Total] 模型总参数层数: {stats['total_params']}")
    print(f"   -------------------------------------------")
    print(f"   🔥 [High LR Group] (Target Modules, lr={target_lr})")
    print(f"       - 包含: {target_prefixes}")
    print(f"       - 数量: {stats['high_lr_count']}")
    print(f"       - 置零验证: {stats['zero_value_count']} / {stats['high_lr_count']} (应相等)")
    
    print(f"   ❄️ [Low LR Group] (Backbone SFT, lr={base_lr})")
    print(f"       - 数量: {stats['low_lr_count']}")
    print(f"       - 说明: 这些是SFT权重中原本开启梯度的部分")
    
    print(f"   🧊 [Skipped/Frozen] (Not Training)")
    print(f"       - 数量: {stats['frozen_skipped']}")
    print(f"       - 说明: 这些参数保持冻结，不消耗显存存梯度 (如TextEncoder)")
    print(f"   -------------------------------------------")

    # 5. 构建优化器所需的参数组列表
    optimizer_grouped_parameters = [
        {
            "params": low_lr_params, 
            "lr": base_lr,
            "name": "backbone_low_lr"
        },
        {
            "params": high_lr_params, 
            "lr": target_lr,
            "name": "audio_new_high_lr"
        }
    ]
    
    return optimizer_grouped_parameters

def launch_training_task(
    accelerator: Accelerator,
    dataset: torch.utils.data.Dataset,
    model: DiffusionTrainingModule,
    model_logger: ModelLogger,
    learning_rate: float = 1e-5,
    weight_decay: float = 1e-2,
    num_workers: int = 1,
    save_steps: int = None,
    num_epochs: int = 1,
    args = None,
):
    if args is not None:
        # small_lr_rate = 1e-5
        learning_rate = args.learning_rate
        weight_decay = args.weight_decay
        num_workers = args.dataset_num_workers
        save_steps = args.save_steps
        num_epochs = args.num_epochs
        debug = args.debug
    
    if debug:
        diagnose_default_training_status(model)
    # optimizer = torch.optim.AdamW(model.trainable_modules(), lr=learning_rate, weight_decay=weight_decay)
    optimizer_grouped_parameters = prepare_model_and_optimizer_groups(
        model, 
        base_lr=1e-5, 
        target_lr=learning_rate
    )
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer)
    dataloader = torch.utils.data.DataLoader(dataset, shuffle=False, collate_fn=lambda x: x[0], num_workers=num_workers) if debug else torch.utils.data.DataLoader(dataset, shuffle=True, collate_fn=lambda x: x[0], num_workers=num_workers)
    
    model, optimizer, dataloader, scheduler = accelerator.prepare(model, optimizer, dataloader, scheduler)
    
    if debug:
        model_logger.on_training_start(accelerator, model)
        log_name = f"perf_debug_{datetime.now().strftime('%m%d_%H%M')}.log"
        timer = StepTimer(log_file=log_name)
        for epoch_id in range(num_epochs):
            end_time = time.perf_counter()
            for step_index, data in enumerate(tqdm(dataloader, desc=f"Epoch {epoch_id}", disable=not accelerator.is_main_process)):
                
                # 1. 提前退出判断
                if step_index > 50:
                    break
                
                # 2. 记录数据加载时间
                data_load_time = time.perf_counter() - end_time
                timer.record("data_loading", data_load_time)
                
                with accelerator.accumulate(model):
                    
                    with timer.time_step("zero_grad"):
                        optimizer.zero_grad()
                    
                    with timer.time_step("forward"):
                        loss = model(data)
                    
                    with timer.time_step("backward"):
                        accelerator.backward(loss)
                    
                    if accelerator.sync_gradients:
                        with timer.time_step("optimizer.step"):
                            optimizer.step()
                        
                        with timer.time_step("model_logger"):
                            model_logger.on_step_end(accelerator, model, save_steps)
                        
                        with timer.time_step("scheduler.step"):
                            scheduler.step()
                    else:
                        # 如果没有执行更新，填充 0 以保持 Timer 内部列表对齐
                        timer.record("optimizer.step", 0)
                        timer.record("model_logger", 0)
                        timer.record("scheduler.step", 0)

                # 重置 end_time 用于下一轮 data_loading 统计
                end_time = time.perf_counter()

        # 打印并保存结果
        accelerator.wait_for_everyone() # 确保所有进程完成
        timer.print_summary(accelerator)
        model_logger.on_training_end(accelerator, model, save_steps)

    else:
        model_logger.on_training_start(accelerator, model)
    
        for epoch_id in range(num_epochs):
            for data in tqdm(dataloader):
                with accelerator.accumulate(model):
                    optimizer.zero_grad()
                    loss = model(data)
                    accelerator.backward(loss)
                    optimizer.step()
                    model_logger.on_step_end(accelerator, model, save_steps)
                    scheduler.step() # 这一步为什么这么慢
        model_logger.on_training_end(accelerator, model, save_steps)


def launch_data_process_task(
    accelerator: Accelerator,
    dataset: torch.utils.data.Dataset,
    model: DiffusionTrainingModule,
    model_logger: ModelLogger,
    num_workers: int = 8,
    args = None,
):
    if args is not None:
        num_workers = args.dataset_num_workers
        
    dataloader = torch.utils.data.DataLoader(dataset, shuffle=False, collate_fn=lambda x: x[0], num_workers=num_workers)
    model, dataloader = accelerator.prepare(model, dataloader)
    
    for data_id, data in enumerate(tqdm(dataloader)):
        with accelerator.accumulate(model):
            with torch.no_grad():
                folder = os.path.join(model_logger.output_path, str(accelerator.process_index))
                os.makedirs(folder, exist_ok=True)
                save_path = os.path.join(model_logger.output_path, str(accelerator.process_index), f"{data_id}.pth")
                data = model(data)
                torch.save(data, save_path)
