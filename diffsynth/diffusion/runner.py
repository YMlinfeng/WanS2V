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

class StepTimer:
    def __init__(self):
        self.times = defaultdict(list)
    
    @contextmanager
    def time_step(self, name):
        start = time.perf_counter()
        yield
        elapsed = time.perf_counter() - start
        self.times[name].append(elapsed)
    
    def print_summary(self):
        print("\n" + "="*60)
        print("计时统计摘要")
        print("="*60)
        for name, times in self.times.items():
            avg = sum(times) / len(times)
            total = sum(times)
            print(f"{name:25s}: 平均 {avg*1000:8.2f}ms | 总计 {total:8.2f}s | 次数 {len(times)}")
        print("="*60)

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
    
    # model_logger.on_training_start(accelerator, model)
    
    # for epoch_id in range(num_epochs):
    #     for data in tqdm(dataloader):
    #         with accelerator.accumulate(model):
    #             optimizer.zero_grad()
    #             loss = model(data)
    #             accelerator.backward(loss)
    #             optimizer.step()
    #             model_logger.on_step_end(accelerator, model, save_steps)
    #             scheduler.step() # 这一步为什么这么慢
    # model_logger.on_training_end(accelerator, model, save_steps)

    timer = StepTimer()

    for epoch_id in range(num_epochs):
        # for data in tqdm(dataloader):
        for step_index, data in enumerate(tqdm(dataloader, desc=f"Epoch {epoch_id}")):
            with accelerator.accumulate(model):
                
                with timer.time_step("zero_grad"):
                    optimizer.zero_grad()
                
                with timer.time_step("forward"):
                    loss = model(data)
                
                with timer.time_step("backward"):
                    accelerator.backward(loss)
                
                with timer.time_step("optimizer.step"):
                    optimizer.step()
                
                with timer.time_step("model_logger"):
                    model_logger.on_step_end(accelerator, model, save_steps)
                
                with timer.time_step("scheduler.step"):
                    scheduler.step()

    model_logger.on_training_end(accelerator, model, save_steps)

    # 打印统计结果
    timer.print_summary()



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
