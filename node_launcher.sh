#!/bin/bash

# -------- 环境设置 --------
CONDA_ENV_PATH="/m2v_intern/mengzijie/env/wan2.2"
export PATH="${CONDA_ENV_PATH}/bin:$PATH"
export LD_LIBRARY_PATH="${CONDA_ENV_PATH}/lib:$LD_LIBRARY_PATH"

export http_proxy=http://10.66.16.238:11080 
export https_proxy=http://10.66.16.238:11080 
export no_proxy=localhost,127.0.0.1,localaddress,localdomain.com,internal,corp.kuaishou.com,test.gifshow.com,staging.kuaishou.com

cd /m2v_intern/mengzijie/DiffSynth-Studio/
export PYTHONPATH=$PYTHONPATH:$(pwd)

# -------- 获取节点编号（从 MPI 获取后立即保存） --------
NODE_RANK=${OMPI_COMM_WORLD_RANK:-0}

echo ""
echo "=========================================="
echo "🖥️  Node $NODE_RANK / $((NUM_NODES - 1)) starting..."
echo "    Hostname: $(hostname)"
echo "    MASTER_ADDR: $MASTER_ADDR"
echo "    MASTER_PORT: $MASTER_PORT"
echo "    NUM_NODES: $NUM_NODES"
echo "    GPUS_PER_NODE: $GPUS_PER_NODE"
echo "    TOTAL_GPUS: $TOTAL_GPUS"
echo "=========================================="
echo ""

# ======================================================
# 🔑 关键修复：清除 MPI 环境变量，避免与 DeepSpeed 冲突
# ======================================================
echo "🧹 Unsetting MPI environment variables to avoid DeepSpeed conflict..."
unset OMPI_COMM_WORLD_RANK
unset OMPI_COMM_WORLD_SIZE
unset OMPI_COMM_WORLD_LOCAL_RANK
unset OMPI_COMM_WORLD_LOCAL_SIZE
unset OMPI_COMM_WORLD_NODE_RANK
unset PMI_RANK
unset PMI_SIZE
unset PMI_LOCAL_RANK
unset PMI_LOCAL_SIZE
unset PMIX_RANK
unset MPI_LOCALRANKID
unset MPI_LOCALNRANKS

# -------- 清理本节点的旧配置 --------
rm -f ~/.cache/huggingface/accelerate/default_config.yaml 2>/dev/null

# -------- 为当前节点创建 accelerate 配置文件 --------
ACCELERATE_CONFIG="/tmp/accelerate_config_node_${NODE_RANK}.yaml"
cat > $ACCELERATE_CONFIG << ACCEOF
compute_environment: LOCAL_MACHINE
debug: false
deepspeed_config:
  deepspeed_config_file: ${DS_CONFIG}
  zero3_init_flag: false
distributed_type: DEEPSPEED
downcast_bf16: 'no'
machine_rank: ${NODE_RANK}
main_process_ip: ${MASTER_ADDR}
main_process_port: ${MASTER_PORT}
main_training_function: main
num_machines: ${NUM_NODES}
num_processes: ${TOTAL_GPUS}
rdzv_backend: static
same_network: true
use_cpu: false
ACCEOF

echo "📄 Accelerate config for node $NODE_RANK:"
cat $ACCELERATE_CONFIG
echo ""

# -------- 使用配置文件启动 --------
accelerate launch --config_file $ACCELERATE_CONFIG \
    examples/wanvideo/model_training/train.py \
        --dataset_base_path "" \
        --dataset_metadata_path "/m2v_intern/mengzijie/DiffSynth-Studio/emo_ge81f_verified.csv" \
        --data_file_keys "video_path,audio_path" \
        --dataset_num_workers 8 \
        --save_steps 50 \
        --height 640 \
        --width 480 \
        --tgt_fps 15 \
        --num_frames 57 \
        --dataset_repeat 1 \
        --model_id_with_origin_paths "Wan-AI/Wan2.2-S2V-14B:diffusion_pytorch_model*.safetensors,Wan-AI/Wan2.2-S2V-14B:wav2vec2-large-xlsr-53-english/model.safetensors,Wan-AI/Wan2.2-S2V-14B:models_t5_umt5-xxl-enc-bf16.pth,Wan-AI/Wan2.2-S2V-14B:Wan2.1_VAE.pth" \
        --learning_rate 1e-4 \
        --num_epochs 1 \
        --trainable_models "dit" \
        --remove_prefix_in_ckpt "pipe.dit." \
        --output_path "/m2v_intern/mengzijie/DiffSynth-Studio/models/train/Wan2.2-S2V-14B_t16" \
        --extra_inputs "input_image,input_audio" \
        --use_gradient_checkpointing_offload
