source /video/anaconda/etc/profile.d/conda.sh
conda activate /m2v_intern/mengzijie/env/wan2.2/
export PATH="/m2v_intern/mengzijie/env/wan2.2/bin:$PATH"
# source /m2v_intern/mengzijie/env/wan2.2/bin/activate

export http_proxy=http://10.66.16.238:11080 
export https_proxy=http://10.66.16.238:11080 
export no_proxy=localhost,127.0.0.1,localaddress,localdomain.com,internal,corp.kuaishou.com,test.gifshow.com,staging.kuaishou.com

# 确保 NCCL 通信正常
export NCCL_IB_DISABLE=0
export NCCL_SOCKET_IFNAME=eth01  # 根据实际网卡名修改
export NCCL_DEBUG=INFO

cd /m2v_intern/mengzijie/DiffSynth-Studio

accelerate launch --config_file accelerate_config_node0.yaml \
  examples/wanvideo/model_training/train.py \
  --dataset_base_path "" \
  --dataset_metadata_path "/m2v_intern/mengzijie/DiffSynth-Studio/720_filter_emo_40w_30fps_split_121f_with_md5_vae_caption_te_0925_40w_fix_pose.csv" \
  --data_file_keys "video_path,audio_path" \
  --dataset_num_workers 0 \
  --save_steps 30 \
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
  --output_path "/m2v_intern/mengzijie/DiffSynth-Studio/models/train/Wan2.2-S2V-14B_debug" \
  --extra_inputs "input_image,input_audio" \
  --use_gradient_checkpointing_offload