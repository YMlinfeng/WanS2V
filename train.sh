wan
export MODELSCOPE_CACHE=/m2v_intern/mengzijie/Wan2.2/

# accelerate launch --config_file examples/wanvideo/model_training/full/accelerate_config_14B.yaml examples/wanvideo/model_training/train.py \
#   --dataset_base_path /m2v_intern/mengzijie/data/example_video_dataset \
#   --dataset_metadata_path /m2v_intern/mengzijie/data/example_video_dataset/metadata_s2v.csv \
#   --data_file_keys "video,input_audio,s2v_pose_video" \
#   --height 720 \
#   --width 480 \
#   --num_frames 81 \
#   --dataset_repeat 3 \
#   --model_id_with_origin_paths "Wan-AI/Wan2.2-S2V-14B:diffusion_pytorch_model*.safetensors,Wan-AI/Wan2.2-S2V-14B:wav2vec2-large-xlsr-53-english/model.safetensors,Wan-AI/Wan2.2-S2V-14B:models_t5_umt5-xxl-enc-bf16.pth,Wan-AI/Wan2.2-S2V-14B:Wan2.1_VAE.pth" \
#   --learning_rate 1e-5 \
#   --num_epochs 1 \
#   --trainable_models "dit" \
#   --remove_prefix_in_ckpt "pipe.dit." \
#   --output_path "/m2v_intern/mengzijie/DiffSynth-Studio/models/train/Wan2.2-S2V-14B_full" \
#   --extra_inputs "input_image,input_audio,s2v_pose_video" \
#   --use_gradient_checkpointing_offload