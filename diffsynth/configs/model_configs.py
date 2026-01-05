
MODEL_CONFIGS  = [
    {
        # Example: ModelConfig(model_id="Wan-AI/Wan2.1-T2V-14B", origin_file_pattern="models_t5_umt5-xxl-enc-bf16.pth")
        "model_hash": "9c8818c2cbea55eca56c7b447df170da",
        "model_name": "wan_video_text_encoder",
        "model_class": "diffsynth.models.wan_video_text_encoder.WanTextEncoder",
    },
    {
        # Example: ModelConfig(model_id="Wan-AI/Wan2.1-T2V-14B", origin_file_pattern="Wan2.1_VAE.pth")
        "model_hash": "ccc42284ea13e1ad04693284c7a09be6",
        "model_name": "wan_video_vae",
        "model_class": "diffsynth.models.wan_video_vae.WanVideoVAE",
        "state_dict_converter": "diffsynth.utils.state_dict_converters.wan_video_vae.WanVideoVAEStateDictConverter",
    },
    {
        # Example: ModelConfig(model_id="Wan-AI/Wan2.1-I2V-14B-480P", origin_file_pattern="models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth")
        "model_hash": "5941c53e207d62f20f9025686193c40b",
        "model_name": "wan_video_image_encoder",
        "model_class": "diffsynth.models.wan_video_image_encoder.WanImageEncoder",
        "state_dict_converter": "diffsynth.utils.state_dict_converters.wan_video_image_encoder.WanImageEncoderStateDictConverter"
    },
    {
        # Example: ModelConfig(model_id="Wan-AI/Wan2.1-T2V-14B", origin_file_pattern="diffusion_pytorch_model*.safetensors")
        "model_hash": "aafcfd9672c3a2456dc46e1cb6e52c70",
        "model_name": "wan_video_dit",
        "model_class": "diffsynth.models.wan_video_dit.WanModel",
        "extra_kwargs": {'has_image_input': False, 'patch_size': [1, 2, 2], 'in_dim': 16, 'dim': 5120, 'ffn_dim': 13824, 'freq_dim': 256, 'text_dim': 4096, 'out_dim': 16, 'num_heads': 40, 'num_layers': 40, 'eps': 1e-06}
    },
    {
        # Example: ModelConfig(model_id="Wan-AI/Wan2.2-I2V-A14B", origin_file_pattern="high_noise_model/diffusion_pytorch_model*.safetensors")
        "model_hash": "5b013604280dd715f8457c6ed6d6a626",
        "model_name": "wan_video_dit",
        "model_class": "diffsynth.models.wan_video_dit.WanModel",
        "extra_kwargs": {'has_image_input': False, 'patch_size': [1, 2, 2], 'in_dim': 36, 'dim': 5120, 'ffn_dim': 13824, 'freq_dim': 256, 'text_dim': 4096, 'out_dim': 16, 'num_heads': 40, 'num_layers': 40, 'eps': 1e-06, 'require_clip_embedding': False}
    },
    {
        # Example: ModelConfig(model_id="Wan-AI/Wan2.2-S2V-14B", origin_file_pattern="diffusion_pytorch_model*.safetensors")
        "model_hash": "966cffdcc52f9c46c391768b27637614",
        "model_name": "wan_video_dit",
        "model_class": "diffsynth.models.wan_video_dit_s2v.WanS2VModel",
        "extra_kwargs": {'dim': 5120, 'in_dim': 16, 'ffn_dim': 13824, 'out_dim': 16, 'text_dim': 4096, 'freq_dim': 256, 'eps': 1e-06, 'patch_size': (1, 2, 2), 'num_heads': 40, 'num_layers': 40, 'cond_dim': 16, 'audio_dim': 1024, 'num_audio_token': 4}
    },
    {
        # Example: ModelConfig(model_id="Wan-AI/Wan2.2-S2V-14B", origin_file_pattern="wav2vec2-large-xlsr-53-english/model.safetensors")
        "model_hash": "06be60f3a4526586d8431cd038a71486",
        "model_name": "wans2v_audio_encoder",
        "model_class": "diffsynth.models.wav2vec.WanS2VAudioEncoder",
        "state_dict_converter": "diffsynth.utils.state_dict_converters.wans2v_audio_encoder.WanS2VAudioEncoderStateDictConverter",
    },
]

