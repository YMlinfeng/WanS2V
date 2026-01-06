import torch
from PIL import Image
import librosa
from diffsynth.utils.data import VideoData, save_video_with_audio
from diffsynth.core import load_state_dict
from safetensors.torch import load_file as load_safetensors
from diffsynth.pipelines.wan_video import WanVideoPipeline, ModelConfig


import os
print("MODELSCOPE_CACHE:", os.environ.get("MODELSCOPE_CACHE"))

print("start downloading...")
pipe = WanVideoPipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[
        ModelConfig(model_id="Wan-AI/Wan2.2-S2V-14B", origin_file_pattern="diffusion_pytorch_model*.safetensors"),
        ModelConfig(model_id="Wan-AI/Wan2.2-S2V-14B", origin_file_pattern="wav2vec2-large-xlsr-53-english/model.safetensors"),
        ModelConfig(model_id="Wan-AI/Wan2.2-S2V-14B", origin_file_pattern="models_t5_umt5-xxl-enc-bf16.pth"),
        ModelConfig(model_id="Wan-AI/Wan2.2-S2V-14B", origin_file_pattern="Wan2.1_VAE.pth"),
    ],
    audio_processor_config=ModelConfig(model_id="Wan-AI/Wan2.2-S2V-14B", origin_file_pattern="wav2vec2-large-xlsr-53-english/"),
)

# 只在主进程开启调试（避免多进程冲突）
if os.environ.get("LOCAL_RANK", "0") == "0":
    import debugpy
    debugpy.listen(("0.0.0.0", 5678))
    print("=" * 50)
    print("Waiting for debugger to attach on port 5678...")
    print("=" * 50)
    debugpy.wait_for_client()  
    print("Debugger attached! Continuing...")

# num_frames = 57 # 4n+1
# height = 640
# width = 480

num_frames = 9 # 4n+1
height = 128
width = 64

#原始

state_dict = load_state_dict("/m2v_intern/mengzijie/DiffSynth-Studio/models/train/Wan2.2-S2V-14B_T16/initial.safetensors")
# state_dict = load_state_dict([f"/m2v_intern/mengzijie/DiffSynth-Studio/models/Wan-AI/Wan2.2-S2V-14B/diffusion_pytorch_model-0000{i}-of-00004.safetensors" for i in range(1, 5)])
# state_dict = load_state_dict("/m2v_intern/mengzijie/DiffSynth-Studio/models/train/Wan2.2-S2V-14B_T16/step-1410.safetensors")

missing, unexpected = pipe.dit.load_state_dict(state_dict, strict=True)




prompt = "a person is singing"
# prompt = "FPS-30. The video plays at normal speed. A person speaks directly to the camera in a well-lit space, delivering content in a clear, steady tone that suggests a speech or casual address. The individual’s appearance details are not specified, with the focus entirely on their verbal expression. The background is not clearly defined, presenting a simple and uncluttered setting that does not distract from the speaker. The person maintains a relatively upright posture while talking, occasionally making slight head or hand gestures that match the rhythm of their speech to emphasize key points. realistic, appearing as a short clip, possibly from a personal talk, a class presentation or a casual online sharing video, with natural lighting that highlights the subject. The scene is characterized by medium saturation, moderate contrast, moderate brightness and neutral-toned colors. The camera is fixed, capturing a medium shot of the person, with the lens at eye level. The speaker is positioned in the center of the frame. The camera remains stationary, with a medium depth of field, keeping the person in sharp focus while the background has a subtle blur, emphasizing the verbal delivery of the subject."
negative_prompt = "画面模糊，最差质量，画面模糊，细节模糊不清，情绪激动剧烈，手快速抖动，字幕，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"
input_image = Image.open("/m2v_intern/mengzijie/data/example_video_dataset/wans2v/pose.png").convert("RGB").resize((width, height))
# s2v audio input, recommend 16kHz sampling rate
audio_path = '/m2v_intern/mengzijie/data/example_video_dataset/wans2v/sing.MP3'
# audio_path = '/m2v_intern/mengzijie/DiffSynth-Studio/debug_vis_output_debug/sample_0001_start3_temp_audio.wav'
input_audio, sample_rate = librosa.load(audio_path, sr=16000)
# S2V pose video input 
# pose_video_path = '/m2v_intern/mengzijie/data/example_video_dataset/wans2v/pose.mp4'
# pose_video = VideoData(pose_video_path, height=height, width=width)

# Speech-to-video with pose
video = pipe(
    prompt=prompt,
    input_image=input_image,
    negative_prompt=negative_prompt,
    seed=0,
    num_frames=num_frames,
    height=height,
    width=width,
    audio_sample_rate=sample_rate,
    input_audio=input_audio,
    # s2v_pose_video=pose_video,
    # num_inference_steps=40,
    num_inference_steps=40
)
save_video_with_audio(video[1:], "ori.mp4", audio_path, fps=16, quality=5)



