python debug_dataset.py  \
  --dataset_base_path "" \
  --dataset_metadata_path "/m2v_intern/mengzijie/data/example_video_dataset/test_emo.csv" \
  --data_file_keys "video_path,audio_path" \
  --height 740 \
  --width 480 \
  --num_frames 81 \
  --dataset_repeat 1 \
  --extra_inputs "input_image,input_audio"