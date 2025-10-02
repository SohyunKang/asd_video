import time
import os
import numpy as np

from models.model_gazelle import load_model, gaze_inference
from utils.utils_gazelle import load_timeseg, load_video, visualize_heatmap_to_video_save_prob, visualize_heatmap_to_video_with_sound

import sys

if len(sys.argv) < 2:
    print("❌ mp4 파일 이름을 입력하세요")
    sys.exit(1)

f_name = sys.argv[1]  # 첫 번째 인자
print("받은 영상 파일:", f_name)

start = time.time()
# f_name = 'IF2001_3_1_1024060343_0.mp4'  # 통합 필요

input_video_path = f_name
output_video_path = f"results/{input_video_path.split('/')[-1].split('.')[0]}_gazed.mp4"
timeseg_file_path = "data/samples_timesegments_total.json"

frames, width, height, fps, frame_count = load_video(input_video_path)
start_sec = load_timeseg(timeseg_file_path, input_video_path)
end = time.time()

# data load
bboxes_all = np.load(f"results/save_outputs/{input_video_path.split('/')[-1].split('.')[0]}_bboxes_all.npy")
valid_frame_num = np.load(f"results/save_outputs/{input_video_path.split('/')[-1].split('.')[0]}_valid_frame_num.npy")

print(f'---> Face Recog Data Load Time: {time.time()-end:.4f}')
end = time.time()

model, transform, device = load_model()

print(f'---> Gazelle Model Load Time: {time.time()-end:.4f}')
end = time.time()

# gazelle 
outputs = gaze_inference(bboxes_all, valid_frame_num, frames, transform, device, width, height, model)

print(f'---> Gazelle Inference Time: {time.time()-end:.4f}')
end = time.time()

os.makedirs("results", exist_ok=True)
end_sec = visualize_heatmap_to_video_save_prob(frames, output_video_path, outputs, valid_frame_num, fps, start_sec, bboxes_all, model, input_video_path)
print(f'---> Visualization 1 Time: {time.time()-end:.4f} / Save to {output_video_path}')
print(f'****** After Time 1 ******: {time.time()-start:.4f}')
end = time.time()

output_video_with_sound_path = f"results/{input_video_path.split('/')[-1].split('.')[0]}_gaze_sound.mp4"

visualize_heatmap_to_video_with_sound(frames, output_video_with_sound_path, outputs, valid_frame_num, fps, start_sec, bboxes_all, model, input_video_path, end_sec)

print(f'---> Visualization 2 Time: {time.time()-end:.4f} / Save to {output_video_with_sound_path}')

print(f'****** After Time 2 ******: {time.time()-start:.4f}')