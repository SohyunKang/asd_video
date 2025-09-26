import time
import os
import numpy as np

from models.model_gazelle import load_model, gaze_inference
from utils.utils_gazelle import load_timeseg, load_video, visualize_heatmap_to_video_save_prob, visualize_heatmap_to_video_with_sound

start = time.time()
f_name = 'IF2001_2_1_1024080292_0.mp4'

input_video_path = f'data/{f_name}'
output_video_path = f"results/{f_name[:-4]}_gazed.mp4"
timeseg_file_path = "data/samples_timesegments_total.json"

frames, width, height, fps, frame_count = load_video(input_video_path)
start_sec = load_timeseg(timeseg_file_path, input_video_path)
end = time.time()

# data load
bboxes_all = np.load("models/save_outputs/bboxes_all.npy")
valid_frame_num = np.load("models/save_outputs/valid_frame_num.npy")

print(f'---> Face Recog Data Load Time: {time.time()-end:.4f}')
end = time.time()

model, transform, device = load_model()

print(f'---> Gazelle Model Load Time: {time.time()-end:.4f}')
end = time.time()

# gazelle 
outputs = gaze_inference(bboxes_all, valid_frame_num, frames, transform, device, width, height, model)

print(f'---> Gazelle Inference Time: {time.time()-end:.4f}')
end = time.time()

end_sec = visualize_heatmap_to_video_save_prob(frames, output_video_path, outputs, valid_frame_num, fps, start_sec, bboxes_all, model, input_video_path)
print(f'---> Visualization 1 Time: {time.time()-end:.4f} / Save to {output_video_path}')
print(f'****** After Time 1 ******: {time.time()-start:.4f}')
end = time.time()

output_video_with_sound_path = f"results/{input_video_path.split('/')[1][:-4]}_gaze_sound.mp4"

visualize_heatmap_to_video_with_sound(frames, output_video_with_sound_path, outputs, valid_frame_num, fps, start_sec, bboxes_all, model, input_video_path, end_sec)

print(f'---> Visualization 2 Time: {time.time()-end:.4f} / Save to {output_video_with_sound_path}')

print(f'****** After Time 2 ******: {time.time()-start:.4f}')