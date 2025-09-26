import time
import os

from models.model_face_recog import face_recog
from utils.utils_face_recog import load_timeseg, load_video

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf

start = time.time()
f_name = 'IF2001_2_1_1024080292_0.mp4'
timeseg_file_path = "data/samples_timesegments_total.json"

input_video_path = f'data/{f_name}'

print(f'****** Input Video: {input_video_path} ******')

frames, width, height, fps, frame_count = load_video(input_video_path)

print(f'---> Data Load Time: {time.time()-start:.4f}')
end = time.time()

start_sec = load_timeseg(timeseg_file_path, input_video_path)

# print(f'---> Time Segment Load Time: {time.time()-end:.4f}')
# end = time.time()

bboxes_all, valid_frame_num = face_recog(frames, start_sec, fps)

print(f'---> Face Recognition Time: {time.time()-end:.4f}')


print(f'****** Before Time ******: {time.time()-start:.4f}')