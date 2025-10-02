import time
import os

from models.model_face_recog import face_recog
from utils.utils_face_recog import load_timeseg, load_video

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf

import sys

if len(sys.argv) < 2:
    print("❌ mp4 파일 이름을 입력하세요")
    sys.exit(1)

mp4_path = sys.argv[1]  # 첫 번째 인자
print("받은 영상 파일:", mp4_path)


# f_name = 'IF2001_3_1_1024060343_0.mp4'  # 통합 필요
timeseg_file_path = "data/samples_timesegments_total.json"

input_video_path = mp4_path

print(f'****** Input Video: {input_video_path} ******')
start = time.time()

frames, width, height, fps, frame_count = load_video(input_video_path)

print(f'---> Data Load Time: {time.time()-start:.4f}')
end = time.time()

start_sec = load_timeseg(timeseg_file_path, input_video_path)

# print(f'---> Time Segment Load Time: {time.time()-end:.4f}')
# end = time.time()

bboxes_all, valid_frame_num = face_recog(frames, start_sec, fps, input_video_path)

print(f'---> Face Recognition Time: {time.time()-end:.4f}')


print(f'****** Before Time ******: {time.time()-start:.4f}')