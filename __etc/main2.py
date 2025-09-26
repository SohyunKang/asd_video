import time
import os

from model import load_model, face_recog
from utils import load_timeseg, load_video

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import tensorflow as tf

# ath = "./IF2001_1_1_1023041311_1.mp4"
input_list = os.listdir('./data')

for f_name in input_list:
    if '.mp4' in f_name and 'gazed' not in f_name:
        f_name = 'IF2001_2_1_1024080292_0.mp4'
        start = time.time()
        # model, transform, device = load_model()
        end = time.time()

        print(f'---> Model Load Time: {end-start:.4f}')

        input_video_path = f'data/{f_name}'
        output_video_path = f"results/{f_name[:-4]}_gazed.mp4"
        timeseg_file_path = "data/samples_timesegments_total.json"
        
        print('Input Video: ', input_video_path)
        print('Output Video: ', output_video_path)
        
        start_sec = load_timeseg(timeseg_file_path, input_video_path)

        print(f'---> Time Segment Load Time: {time.time()-end:.4f}')
        end = time.time()

        frames, width, height, fps, frame_count = load_video(input_video_path)

        print(f'---> Data Load Time: {time.time()-end:.4f}')
        end = time.time()

        bboxes_all, valid_frame_num = face_recog(frames)

        # # bboxes_all, valid_frame_num = face_recog_torch_biubug(frames)

        print(f'Face Recognition Time: {time.time()-end:.4f}')
        # end = time.time()

       