
import time
import os

from model import load_model, face_recog, face_recog_torch_biubug, gaze_inference
from utils import load_timeseg, load_video, visualize_heatmap_to_video_save_prob, visualize_heatmap_to_video_with_sound

# ath = "./IF2001_1_1_1023041311_1.mp4"
input_list = os.listdir('./data')

for f_name in input_list:
    if '.mp4' in f_name and 'gazed' not in f_name:
        f_name = 'IF2001_2_1_1024080292_0.mp4'
        start = time.time()
        model, transform, device = load_model()
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

        # bboxes_all, valid_frame_num = face_recog(frames)

        bboxes_all, valid_frame_num = face_recog_torch_biubug(frames)

        print(f'Face Recognition Time: {time.time()-end:.4f}')
        end = time.time()

        # gazelle 
        outputs = gaze_inference(bboxes_all, valid_frame_num, frames, transform, device, width, height, model)

        print(f'Gazelle Inference Time: {time.time()-end:.4f}')
        end = time.time()

        end_sec = visualize_heatmap_to_video_save_prob(frames, output_video_path, outputs, valid_frame_num, fps, start_sec, bboxes_all, model, input_video_path)

        print(f'Save Time: {time.time()-end:.4f}')
        print(f'Total Time: {time.time()-start:.4f}')

        output_video_with_sound_path = f"results/{input_video_path.split('/')[1][:-4]}_gaze_sound.mp4"

        visualize_heatmap_to_video_with_sound(frames, output_video_with_sound_path, outputs, valid_frame_num, fps, start_sec, bboxes_all, model, input_video_path, end_sec)
