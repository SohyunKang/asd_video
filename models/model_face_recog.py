
from retinaface import RetinaFace
import numpy as np
import os

def face_recog(frames, start_sec, fps, f_name):
    bboxes_all = []
    valid_frame_num = []
    for n, frame in enumerate(frames):
        if n >= start_sec*fps:
            image = frame
            resp = RetinaFace.detect_faces(np.array(image))
            bboxes = [resp[key]['facial_area'] for key in resp.keys()]
            if bboxes:
                bboxes_all.append(bboxes)
                valid_frame_num.append(n)
    
    # print(bboxes_all)
    # print(len(bboxes_all), len(bboxes_all[0]), len(bboxes_all[0][0]))
    os.makedirs("results", exist_ok=True)
    os.makedirs("results/save_outputs", exist_ok=True)
    np.save(f"results/save_outputs/{f_name.split('/')[-1].split('.')[0]}_bboxes_all", bboxes_all)
    np.save(f"results/save_outputs/{f_name.split('/')[-1].split('.')[0]}_valid_frame_num", valid_frame_num)

    return bboxes_all, valid_frame_num
