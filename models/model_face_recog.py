
from retinaface import RetinaFace
import numpy as np

def face_recog(frames, start_sec, fps):
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
    np.save("models/save_outputs/bboxes_all", bboxes_all)
    np.save("models/save_outputs/valid_frame_num", valid_frame_num)

    return bboxes_all, valid_frame_num
