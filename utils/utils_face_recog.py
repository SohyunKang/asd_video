import json
import cv2
import numpy as np

from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import scipy.io as io
import os

def load_timeseg(timeseg_file_path, input_video_path):
    with open(timeseg_file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    starts, ends = [], []

    target_word = ""
    for entry in data:
        if entry['id'] == input_video_path.split('/')[1][:-4]:
            target_word = entry['hybrid_segments'][0]['word']
            print('---> 호명 :', target_word)
        
            for wse in entry["hybrid_segments"]:
                if wse["word"] == target_word:
                    starts.append(float(wse["start"]))
                    ends.append(float(wse["end"]))
            break

    if starts and ends:
        start_sec = (starts[0]+ends[0])/2
        print(f"---> 호명 시작 시간 : {start_sec:.2f}s")
        return start_sec
    else:
        raise Exception(f"Target word of ID {input_video_path.split('/')[1][:-4]} doesn't exist.")


def load_video(input_video_path):
    cap = cv2.VideoCapture(input_video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f'---> FPS: {fps:.2f}, Frame_count: {frame_count}')

    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        width, height, _ = frame.shape
        frames.append(frame)
    cap.release()

    return frames, width, height, fps, frame_count
    