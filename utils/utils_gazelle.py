import json
import cv2
import torch
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
            for wse in entry["hybrid_segments"]:
                if wse["word"] == target_word:
                    starts.append(float(wse["start"]))
                    ends.append(float(wse["end"]))
            break

    if starts and ends:
        start_sec = (starts[0]+ends[0])/2
        return start_sec
    else:
        raise Exception(f"Target word of ID {input_video_path.split('/')[1][:-4]} doesn't exist.")


def load_video(input_video_path):
    cap = cv2.VideoCapture(input_video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
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

# visualize predicted gaze heatmap for each person and gaze in/out of frame score
def visualize_heatmap(pil_image, heatmap, width, height, bbox=None, inout_score=None):
    if isinstance(heatmap, torch.Tensor):
        heatmap = heatmap.detach().cpu().numpy()
    heatmap = Image.fromarray((heatmap * 255).astype(np.uint8)).resize(pil_image.size, Image.Resampling.BILINEAR)
    heatmap = plt.cm.jet(np.array(heatmap) / 255.)
    heatmap = (heatmap[:, :, :3] * 255).astype(np.uint8)
    heatmap = Image.fromarray(heatmap).convert("RGBA")
    heatmap.putalpha(90)
    overlay_image = Image.alpha_composite(pil_image.convert("RGBA"), heatmap)

    if bbox is not None:
        width, height = pil_image.size
        xmin, ymin, xmax, ymax = bbox
        draw = ImageDraw.Draw(overlay_image)
        draw.rectangle([xmin * width, ymin * height, xmax * width, ymax * height], outline="lime", width=int(min(width, height) * 0.01))

        if inout_score is not None:
            text = f"in-frame: {inout_score:.2f}"
            text_width = draw.textlength(text)
            text_height = int(height * 0.01)
            text_x = xmin * width
            text_y = ymax * height + text_height
            draw.text((text_x, text_y), text, fill="lime", font=ImageFont.load_default(size=int(min(width, height) * 0.05)))

    return overlay_image

def visualize_heatmap_to_video_save_prob(frames, output_video_path, outputs, valid_frame_num, fps, start_sec, bboxes_all, model, input_video_path):
    height, width = frames[0].shape[:2]  # (H, W)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))  # 20fps
    # print(outputs, valid_frame_num)
    
    # valid_frame_num을 빠르게 lookup 할 수 있도록 dict 생성
    frame_to_output = {n: outputs[idx] for idx, n in enumerate(valid_frame_num)}
    idx = 0

    for v_f in valid_frame_num:
        if v_f/fps >= start_sec:
            end_sec = v_f/fps
            print(f"---> 반응 시간 : {end_sec:.2f}s")
            break
    hs = []
    inoutlist = []
    idxss = []

    for f_idx in range(len(frames)):
        pil_img = Image.fromarray(frames[f_idx]) # 원본 프레임
        current_time = f_idx / fps
        if f_idx in frame_to_output:  # 결과가 있는 프레임이면 heatmap 시각화
            if current_time >= start_sec:
                output = frame_to_output[f_idx]
                bboxes = bboxes_all[idx]
                idx += 1
                heatmaps = output['heatmap'][0]
                inouts = output['inout'][0] if model.inout else [None]*len(bboxes)
                hs.append(list(heatmaps.detach().cpu()))
                inoutlist.append(list(inouts.detach().cpu()))
                idxss.append(f_idx)
                
                overlay = pil_img
                for i in range(len(bboxes)):
                    overlay = visualize_heatmap(
                        overlay,
                        heatmaps[i],
                        width, height, np.array(bboxes[i]) / np.array([width, height, width, height]),  # 정규화
                        inout_score=inouts[i] if inouts is not None else None
                    )
            else:
                idx += 1
                overlay = pil_img

        else:  # 결과 없는 프레임은 원본만 사용
            overlay = pil_img

        if current_time >= start_sec:
            elapsed = min(current_time - start_sec, end_sec - start_sec)
            draw = ImageDraw.Draw(overlay)
            text = f"Reaction time: {elapsed:.1f} sec"
            font = ImageFont.load_default(size=int(min(width, height) * 0.06))
            font = ImageFont.load_default(size=int(min(width, height) * 0.06))
            text_bbox = draw.textbbox((0, 0), text, font=font)  # (x0, y0, x1, y1)
            text_width = text_bbox[2] - text_bbox[0]
            text_height = text_bbox[3] - text_bbox[1]

            # 오른쪽 상단 위치 계산 (여백 50)
            x = width - text_width - 50
            y = 50

            draw.text((x, y), text, fill="yellow", font=font)

        overlay_rgb = np.array(overlay.convert("RGB"), dtype=np.uint8)
        frame_bgr = cv2.cvtColor(overlay_rgb, cv2.COLOR_RGB2BGR)
        out.write(frame_bgr)

    io.savemat(f'{output_video_path[:-4]}.mat',
        {
            "heatmap": hs,
            "iolist": inoutlist,
            "idx": idxss
        }
            )

    # print("VideoWriter size:", width, height)
    # print("Frame size:", frame_bgr.shape, frame_bgr.dtype)
    out.release()
    
    return end_sec

# 소리 복원 하는 곳

def visualize_heatmap_to_video_with_sound(frames, output_video_with_sound_path, outputs, valid_frame_num, fps, start_sec, bboxes_all, model, input_video_path, end_sec):
    # --- 1) overlay된 프레임을 PNG 시퀀스로 저장 ---
    overlay_folder_name = f"{input_video_path[:-4]}_overlay_frames"
    os.makedirs(overlay_folder_name, exist_ok=True)

    height, width = frames[0].shape[:2]  # (H, W)
    frame_to_output = {n: outputs[idx] for idx, n in enumerate(valid_frame_num)}
    idx = 0
    for f_idx in range(len(frames)):
        pil_img = Image.fromarray(frames[f_idx]) # 원본 프레임
        current_time = f_idx / fps
        if f_idx in frame_to_output:  # 결과가 있는 프레임이면 heatmap 시각화
            if current_time >= start_sec:
                output = frame_to_output[f_idx]
                bboxes = bboxes_all[idx]
                idx += 1
                heatmaps = output['heatmap'][0]
                inouts = output['inout'][0] if model.inout else [None]*len(bboxes)

                overlay = pil_img
                for i in range(len(bboxes)):
                    overlay = visualize_heatmap(
                        overlay,
                        heatmaps[i],
                        width, height, np.array(bboxes[i]) / np.array([width, height, width, height]),  # 정규화
                        inout_score=inouts[i] if inouts is not None else None
                    )
            else:
                idx += 1
                overlay = pil_img
        else:
            overlay = pil_img

        if current_time >= start_sec:
            elapsed = min(current_time - start_sec, end_sec - start_sec)
            draw = ImageDraw.Draw(overlay)
            text = f"Reaction time: {elapsed:.1f} sec"
            font = ImageFont.load_default(size=int(min(width, height) * 0.06))
            text_bbox = draw.textbbox((0, 0), text, font=font)  # (x0, y0, x1, y1)
            text_width = text_bbox[2] - text_bbox[0]
            text_height = text_bbox[3] - text_bbox[1]

            # 오른쪽 상단 위치 계산 (여백 50)
            x = width - text_width - 50
            y = 50

            font = ImageFont.load_default(size=int(min(width, height) * 0.06))
            draw.text((x, y), text, fill="yellow", font=font)

        # 프레임을 PNG 파일로 저장
        overlay.save(f"{overlay_folder_name}/frame_{f_idx:05d}.png")

    # print("총 저장된 프레임 수:", len(os.listdir(overlay_folder_name)))

    # --- 2) ffmpeg로 원본 비디오 + overlay PNG 합성 ---


    # # overlay는 단순히 PNG를 위에 덮어씀
    os.system(
        f'ffmpeg-4.4.1-amd64-static/ffmpeg -y '
        f'-i "{input_video_path}" '
        f'-framerate {fps} -i {overlay_folder_name}/frame_%05d.png '
        '-filter_complex "[1:v]format=rgb24[ov];[0:v][ov]overlay=0:0:format=auto:eof_action=pass:repeatlast=0,format=yuv420p" '
        f'-c:a copy '
        '-hide_banner -loglevel error '
        f'"{output_video_with_sound_path}"'
    )