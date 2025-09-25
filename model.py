import torch
from retinaface import RetinaFace
import numpy as np
from Pytorch_Retinaface.models.retinaface import RetinaFace as RetinaFace_torch

from Pytorch_Retinaface.layers.functions.prior_box import PriorBox
from Pytorch_Retinaface.utils.box_utils import decode
from Pytorch_Retinaface.utils.nms.py_cpu_nms import py_cpu_nms
from Pytorch_Retinaface.data import cfg_re50
from torchvision.ops import nms  # GPU NMS

def load_model():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(device)

    # load Gaze-LLE model
    _ = torch.randn(1).cuda()  # GPU warmup
    # 최초 1회 실행 (가중치 다운로드 → ~/.cache/torch/hub에 저장)
    # torch.hub.load('fkryan/gazelle', 'gazelle_dinov2_vitl14_inout', force_reload=True)
    model, transform = torch.hub.load('fkryan/gazelle', 'gazelle_dinov2_vitl14_inout')
    model.eval()
    model.to(device)
    
    return model, transform, device


def face_recog(frames):
    bboxes_all = []
    valid_frame_num = []
    for n, frame in enumerate(frames):
        image = frame
        resp = RetinaFace.detect_faces(np.array(image))
        bboxes = [resp[key]['facial_area'] for key in resp.keys()]
        if bboxes:
            bboxes_all.append(bboxes)
            valid_frame_num.append(n)
    
    print(bboxes_all)
    print(len(bboxes_all), len(bboxes_all[0]), len(bboxes_all[0][0]))
    return bboxes_all, valid_frame_num


def face_recog_torch_biubug(
    frames, 
    device='cuda', 
    batch_size=16, 
    cfg=cfg_re50, 
    score_thresh=0.85, 
    nms_thresh=0.5, 
    n_topk=1
):
    """
    RetinaFace 배치 기반 얼굴 인식 (frames는 RGB 이미지)
    return: bboxes_all, valid_frame_num
    """

    # ---------------------------
    # 1. 모델 로드
    # ---------------------------
    net = RetinaFace_torch(cfg=cfg, phase='test').to(device)
    net.eval()

    # ---------------------------
    # 2. PriorBox 생성 (영상 해상도 고정 가정)
    # frames[0].shape = (H, W, 3)
    im_height, im_width, _ = frames[0].shape  
    priorbox = PriorBox(cfg, image_size=(im_height, im_width))
    priors = priorbox.forward().to(device)

    # ---------------------------
    # 3. 결과 저장
    # ---------------------------
    bboxes_all = []
    valid_frame_num = []

    # ---------------------------
    # 4. 배치 단위 추론
    # ---------------------------
    num_frames = len(frames)

    for start in range(0, num_frames, batch_size):
        batch_frames = frames[start:start+batch_size]

        # ---- 전처리 (RGB mean subtraction) ----
        imgs = []
        for frame in batch_frames:
            img = np.float32(frame)
            img -= (123, 117, 104)  # RGB mean
            # img -= (104, 117, 123)  # BGR mean
            img = img.transpose(2, 0, 1)  # HWC → CHW
            imgs.append(torch.from_numpy(img))

        imgs = torch.stack(imgs, dim=0).to(device)  # (B,3,H,W)

        with torch.no_grad():
            loc, conf, landms = net(imgs)

        # ---------------------------
        # 5. 후처리 (프레임별)
        # ---------------------------
        for i in range(len(batch_frames)):
            # scale 순서 맞게 (W,H,W,H)
            scale = torch.tensor([im_width, im_height, im_width, im_height], device=device)

            # decode
            boxes = decode(loc[i], priors, [0.1, 0.2]) * scale
            scores = conf[i][:, 1]

            # 스코어 threshold
            mask = scores > score_thresh
            boxes = boxes[mask]
            scores = scores[mask]

            if boxes.shape[0] == 0:
                continue

            # NMS
            keep = nms(boxes, scores, nms_thresh)
            boxes = boxes[keep]
            scores = scores[keep]

            # 상위 N개 선택
            topk = torch.argsort(scores, descending=True)[:n_topk]
            boxes = boxes[topk]
            scores = scores[topk]

            # numpy 변환
            dets = torch.cat([boxes, scores.unsqueeze(1)], dim=1).cpu().numpy()

            # clip 해서 이미지 밖 좌표 방지
            dets[:, :4] = np.clip(
                dets[:, :4], 
                [0, 0, 0, 0], 
                [im_width, im_height, im_width, im_height]
            )

            bboxes = dets[:, :4].astype(int).tolist()

            if bboxes:
                bboxes_all.append(bboxes)
                valid_frame_num.append(start + i)

    print(bboxes_all, valid_frame_num)
    print(len(bboxes_all), len(bboxes_all[0]), len(bboxes_all[0][0]))
    return bboxes_all, valid_frame_num



    



# # ---------------------------
# # RetinaFace 추론 함수 (bbox만 반환)
# # ---------------------------
# def detect_faces_torch(img, cfg, net, device='cuda'):
#     im_height, im_width, _ = img.shape
#     scale = torch.Tensor([im_width, im_height, im_width, im_height])

#     # 전처리
#     img_raw = img.copy()
#     img = np.float32(img)
#     img -= (104, 117, 123)  # mean subtraction
#     img = img.transpose(2, 0, 1)  # HWC → CHW
#     img = torch.from_numpy(img).unsqueeze(0).to(device)

#     # 추론
#     with torch.no_grad():
#         loc, conf, landms = net(img)

#     # prior box 생성
#     priorbox = PriorBox(cfg=cfg, image_size=(im_height, im_width))
#     priors = priorbox.forward().to(device)

#     # 디코딩
#     boxes = decode(loc.data.squeeze(0), priors.data, [0.1, 0.2])
#     boxes = boxes * scale.to(device)
#     scores = conf.squeeze(0).data.cpu().numpy()[:, 1]  # face confidence

#     # 스코어 threshold
#     inds = np.where(scores > 0.5)[0]
#     boxes = boxes[inds]
#     scores = scores[inds]

#     if boxes.shape[0] == 0:
#         return []

#     # NMS
#     dets = np.hstack((boxes.cpu().numpy(), scores[:, np.newaxis]))
#     keep = py_cpu_nms(dets, 0.4)
#     dets = dets[keep, :]

#     # 결과 bbox만 반환
#     bboxes = dets[:, :4].astype(int).tolist()
#     return bboxes

# # ---------------------------
# # face_recog 함수 (Torch 버전)
# # ---------------------------

# def face_recog_torch(frames, device='cuda'):
#     # RetinaFace 모델 로드 (ResNet50 backbone 예시)
#     net = RetinaFace_torch(cfg=cfg_re50, phase='test').to(device)
#     net.eval()

#     bboxes_all = []
#     valid_frame_num = []

#     for n, frame in enumerate(frames):
#         print(n)
#         bboxes = detect_faces_torch(frame, cfg_re50, net, device=device)
#         if bboxes:  # 얼굴이 하나라도 감지된 경우
#             bboxes_all.append(bboxes)
#             valid_frame_num.append(n)

#     return bboxes_all, valid_frame_num

def gaze_inference(bboxes_all, valid_frame_num, frames, transform, device, width, height, model):
    outputs = []
    for bboxes, n in zip(bboxes_all, valid_frame_num):
        image = frames[n]
        img_tensor = transform(image).unsqueeze(0).to(device)
        norm_bboxes = [[np.array(bbox) / np.array([width, height, width, height]) for bbox in bboxes]]

        input = {
            "images": img_tensor, # [num_images, 3, 448, 448]
            "bboxes": norm_bboxes # [[img1_bbox1, img1_bbox2...], [img2_bbox1, img2_bbox2]...]
        }

        with torch.no_grad():
            output = model(input)

        outputs.append(output)

        img1_person1_heatmap = output['heatmap'][0][0] # [64, 64] heatmap

        if model.inout:
            img1_person1_inout = output['inout'][0][0] # gaze in frame score (if model supports inout prediction)

    return outputs