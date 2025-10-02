import torch
import numpy as np
import ssl
ssl._create_default_https_context = ssl._create_unverified_context  # ssl 에러 방지

def load_model():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # print(device)

    # load Gaze-LLE model
    _ = torch.randn(1).cuda()  # GPU warmup
    # 최초 1회 실행 (가중치 다운로드 → ~/.cache/torch/hub에 저장)
    # torch.hub.load('fkryan/gazelle', 'gazelle_dinov2_vitl14_inout', force_reload=True)
    model, transform = torch.hub.load('fkryan/gazelle', 'gazelle_dinov2_vitl14_inout')
    model.eval()
    model.to(device)
    
    return model, transform, device

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