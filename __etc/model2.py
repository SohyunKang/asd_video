import torch

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