import cv2
import mediapipe as mp
import numpy as np

mp_face_detection = mp.solutions.face_detection

def estimate_head_pose(video_path, output_path="output.mp4", stable_threshold=50):
    cap = cv2.VideoCapture(video_path)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # 안정화 변수
    stable_count = 0
    current_state = "BACK"
    print('# frame:', int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
    with mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.3) as fd:
        while True:
            ret, frame = cap.read()
            
            if not ret:
                break

            h, w = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = fd.process(rgb)

            detected_state = "BACK"  # 기본값

            if result.detections:
                for detection in result.detections:
                    bboxC = detection.location_data.relative_bounding_box
                    x1, y1 = int(bboxC.xmin * w), int(bboxC.ymin * h)
                    x2, y2 = int((bboxC.xmin + bboxC.width) * w), int((bboxC.ymin + bboxC.height) * h)

                    keypoints = detection.location_data.relative_keypoints
                    nose = (int(keypoints[0].x * w), int(keypoints[0].y * h))
                    left_eye = (int(keypoints[1].x * w), int(keypoints[1].y * h))
                    right_eye = (int(keypoints[2].x * w), int(keypoints[2].y * h))
                    mouth = (int(keypoints[3].x * w), int(keypoints[3].y * h))
                    left_ear = (int(keypoints[4].x * w), int(keypoints[4].y * h))
                    right_ear = (int(keypoints[5].x * w), int(keypoints[5].y * h))

                    face_2d = np.array([nose, left_eye, right_eye, mouth, left_ear, right_ear], dtype=np.float64)
                    if any(p[0] <= 0 or p[1] <= 0 for p in face_2d):
                        detected_state = "BACK"
                        continue

                    face_3d = np.array([
                        [0.0, 0.0, 0.0],        # nose
                        [-30.0, -30.0, -30.0],  # left eye
                        [30.0, -30.0, -30.0],   # right eye
                        [0.0, 30.0, -30.0],     # mouth
                        [-60.0, 0.0, -30.0],    # left ear
                        [60.0, 0.0, -30.0]      # right ear
                    ], dtype=np.float64)

                    focal_length = 1 * w
                    cam_matrix = np.array([[focal_length, 0, w/2],
                                           [0, focal_length, h/2],
                                           [0, 0, 1]])
                    dist_matrix = np.zeros((4,1), dtype=np.float64)

                    success, rvec, tvec = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix)

                    if success:
                        rmat, _ = cv2.Rodrigues(rvec)
                        angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)
                        pitch, yaw, roll = angles
                        yaw_deg = yaw * 180
                        pitch_deg = pitch * 180

                        # 상태 분류
                        if yaw > 0.2:
                            detected_state = "RIGHT"
                        elif yaw < -0.2:
                            detected_state = "LEFT"
                        else:
                            detected_state = "FRONT"

                        # bounding box
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
                        # nose 방향축 그리기
                        nose_3d = np.array([[0, 0, 100]], dtype=np.float64)
                        nose_2d, _ = cv2.projectPoints(nose_3d, rvec, tvec, cam_matrix, dist_matrix)
                        p1 = (int(nose[0]), int(nose[1]))
                        p2 = (int(nose_2d[0][0][0]), int(nose_2d[0][0][1]))
                        cv2.line(frame, p1, p2, (255, 0, 0), 3)

                        # 각도 표시
                        cv2.putText(frame, f"Yaw:{yaw_deg:.1f} Pitch:{pitch_deg:.1f}",
                                    (x1, y2+25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)

            # --- 안정화 처리 ---
            if detected_state != "BACK":
                stable_count += 1
                if stable_count >= stable_threshold:
                    current_state = detected_state
            else:
                stable_count = 0
                current_state = "BACK"

            # 텍스트 표시
            cv2.putText(frame, f"Gaze: {current_state}", (50,50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255) if current_state=="BACK" else (0,255,0), 2)

            out.write(frame)

    cap.release()
    out.release()
    print(f"✅ 결과 저장 완료: {output_path}")


if __name__ == "__main__":
    video_file = "/home/sohyunkang/asd_video/head_turn/KakaoTalk_Video_2025-09-15-18-19-32.mp4"   # 여기에 분석할 mp4 경로 입력
    estimate_head_pose(video_file, "output_fd_pose_50.mp4")



