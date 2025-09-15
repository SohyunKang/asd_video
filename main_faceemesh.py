import cv2
import mediapipe as mp
import numpy as np

mp_face_detection = mp.solutions.face_detection
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

FACE_3D_IDX = [1, 33, 263, 61, 291, 199]  # nose, eyes, mouth

def facemesh_headpose(video_path, output_path="facemesh_output.mp4"):
    cap = cv2.VideoCapture(video_path)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    with mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.3) as fd, \
         mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        ) as fm:

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            h, w = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # 1. 얼굴 감지 먼저
            fd_result = fd.process(rgb)

            if not fd_result.detections:
                # 얼굴 없으면 → 뒤돌아있거나 안보이는 상태 → 스킵
                out.write(frame)
                continue

            # 2. 얼굴이 잡혔을 때만 FaceMesh 실행
            fm_result = fm.process(rgb)

            if fm_result.multi_face_landmarks:
                landmarks = fm_result.multi_face_landmarks[0].landmark

                # 윤곽선 시각화
                mp_drawing.draw_landmarks(
                    frame,
                    fm_result.multi_face_landmarks[0],
                    mp_face_mesh.FACEMESH_CONTOURS,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style()
                )

                # solvePnP 준비
                face_2d, face_3d = [], []
                for idx in FACE_3D_IDX:
                    x, y = int(landmarks[idx].x * w), int(landmarks[idx].y * h)
                    face_2d.append([x, y])
                    face_3d.append([x, y, landmarks[idx].z * 3000])  # Z scaling

                face_2d = np.array(face_2d, dtype=np.float64)
                face_3d = np.array(face_3d, dtype=np.float64)

                focal_length = 1 * w
                cam_matrix = np.array([[focal_length, 0, w / 2],
                                       [0, focal_length, h / 2],
                                       [0, 0, 1]])
                dist_matrix = np.zeros((4, 1), dtype=np.float64)

                success, rvec, tvec = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix)
                rmat, _ = cv2.Rodrigues(rvec)
                angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)
                pitch, yaw, roll = angles

                # 방향 판정
                if yaw > 0.25:
                    gaze = "머리 → 오른쪽"
                elif yaw < -0.25:
                    gaze = "머리 → 왼쪽"
                else:
                    gaze = "정면"

                cv2.putText(frame, f"Gaze: {gaze}", (50, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                # 3D 축 가시화
                nose = tuple(face_2d[0].astype(int))
                axis = np.float32([[0, 0, 1000]]).reshape(-1, 3)
                nose_end, _ = cv2.projectPoints(axis, rvec, tvec, cam_matrix, dist_matrix)
                p2 = tuple(nose_end[0].ravel().astype(int))
                cv2.line(frame, nose, p2, (255, 0, 0), 3)

            out.write(frame)

    cap.release()
    out.release()
    print(f"✅ 결과 저장 완료: {output_path}")


if __name__ == "__main__":
    facemesh_headpose("/home/sohyunkang/asd_video/head_turn/IF2001_1_1_1023041311_1.mp4", "facemesh_output.mp4")
