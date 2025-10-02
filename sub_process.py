import subprocess
import os
import glob

# ===== 설정 =====
root_dir = "/storage/ASD/ASD_movies/251002/아이 AI 플랫폼 1단계 상호작용 파일/영유아/보류군 외(정상군,고위험군,자폐군) 복호화파일"   # 환자별 폴더들이 들어있는 상위 폴더

pattern = "IF2001*.mp4" # 찾을 파일 패턴

# ===== 파일 탐색 =====
mp4_files = glob.glob(os.path.join(root_dir, "*", pattern))  # 환자폴더/*/IF2001*.mp4
print(f"🔎 Found {len(mp4_files)} files")

# ===== subprocess 실행 =====
for mp4_path in mp4_files:
    mp4_name = os.path.basename(mp4_path)   # 파일 이름만 추출
    print(f"▶ Running on {mp4_name}")

    # # coco server ver.
    # # TensorFlow 쪽 실행
    # subprocess.run(["conda", "run", "-n", "facerecog", "python3", "before.py"])

    # print('------- Next ------->\n')

    # # Pytorch 쪽 실행
    # subprocess.run(["conda", "run", "-n", "gazelle", "python3", "after.py"])

    # workstation1 ver.
    # TensorFlow 쪽 실행
    subprocess.run(["conda", "run", "-n", "before", "python3", "before.py", mp4_path])

    print('------- Next ------->\n')

    # Pytorch 쪽 실행
    subprocess.run(["conda", "run", "-n", "after", "python3", "after.py", mp4_path])
    break
