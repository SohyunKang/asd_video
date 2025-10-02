import os
import pandas as pd

# 분석할 디렉토리
data_dir = '/storage/ASD/ASD_movies/251002/아이 AI 플랫폼 1단계 상호작용 파일/영유아/보류군 외(정상군,고위험군,자폐군) 복호화파일'

records = []

for root, dirs, files in os.walk(data_dir):
    for f in files:
        if f.endswith(".mp4"):
            patient = os.path.basename(root)

            parts = f.split("_")
            if len(parts) < 2:
                continue

            # task 번호 추출
            task = parts[0][2:]
            try:
                task = int(task)
            except:
                continue

            # 라벨 추출 (0=정상, 1=자폐)
            try:
                label = int(f.split(".mp4")[0].split("_")[-1])
            except:
                continue

            records.append({"Patient": patient, "Task": task, "Label": label})

# DataFrame으로 정리
df = pd.DataFrame(records)

# ----- 전체 요약 -----
n_patients = df["Patient"].nunique()
n_videos = len(df)

# Task별 영상 수 (정렬)
task_counts = df["Task"].value_counts().sort_index()

# 영상 수 기준 라벨 분포
label_counts_video = df["Label"].value_counts().sort_index()

# 환자별 라벨 → 다수결로 결정
patient_labels = df.groupby("Patient")["Label"].agg(lambda x: x.mode()[0])
label_counts_patient = patient_labels.value_counts().sort_index()

# ----- 출력 -----
print("===== Dataset Summary =====")
print(f"전체 환자 수: {n_patients}")
print(f"전체 영상 수: {n_videos}")

print("\nTask별 영상 수 (정렬됨):")
print(task_counts.to_string())

print("\n영상 기준 정상/자폐 분포:")
print(label_counts_video.rename({0: "자폐(0)", 1: "정상(1)"}).to_string())

print("\n환자 기준 정상/자폐 분포 (다수결):")
print(label_counts_patient.rename({0: "자폐(0)", 1: "정상(1)"}).to_string())