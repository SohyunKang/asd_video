import subprocess

# TensorFlow 쪽 실행
subprocess.run(["conda", "run", "-n", "facerecog", "python3", "before.py"])

print('------- Next ------->\n')

# Pytorch 쪽 실행
subprocess.run(["conda", "run", "-n", "gazelle", "python3", "after.py"])
