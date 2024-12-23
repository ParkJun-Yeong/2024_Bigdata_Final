import wandb
import os
import requests
from tqdm import tqdm
import shutil

# W&B에서 API 키로 로그인

# 프로젝트 및 런 정보 입력
entity = "goguma"  # W&B 팀 또는 사용자 이름
project = "vfss_segmentation"  # 프로젝트 이름
# run_id = "Test of First Run"  # 다운로드할 특정 run ID
run_id = "UnVuOnYxOjA3dmlxa3FnOnZmc3Nfc2VnbWVudGF0aW9uOmdvZ3VtYQ=="
# W&B Run 객체 가져오기
api = wandb.Api()
# run = api.runs(f"{entity}/{project}/{run_id}")
runs = api.runs(f"{entity}/{project}")

run = None
for i, r in enumerate(runs):
    if i == 50:
        run = r
        print("download images in run: ", r.project + '/' + r.name)
        break

# 이미지를 저장할 디렉터리
save_dir = os.path.join("/mnt/ssd01_250gb/juny/vfss/SAMed/images/", run.project, run.name)
os.makedirs(save_dir, exist_ok=True)

gt_path = os.path.join(save_dir, 'gt')
pd_path = os.path.join(save_dir, 'prediction')
img_path = os.path.join(save_dir, 'img')
os.makedirs(gt_path, exist_ok=True)
os.makedirs(img_path, exist_ok=True)
os.makedirs(pd_path, exist_ok=True)


split = "test"

# run에서 파일 가져오기
for file in tqdm(run.files()):
    if file.name.endswith((".png", ".jpg", ".jpeg", ".gif")):  # 이미지 파일 필터링
        file_url = file.url
        response = requests.get(file_url)

        filename = file.name.replace('media/images/', '')

        if 'groundtruth' in filename:
            file_path = os.path.join(gt_path, filename)
        elif 'image' in filename:
            file_path = os.path.join(img_path, filename)
        elif 'prediction' in filename:
            file_path = os.path.join(pd_path, filename)
        else:
            print("not image/prediction/groundtruth")
            continue

        with open(file_path, "wb") as img_file:
            sp = file_path.split('/')
            ss = ""
            for s in sp[:-1]:
                ss += (s + '/')
            if not os.path.exists(ss):
                os.makedirs(ss, exist_ok=True)
            img_file.write(response.content)

        print(f"Downloaded {file.name}")


print("모든 이미지가 다운로드되었습니다.")

