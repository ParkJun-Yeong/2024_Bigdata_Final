import cv2
import numpy as np
from pycocotools import mask
import json
from skimage import measure
import os
from tqdm import tqdm

from shapely.geometry import Polygon



# tif mask를 polygon annotation으로 전환 (coco annotator로 수정을 위함)
def tif_to_coco_segmentation(png_mask_path, image_id, category_id):
    # 마스크 이미지 불러오기 (0-255 범위로 이진화된 마스크)
    mask_image = cv2.imread(png_mask_path, cv2.IMREAD_GRAYSCALE)

    # 윤곽선을 추출 (contours 찾기)
    contours, _ = cv2.findContours(mask_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    segmentation = []
    for contour in contours:
        contour = contour.flatten().tolist()  # 다각형 좌표를 1차원 배열로 변환
        segmentation.append(contour)

    # COCO 형식의 annotation 생성
    try:
        annotation = {
            "segmentation": segmentation,
            "area": int(np.sum(mask_image > 0)),  # 마스크 영역의 픽셀 수 (면적)
            "iscrowd": 0,
            "image_id": image_id,  # 이미지 ID
            "category_id": category_id,  # 클래스 ID
            "bbox": cv2.boundingRect(mask_image),  # 바운딩 박스
            "id": np.random.randint(1, 1000000),  # 유일한 ID 생성
        }
    except:
        print()

    return annotation, mask_image.shape


def convert_mask_to_coco_polygon():
    # 이미지 및 클래스 정보
    # image_id = 1  # 이미지 ID 설정
    category_id = 1  # 클래스 ID 설정
    base_path = '/mnt/ssd03_4tb/juny/vfss/vfss_cauh/annotation/pa/'  # 마스크 이미지 경로
    annot_folder_list = [os.path.join(base_path, folder) for folder in os.listdir(base_path)]
    annot_file_list = []
    for annot_folder in annot_folder_list:
        if '.DS_Store' not in annot_folder:
            annot_file = [os.path.join(annot_folder, file) for file in os.listdir(annot_folder) 
                          if ('.DS_Store' not in file) and (file.split('.')[-1] == 'tif')]
            annot_file_list.extend(annot_file)
    
    for i, tif_mask_path in tqdm(enumerate(annot_file_list)):
        image_id = i
    # 변환 수행
        coco_annotation, image_size = tif_to_coco_segmentation(tif_mask_path, image_id, category_id)

        image_path = os.path.join(tif_mask_path.replace('annotation/pa', 'frame_cut_info/frame_per_second').replace('.tif', '.jpg')).replace('_X', '')
        if not os.path.exists(image_path):
            print("Image Not Exists: ", image_path)

        im_split = image_path.split('/')
        # COCO 포맷을 JSON으로 저장
        coco_output = {
            "annotations": [coco_annotation],
            "images": [{
                "folder": im_split[-2],
                "file_name": im_split[-1],  # 이미지 파일명
                "id": image_id,
                "height": image_size[0],  # 이미지 높이
                "width": image_size[1]   # 이미지 너비
            }],
            "categories": [{
                "id": category_id,
                "name": "pa",  # 클래스 이름
                "supercategory": "none"
            }]
        }

        
        # 저장할 폴더 존재 확인
        save_path = tif_mask_path.replace('pa', 'pa_polygon').replace('tif', 'json')
        save_filename = tif_mask_path.split('/')[-1].replace('.tif', '.json')
        save_folder = save_path.replace(save_filename, '')

        # 241108 하위폴더 만들지 않음.
        # if not os.path.exists(save_folder):
        #     os.makedirs(save_folder, exist_ok=True)

        save_path = save_path.replace(save_folder.split('/')[-2]+'/', '')        

        # 결과를 JSON 파일로 저장
        with open(os.path.join(save_path), 'w') as f:
            json.dump(coco_output, f, ensure_ascii=False, indent=4)

        f.close()

def tif_to_labelme_segmentation(png_mask_path):
    # 마스크 이미지 불러오기 (0-255 범위로 이진화된 마스크)
    mask_image = cv2.imread(png_mask_path, cv2.IMREAD_GRAYSCALE)

    # 윤곽선을 추출 (contours 찾기)
    contours, _ = cv2.findContours(mask_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    shapes = []

    for contour in contours:
        contour = contour.squeeze()  # 불필요한 차원 제거
        points = contour.tolist()
        shape = {
            "label": "pa",  # 필요한 라벨 이름 입력
            "points": points,
            "group_id": None,
            "shape_type": "polygon",
            "flags": {}
        }
        shapes.append(shape)


    return shapes

def convert_mask_to_labelme_polygon():

    mask_path = "/mnt/ssd03_4tb/juny/vfss/vfss_cauh/experiment/train_mask"
    save_path = "/mnt/ssd03_4tb/juny/vfss/vfss_cauh/experiment/train_json"

    # tif 마스크 이미지 불러오기
    files = os.listdir(mask_path)
    files = [os.path.join(mask_path, f) for f in files]

   # 이미지 및 클래스 정보
    # image_id = 1  # 이미지 ID 설정
    category_id = 1  # 클래스 ID 설정
    base_path = '/mnt/ssd03_4tb/juny/vfss/vfss_cauh/annotation/pa/'  # 마스크 이미지 경로
    annot_folder_list = [os.path.join(base_path, folder) for folder in os.listdir(base_path)]
    annot_file_list = []
    for annot_folder in annot_folder_list:
        if '.DS_Store' not in annot_folder:
            annot_file = [os.path.join(annot_folder, file) for file in os.listdir(annot_folder) 
                          if ('.DS_Store' not in file) and (file.split('.')[-1] == 'tif')]
            annot_file_list.extend(annot_file)
    
    for tif_mask_path in tqdm(annot_file_list):
        # image_id = i
    # 변환 수행
        shapes = tif_to_labelme_segmentation(tif_mask_path)

        # image_path = os.path.join(tif_mask_path.replace('annotation/pa', 'frame_cut_info/frame_per_second').replace('.tif', '.jpg')).replace('_X', '')
        # if not os.path.exists(image_path):
        #     print("Image Not Exists: ", image_path)

        # im_split = image_path.split('/')

                # 폴리곤 좌표 저장할 리스트 생성

        # JSON 형식으로 변환하여 저장
        output_data = {
            "version": "4.5.9",  # labelme 버전에 따라 변경 가능
            "flags": {},
            "shapes": shapes,
            "imagePath": tif_mask_path.split('/')[-1].replace('.tif', '.png'),
            "imageData": None
        }

        with open(os.path.join(save_path, tif_mask_path.split('/')[-1].replace('.tif', '.json')), "w") as f:
            json.dump(output_data, f)


        
        # # 저장할 폴더 존재 확인
        # save_path = tif_mask_path.replace('pa', 'pa_polygon').replace('tif', 'json')
        
        # save_folder = save_path.replace(save_filename, '')
        # if not os.path.exists(save_folder):
        #     os.makedirs(save_folder, exist_ok=True)

        # # 결과를 JSON 파일로 저장
        # with open(os.path.join(save_path), 'w') as f:
        #     json.dump(coco_output, f, ensure_ascii=False, indent=4)

        f.close()


def convert_single_coco():

    # 경로 설정
    input_dir = '/mnt/ssd03_4tb/juny/vfss/vfss_cauh/annotation/pa_polygon'  # 각 이미지의 JSON 파일들이 저장된 폴더
    output_file = '/mnt/ssd03_4tb/juny/vfss/vfss_cauh/annotation/coco_annotations.json'

    # COCO 형식 초기화
    coco_format = {
        "images": [],
        "annotations": [],
        "categories": []
    }

    # 카테고리 정보 설정 (필요에 따라 수정)
    category_id_mapping = {}
    category_id = 1  # 카테고리 ID 시작 값

    # 어노테이션 ID
    annotation_id = 1

    # JSON 파일 순회
    for idx, filename in enumerate(os.listdir(input_dir)):
        if filename.endswith('.json'):
            with open(os.path.join(input_dir, filename)) as f:
                data = json.load(f)

            # 이미지 정보 추가
            image_info = {
                "id": idx,
                "file_name": data["images"][0]["file_name"],
                "height": data["images"][0]["height"],
                "width": data["images"][0]["width"]
            }
            coco_format["images"].append(image_info)

            # 어노테이션 추가
            for annotation in data["annotations"]:
                # 카테고리 ID 매핑
                category_name = data["categories"][0]["name"]
                if category_name not in category_id_mapping:
                    category_id_mapping[category_name] = category_id
                    category_info = {
                        "id": category_id,
                        "name": category_name,
                        "supercategory": "none"
                    }
                    coco_format["categories"].append(category_info)
                    category_id += 1

                seg = [[]]
                for a in annotation['segmentation']:
                    seg[0].extend(a)
                
                p = []
                for s in range(0, len(seg[0]), 2):
                    p.append((seg[0][s], seg[0][s+1]))

                polygon = Polygon(p)  # 원래 포인트 리스트로 Polygon 생성
                simplified_polygon = polygon.simplify(1.0)  # 적절한 tolerance 값을 설정하여 단순화
                simplified= list(simplified_polygon.exterior.coords)  # 단순화된 포인트 추출
                simplified_points = [[]]
                for p in simplified:
                    simplified_points[0].extend(p)

                    
                
                ann = {
                    "id": annotation_id,
                    "image_id": idx,
                    "category_id": category_id_mapping[category_name],
                    # "segmentation": annotation["segmentation"],
                    "segmentation": simplified_points,
                    "area": annotation["area"],
                    "bbox": annotation["bbox"],
                    "iscrowd": annotation.get("iscrowd", 0)
                }
                coco_format["annotations"].append(ann)
                annotation_id += 1

    # COCO 형식으로 저장
    with open(output_file, 'w') as f:
        json.dump(coco_format, f)






if __name__ == "__main__":
    # convert_mask_to_coco_polygon()
    # convert_mask_to_labelme_polygon()
    convert_single_coco()