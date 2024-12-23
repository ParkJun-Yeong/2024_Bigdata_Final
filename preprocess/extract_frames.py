import os
from tqdm import tqdm
import cv2
from change_name import find_in_cut_info


def extract_frames():
    years = ['2023', '2024']
    base_path = "/mnt/ssd03_4tb/juny/vfss/vfss_cauh/cut_info"

    # # dataset = {}
    # data_path = []
    # print("===== Get data path... =======")
    # for year in years:
    #     months = os.listdir(os.path.join(base_path, year))
    #     for month in months:
    #         days = os.listdir(os.path.join(base_path, year, month))
    #         for day in days:
    #             patients = os.listdir(os.path.join(base_path, year, month, day))
    #             for patient in patients:
    #                 datas = os.listdir(os.path.join(base_path, year, month, day, patient))
    #                 for data in datas:
    #                     data_path.append(os.path.join(base_path, year, month, day, patient, data))

    data_path = os.listdir(base_path)
    frame_path = '/mnt/ssd03_4tb/juny/vfss/vfss_cauh/frame_cut_info/all'            # extract all frames
    # frame_path = '/mnt/ssd03_4tb/juny/vfss/vfss_cauh/frame_cut_info/frame_per_second'             # extract frame per second
    # 프레임이 저장될 디렉토리 생성
    if not os.path.exists(frame_path):
        os.makedirs(frame_path, exists_ok=True)

    print("===== Extract image frames... =======")
    for data in tqdm(data_path):
        # patient = data.split('/')[10]
        # bolus_type = data.split('/')[-1].split('.')[0]
        # year = data.split('/')[7]

        id = data.split('.')[0]
        
        save_path = os.path.join(frame_path, id)

        if os.path.exists(save_path):
            continue
        if not os.path.exists(save_path):
            os.makedirs(save_path, exist_ok=True)
        if os.path.exists(save_path):
            if len(os.listdir(save_path)):
                continue
                

        print(f"extract frames: {save_path}")        
        cap = cv2.VideoCapture(os.path.join(base_path, data))
        
        height = cap.get (cv2.CAP_PROP_FRAME_HEIGHT)
        width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        fps = cap.get (cv2.CAP_PROP_FPS)
        n_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        
        frames = []
        # ret, frame = cap.read()
        
        count = 0
        while(cap.isOpened()):
            ret, image = cap.read()
            if ret: 
                #if(int(cap.get(1)) % int(fps) == 0):               # extract frame per second. comment this line if you want to extract all frames
                cv2.imwrite(os.path.join(save_path, id+str(count)+'.png'), image)
                count += 1
            else:
                break
                
        cap.release()


def extract_frames_with_cut_info():

    years = ['2023', '2024']
    base_path = "/mnt/ssd03_4tb/juny/vfss/vfss_cauh/raw"

    data_path = []
    print("===== Get data path... =======")
    for year in years:
        months = os.listdir(os.path.join(base_path, year))
        for month in months:
            days = os.listdir(os.path.join(base_path, year, month))
            for day in days:
                patients = os.listdir(os.path.join(base_path, year, month, day))
                for patient in patients:
                    datas = os.listdir(os.path.join(base_path, year, month, day, patient))
                    for data in datas:
                        data_path.append(os.path.join(base_path, year, month, day, patient, data))


    frame_path = "/mnt/ssd03_4tb/juny/vfss/vfss_cauh/frame_cut_info/frame_per_second_png"

    # frame_per-_second_png 파일 리스트 가져오는거. 필요할때만 활성화 시키기.
    if True:
        from glob import glob
        fpsp = glob(frame_path + '/**/*.tiff')



    # data_path = os.listdir(base_path)

    # 프레임이 저장될 디렉토리 생성
    if not os.path.exists(frame_path):
        os.makedirs(frame_path, exist_ok=True)

    print("===== Extract image frames... =======")
    crop_left = 200
    crop_right = 1150

    for data in tqdm(data_path):

        patient_sp = data.split('/')[10].split('_')
        patient = patient_sp[0] + '_' + patient_sp[2]
        bolus_type = data.split('/')[-1].split('.')[0].split('-')[-1]
        year = data.split('/')[7]

        id = year+'_'+patient+'_'+bolus_type

        save_path = os.path.join(frame_path, id)

        
        
        extension = 'tiff'

        if '요플레' in save_path:
            print(f'replace {"요플레"}')
            save_path = save_path.replace('요플레', 'yogurt')
        
        if '이유식' in save_path:
            print(f'replace {"이유식"}')
            save_path = save_path.replace('이유식', 'babyfood')
        
        if '2024_0202_남궁숙희' in save_path:
            print(f'replace {"남궁숙희"}')
            save_path = save_path.replace('2024_0202_남궁숙희', '2024_0202_00810990')
        
        if ' 시행 후 ft3 주기 직전 aspiration' in save_path:
            print(f'replace {"aspiration"}')
            save_path = save_path.replace(' 시행 후 ft3 주기 직전 aspiration', 'aspiration')

        if ' ' in save_path:
            print(f'replace {"blank space"}')
            save_path = save_path.replace(' ', '')
        
        # save_folder = os.path.join(frame_path, id)

        # 241203
        # if os.path.exists(save_path):
        #     continue

        if not os.path.exists(save_path):
            os.makedirs(save_path, exist_ok=True)

        print(f"extract frames: {save_path}")       
        cap = cv2.VideoCapture(os.path.join(base_path, data))
        
        height = cap.get (cv2.CAP_PROP_FRAME_HEIGHT)
        width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        fps = cap.get (cv2.CAP_PROP_FPS)
        n_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        
        frames = []
        # ret, frame = cap.read()
        
        count = 0
        while(cap.isOpened()):
            # # 오래걸림
            # if os.path.exists(os.path.join(save_path, id+'_'+str(count)+'.tiff')):
            #     continue
                # print("**** extract idx: ", count)

            ret, image = cap.read()
            if ret: 
                if(int(cap.get(1)) % int(fps) == 0):               # extract frame per second. comment this line if you want to extract all frames
                    if os.path.join(os.path.join(save_path, id+'_'+str(count)+'.tiff')) in fpsp:
                        count += 1
                        continue
                    cv2.imwrite(os.path.join(save_path, id+'_'+str(count)+'.tiff'), image[:,crop_left:crop_right, :])
                    print("write ", count)
                    count += 1
            else:
                break
                
        cap.release()


extract_frames_with_cut_info()
print("Extracting Ended. Start Validating Names")
# unvalid_names = find_in_cut_info(frame_per_second_path="/mnt/ssd03_4tb/juny/vfss/vfss_cauh/frame_cut_info/frame_per_second_png")
# print("Unvalid Names: ", unvalid_names)