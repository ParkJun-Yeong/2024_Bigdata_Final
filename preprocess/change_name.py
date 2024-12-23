import os
import shutil
from glob import glob

def change_in_frame_per_second():
    base = "frame_per_second"
    for folder in os.listdir(base):
        new_name = folder.replace()
        os.rename(os.path.join(base, folder), os.path.join(base, folder))


# 241111 cut_info와 frame_per_second의 이름 동일한지 확인 (후자는 이전에 수작업으로 이름 정리하여 학습에 사용, 전자는 이름 정제가 안 돼 있음.)
def find_in_cut_info(base=None, frame_per_second_path=None):
    base = "/mnt/ssd03_4tb/juny/vfss/vfss_cauh/cut_info"
    if frame_per_second_path == None:
        frame_per_second_path = "/mnt/ssd03_4tb/juny/vfss/vfss_cauh/frame_cut_info/frame_per_second"


    avis = os.listdir(base)
    if 'nofolder' in frame_per_second_path:
        frame_seconds = [f+'.avi' for f in os.listdir(frame_per_second_path)]
    else:
        frame_seconds = glob()

    name_unmatched = []
    for avi in avis:
        if avi not in frame_seconds:
            # print("name unmatched: ", avi)
            name_unmatched.append(avi)
    
    print("=====FINAL RESULT=====")
    print(name_unmatched)

    return name_unmatched


def change_in_cut_info(name_unmatched=None):
    base = "/mnt/ssd03_4tb/juny/vfss/vfss_cauh/cut_info"
    frame_per_second_path = "/mnt/ssd03_4tb/juny/vfss/vfss_cauh/frame_cut_info/frame_per_second"

    frame_seconds = [f+'.avi' for f in os.listdir(frame_per_second_path)]

    if name_unmatched is None:
        name_unmatched = find_in_cut_info()
    
    
    for name in name_unmatched:
        
        # 이상한 단어 먼저 제거
        new_name = name.replace('01102500_요플레', '01102500_yogurt')
        new_name = new_name.replace('이유식', 'babyfood')
        new_name = new_name.replace('2024_0202_남궁숙희', '2024_0202_00810990')
        new_name = new_name.replace(' 시행 후 ft3 주기 직전 aspiration.avi', 'aspiration.avi')

        new_name = new_name.replace(' ', '')

        shutil.move(os.path.join(base, name), os.path.join(base, new_name))
        
        
        
        # name 제대로 바뀌었는지 확인
        # if new_name not in frame_seconds:
        #     print(new_name)

    # print(find_in_cut_info())


    



if __name__ == "__main__":
    name_unmatched = find_in_cut_info()
    # change_in_cut_info()
    
        

