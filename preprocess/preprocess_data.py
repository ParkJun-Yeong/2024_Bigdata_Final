import os
import time
import argparse
from glob import glob

import h5py
import nibabel as nib
import numpy as np
from tqdm import tqdm
# from icecream import ic
import cv2 as cv


parser = argparse.ArgumentParser()
parser.add_argument('--src_path', type=str,
                   default='/mnt/ssd03_4tb/juny/vfss/vfss_cauh', help='download path for Synapse data')
parser.add_argument('--dst_path', type=str,
                   default='/mnt/ssd03_4tb/juny/vfss/vfss_cauh/experiment', help='root dir for data')
parser.add_argument('--use_normalize', action='store_true', default=True,
                   help='use normalize')
args = parser.parse_args()

test_data = [1, 2, 3, 4, 8, 22, 25, 29, 32, 35, 36, 38]

hashmap = {1:1, 2:2, 3:3, 4:4, 5:0, 6:5, 7:6, 8:7, 9:0, 10:0, 11:8, 12:0, 13:0}


# tif annotation
def preprocess_train_tif(image_files: list, label_files: list) -> None:
    os.makedirs(f"{args.dst_path}/train", exist_ok=True)

    # a_min, a_max = -125, 275
    b_min, b_max = 0.0, 1.0         # binary

    a_min, a_max = 0, 255


    pbar = tqdm(zip(image_files, label_files), total=len(image_files))
    for image_file, label_file in pbar:

        image_data = cv.imread(image_file)
        image_data = cv.cvtColor(image_data, cv.COLOR_BGR2RGB)
        # image_data = cv.cvtColor(image_data)


        label_data = cv.imread(label_file, cv.IMREAD_GRAYSCALE)
        _, label_data = cv.threshold(label_data, 127, 255, cv.THRESH_BINARY)

        # image_data = nib.load(image_file).get_fdata()
        # label_data = nib.load(label_file).get_fdata()

        lower_bound, upper_bound = np.percentile(
                image_data[image_data > 0], 0.5
            ), np.percentile(image_data[image_data > 0], 99.5)
        image_data_pre = np.clip(image_data, lower_bound, upper_bound)
        # normalize
        image_data_pre = (
            (image_data_pre - np.min(image_data_pre))
            / (np.max(image_data_pre) - np.min(image_data_pre))
            * 255.0
        )
        image_data_pre[image_data == 0] = 0

        image_data_pre = np.uint8(image_data_pre)

        image_data = image_data_pre
        
        # label value clipping to binary
        label_data = np.clip(label_data, b_min, b_max)
        # if args.use_normalize:
        #     assert a_max != a_min
        #     image_data = (image_data - a_min) / (a_max - a_min)

        H, W, D = image_data.shape

        image_data = np.transpose(image_data, (2, 0, 1))  # [D, W, H]
        # label_data = np.transpose(label_data, (1, 0))

        counter = 1
        for k in sorted(hashmap.keys()):
            assert counter == k
            counter += 1
            label_data[label_data == k] = hashmap[k]

        # x-ray 이므로 slice로 나눌 필요 없음.
        save_path = f"{args.dst_path}/train/{label_file.split('/')[-1].replace('tif', 'npz')}"
        np.savez(save_path, label=label_data, image=image_data)
        # for dep in range(D):
        #     save_path = f"{args.dst_path}/train/{label_file.split('/')[-1].replace('tif', 'npz')}"
        #     np.savez(save_path, label=label_data, image=image_data)
        #     # np.savez(save_path, label=label_data[dep,:,:], image=image_data[dep,:,:])

    pbar.close()


def preprocess_valid_image(image_files: str, label_files: str) -> None:
    os.makedirs(f"{args.dst_path}/test", exist_ok=True)

    # a_min, a_max = -125, 275
    # b_min, b_max = 0.0, 1.0

    b_min, b_max = 0.0, 1.0         # binary
    a_min, a_max = 0, 255

    pbar = tqdm(zip(image_files, label_files), total=len(image_files))
    for image_file, label_file in pbar:
        # **/imgXXXX.nii.gz -> parse XXXX
        number = image_file.split('/')[-1][3:7]

        if int(number) not in test_data:
            continue

        image_data = nib.load(image_file).get_fdata()
        label_data = nib.load(label_file).get_fdata()

        image_data = image_data.astype(np.float32)
        label_data = label_data.astype(np.float32)

        image_data = np.clip(image_data, a_min, a_max)
        if args.use_normalize:
            assert a_max != a_min
            image_data = (image_data - a_min) / (a_max - a_min)

        H, W, D = image_data.shape

        image_data = np.transpose(image_data, (2, 1, 0))
        label_data = np.transpose(label_data, (2, 1, 0))

        counter = 1
        for k in sorted(hashmap.keys()):
            assert counter == k
            counter += 1
            label_data[label_data == k] = hashmap[k]

        save_path = f"{args.dst_path}/test_vol_h5/case{number}.npy.h5"
        f = h5py.File(save_path, 'w')
        f['image'] = image_data
        f['label'] = label_data
        f.close()
    pbar.close()


# tif annotation
def preprocess_vfss_train_tif(image_files: list, label_files: list) -> None:
    os.makedirs(f"{args.dst_path}/train", exist_ok=True)

    # a_min, a_max = -125, 275
    b_min, b_max = 0.0, 1.0         # binary

    a_min, a_max = 0, 255


    pbar = tqdm(zip(image_files, label_files), total=len(image_files))
    for image_file, label_file in pbar:
        # **/imgXXXX.nii.gz -> parse XXXX
        # number = image_file.split('/')[-1][3:7]

        # if int(number) in test_data:
        #     continue

        image_data = cv.imread(image_file)
        try:
            image_data = cv.cvtColor(image_data, cv.COLOR_BGR2RGB)
        except:
            print()
        # image_data = cv.cvtColor(image_data)


        label_data = cv.imread(label_file, cv.IMREAD_GRAYSCALE)
        _, label_data = cv.threshold(label_data, 127, 255, cv.THRESH_BINARY)

        # image_data = nib.load(image_file).get_fdata()
        # label_data = nib.load(label_file).get_fdata()

        image_data = image_data.astype(np.float32)
        label_data = label_data.astype(np.float32)

        image_data = np.clip(image_data, a_min, a_max)
        # label value clipping to binary
        label_data = np.clip(label_data, b_min, b_max)
        if args.use_normalize:
            assert a_max != a_min
            image_data = (image_data - a_min) / (a_max - a_min)

        H, W, D = image_data.shape

        image_data = np.transpose(image_data, (2, 0, 1))  # [D, W, H]
        # label_data = np.transpose(label_data, (1, 0))

        counter = 1
        for k in sorted(hashmap.keys()):
            assert counter == k
            counter += 1
            label_data[label_data == k] = hashmap[k]

        # x-ray 이므로 slice로 나눌 필요 없음.
        save_path = f"{args.dst_path}/train/{label_file.split('/')[-1].replace('tif', 'npz')}"
        np.savez(save_path, label=label_data, image=image_data)
        # for dep in range(D):
        #     save_path = f"{args.dst_path}/train/{label_file.split('/')[-1].replace('tif', 'npz')}"
        #     np.savez(save_path, label=label_data, image=image_data)
        #     # np.savez(save_path, label=label_data[dep,:,:], image=image_data[dep,:,:])

    pbar.close()


# png annotation
def preprocess_train_png(image_files: list, label_files: list) -> None:
    os.makedirs(f"{args.dst_path}/train", exist_ok=True)

    # a_min, a_max = -125, 275
    b_min, b_max = 0.0, 1.0         # binary

    a_min, a_max = 0, 255


    pbar = tqdm(zip(image_files, label_files), total=len(image_files))
    for image_file, label_file in pbar:

        image_data = cv.imread(image_file)
        try:
            image_data = cv.cvtColor(image_data, cv.COLOR_BGR2RGB)
        except:
            print()
        # image_data = cv.cvtColor(image_data)


        label_data = cv.imread(label_file, cv.IMREAD_GRAYSCALE)
        # tif와 다른부분
        threshold = 37
        _, label_data = cv.threshold(label_data, threshold, 255, cv.THRESH_BINARY)



        lower_bound, upper_bound = np.percentile(
                image_data[image_data > 0], 0.5
            ), np.percentile(image_data[image_data > 0], 99.5)
        image_data_pre = np.clip(image_data, lower_bound, upper_bound)
        # normalize
        image_data_pre = (
            (image_data_pre - np.min(image_data_pre))
            / (np.max(image_data_pre) - np.min(image_data_pre))
            * 255.0
        )
        image_data_pre[image_data == 0] = 0

        image_data_pre = np.uint8(image_data_pre)

        image_data = image_data_pre
        
        # label value clipping to binary
        label_data = np.clip(label_data, b_min, b_max)
        # if args.use_normalize:
        #     assert a_max != a_min
        #     image_data = (image_data - a_min) / (a_max - a_min)

        H, W, D = image_data.shape

        image_data = np.transpose(image_data, (2, 0, 1))  # [D, W, H]
        # label_data = np.transpose(label_data, (1, 0))

        counter = 1
        for k in sorted(hashmap.keys()):
            assert counter == k
            counter += 1
            label_data[label_data == k] = hashmap[k]

        # x-ray 이므로 slice로 나눌 필요 없음.
        save_path = f"{args.dst_path}/train/{label_file.split('/')[-1].replace('png', 'npz')}"
        np.savez(save_path, label=label_data, image=image_data)
        # for dep in range(D):
        #     save_path = f"{args.dst_path}/train/{label_file.split('/')[-1].replace('tif', 'npz')}"
        #     np.savez(save_path, label=label_data, image=image_data)
        #     # np.savez(save_path, label=label_data[dep,:,:], image=image_data[dep,:,:])

    pbar.close()



if __name__ == "__main__":
    # data_root = f"{args.src_path}/Training"
    data_root = f"{args.src_path}/frame_cut_info/frame_per_second_png/"
    # label_root = f"{args.src_path}/annotation/pa/"
    label_root = f"{args.src_path}/annotation/241120"

    # tif annotation
    # image_files = sorted(glob(f"{data_root}/**/*.jpg", recursive=True))
    # label_files = sorted(glob(f"{label_root}/**/*.tif", recursive=True))
    label_files = sorted(glob(f"{label_root}/**/*.png", recursive=True))
    # image_files = [label_file.replace('_X', '').replace('annotation/pa', 'frame_cut_info/frame_per_second').replace('tif', 'jpg') for label_file in label_files]
    image_files = [label_file.replace('_X', '').replace('annotation/241120', 'frame_cut_info/frame_per_second_refined_nofolder').replace('png', 'jpg') for label_file in label_files]


    # preprocess_train_tif(image_files, label_files)

    # png annotation
    preprocess_train_png(image_files, label_files)