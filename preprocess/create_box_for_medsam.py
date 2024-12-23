import os
import glob
import cv2 as cv
import numpy as np

path = "/mnt/ssd03_4tb/juny/vfss/vfss_cauh/experiment"
list = "/mnt/ssd01_250gb/juny/vfss/SAMed/lists/lists_vfss/test.txt"

list_file = open(list, 'r')
file_names = list_file.readlines()
file_names = [f.replace('\n', '') for f in file_names]

    # for file_name in file_names:
    #     data = np.load(os.path.join(path, file_names[0]+'.npz'))
def create_bbox(label):
    nonz = np.nonzero(label)
    ys, xs = nonz
    left = xs.min()
    right = xs.max()
    top = ys.min()
    bottom = ys.max()
    
    origin = (left, top)
    height = bottom - top
    width = right - left

    box = (left, top, right, bottom)

    # box = (left, top, width, height)

    return box