import os
import random
# import h5py
import numpy as np
import torch
from scipy import ndimage
from scipy.ndimage.interpolation import zoom
from torch.utils.data import Dataset
from einops import repeat
from icecream import ic


def random_rot_flip(image, label):
    k = np.random.randint(0, 4)
    image = np.rot90(image, k, axes=(-2,-1))
    label = np.rot90(label, k, axes=(-2,-1))
    axis = np.random.randint(0, 2)
    image = np.flip(image, axis=axis).copy()
    label = np.flip(label, axis=axis).copy()
    return image, label


def random_rotate(image, label):
    angle = np.random.randint(-20, 20)
    image = ndimage.rotate(image, angle, order=0, reshape=False, axes=(-2,-1))
    label = ndimage.rotate(label, angle, order=0, reshape=False, axes=(-2,-1))
    return image, label


class RandomGenerator(object):
    def __init__(self, output_size, low_res):
        self.output_size = output_size
        self.low_res = low_res

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        # flip은 vfss 환경에서는 필요 없을 것 같아서 사용 X
        # if random.random() > 0.5:
        #     image, label = random_rot_flip(image, label)
        if random.random() > 0.5:
            image, label = random_rotate(image, label)
        d, x, y = image.shape
        if x != self.output_size[0] or y != self.output_size[1]:
            image = zoom(image, (1, self.output_size[0] / x, self.output_size[1] / y), order=3)  # why not 3?
            label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        label_h, label_w = label.shape
        low_res_label = zoom(label, (self.low_res[0] / label_h, self.low_res[1] / label_w), order=0)
        image = torch.from_numpy(image.astype(np.uint8)).unsqueeze(0)
        # image = repeat(image, 'b c h w -> (repeat b) c h w', repeat=3)
        label = torch.from_numpy(label.astype(np.uint8))
        low_res_label = torch.from_numpy(low_res_label.astype(np.uint8))
        sample = {'image': image, 'label': label, 'low_res_label': low_res_label}
        return sample


class Vfss_dataset(Dataset):
    def __init__(self, base_dir="/mnt/ssd03_4tb/juny/vfss/vfss_cauh/experiment", list_dir="/mnt/ssd01_250gb/juny/vfss/SAMed/lists_vfss/", split='train', transform=None):
        self.transform = transform  # using transform in torch!
        self.split = split
        self.sample_list = open(os.path.join(list_dir, self.split+'.txt')).readlines()
        # self.data_dir = os.path.join(base_dir, self.split)
        self.data_dir = base_dir

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        if (self.split == "train") or ('train' in self.split):
            slice_name = self.sample_list[idx].strip('\n')
            data_path = os.path.join(self.data_dir, slice_name+'.npz')
            data = np.load(data_path)
            image, label = data['image'], data['label']
            # image, label = np.transpose(data['image'], (0,2,1)), np.transpose(data['label'], (1,0))
        else:
            vol_name = self.sample_list[idx].strip('\n')
            filepath = self.data_dir + "/{}.npz".format(vol_name)
            # data = h5py.File(filepath)
            data = np.load(filepath)
            image, label = data['image'], data['label']
            # image, label = np.transpose(data['image'], (0,2,1)), np.transpose(data['label'], (1,0))

        # Input dim should be consistent
        # Since the channel dimension of nature image is 3, that of medical image should also be 3

        sample = {'image': image, 'label': label}
        if self.transform:
            sample = self.transform(sample)
        sample['case_name'] = self.sample_list[idx].strip('\n')

        if sample['image'].shape[1] != 3:
            print()
        return sample
