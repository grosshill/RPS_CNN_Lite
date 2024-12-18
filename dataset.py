import os
import cv2
import torch
from torch.utils.data import Dataset
from natsort import natsorted as nst
from typing import Tuple
from matplotlib import pyplot as plt


class rps_dataset(Dataset):
    def __init__(self, data_dir: str, data_type: str = 'test', transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.classes = ['rock', 'paper', 'scissors']
        self.data_list = []

        if data_type == 'train':
            re_dir = 'rps'
        elif data_type == 'test':
            re_dir = 'rps-test-set'
        else:
            raise ValueError('data_type must be either "train" or "test", got {}'.format(data_type))

        # load the dataset
        for idx, name in enumerate(self.classes):
            this_abs_dir = os.path.join(data_dir, re_dir, name)
            this_data = os.listdir(this_abs_dir)
            this_data = nst(this_data)
            for data in this_data:
                self.data_list.append((os.path.join(this_abs_dir, data), idx))

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, str]:
        img_dir, label = self.data_list[idx]
        # img = cv2.imread(img_dir, cv2.IMREAD_UNCHANGED)
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        img = cv2.imread(img_dir, cv2.IMREAD_GRAYSCALE)
        img = cv2.threshold(img, 225, 255, cv2.THRESH_BINARY)[1]
        # plt.imshow(img, cmap='gray')
        # plt.show()
        # implement transform
        if self.transform is not None:
            img = self.transform(img)
        return img, label, img_dir

    def __len__(self) -> int:
        return len(self.data_list)
