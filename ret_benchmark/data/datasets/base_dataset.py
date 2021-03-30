# encoding: utf-8

import os
import re
from collections import defaultdict
import numpy as np
from torch.utils.data import Dataset
from ret_benchmark.utils.img_reader import read_image
import torch
def find_clean_dataset(img_source):
    idx=img_source.find('noised_')+len('noised_')
    return img_source[:idx]+'cleaned_train.csv'
def get_is_noise(img_source,path_list):
    clean_img_source=find_clean_dataset(img_source)
    clean_path_list=list()
    with open(clean_img_source, "r") as f:
        for line in f:
            _path, _label = re.split(r",", line.strip())
            clean_path_list.append(_path)
    clean_path_list=np.asarray(clean_path_list)
    path_list=np.asarray(path_list)
    is_noise=np.isin(path_list,clean_path_list)
    return torch.BoolTensor(~is_noise)

class BaseDataSet(Dataset):
    """
    Basic Dataset read image path from img_source
    img_source: list of img_path and label
    """

    def __init__(self, img_source, transforms=None, mode="RGB",leng=-1,is_train=False):
        self.mode = mode
        self.transforms = transforms
        self.root = os.path.dirname(img_source)
        
        if 'noise' in img_source and 'cleaned' not in img_source and not is_train:
            img_source=find_clean_dataset(img_source)
        assert os.path.exists(img_source), f"{img_source} NOT found."
        self.img_source = img_source

        self.label_list = list()
        self.path_list = list()
        self._load_data()
        if 'noise' in img_source and 'cleaned' not in img_source and is_train:
            self.is_noise = get_is_noise(img_source,self.path_list)
        self.label_index_dict = self._build_label_index_dict()
        if leng!=-1:
            self.path_list=self.path_list[:leng]
            self.label_list=self.label_list[:leng]

    def __len__(self):
        return len(self.label_list)

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return f"| Dataset Info |datasize: {self.__len__()}|num_labels: {len(set(self.label_list))}|"

    def _load_data(self):
        with open(self.img_source, "r") as f:
            for line in f:
                _path, _label = re.split(r",", line.strip())
                self.path_list.append(_path)
                self.label_list.append(_label)

    def _build_label_index_dict(self):
        index_dict = defaultdict(list)
        for i, label in enumerate(self.label_list):
            index_dict[label].append(i)
        return index_dict

    def __getitem__(self, index):
        path = self.path_list[index]
        img_path = os.path.join(self.root, path)
        label = self.label_list[index]

        img = read_image(img_path, mode=self.mode)
        if self.transforms is not None:
            img = self.transforms(img)
        return img, label, index
