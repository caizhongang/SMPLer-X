import os
import pickle
import random
from easydict import EasyDict as edict
import numpy as np
import torch.utils.data as data
from PIL import Image
from petrelbox.io import PetrelHelper
from core.data.transforms.pedattr_transforms import PedAttrAugmentation, PedAttrTestAugmentation, PedAttrRandomAugmentation

__all__ = ['AttrDataset', 'MultiAttrDataset']

def merge_pedattr_datasets(data_path_list, root_path_list, dataset_name_list, train, data_use_ratio):
    total_img_id = []
    total_attr_num = 0
    total_img_num = 0
    total_attr_begin = []
    total_attr_end = []
    total_img_begin = []
    total_img_end = []

    for data_path, root_path, dataset_name in zip(data_path_list, root_path_list, dataset_name_list):
        assert dataset_name in ['peta', 'PA-100k', 'rap', 'rap2', 'uavhuman', 'HARDHC', 'ClothingAttribute', 'parse27k', 'duke', 'market'], 'dataset name {} is not exist'.format(dataset_name)
        
        dataset_info = PetrelHelper.pickle_load(data_path)
        dataset_info = edict(dataset_info)

        img_id = dataset_info.image_name
        attr_label = dataset_info.label

        if train:
            split = 'trainval'
        else:
            split = 'test'

        assert split in dataset_info.partition.keys(), f'split {split} is not exist'

        attr_id = dataset_info.attr_name
        attr_num = len(attr_id)

        img_idx = dataset_info.partition[split]

        if isinstance(img_idx, list):
            img_idx = img_idx[0]  # default partition 0

        if data_use_ratio != 1:
            img_idx = random.sample(list(img_idx), int(len(img_idx) * data_use_ratio))
        
        img_num = len(img_idx)
        img_idx = np.array(img_idx)

        img_id = [os.path.join(root_path, img_id[i]) for i in img_idx]
        label = attr_label[img_idx]

        total_attr_begin.append(total_attr_num)
        total_img_begin.append(total_img_num)
        total_attr_num = total_attr_num + attr_num
        total_img_num = total_img_num + len(img_id)
        total_attr_end.append(total_attr_num)
        total_img_end.append(total_img_num)


    total_label = np.full((total_img_num, total_attr_num), -1, dtype=np.int)

    for index, (data_path, root_path, dataset_name) in enumerate(zip(data_path_list, root_path_list, dataset_name_list)):
        assert dataset_name in ['peta', 'PA-100k', 'rap', 'rap2', 'uavhuman', 'HARDHC', 'ClothingAttribute', 'parse27k', 'duke', 'market'], 'dataset name {} is not exist'.format(dataset_name)
        dataset_info = PetrelHelper.pickle_load(data_path)
        dataset_info = edict(dataset_info)

        img_id = dataset_info.image_name
        attr_label = dataset_info.label

        if train:
            split = 'trainval'
        else:
            split = 'test'

        assert split in dataset_info.partition.keys(), f'split {split} is not exist'

        attr_id = dataset_info.attr_name
        attr_num = len(attr_id)

        img_idx = dataset_info.partition[split]

        if isinstance(img_idx, list):
            img_idx = img_idx[0]  # default partition 0

        if data_use_ratio != 1:
            img_idx = random.sample(list(img_idx), int(len(img_idx) * data_use_ratio))

        img_num = len(img_idx)
        img_idx = np.array(img_idx)

        img_id = [os.path.join(root_path, img_id[i]) for i in img_idx]
        label = attr_label[img_idx]
        total_label[total_img_begin[index]: total_img_end[index], total_attr_begin[index]: total_attr_end[index]] = label
        total_img_id.extend(img_id)

    return total_img_id, total_label



class MultiAttrDataset(data.Dataset):

    def __init__(self, ginfo, augmentation, task_spec, train=True, data_use_ratio=1, **kwargs):
        data_path = task_spec.data_path
        root_path = task_spec.root_path
        dataset_name = task_spec.dataset
        # import pdb; pdb.set_trace()
        self.img_id, self.label = merge_pedattr_datasets(data_path, root_path, dataset_name, train, data_use_ratio)
        height = augmentation.height
        width = augmentation.width
        self.img_num = len(self.img_id)

        if train:
            self.transform = PedAttrAugmentation(height, width)
            if augmentation.use_random_aug:
                self.transform = PedAttrRandomAugmentation(height, width, \
                    augmentation.use_random_aug.m, augmentation.use_random_aug.n)
        else:
            self.transform = PedAttrTestAugmentation(height, width)


        self.task_name = ginfo.task_name

    def __getitem__(self, index):
        return self.read_one(index)

    def __len__(self):
        return len(self.img_id)

    def read_one(self, idx=None):
        if idx == None:
            idx = np.random.randint(self.img_num)

        imgname, gt_label = self.img_id[idx], self.label[idx]
        imgpath = imgname

        try:
            img = PetrelHelper.pil_open(imgpath, "RGB")
            if self.transform is not None:
                img = self.transform(img)

            gt_label = gt_label.astype(np.float32)

            output = {}
            output = {'image': img, 'label': gt_label, 'filename': imgname}
            return output
        except:
            print('{} load failed'.format(imgpath))
            return self.read_one()

class AttrDataset(data.Dataset):

    def __init__(self, ginfo, augmentation, task_spec, train=True, data_use_ratio=1, **kwargs):

        assert task_spec.dataset in ['peta', 'PA-100k', 'rap', 'rap2', 'uavhuman', 'HARDHC', 'ClothingAttribute', 'parse27k', 'duke', 'market'], \
            f'dataset name {task_spec.dataset} is not exist'

        data_path = task_spec.data_path

        dataset_info = PetrelHelper.pickle_load(data_path)
        # dataset_info = pickle.load(open(data_path, 'rb+'))
        dataset_info = edict(dataset_info)

        img_id = dataset_info.image_name
        attr_label = dataset_info.label

        if train:
            split = 'trainval'
        else:
            split = 'test'

        assert split in dataset_info.partition.keys(), f'split {split} is not exist'

        height = augmentation.height
        width = augmentation.width

        self.dataset = task_spec.dataset
        self.root_path = task_spec.root_path

        if train:
            self.transform = PedAttrAugmentation(height, width)
            if augmentation.use_random_aug:
                self.transform = PedAttrRandomAugmentation(height, width, \
                    augmentation.use_random_aug.m, augmentation.use_random_aug.n)
        else:
            self.transform = PedAttrTestAugmentation(height, width)

        self.attr_id = dataset_info.attr_name
        self.attr_num = len(self.attr_id)

        self.img_idx = dataset_info.partition[split]

        if isinstance(self.img_idx, list):
            self.img_idx = self.img_idx[0]  # default partition 0

        if data_use_ratio != 1:
            self.img_idx = random.sample(list(self.img_idx), int(len(self.img_idx) * data_use_ratio))

        self.img_num = len(self.img_idx)
        self.img_idx = np.array(self.img_idx)

        self.img_id = [img_id[i] for i in self.img_idx]
        self.label = attr_label[self.img_idx]
        self.task_name = ginfo.task_name

    def __getitem__(self, index):
        return self.read_one(index)

    def __len__(self):
        return len(self.img_id)

    def read_one(self, idx=None):
        if idx == None:
            idx = np.random.randint(self.img_num)

        imgname, gt_label, imgidx = self.img_id[idx], self.label[idx], self.img_idx[idx]
        imgpath = os.path.join(self.root_path, imgname)

        try:
            img = PetrelHelper.pil_open(imgpath, "RGB")
            if self.transform is not None:
                img = self.transform(img)

            gt_label = gt_label.astype(np.float32)

            output = {}
            output = {'image': img, 'label': gt_label, 'filename': imgname}
            return output
        except:
            print('{} load failed'.format(imgpath))
            return self.read_one()
