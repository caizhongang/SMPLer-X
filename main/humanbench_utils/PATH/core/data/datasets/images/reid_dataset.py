try:
    import mc
except:
    print("no mc")
from torch.utils.data import Dataset
import numpy as np
import random
import struct
import os
import cv2
import os.path as osp
import torch
import time
from core.utils import cv2_loader, pil_loader
from core.data.transforms.reid_transforms import ReidAugmentation, NoneAugmentation, \
                    ReidAugmentationCV2, BodySplit, ReidTestAugmentationCV2, \
                    ReidTestAugmentation, ReidAugmentationTimm
from core import distributed_utils as dist

from core.utils import sync_print
try:
    from petrel_client.client import Client as Client

    s3client = Client(boto=True,
                      enable_multi_cluster=True,
                      enable_mc = True)
except:
    print("ceph can not be used")


def pack_labels(labels, begin_index=0):
    """
        if labels are not continuously, the function assign the labels from
        begin_index
    """
    if isinstance(labels, list):
        labels = np.array(labels)
    [a, b, c, d] = np.unique(
        labels, return_index=True, return_inverse=True, return_counts=True)
    a = np.array((range(begin_index, begin_index + len(a))))
    new_labels = a[np.array(c)]
    return new_labels

def merge_sub_datasets(task_spec, ginfo):
    output = {}
    task_list = task_spec['list']
    task_meta = task_spec['meta']
    task_pref = task_spec['prefix']

    num_sub_datasets = len(task_list)
    assert num_sub_datasets == len(task_meta)
    assert num_sub_datasets == len(task_pref)

    sync_print('task id: {}, Dataset: no split'.format(ginfo.task_id))
    label_base = 0
    images = []
    labels = []
    for list_file, meta_file, prefix in zip(task_list, task_meta, task_pref):
        if list_file.startswith('/mnt/') or  list_file.startswith('/data/'):
            with open(list_file) as f:
                list_lines = f.readlines()
                list_lines = [osp.join(prefix, x.strip()) for x in list_lines]
            with open(meta_file) as f:
                meta_lines = f.readlines()
                meta_lines = [label_base+int(x.strip()) for x in meta_lines[1:]]
                label_base = max(meta_lines) + 1 # update label_base
        else:
            list_lines = s3client.Get(list_file)
            if not list_lines:
                print('File not exist', list_file)
                import pdb; pdb.set_trace()
                raise IOError('File not exist', list_file)
            list_lines = list_lines.decode('ascii')
            list_lines = list_lines.split('\n')
            list_lines = list_lines[:-1]
            list_lines = [osp.join(prefix, x) for x in list_lines]

            meta_lines = s3client.Get(meta_file)
            if not meta_lines:
                print('File not exist', meta_file)
                import pdb; pdb.set_trace()
                raise IOError('File not exist', meta_file)
            meta_lines = meta_lines.decode('ascii')
            meta_lines = meta_lines.split('\n')
            meta_lines = [label_base+int(x) for x in meta_lines[1:-1]]
            label_base = max(meta_lines) + 1 # update label_base

        assert len(list_lines) == len(meta_lines)
        images.extend(list_lines)
        labels.extend(meta_lines)

    output['images'] = images
    labels = np.array(labels)
    labels = pack_labels(labels)
    output['labels'] = labels
    output['num_classes'] = np.max(labels) + 1
    return output


class ReIDDataset(Dataset):
    def __init__(self, ginfo, augmentation, task_spec, train=True, loader='cv2', data_use_ratio=1, **kwargs):
        width = augmentation.get('width', None)
        height = augmentation.get('height', None)
        assert width is not None, 'width not set!'
        assert height is not None, 'height not set!'
        earser = augmentation.get('earser', True)
        bright_aug=augmentation.get('brightness',False)
        contrast_aug=augmentation.get('contrast',False)
        vit = augmentation.get('vit', False)
        if loader == 'cv2':
            if train:
                #dist.barrier()
                sync_print('Train - task {}: brightness aug: {} contrast_aug:{}'.format(ginfo.task_name,bright_aug,contrast_aug))
                if earser:
                    sync_print('Train - task {}: using random eraser'.format(ginfo.task_name))
                    self.reid_aug = ReidAugmentationCV2(height, width, earser,bright_aug,contrast_aug, vit=vit)
                else:
                    sync_print('Train - task {}: Not using random eraser'.format(ginfo.task_name))
                    self.reid_aug = ReidAugmentationCV2(height, width, earser,bright_aug,contrast_aug, vit=vit)

                split_aug_config = augmentation.get('split', None)
                if split_aug_config is None:
                    sync_print('Train - task {}: not using extra info'.format(ginfo.task_name))

                self.body_split = None
                if split_aug_config is not None:
                    split_aug_flag = split_aug_config.get('flag', 'default')
                    extra_info_file = split_aug_config.get('extra_info', None)
                    aug_type = split_aug_config.get('aug_type', -1)
                    if split_aug_flag == 'default':
                        bg_type = split_aug_config.get('bg_type', 0)
                        split_prob = split_aug_config.get('split_prob', 0.5)
                        sync_print('Train - [body split crop-aug]task {}: using extra info {}, bg_type {},aug_type {},aug prob {}'.
                                   format(ginfo.task_name, extra_info_file, bg_type, aug_type, split_prob))
                        self.body_split = BodySplit(extra_info_file, bg_type=bg_type, aug_type=aug_type, split_prob=split_prob)
                    else:
                        raise ValueError('unknown split aug_flag:{}'.format(split_aug_flag))
            else:
                sync_print('Val - task {}: width {}, height {}'.format(ginfo.task_name, width, height))
                self.body_split = None
                self.reid_aug = ReidTestAugmentationCV2(height, width, vit=vit)
        else:
            if train:
                if earser:
                    sync_print('Train - task {}: using random eraser'.format(ginfo.task_name))
                    self.reid_aug = ReidAugmentation(height, width, earser, vit=vit)
                    self.body_split = None
                else:
                    sync_print('Train - task {}: Not using random eraser'.format(ginfo.task_name))
                    self.reid_aug = ReidAugmentation(height, width, earser, vit=vit)
                    self.body_split = None
            else:
                sync_print('Val - task {}: width {}, height {}'.format(ginfo.task_name, width, height))
                self.reid_aug = ReidTestAugmentation(height, width, earser, vit=vit)
                self.body_split = None

        self.meta_data = merge_sub_datasets(task_spec, ginfo)

        ginfo.task_num_classes = self.meta_data['num_classes']
        self.images = self.meta_data['images']
        self.labels = self.meta_data['labels']

        if data_use_ratio != 1:
            img_nums = len(self.images)
            index_selected = random.sample(list(range(img_nums)), int(img_nums * data_use_ratio))

            self.images = np.array(self.images)
            self.labels = np.array(self.labels)

            self.images = list(self.images[index_selected])
            self.labels = list(self.labels[index_selected])

        self.num = len(self.images)
        self.num_classes = self.meta_data['num_classes']
        self.task_name = ginfo.task_name
        self.loader = loader
        self.initialized = False

        self.use_ceph = True

        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()

    def __len__(self):
        return self.num

    def _init_memcached(self):
        if not self.initialized:
            ## only use mc default
            print("==> will load files from local machine")
            server_list_config_file = "/mnt/lustre/share/memcached_client/server_list.conf"
            client_config_file = "/mnt/lustre/share/memcached_client/client.conf"
            self.memcached_mclient = mc.MemcachedClient.GetInstance(server_list_config_file, client_config_file)
            ## mc-support-ceph
            print('mc-support-ceph')
            self.ceph_mclient = s3client

            self.initialized = True

    def _read_one(self, idx=None):
        if idx == None:
            idx = np.random.randint(self.num)

        ## try read image first
        filename = self.images[idx]

        try:
            if filename.startswith('/mnt/') or filename.startswith('/data/'):
                value = mc.pyvector()
                self.memcached_mclient.Get(filename, value)
                value_str = mc.ConvertBuffer(value)
                if self.loader == 'cv2':
                    img = cv2_loader(value_str)
                else:
                    img = pil_loader(value_str)
            else:
                value = self.ceph_mclient.Get(filename)
                if value:
                    value_str = np.fromstring(value, np.uint8)
                if self.loader == 'cv2':
                    img = cv2_loader(value_str)
                else:
                    img = pil_loader(value_str)
            if img is None:
                raise Exception('None Image')
        except:
            outputName = "failed_to_read_in_train.txt"
            with open(outputName,"a") as g:
                g.write("%s\n"%(filename))
            print('Read image[{}] failed ({})'.format(idx, filename))
            ## if fail then recursive call _read_one without idx
            return self._read_one()
        else:
            output = dict()
            ##set random_seed with img idx
            random.seed(idx+self.rank)
            np.random.seed(idx+self.rank)

            if self.body_split is not None:
                img = self.body_split(img, filename)

            img = self.reid_aug(img)

            output['image'] = img

            label = self.labels[idx]
            output['label'] = label
            output['filename'] = filename

            return output

    def __getitem__(self, idx):

        self._init_memcached()
        return self._read_one(idx)

    def __repr__(self):
        return self.__class__.__name__ + \
            'task: {} dataset_num:{} id_num:{}'.format(self.task_name,
            self.num, self.num_classes)
