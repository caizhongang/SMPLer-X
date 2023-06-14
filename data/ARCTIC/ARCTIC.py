import os
import os.path as osp
import numpy as np
import torch
import cv2
import json
import copy
from pycocotools.coco import COCO
from config import cfg
from utils.human_models import smpl_x
from utils.preprocessing import load_img, process_bbox, augmentation, process_db_coord, process_human_model_output, \
    get_fitting_error_3D
from utils.transforms import world2cam, cam2pixel, rigid_align
from humandata import HumanDataset


class ARCTIC(HumanDataset):
    def __init__(self, transform, data_split):
        super(ARCTIC, self).__init__(transform, data_split)

        if getattr(cfg, 'eval_on_train', False):
            self.data_split = 'eval_train'
            print("Evaluate on train set.")

        self.use_cache = getattr(cfg, 'use_cache', False)
        self.annot_path_cache = osp.join(cfg.data_dir, 'cache', f'arctic_{self.data_split}.npz')
        self.img_shape = None  # (h, w)
        self.cam_param = {}

        if self.use_cache and osp.isfile(self.annot_path_cache):
            print(f'[{self.__class__.__name__}] loading cache from {self.annot_path_cache}')
            self.datalist = self.load_cache(self.annot_path_cache)
        else:
            if self.use_cache:
                print(f'[{self.__class__.__name__}] Cache not found, generating cache...')

            if 'train' in data_split:
                filename = getattr(cfg, 'filename', 'p1_train.npz')
            else:
                filename = getattr(cfg, 'filename', 'p1_val.npz')

            self.img_dir = osp.join(cfg.data_dir, 'ARCTIC')
            self.annot_path = osp.join(cfg.data_dir, 'preprocessed_datasets', filename)

            # load data
            self.datalist = self.load_data(
                train_sample_interval=getattr(cfg, f'{self.__class__.__name__}_train_sample_interval', 1),
                test_sample_interval=getattr(cfg, f'{self.__class__.__name__}_test_sample_interval', 10))

            if self.use_cache:
                self.save_cache(self.annot_path_cache, self.datalist)
