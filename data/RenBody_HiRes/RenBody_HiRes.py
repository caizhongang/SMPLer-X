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


class RenBody_HiRes(HumanDataset):
    def __init__(self, transform, data_split):
        super(RenBody_HiRes, self).__init__(transform, data_split)
        self.datalist = []
        if getattr(cfg, 'eval_on_train', False):
            self.data_split = 'eval_train'
            print("Evaluate on train set.")

        self.use_cache = getattr(cfg, 'use_cache', False)
        self.annot_path_cache = osp.join(cfg.data_dir, 'cache', f'renbody_{self.data_split}_highrescam_230517_399_fix_betas.npz')
        self.img_shape = None  # (h, w)
        self.cam_param = {}

        # load data or cache
        if self.use_cache and osp.isfile(self.annot_path_cache):
            print(f'[{self.__class__.__name__}] loading cache from {self.annot_path_cache}')
            self.datalist = self.load_cache(self.annot_path_cache)

        else:
            if self.use_cache:
                print(f'[{self.__class__.__name__}] Cache not found, generating cache...')

            for idx in range(2):
                if 'train' in self.data_split:
                    pre_prc_file_train = f'renbody_train_highrescam_230517_399_{idx}_fix_betas.npz'
                    filename = getattr(cfg, 'filename', pre_prc_file_train)
                else:
                    if idx > 0: continue
                    pre_prc_file_test = f'renbody_test_highrescam_230517_78_{idx}_fix_betas.npz'
                    filename = getattr(cfg, 'filename', pre_prc_file_test)

                self.img_dir = osp.join(cfg.data_dir, 'RenBody')
                self.annot_path = osp.join(cfg.data_dir, 'preprocessed_datasets', filename)

                if self.use_cache:
                    print(f'[{self.__class__.__name__}] Cache not found, generating cache...')
                
                data_split = self.load_data(
                    train_sample_interval=getattr(cfg, f'{self.__class__.__name__}_{self.data_split}_sample_interval', 1),
                    test_sample_interval=getattr(cfg, f'{self.__class__.__name__}_{self.data_split}_sample_interval', 10))
                self.datalist.extend(data_split)


            if self.use_cache:
                self.save_cache(self.annot_path_cache, self.datalist)