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

# issue: 4 IndexError: index 432000 is out of bounds for axis 0 with size 432000 (bbox = bbox_xywh[i][:4])
class RenBody(HumanDataset):
    def __init__(self, transform, data_split):
        super(RenBody, self).__init__(transform, data_split)

        self.use_cache = getattr(cfg, 'use_cache', False)
        if self.data_split == 'train':
                self.annot_path_cache = osp.join(cfg.data_dir, 'cache', 'renbody_train_230525_399_ds10_fix_betas.npz')
        else:
            self.annot_path_cache = osp.join(cfg.data_dir, 'cache', 'renbody_test_230525_78_ds10_fix_betas.npz')
        self.img_shape = None  # (h, w)
        self.cam_param = {}
        
        if self.use_cache and osp.isfile(self.annot_path_cache):
                print(f'[{self.__class__.__name__}] loading cache from {self.annot_path_cache}')
                self.datalist = self.load_cache(self.annot_path_cache)
        else:
            if self.use_cache:
                print(f'[{self.__class__.__name__}] Cache not found, generating cache...')
            # load data or cache
            self.datalist = []
            for idx in range(10):
                if self.data_split == 'train':
                    pre_prc_file_train = f'renbody_train_230525_399_{idx}.npz'
                    filename = getattr(cfg, 'filename', pre_prc_file_train)
                else:
                    if idx > 1: continue
                    pre_prc_file_test = f'renbody_test_230525_78_{idx}.npz'
                    filename = getattr(cfg, 'filename', pre_prc_file_test)

                self.img_dir = osp.join(cfg.data_dir, 'RenBody')
                self.annot_path = osp.join(cfg.data_dir, 'preprocessed_datasets', filename)
                
                # load data
                datalist_slice = self.load_data(
                    train_sample_interval=getattr(cfg, f'{self.__class__.__name__}_train_sample_interval', 1),
                    test_sample_interval=getattr(cfg, f'{self.__class__.__name__}_test_sample_interval', 1))
                
                self.datalist.extend(datalist_slice)
            
            if self.use_cache:
                self.save_cache(self.annot_path_cache, self.datalist)


            
