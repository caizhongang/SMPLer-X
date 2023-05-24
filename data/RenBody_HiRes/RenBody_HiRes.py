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

        for idx in range(2):
            if self.data_split == 'train':
                pre_prc_file_train = f'renbody_train_highrescam_230517_399_{idx}_fix_betas.npz'
                filename = getattr(cfg, 'filename', pre_prc_file_train)
            else:
                if idx > 0: continue
                pre_prc_file_test = f'renbody_test_highrescam_230517_78_{idx}_fix_betas.npz'
                filename = getattr(cfg, 'filename', pre_prc_file_test)

            self.img_dir = osp.join(cfg.data_dir, 'RenBody')
            self.annot_path = osp.join(cfg.data_dir, 'preprocessed_datasets', filename)
            self.img_shape = None # (h, w)
            self.cam_param = {}

            # check image shape
            # img_path = osp.join(self.img_dir, np.load(self.annot_path)['image_path'][0])
            # img_shape = cv2.imread(img_path).shape[:2]
            # assert self.img_shape == img_shape, 'image shape is incorrect: {} vs {}'.format(self.img_shape, img_shape)

            # load data
            datalist_slice = self.load_data(
                train_sample_interval=getattr(cfg, f'{self.__class__.__name__}_train_sample_interval', 1))
            self.datalist.extend(datalist_slice)
