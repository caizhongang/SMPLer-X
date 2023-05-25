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

        if self.data_split == 'train':
            filename = getattr(cfg, 'filename', 'p1_train.npz')
        else:
            raise ValueError('ARCTIC test set is not support')

        self.img_dir = osp.join(cfg.data_dir, 'ARCTIC')
        self.annot_path = osp.join(cfg.data_dir, 'preprocessed_datasets', filename)
        self.img_shape = None #1024, 1024)  # (h, w) 
        self.cam_param = {}

        # load data
        self.datalist = self.load_data(
            train_sample_interval=getattr(cfg, f'{self.__class__.__name__}_train_sample_interval', 1))