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


class GTA_Human2(HumanDataset):
    def __init__(self, transform, data_split, filename='gta_human2multiple_230406_04000_0.npz'):
        super(GTA_Human2, self).__init__(transform, data_split)

        self.img_dir = osp.join(cfg.data_dir, 'GTA_Human2')
        self.annot_path = osp.join(cfg.data_dir, 'preprocessed_datasets', filename)
        self.img_shape = (1080, 1920)  # (h, w)
        self.cam_param = {
            'focal': (1158.0337, 1158.0337),  # (fx, fy)
            'princpt': (960, 540)  # (cx, cy)
        }

        # check image shape
        img_path = osp.join(self.img_dir, np.load(self.annot_path)['image_path'][0])
        img_shape = cv2.imread(img_path).shape[:2]
        assert self.img_shape == img_shape, 'image shape is incorrect: {} vs {}'.format(self.img_shape, img_shape)

        # load data
        self.datalist = self.load_data()