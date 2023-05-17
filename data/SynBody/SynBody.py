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


class SynBody(HumanDataset):
    def __init__(self, transform, data_split, filename='synbody_rc_230301_04000.npz'):
        super(SynBody, self).__init__(transform, data_split)

        self.img_dir = osp.join(cfg.data_dir, 'SynBody')
        self.annot_path = osp.join(cfg.data_dir, 'preprocessed_datasets', filename)
        self.img_shape = (720, 1280)  # (h, w)
        self.cam_param = {
            'focal': (540, 540),  # (fx, fy)
            'princpt': (640, 360)  # (cx, cy)
        }

        # check image shape
        img_path = osp.join(self.img_dir, np.load(self.annot_path)['image_path'][0])
        img_shape = cv2.imread(img_path).shape[:2]
        assert self.img_shape == img_shape, 'image shape is incorrect: {} vs {}'.format(self.img_shape, img_shape)

        # load data
        self.datalist = self.load_data(train_sample_interval=getattr(cfg, 'SynBody_train_sample_interval', 1))