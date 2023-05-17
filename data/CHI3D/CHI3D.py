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


class CHI3D(HumanDataset):
    def __init__(self, transform, data_split):
        super(CHI3D, self).__init__(transform, data_split)

        self.datalist = []

        for pre_prc_file in ['CHI3D_train_230511_1492_0.npz','CHI3D_train_230511_1492_1.npz']:
            if self.data_split == 'train':
                filename = getattr(cfg, 'filename', pre_prc_file)
            else:
                raise ValueError('CHI3D test set is not support')

            self.img_dir = cfg.data_dir # CHI3D included
            self.annot_path = osp.join(cfg.data_dir, 'preprocessed_datasets', filename)
            self.img_shape = (900, 900)  # (h, w)
            self.cam_param = {}

            # check image shape
            img_path = osp.join(self.img_dir, np.load(self.annot_path)['image_path'][0])
            img_shape = cv2.imread(img_path).shape[:2]
            assert self.img_shape == img_shape, 'image shape is incorrect: {} vs {}'.format(self.img_shape, img_shape)

            # load data
            datalist_slice = self.load_data()
            self.datalist.extend(datalist_slice)
