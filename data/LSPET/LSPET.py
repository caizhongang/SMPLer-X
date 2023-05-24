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


class LSPET(HumanDataset):
    def __init__(self, transform, data_split):
        super(LSPET, self).__init__(transform, data_split)

        if self.data_split == 'train':
            filename = getattr(cfg, 'filename', 'eft_lspet.npz')
        else:
            raise ValueError('LSPET test set is not support')

        self.img_dir = osp.join(cfg.data_dir, 'LSPET')
        self.annot_path = osp.join(cfg.data_dir, 'preprocessed_datasets', filename)
        self.img_shape = None # (h, w)
        self.cam_param = {}

        # load data
        self.datalist = self.load_data()