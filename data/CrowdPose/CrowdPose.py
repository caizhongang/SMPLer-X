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


class CrowdPose(HumanDataset):
    def __init__(self, transform, data_split):
        super(CrowdPose, self).__init__(transform, data_split)

        self.datalist = []

        pre_prc_file = 'crowdpose_neural_annot_train_new.npz'
        if self.data_split == 'train':
            filename = getattr(cfg, 'filename', pre_prc_file)
        else:
            raise ValueError('CrowdPose test set is not support')

        self.img_dir = osp.join(cfg.data_dir, 'CrowdPose')
        self.annot_path = osp.join(cfg.data_dir, 'preprocessed_datasets', filename)
        self.img_shape = None
        self.cam_param = {}
        print("Various image shape in CrowdPose dataset.")

        # load data
        datalist_slice = self.load_data(
            train_sample_interval=getattr(cfg, f'{self.__class__.__name__}_train_sample_interval', 1))
        self.datalist.extend(datalist_slice)