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

# 'talkshow_smplx_chemistry_path.npz' zipfile.BadZipFile: File is not a zip file
# ['talkshow_smplx_conan.npz',
#                             'talkshow_smplx_oliver_path.npz', 'talkshow_smplx_seth.npz']:

class Talkshow(HumanDataset):
    def __init__(self, transform, data_split):
        super(Talkshow, self).__init__(transform, data_split)
        sample_rate = getattr(cfg, 'Talkshow_train_sample_interval', 1)
        self.use_cache = getattr(cfg, 'use_cache', False)
        self.annot_path_cache = osp.join(cfg.data_dir, 'cache', 'talkshow_smplx.npz')
        self.img_shape = None  # (h, w)
        self.cam_param = {}

        # load data or cache
        if self.use_cache and osp.isfile(self.annot_path_cache):
            print(f'[{self.__class__.__name__}] loading cache from {self.annot_path_cache}')
            self.datalist = self.load_cache(self.annot_path_cache)

        else:
            if self.use_cache:
                print(f'[{self.__class__.__name__}] Cache not found, generating cache...')

            self.datalist = []
            for pre_prc_file in ['talkshow_smplx_chemistry.npz', 'talkshow_smplx_conan.npz',
                                'talkshow_smplx_oliver.npz', 'talkshow_smplx_seth.npz']:
                if self.data_split == 'train':
                    filename = getattr(cfg, 'filename', pre_prc_file)
                else:
                    raise ValueError('Talkshow test set is not support')

                self.img_dir = osp.join(cfg.data_dir, 'Talkshow')
                self.annot_path = osp.join(cfg.data_dir, 'preprocessed_datasets', filename)

                # check image shape
                # img_path = osp.join(self.img_dir, np.load(self.annot_path)['image_path'][0])
                # img_shape = cv2.imread(img_path).shape[:2]
                # assert self.img_shape == img_shape, 'image shape is incorrect: {} vs {}'.format(self.img_shape, img_shape)

                 # load data
                datalist_slice = self.load_data(sample_rate)
                self.datalist.extend(datalist_slice)

            if self.use_cache:
                self.save_cache(self.annot_path_cache, self.datalist)