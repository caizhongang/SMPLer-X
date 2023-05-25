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


class MPI_INF_3DHP(HumanDataset):
    def __init__(self, transform, data_split):
        super(MPI_INF_3DHP, self).__init__(transform, data_split)

        if data_split != 'train':
            raise NotImplementedError('MPI_INF_3DHP test set is not supported')

        self.use_cache = getattr(cfg, 'use_cache', False)
        self.annot_path_cache = osp.join(cfg.data_dir, 'cache', 'mpi_inf_3dhp_neural_annot.npz')
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
            for pre_prc_file in ['mpi_inf_3dhp_neural_annot_part1.npz', 'mpi_inf_3dhp_neural_annot_part2.npz',
                                'mpi_inf_3dhp_neural_annot_part3.npz']:

                if self.data_split == 'train':
                    filename = getattr(cfg, 'filename', pre_prc_file)
                else:
                    raise ValueError('MPI_INF_3DHP test set is not support')

                self.img_dir = osp.join(cfg.data_dir, 'MPI_INF_3DHP')
                self.annot_path = osp.join(cfg.data_dir, 'preprocessed_datasets', filename)

                # load data
                datalist_slice = self.load_data(
                    train_sample_interval=getattr(cfg, f'{self.__class__.__name__}_train_sample_interval', 1))
                self.datalist.extend(datalist_slice)

            if self.use_cache:
                self.save_cache(self.annot_path_cache, self.datalist)