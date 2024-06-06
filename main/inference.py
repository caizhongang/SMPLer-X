import os
import sys
import os.path as osp
import argparse
import json
from tqdm import tqdm

import cv2
import numpy as np

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
from torch.utils.data import Dataset

CUR_DIR = osp.dirname(os.path.abspath(__file__))
sys.path.insert(0, osp.join(CUR_DIR, '..', 'main'))
sys.path.insert(0, osp.join(CUR_DIR , '..', 'common'))
from config import cfg, model_path_dict
from base import Demoer
from utils.preprocessing import process_bbox, generate_patch_image
from utils.human_models import smpl_x

class SmplerxData(Dataset):
    def __init__(self, annotations):
        self.annotations = annotations

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        img_path = self.annotations[idx]['image_path']
        image = cv2.imread(img_path)
        bbox = self.annotations[idx]['bbox']
        if bbox[2] < 50 or bbox[3] < 150:
            return None
        img_shape = image.shape  # (width, height)

        # prepare input image
        transform = transforms.ToTensor()
        original_img_height, original_img_width = image.shape[:2]
        bbox = process_bbox(bbox, original_img_width, original_img_height)
        img, img2bb_trans, bb2img_trans = generate_patch_image(image, bbox, 1.0, 0.0, False, (512, 384))
        img = transform(img.astype(np.float32))/255
        
        sample = {'image': img, 'bbox': bbox, 'shape': img_shape, 'path': img_path}
        
        return sample


class Inferer:

    def __init__(self, pretrained_model, num_gpus, data_parallel=True,
                 output_folder=osp.join(CUR_DIR, '..', 'demo_out')):

        self.output_folder = output_folder
        self.data_parallel = data_parallel
        self.device = torch.device('cuda') if (num_gpus > 0) else torch.device('cpu')

        # load config and model path
        ckpt_path = model_path_dict[pretrained_model]
        config_path = osp.join(CUR_DIR, 'config', f'config_{pretrained_model}.py')

        cfg.get_config_fromfile(config_path)
        cfg.update_config(num_gpus, ckpt_path, output_folder, self.device)
        self.cfg = cfg
        cudnn.benchmark = True
        # load model
        demoer = Demoer()
        # if num_gpus > 1:
        demoer._make_model()
        if self.data_parallel:
            demoer.model = nn.DataParallel(demoer.model)
        demoer.model.eval()
        self.demoer = demoer


    def _get_focal(self, bbox):
        bbox = bbox.cpu().numpy()
        focal = [self.cfg.focal[0] / self.cfg.input_body_shape[1] * bbox[2],
                 self.cfg.focal[0] / self.cfg.input_body_shape[0] * bbox[3]]
        return focal
    

    def _get_princpt(self, bbox):
        bbox = bbox.cpu().numpy()
        princpt = [self.cfg.princpt[0] / self.cfg.input_body_shape[1] * bbox[2] + bbox[0],
                   self.cfg.princpt[1] / self.cfg.input_body_shape[0] * bbox[3] + bbox[1]]
        return princpt


    def batch_infer_given_bbox(self, img, bbox, return_mesh=False):

        batch_size = img.shape[0]
        inputs = {'img': img}
        targets = {}
        meta_info = {}

        # mesh recovery
        with torch.no_grad():
            out = self.demoer.model(inputs, targets, meta_info, 'test')

        ## save mesh
        if return_mesh:
            mesh = out['smplx_mesh_cam'].detach().cpu().numpy()[0]
        else:
            mesh = None

        ## save single person param
        smplx_pred = {}
        smplx_pred['global_orient'] = out['smplx_root_pose'].reshape(batch_size,-1,3).cpu().numpy()
        smplx_pred['body_pose'] = out['smplx_body_pose'].reshape(batch_size,-1,3).cpu().numpy()
        smplx_pred['left_hand_pose'] = out['smplx_lhand_pose'].reshape(batch_size,-1,3).cpu().numpy()
        smplx_pred['right_hand_pose'] = out['smplx_rhand_pose'].reshape(batch_size,-1,3).cpu().numpy()
        smplx_pred['jaw_pose'] = out['smplx_jaw_pose'].reshape(batch_size,-1,3).cpu().numpy()
        smplx_pred['leye_pose'] = np.zeros((batch_size,1,3))
        smplx_pred['reye_pose'] = np.zeros((batch_size,1,3))
        smplx_pred['betas'] = out['smplx_shape'].reshape(batch_size,-1,10).cpu().numpy()
        smplx_pred['expression'] = out['smplx_expr'].reshape(batch_size,-1,10).cpu().numpy()
        smplx_pred['transl'] =  out['cam_trans'].reshape(batch_size,-1,3).cpu().numpy()

        ## save meta
        meta = {}
        meta['focal_length'] = [(self._get_focal(box)) for box in bbox]
        meta['principal_point'] = [(self._get_princpt(box)) for box in bbox]

        return smplx_pred, meta, mesh
