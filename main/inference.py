import os
import sys
import os.path as osp
import argparse
import numpy as np
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn
import torch
CUR_DIR = osp.dirname(os.path.abspath(__file__))
sys.path.insert(0, osp.join(CUR_DIR, '..', 'main'))
sys.path.insert(0, osp.join(CUR_DIR , '..', 'common'))
from config import cfg
import cv2
from tqdm import tqdm
import json
from typing import Literal, Union
from mmdet.apis import init_detector, inference_detector
from utils.inference_utils import process_mmdet_results, non_max_suppression

class Inferer:

    def __init__(self, pretrained_model, num_gpus, output_folder):
        self.output_folder = output_folder
        self.device = torch.device('cuda') if (num_gpus > 0) else torch.device('cpu')
        config_path = osp.join(CUR_DIR, './config', f'config_{pretrained_model}.py')
        ckpt_path = osp.join(CUR_DIR, '../pretrained_models', f'{pretrained_model}.pth.tar')
        cfg.get_config_fromfile(config_path)
        cfg.update_config(num_gpus, ckpt_path, output_folder, self.device)
        self.cfg = cfg
        cudnn.benchmark = True
        
        # load model
        from base import Demoer
        demoer = Demoer()
        demoer._make_model()
        demoer.model.eval()
        self.demoer = demoer
        checkpoint_file = osp.join(CUR_DIR, '../pretrained_models/mmdet/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth')
        config_file= osp.join(CUR_DIR, '../pretrained_models/mmdet/mmdet_faster_rcnn_r50_fpn_coco.py')
        model = init_detector(config_file, checkpoint_file, device=self.device)  # or device='cuda:0'
        self.model = model

    def infer(self, original_img, iou_thr, frame, multi_person=False, mesh_as_vertices=False):
        from utils.preprocessing import process_bbox, generate_patch_image
        from utils.vis import render_mesh, save_obj
        from utils.human_models import smpl_x
        mesh_paths = []
        smplx_paths = []
        # prepare input image
        transform = transforms.ToTensor()
        vis_img = original_img.copy()
        original_img_height, original_img_width = original_img.shape[:2]

        ## mmdet inference
        mmdet_results = inference_detector(self.model, original_img)
        
        pred_instance = mmdet_results.pred_instances.cpu().numpy()
        bboxes = np.concatenate(
            (pred_instance.bboxes, pred_instance.scores[:, None]), axis=1)
        bboxes = bboxes[pred_instance.labels == 0]
        bboxes = np.expand_dims(bboxes, axis=0)
        mmdet_box = process_mmdet_results(bboxes, cat_id=0, multi_person=True)

        # save original image if no bbox
        if len(mmdet_box[0])<1:
            return original_img, [], []
        
        if not multi_person:
            # only select the largest bbox
            num_bbox = 1
            mmdet_box = mmdet_box[0]
        else:
            # keep bbox by NMS with iou_thr
            mmdet_box = non_max_suppression(mmdet_box[0], iou_thr)
            num_bbox = len(mmdet_box)
        
        ## loop all detected bboxes
        for bbox_id in range(num_bbox):
            mmdet_box_xywh = np.zeros((4))
            mmdet_box_xywh[0] = mmdet_box[bbox_id][0]
            mmdet_box_xywh[1] = mmdet_box[bbox_id][1]
            mmdet_box_xywh[2] =  abs(mmdet_box[bbox_id][2]-mmdet_box[bbox_id][0])
            mmdet_box_xywh[3] =  abs(mmdet_box[bbox_id][3]-mmdet_box[bbox_id][1]) 

            # skip small bboxes by bbox_thr in pixel
            if mmdet_box_xywh[2] < 50 or mmdet_box_xywh[3] < 150:
                continue

            bbox = process_bbox(mmdet_box_xywh, original_img_width, original_img_height)
            img, img2bb_trans, bb2img_trans = generate_patch_image(original_img, bbox, 1.0, 0.0, False, self.cfg.input_img_shape)
            img = transform(img.astype(np.float32))/255
            img = img.to(cfg.device)[None,:,:,:]
            inputs = {'img': img}
            targets = {}
            meta_info = {}

            # mesh recovery
            with torch.no_grad():
                out = self.demoer.model(inputs, targets, meta_info, 'test')
            mesh = out['smplx_mesh_cam'].detach().cpu().numpy()[0]

            ## save mesh
            save_path_mesh = os.path.join(self.output_folder, 'mesh')
            os.makedirs(save_path_mesh, exist_ok= True)
            obj_path = os.path.join(save_path_mesh, f'{frame:05}_{bbox_id}.obj')
            save_obj(mesh, smpl_x.face, obj_path)
            mesh_paths.append(obj_path)
            ## save single person param
            smplx_pred = {}
            smplx_pred['global_orient'] = out['smplx_root_pose'].reshape(-1,3).cpu().numpy()
            smplx_pred['body_pose'] = out['smplx_body_pose'].reshape(-1,3).cpu().numpy()
            smplx_pred['left_hand_pose'] = out['smplx_lhand_pose'].reshape(-1,3).cpu().numpy()
            smplx_pred['right_hand_pose'] = out['smplx_rhand_pose'].reshape(-1,3).cpu().numpy()
            smplx_pred['jaw_pose'] = out['smplx_jaw_pose'].reshape(-1,3).cpu().numpy()
            smplx_pred['leye_pose'] = np.zeros((1, 3))
            smplx_pred['reye_pose'] = np.zeros((1, 3))
            smplx_pred['betas'] = out['smplx_shape'].reshape(-1,10).cpu().numpy()
            smplx_pred['expression'] = out['smplx_expr'].reshape(-1,10).cpu().numpy()
            smplx_pred['transl'] =  out['cam_trans'].reshape(-1,3).cpu().numpy()
            save_path_smplx = os.path.join(self.output_folder, 'smplx')
            os.makedirs(save_path_smplx, exist_ok= True)

            npz_path = os.path.join(save_path_smplx, f'{frame:05}_{bbox_id}.npz')
            np.savez(npz_path, **smplx_pred)
            smplx_paths.append(npz_path)

            ## render single person mesh
            focal = [self.cfg.focal[0] / self.cfg.input_body_shape[1] * bbox[2], self.cfg.focal[1] / self.cfg.input_body_shape[0] * bbox[3]]
            princpt = [self.cfg.princpt[0] / self.cfg.input_body_shape[1] * bbox[2] + bbox[0], self.cfg.princpt[1] / self.cfg.input_body_shape[0] * bbox[3] + bbox[1]]
            vis_img = render_mesh(vis_img, mesh, smpl_x.face, {'focal': focal, 'princpt': princpt}, 
                                  mesh_as_vertices=mesh_as_vertices)
            vis_img = vis_img.astype('uint8') 
        return vis_img, mesh_paths, smplx_paths

