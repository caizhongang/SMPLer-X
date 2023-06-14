import os
import os.path as osp
import numpy as np
from config import cfg
import copy
import json
import cv2
import torch
from pycocotools.coco import COCO
from utils.human_models import smpl_x
from utils.preprocessing import load_img, process_bbox, augmentation, process_db_coord, \
    process_human_model_output
import random
from humandata import Cache
# from utils.vis import vis_keypoints, vis_mesh, save_obj

class MPII(torch.utils.data.Dataset):
    def __init__(self, transform, data_split):
        self.transform = transform
        self.data_split = data_split
        self.img_path = osp.join(cfg.data_dir, 'MPII', 'data')
        self.annot_path = osp.join(cfg.data_dir, 'MPII', 'data', 'annotations')

        # mpii skeleton
        self.joint_set = {
                        'joint_num': 16,
                        'joints_name': ('R_Ankle', 'R_Knee', 'R_Hip', 'L_Hip', 'L_Knee', 'L_Ankle', 'Pelvis', 'Thorax', 'Neck', 'Head_top', 'R_Wrist', 'R_Elbow', 'R_Shoulder', 'L_Shoulder', 'L_Elbow', 'L_Wrist'),
                        'flip_pairs': ( (0,5), (1,4), (2,3), (10,15), (11,14), (12,13) ),
                        }

        # self.datalist = self.load_data()

        # load data or cache
        self.use_cache = getattr(cfg, 'use_cache', False)
        self.annot_path_cache = osp.join(cfg.data_dir, 'cache', f'MPII_{data_split}.npz')
        if self.use_cache and osp.isfile(self.annot_path_cache):
            print(f'[{self.__class__.__name__}] loading cache from {self.annot_path_cache}')
            datalist = Cache(self.annot_path_cache)
            assert datalist.data_strategy == getattr(cfg, 'data_strategy', None), \
                f'Cache data strategy {datalist.data_strategy} does not match current data strategy ' \
                f'{getattr(cfg, "data_strategy", None)}'
            self.datalist = datalist
        else:
            if self.use_cache:
                print(f'[{self.__class__.__name__}] Cache not found, generating cache...')
            self.datalist = self.load_data()
            if self.use_cache:
                print(f'[{self.__class__.__name__}] Caching datalist to {self.annot_path_cache}...')
                Cache.save(
                    self.annot_path_cache,
                    self.datalist,
                    data_strategy=getattr(cfg, 'data_strategy', None)
                )



    def load_data(self):
        db = COCO(osp.join(self.annot_path, 'train.json'))
        with open(osp.join(self.annot_path, 'MPII_train_SMPLX_NeuralAnnot.json')) as f:
            smplx_params = json.load(f)

        datalist = []
        i = 0
        for aid in db.anns.keys():

            i += 1
            if self.data_split == 'train' and i % getattr(cfg, 'MPII_train_sample_interval', 1) != 0:
                continue

            ann = db.anns[aid]
            img = db.loadImgs(ann['image_id'])[0]
            imgname = img['file_name']
            img_path = osp.join(self.img_path, imgname)
            
            # bbox
            bbox = process_bbox(ann['bbox'], img['width'], img['height'], ratio=getattr(cfg, 'bbox_ratio', 1.25)) 
            if bbox is None: continue
            
            # joint coordinates
            joint_img = np.array(ann['keypoints'], dtype=np.float32).reshape(-1,3)
            joint_valid = joint_img[:,2:].copy()
            joint_img[:,2] = 0

            # smplx parameter	    
            if str(aid) in smplx_params:
                smplx_param = smplx_params[str(aid)]
            else:
                smplx_param = None

            datalist.append({
                'img_path': img_path,
                'img_shape': (img['height'], img['width']),
                'bbox': bbox,
                'joint_img': joint_img,
                'joint_valid': joint_valid,
                'smplx_param': smplx_param
            })

        if self.data_split == 'train':
            print('[MPII train] original size:', len(db.anns.keys()),
                  '. Sample interval:', getattr(cfg, 'MPII_train_sample_interval', 1),
                  '. Sampled size:', len(datalist))
        
        if getattr(cfg, 'data_strategy', None) == 'balance' and self.data_split == 'train':
            print(f'[MPII] Using [balance] strategy with datalist shuffled...')
            random.shuffle(datalist)

        return datalist
    
    def __len__(self):
        return len(self.datalist)

    def __getitem__(self, idx):
        data = copy.deepcopy(self.datalist[idx])
        img_path, img_shape, bbox = data['img_path'], data['img_shape'], data['bbox']

        # image load and affine transform
        img = load_img(img_path)
        img, img2bb_trans, bb2img_trans, rot, do_flip = augmentation(img, bbox, self.data_split)
        img = self.transform(img.astype(np.float32))/255.
        
        # mpii gt
        dummy_coord = np.zeros((self.joint_set['joint_num'],3), dtype=np.float32)
        joint_img = data['joint_img']
        joint_img = np.concatenate((joint_img[:,:2], np.zeros_like(joint_img[:,:1])),1) # x, y, dummy depth
        joint_img, joint_cam, joint_cam_ra, joint_valid, joint_trunc = process_db_coord(joint_img, dummy_coord, data['joint_valid'], do_flip, img_shape, self.joint_set['flip_pairs'], img2bb_trans, rot, self.joint_set['joints_name'], smpl_x.joints_name)

        # smplx coordinates and parameters
        smplx_param = data['smplx_param']
        if smplx_param is not None:
            smplx_joint_img, smplx_joint_cam, smplx_joint_trunc, smplx_pose, smplx_shape, smplx_expr, smplx_pose_valid, smplx_joint_valid, smplx_expr_valid, smplx_mesh_cam_orig = process_human_model_output(smplx_param['smplx_param'], smplx_param['cam_param'], do_flip, img_shape, img2bb_trans, rot, 'smplx')
            is_valid_fit = True
            
        else:
            # dummy values
            smplx_joint_img = np.zeros((smpl_x.joint_num,3), dtype=np.float32)
            smplx_joint_cam = np.zeros((smpl_x.joint_num,3), dtype=np.float32)
            smplx_joint_trunc = np.zeros((smpl_x.joint_num,1), dtype=np.float32)
            smplx_joint_valid = np.zeros((smpl_x.joint_num), dtype=np.float32)
            smplx_pose = np.zeros((smpl_x.orig_joint_num*3), dtype=np.float32) 
            smplx_shape = np.zeros((smpl_x.shape_param_dim), dtype=np.float32)
            smplx_expr = np.zeros((smpl_x.expr_code_dim), dtype=np.float32)
            smplx_pose_valid = np.zeros((smpl_x.orig_joint_num), dtype=np.float32)
            smplx_expr_valid = False
            is_valid_fit = False
       
        # SMPLX pose parameter validity
        for name in ('L_Ankle', 'R_Ankle', 'L_Wrist', 'R_Wrist'):
            smplx_pose_valid[smpl_x.orig_joints_name.index(name)] = 0
        smplx_pose_valid = np.tile(smplx_pose_valid[:,None], (1,3)).reshape(-1)
        # SMPLX joint coordinate validity
        for name in ('L_Big_toe', 'L_Small_toe', 'L_Heel', 'R_Big_toe', 'R_Small_toe', 'R_Heel'):
            smplx_joint_valid[smpl_x.joints_name.index(name)] = 0
        smplx_joint_valid = smplx_joint_valid[:,None]
        smplx_joint_trunc = smplx_joint_valid * smplx_joint_trunc

        # make zero mask for invalid fit
        if not is_valid_fit:
            smplx_pose_valid[:] = 0
            smplx_joint_valid[:] = 0
            smplx_joint_trunc[:] = 0
            smplx_shape_valid = False
        else:
            smplx_shape_valid = True

        # dummy hand/face bbox
        dummy_center = np.zeros((2), dtype=np.float32)
        dummy_size = np.zeros((2), dtype=np.float32)

        inputs = {'img': img}
        targets = {'joint_img': joint_img, 'smplx_joint_img': smplx_joint_img, 
                   'joint_cam': joint_cam, 'smplx_joint_cam': smplx_joint_cam, 
                   'smplx_pose': smplx_pose, 'smplx_shape': smplx_shape, 'smplx_expr': smplx_expr, 
                   'lhand_bbox_center': dummy_center, 'lhand_bbox_size': dummy_size, 
                   'rhand_bbox_center': dummy_center, 'rhand_bbox_size': dummy_size, 
                   'face_bbox_center': dummy_center, 'face_bbox_size': dummy_size}
        meta_info = {'joint_valid': joint_valid, 'joint_trunc': joint_trunc, 
                     'smplx_joint_valid': smplx_joint_valid, 
                     'smplx_joint_trunc': smplx_joint_trunc, 'smplx_pose_valid': smplx_pose_valid, 
                     'smplx_shape_valid': float(smplx_shape_valid), 
                     'smplx_expr_valid': float(smplx_expr_valid), 'is_3D': float(False), 
                     'lhand_bbox_valid': float(False), 'rhand_bbox_valid': float(False), 
                     'face_bbox_valid': float(False)}
        return inputs, targets, meta_info

