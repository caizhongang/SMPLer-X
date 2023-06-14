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
import tqdm
import time
import random

KPS2D_KEYS = ['keypoints2d', 'keypoints2d_smplx', 'keypoints2d_smpl', 'keypoints2d_original']
KPS3D_KEYS = ['keypoints3d_cam', 'keypoints3d', 'keypoints3d_smplx','keypoints3d_smpl' ,'keypoints3d_original'] 
# keypoints3d_cam with root-align has higher priority, followed by old version key keypoints3d
# when there is keypoints3d_smplx, use this rather than keypoints3d_original

hands_meanr = np.array([ 0.11167871, -0.04289218,  0.41644183,  0.10881133,  0.06598568,
        0.75622   , -0.09639297,  0.09091566,  0.18845929, -0.11809504,
       -0.05094385,  0.5295845 , -0.14369841, -0.0552417 ,  0.7048571 ,
       -0.01918292,  0.09233685,  0.3379135 , -0.45703298,  0.19628395,
        0.6254575 , -0.21465237,  0.06599829,  0.50689423, -0.36972436,
        0.06034463,  0.07949023, -0.1418697 ,  0.08585263,  0.63552827,
       -0.3033416 ,  0.05788098,  0.6313892 , -0.17612089,  0.13209307,
        0.37335458,  0.8509643 , -0.27692273,  0.09154807, -0.49983943,
       -0.02655647, -0.05288088,  0.5355592 , -0.04596104,  0.27735803]).reshape(15, -1)
hands_meanl = np.array([ 0.11167871,  0.04289218, -0.41644183,  0.10881133, -0.06598568,
       -0.75622   , -0.09639297, -0.09091566, -0.18845929, -0.11809504,
        0.05094385, -0.5295845 , -0.14369841,  0.0552417 , -0.7048571 ,
       -0.01918292, -0.09233685, -0.3379135 , -0.45703298, -0.19628395,
       -0.6254575 , -0.21465237, -0.06599829, -0.50689423, -0.36972436,
       -0.06034463, -0.07949023, -0.1418697 , -0.08585263, -0.63552827,
       -0.3033416 , -0.05788098, -0.6313892 , -0.17612089, -0.13209307,
       -0.37335458,  0.8509643 ,  0.27692273, -0.09154807, -0.49983943,
        0.02655647,  0.05288088,  0.5355592 ,  0.04596104, -0.27735803]).reshape(15, -1)

class Cache():
    """ A custom implementation for SMPLer_X pipeline
        Need to run tool/cache/fix_cache.py to fix paths
    """
    def __init__(self, load_path=None):
        if load_path is not None:
            self.load(load_path)

    def load(self, load_path):
        self.load_path = load_path
        self.cache = np.load(load_path, allow_pickle=True)
        self.data_len = self.cache['data_len']
        self.data_strategy = self.cache['data_strategy']
        assert self.data_len == len(self.cache) - 2  # data_len, data_strategy
        self.cache = None

    @classmethod
    def save(cls, save_path, data_list, data_strategy):
        assert save_path is not None, 'save_path is None'
        data_len = len(data_list)
        cache = {}
        for i, data in enumerate(data_list):
            cache[str(i)] = data
        assert len(cache) == data_len
        # update meta
        cache.update({
            'data_len': data_len,
            'data_strategy': data_strategy})

        np.savez_compressed(save_path, **cache)
        print(f'Cache saved to {save_path}.')

    # def shuffle(self):
    #     random.shuffle(self.mapping)

    def __len__(self):
        return self.data_len

    def __getitem__(self, idx):
        if self.cache is None:
            self.cache = np.load(self.load_path, allow_pickle=True)
        # mapped_idx = self.mapping[idx]
        # cache_data = self.cache[str(mapped_idx)]
        cache_data = self.cache[str(idx)]
        data = cache_data.item()
        return data


class HumanDataset(torch.utils.data.Dataset):

    # same mapping for 144->137 and 190->137
    SMPLX_137_MAPPING = [
        0, 1, 2, 4, 5, 7, 8, 12, 16, 17, 18, 19, 20, 21, 60, 61, 62, 63, 64, 65, 59, 58, 57, 56, 55, 37, 38, 39, 66,
        25, 26, 27, 67, 28, 29, 30, 68, 34, 35, 36, 69, 31, 32, 33, 70, 52, 53, 54, 71, 40, 41, 42, 72, 43, 44, 45,
        73, 49, 50, 51, 74, 46, 47, 48, 75, 22, 15, 56, 57, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89,
        90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113,
        114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135,
        136, 137, 138, 139, 140, 141, 142, 143]

    def __init__(self, transform, data_split):
        self.transform = transform
        self.data_split = data_split

        # dataset information, to be filled by child class
        self.img_dir = None
        self.annot_path = None
        self.annot_path_cache = None
        self.use_cache = False
        self.save_idx = 0
        self.img_shape = None  # (h, w)
        self.cam_param = None  # {'focal_length': (fx, fy), 'princpt': (cx, cy)}
        self.use_betas_neutral = False

        self.joint_set = {
            'joint_num': smpl_x.joint_num,
            'joints_name': smpl_x.joints_name,
            'flip_pairs': smpl_x.flip_pairs}
        self.joint_set['root_joint_idx'] = self.joint_set['joints_name'].index('Pelvis')

    def load_cache(self, annot_path_cache):
        datalist = Cache(annot_path_cache)
        assert datalist.data_strategy == getattr(cfg, 'data_strategy', None), \
            f'Cache data strategy {datalist.data_strategy} does not match current data strategy ' \
            f'{getattr(cfg, "data_strategy", None)}'
        return datalist

    def save_cache(self, annot_path_cache, datalist):
        print(f'[{self.__class__.__name__}] Caching datalist to {self.annot_path_cache}...')
        Cache.save(
            annot_path_cache,
            datalist,
            data_strategy=getattr(cfg, 'data_strategy', None)
        )

    def load_data(self, train_sample_interval=1, test_sample_interval=1):

        content = np.load(self.annot_path, allow_pickle=True)
        num_examples = len(content['image_path'])

        if 'meta' in content:
            meta = content['meta'].item()
            print('meta keys:', meta.keys())
        else:
            meta = None
            print('No meta info provided! Please give height and width manually')

        print(f'Start loading humandata {self.annot_path} into memory...\nDataset includes: {content.files}'); tic = time.time()
        image_path = content['image_path']

        if meta is not None and 'height' in meta:
            height = np.array(meta['height'])
            width = np.array(meta['width'])
            image_shape = np.stack([height, width], axis=-1)
        else:
            image_shape = None

        bbox_xywh = content['bbox_xywh']

        if 'smplx' in content:
            smplx = content['smplx'].item()
            as_smplx = 'smplx'
        elif 'smpl' in content:
            smplx = content['smpl'].item()
            as_smplx = 'smpl'
        elif 'smplh' in content:
            smplx = content['smplh'].item()
            as_smplx = 'smplh'

        # TODO: temp solution, should be more general. But SHAPY is very special
        elif self.__class__.__name__ == 'SHAPY':
            smplx = {}

        else:
            raise KeyError('No SMPL for SMPLX available, please check keys:\n'
                        f'{content.files}')

        print('Smplx param', smplx.keys())

        if 'lhand_bbox_xywh' in content and 'rhand_bbox_xywh' in content:
            lhand_bbox_xywh = content['lhand_bbox_xywh']
            rhand_bbox_xywh = content['rhand_bbox_xywh']
        else:
            lhand_bbox_xywh = np.zeros_like(bbox_xywh)
            rhand_bbox_xywh = np.zeros_like(bbox_xywh)

        if 'face_bbox_xywh' in content:
            face_bbox_xywh = content['face_bbox_xywh']
        else:
            face_bbox_xywh = np.zeros_like(bbox_xywh)

        decompressed = False
        if content['__keypoints_compressed__']:
            decompressed_kps = self.decompress_keypoints(content)
            decompressed = True

        keypoints3d = None
        valid_kps3d = False
        keypoints3d_mask = None
        valid_kps3d_mask = False
        for kps3d_key in KPS3D_KEYS:
            if kps3d_key in content:
                keypoints3d = decompressed_kps[kps3d_key][:, self.SMPLX_137_MAPPING, :3] if decompressed \
                else content[kps3d_key][:, self.SMPLX_137_MAPPING, :3]
                valid_kps3d = True

                if f'{kps3d_key}_mask' in content:
                    keypoints3d_mask = content[f'{kps3d_key}_mask'][self.SMPLX_137_MAPPING]
                    valid_kps3d_mask = True
                elif 'keypoints3d_mask' in content:
                    keypoints3d_mask = content['keypoints3d_mask'][self.SMPLX_137_MAPPING]
                    valid_kps3d_mask = True
                break

        for kps2d_key in KPS2D_KEYS:
            if kps2d_key in content:
                keypoints2d = decompressed_kps[kps2d_key][:, self.SMPLX_137_MAPPING, :2] if decompressed \
                    else content[kps2d_key][:, self.SMPLX_137_MAPPING, :2]

                if f'{kps2d_key}_mask' in content:
                    keypoints2d_mask = content[f'{kps2d_key}_mask'][self.SMPLX_137_MAPPING]
                elif 'keypoints2d_mask' in content:
                    keypoints2d_mask = content['keypoints2d_mask'][self.SMPLX_137_MAPPING]
                break

        mask = keypoints3d_mask if valid_kps3d_mask \
                else keypoints2d_mask

        print('Done. Time: {:.2f}s'.format(time.time() - tic))

        datalist = []
        for i in tqdm.tqdm(range(int(num_examples))):
            if self.data_split == 'train' and i % train_sample_interval != 0:
                continue
            if self.data_split == 'test' and i % test_sample_interval != 0:
                continue
            img_path = osp.join(self.img_dir, image_path[i])
            img_shape = image_shape[i] if image_shape is not None else self.img_shape

            bbox = bbox_xywh[i][:4]

            if hasattr(cfg, 'bbox_ratio'):
                bbox_ratio = cfg.bbox_ratio * 0.833 # preprocess body bbox is giving 1.2 box padding
            else:
                bbox_ratio = 1.25
            bbox = process_bbox(bbox, img_width=img_shape[1], img_height=img_shape[0], ratio=bbox_ratio)
            if bbox is None: continue

            # hand/face bbox
            lhand_bbox = lhand_bbox_xywh[i]
            rhand_bbox = rhand_bbox_xywh[i]
            face_bbox = face_bbox_xywh[i]

            if lhand_bbox[-1] > 0:  # conf > 0
                lhand_bbox = lhand_bbox[:4]
                if hasattr(cfg, 'bbox_ratio'):
                    lhand_bbox = process_bbox(lhand_bbox, img_width=img_shape[1], img_height=img_shape[0], ratio=cfg.bbox_ratio)
                if lhand_bbox is not None:
                    lhand_bbox[2:] += lhand_bbox[:2]  # xywh -> xyxy
            else:
                lhand_bbox = None
            if rhand_bbox[-1] > 0:
                rhand_bbox = rhand_bbox[:4]
                if hasattr(cfg, 'bbox_ratio'):
                    rhand_bbox = process_bbox(rhand_bbox, img_width=img_shape[1], img_height=img_shape[0], ratio=cfg.bbox_ratio)
                if rhand_bbox is not None:
                    rhand_bbox[2:] += rhand_bbox[:2]  # xywh -> xyxy
            else:
                rhand_bbox = None
            if face_bbox[-1] > 0:
                face_bbox = face_bbox[:4]
                if hasattr(cfg, 'bbox_ratio'):
                    face_bbox = process_bbox(face_bbox, img_width=img_shape[1], img_height=img_shape[0], ratio=cfg.bbox_ratio)
                if face_bbox is not None:
                    face_bbox[2:] += face_bbox[:2]  # xywh -> xyxy
            else:
                face_bbox = None

            joint_img = keypoints2d[i]
            joint_valid = mask.reshape(-1, 1)
            # num_joints = joint_cam.shape[0]
            # joint_valid = np.ones((num_joints, 1))
            if valid_kps3d:
                joint_cam = keypoints3d[i]
            else:
                joint_cam = None

            smplx_param = {k: v[i] for k, v in smplx.items()}

            smplx_param['root_pose'] = smplx_param.pop('global_orient', None)
            smplx_param['shape'] = smplx_param.pop('betas', None)
            smplx_param['trans'] = smplx_param.pop('transl', np.zeros(3))
            smplx_param['lhand_pose'] = smplx_param.pop('left_hand_pose', None)
            smplx_param['rhand_pose'] = smplx_param.pop('right_hand_pose', None)
            smplx_param['expr'] = smplx_param.pop('expression', None)

            # TODO do not fix betas, give up shape supervision
            if 'betas_neutral' in smplx_param:
                smplx_param['shape'] = smplx_param.pop('betas_neutral')

            # TODO fix shape of poses
            if self.__class__.__name__ == 'Talkshow':
                smplx_param['body_pose'] = smplx_param['body_pose'].reshape(21, 3)
                smplx_param['lhand_pose'] = smplx_param['lhand_pose'].reshape(15, 3)
                smplx_param['rhand_pose'] = smplx_param['lhand_pose'].reshape(15, 3)
                smplx_param['expr'] = smplx_param['expr'][:10]

            if self.__class__.__name__ == 'BEDLAM':
                smplx_param['shape'] = smplx_param['shape'][:10]
                # manually set flat_hand_mean = True
                smplx_param['lhand_pose'] -= hands_meanl
                smplx_param['rhand_pose'] -= hands_meanr


            if as_smplx == 'smpl':
                smplx_param['shape'] = np.zeros(10, dtype=np.float32) # drop smpl betas for smplx
                smplx_param['body_pose'] = smplx_param['body_pose'][:21, :] # use smpl body_pose on smplx

            if as_smplx == 'smplh':
                smplx_param['shape'] = np.zeros(10, dtype=np.float32) # drop smpl betas for smplx

            if smplx_param['lhand_pose'] is None:
                smplx_param['lhand_valid'] = False
            else:
                smplx_param['lhand_valid'] = True
            if smplx_param['rhand_pose'] is None:
                smplx_param['rhand_valid'] = False
            else:
                smplx_param['rhand_valid'] = True
            if smplx_param['expr'] is None:
                smplx_param['face_valid'] = False
            else:
                smplx_param['face_valid'] = True

            if joint_cam is not None and np.any(np.isnan(joint_cam)):
                continue

            datalist.append({
                'img_path': img_path,
                'img_shape': img_shape,
                'bbox': bbox,
                'lhand_bbox': lhand_bbox,
                'rhand_bbox': rhand_bbox,
                'face_bbox': face_bbox,
                'joint_img': joint_img,
                'joint_cam': joint_cam,
                'joint_valid': joint_valid,
                'smplx_param': smplx_param,
                'smplx': smplx})

        # save memory
        del content, image_path, bbox_xywh, lhand_bbox_xywh, rhand_bbox_xywh, face_bbox_xywh, keypoints3d, keypoints2d

        if self.data_split == 'train':
            print(f'[{self.__class__.__name__} train] original size:', int(num_examples),
                  '. Sample interval:', train_sample_interval,
                  '. Sampled size:', len(datalist))

        if (getattr(cfg, 'data_strategy', None) == 'balance' and self.data_split == 'train') or \
                getattr(cfg, 'eval_on_train', False):
            print(f'[{self.__class__.__name__}] Using [balance] strategy with datalist shuffled...')
            random.seed(2023)
            random.shuffle(datalist)

            if getattr(cfg, 'eval_on_train', False):
                return datalist[:10000]

        return datalist

    def __len__(self):
        return len(self.datalist)

    def __getitem__(self, idx):
        try:
            data = copy.deepcopy(self.datalist[idx])
        except Exception as e:
            print(f'[{self.__class__.__name__}] Error loading data {idx}')
            print(e)
            exit(0)

        img_path, img_shape, bbox = data['img_path'], data['img_shape'], data['bbox']

        # img
        img = load_img(img_path)
        img, img2bb_trans, bb2img_trans, rot, do_flip = augmentation(img, bbox, self.data_split)
        img = self.transform(img.astype(np.float32)) / 255.

        if self.data_split == 'train':
            
            # h36m gt
            joint_cam = data['joint_cam']
            if joint_cam is not None:
                dummy_cord = False
                joint_cam = joint_cam - joint_cam[self.joint_set['root_joint_idx'], None, :]  # root-relative
            else:
                # dummy cord as joint_cam
                dummy_cord = True
                joint_cam = np.zeros((self.joint_set['joint_num'], 3), dtype=np.float32)

            joint_img = data['joint_img']
            joint_img = np.concatenate((joint_img[:, :2], joint_cam[:, 2:]), 1)  # x, y, depth
            if not dummy_cord: 
                joint_img[:, 2] = (joint_img[:, 2] / (cfg.body_3d_size / 2) + 1) / 2. * cfg.output_hm_shape[0]  # discretize depth
            
            joint_img_aug, joint_cam_wo_ra, joint_cam_ra, joint_valid, joint_trunc = process_db_coord(
                joint_img, joint_cam, data['joint_valid'], do_flip, img_shape,
                self.joint_set['flip_pairs'], img2bb_trans, rot, self.joint_set['joints_name'], smpl_x.joints_name)
            
            # smplx coordinates and parameters
            smplx_param = data['smplx_param']
            smplx_joint_img, smplx_joint_cam, smplx_joint_trunc, smplx_pose, smplx_shape, smplx_expr, \
            smplx_pose_valid, smplx_joint_valid, smplx_expr_valid, smplx_mesh_cam_orig = process_human_model_output(
                smplx_param, self.cam_param, do_flip, img_shape, img2bb_trans, rot, 'smplx',
                joint_img=None if self.cam_param else joint_img,  # if cam not provided, we take joint_img as smplx joint 2d, which is commonly the case for our processed humandata
            )

            # TODO temp fix keypoints3d for renbody
            if 'RenBody' in self.__class__.__name__:
                joint_cam_ra = smplx_joint_cam.copy()
                joint_cam_wo_ra = smplx_joint_cam.copy()
                joint_cam_wo_ra[smpl_x.joint_part['lhand'], :] = joint_cam_wo_ra[smpl_x.joint_part['lhand'], :] \
                                                                + joint_cam_wo_ra[smpl_x.lwrist_idx, None, :]  # left hand root-relative
                joint_cam_wo_ra[smpl_x.joint_part['rhand'], :] = joint_cam_wo_ra[smpl_x.joint_part['rhand'], :] \
                                                                + joint_cam_wo_ra[smpl_x.rwrist_idx, None, :]  # right hand root-relative
                joint_cam_wo_ra[smpl_x.joint_part['face'], :] = joint_cam_wo_ra[smpl_x.joint_part['face'], :] \
                                                                + joint_cam_wo_ra[smpl_x.neck_idx, None,: ]  # face root-relative

            # change smplx_shape if use_betas_neutral
            # processing follows that in process_human_model_output
            if self.use_betas_neutral:
                smplx_shape = smplx_param['betas_neutral'].reshape(1, -1)
                smplx_shape[(np.abs(smplx_shape) > 3).any(axis=1)] = 0.
                smplx_shape = smplx_shape.reshape(-1)
                
            # SMPLX pose parameter validity
            # for name in ('L_Ankle', 'R_Ankle', 'L_Wrist', 'R_Wrist'):
            #     smplx_pose_valid[smpl_x.orig_joints_name.index(name)] = 0
            smplx_pose_valid = np.tile(smplx_pose_valid[:, None], (1, 3)).reshape(-1)
            # SMPLX joint coordinate validity
            # for name in ('L_Big_toe', 'L_Small_toe', 'L_Heel', 'R_Big_toe', 'R_Small_toe', 'R_Heel'):
            #     smplx_joint_valid[smpl_x.joints_name.index(name)] = 0
            smplx_joint_valid = smplx_joint_valid[:, None]
            smplx_joint_trunc = smplx_joint_valid * smplx_joint_trunc
            if not (smplx_shape == 0).all():
                smplx_shape_valid = True
            else: 
                smplx_shape_valid = False

            # hand and face bbox transform
            lhand_bbox, lhand_bbox_valid = self.process_hand_face_bbox(data['lhand_bbox'], do_flip, img_shape, img2bb_trans)
            rhand_bbox, rhand_bbox_valid = self.process_hand_face_bbox(data['rhand_bbox'], do_flip, img_shape, img2bb_trans)
            face_bbox, face_bbox_valid = self.process_hand_face_bbox(data['face_bbox'], do_flip, img_shape, img2bb_trans)
            if do_flip:
                lhand_bbox, rhand_bbox = rhand_bbox, lhand_bbox
                lhand_bbox_valid, rhand_bbox_valid = rhand_bbox_valid, lhand_bbox_valid
            lhand_bbox_center = (lhand_bbox[0] + lhand_bbox[1]) / 2.
            rhand_bbox_center = (rhand_bbox[0] + rhand_bbox[1]) / 2.
            face_bbox_center = (face_bbox[0] + face_bbox[1]) / 2.
            lhand_bbox_size = lhand_bbox[1] - lhand_bbox[0]
            rhand_bbox_size = rhand_bbox[1] - rhand_bbox[0]
            face_bbox_size = face_bbox[1] - face_bbox[0]

            inputs = {'img': img}
            targets = {'joint_img': joint_img_aug, # keypoints2d
                       'smplx_joint_img': joint_img_aug, #smplx_joint_img, # projected smplx if valid cam_param, else same as keypoints2d
                       'joint_cam': joint_cam_wo_ra, # joint_cam actually not used in any loss, # raw kps3d probably without ra
                       'smplx_joint_cam': smplx_joint_cam if dummy_cord else joint_cam_ra, # kps3d with body, face, hand ra
                       'smplx_pose': smplx_pose,
                       'smplx_shape': smplx_shape,
                       'smplx_expr': smplx_expr,
                       'lhand_bbox_center': lhand_bbox_center, 'lhand_bbox_size': lhand_bbox_size,
                       'rhand_bbox_center': rhand_bbox_center, 'rhand_bbox_size': rhand_bbox_size,
                       'face_bbox_center': face_bbox_center, 'face_bbox_size': face_bbox_size}
            meta_info = {'joint_valid': joint_valid,
                         'joint_trunc': joint_trunc,
                         'smplx_joint_valid': smplx_joint_valid if dummy_cord else joint_valid,
                         'smplx_joint_trunc': smplx_joint_trunc if dummy_cord else joint_trunc,
                         'smplx_pose_valid': smplx_pose_valid,
                         'smplx_shape_valid': float(smplx_shape_valid),
                         'smplx_expr_valid': float(smplx_expr_valid),
                         'is_3D': float(False) if dummy_cord else float(True), 
                         'lhand_bbox_valid': lhand_bbox_valid,
                         'rhand_bbox_valid': rhand_bbox_valid, 'face_bbox_valid': face_bbox_valid}
            
            if self.__class__.__name__  == 'SHAPY':
                meta_info['img_path'] = img_path
            
            return inputs, targets, meta_info

        # TODO: temp solution, should be more general. But SHAPY is very special
        elif self.__class__.__name__  == 'SHAPY':
            inputs = {'img': img}
            if cfg.shapy_eval_split == 'val':
                targets = {'smplx_shape': smplx_shape}
            else:
                targets = {}
            meta_info = {'img_path': img_path}
            return inputs, targets, meta_info

        else:
            joint_cam = data['joint_cam']
            if joint_cam is not None:
                dummy_cord = False
                joint_cam = joint_cam - joint_cam[self.joint_set['root_joint_idx'], None, :]  # root-relative
            else:
                # dummy cord as joint_cam
                dummy_cord = True
                joint_cam = np.zeros((self.joint_set['joint_num'], 3), dtype=np.float32)

            joint_img = data['joint_img']
            joint_img = np.concatenate((joint_img[:, :2], joint_cam[:, 2:]), 1)  # x, y, depth
            if not dummy_cord:
                joint_img[:, 2] = (joint_img[:, 2] / (cfg.body_3d_size / 2) + 1) / 2. * cfg.output_hm_shape[0]  # discretize depth

            joint_img, joint_cam, joint_cam_ra, joint_valid, joint_trunc = process_db_coord(
                joint_img, joint_cam, data['joint_valid'], do_flip, img_shape,
                self.joint_set['flip_pairs'], img2bb_trans, rot, self.joint_set['joints_name'], smpl_x.joints_name)

            # smplx coordinates and parameters
            smplx_param = data['smplx_param']
            smplx_cam_trans = np.array(smplx_param['trans']) if 'trans' in smplx_param else None
            smplx_joint_img, smplx_joint_cam, smplx_joint_trunc, smplx_pose, smplx_shape, smplx_expr, \
            smplx_pose_valid, smplx_joint_valid, smplx_expr_valid, smplx_mesh_cam_orig = process_human_model_output(
                smplx_param, self.cam_param, do_flip, img_shape, img2bb_trans, rot, 'smplx',
                joint_img=None if self.cam_param else joint_img
                )  # if cam not provided, we take joint_img as smplx joint 2d, which is commonly the case for our processed humandata

            inputs = {'img': img}
            targets = {'smplx_pose': smplx_pose,
                       'smplx_shape': smplx_shape,
                       'smplx_expr': smplx_expr,
                       'smplx_cam_trans' : smplx_cam_trans,
                       }
            meta_info = {'img_path': img_path,
                         'bb2img_trans': bb2img_trans,
                         'gt_smplx_transl':smplx_cam_trans}

            return inputs, targets, meta_info

    def process_hand_face_bbox(self, bbox, do_flip, img_shape, img2bb_trans):
        if bbox is None:
            bbox = np.array([0, 0, 1, 1], dtype=np.float32).reshape(2, 2)  # dummy value
            bbox_valid = float(False)  # dummy value
        else:
            # reshape to top-left (x,y) and bottom-right (x,y)
            bbox = bbox.reshape(2, 2)

            # flip augmentation
            if do_flip:
                bbox[:, 0] = img_shape[1] - bbox[:, 0] - 1
                bbox[0, 0], bbox[1, 0] = bbox[1, 0].copy(), bbox[0, 0].copy()  # xmin <-> xmax swap

            # make four points of the bbox
            bbox = bbox.reshape(4).tolist()
            xmin, ymin, xmax, ymax = bbox
            bbox = np.array([[xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax]], dtype=np.float32).reshape(4, 2)

            # affine transformation (crop, rotation, scale)
            bbox_xy1 = np.concatenate((bbox, np.ones_like(bbox[:, :1])), 1)
            bbox = np.dot(img2bb_trans, bbox_xy1.transpose(1, 0)).transpose(1, 0)[:, :2]
            bbox[:, 0] = bbox[:, 0] / cfg.input_img_shape[1] * cfg.output_hm_shape[2]
            bbox[:, 1] = bbox[:, 1] / cfg.input_img_shape[0] * cfg.output_hm_shape[1]

            # make box a rectangle without rotation
            xmin = np.min(bbox[:, 0])
            xmax = np.max(bbox[:, 0])
            ymin = np.min(bbox[:, 1])
            ymax = np.max(bbox[:, 1])
            bbox = np.array([xmin, ymin, xmax, ymax], dtype=np.float32)

            bbox_valid = float(True)
            bbox = bbox.reshape(2, 2)

        return bbox, bbox_valid

    def evaluate(self, outs, cur_sample_idx=None):
        sample_num = len(outs)
        eval_result = {'pa_mpvpe_all': [], 'pa_mpvpe_l_hand': [], 'pa_mpvpe_r_hand': [], 'pa_mpvpe_hand': [], 'pa_mpvpe_face': [], 
                       'mpvpe_all': [], 'mpvpe_l_hand': [], 'mpvpe_r_hand': [], 'mpvpe_hand': [], 'mpvpe_face': [], 
                       'pa_mpjpe_body': [], 'pa_mpjpe_l_hand': [], 'pa_mpjpe_r_hand': [], 'pa_mpjpe_hand': []}

        if getattr(cfg, 'vis', False):
            import csv
            csv_file = f'{cfg.vis_dir}/{cfg.testset}_smplx_error.csv'
            file = open(csv_file, 'a', newline='')
            writer = csv.writer(file)

        for n in range(sample_num):
            out = outs[n]
            mesh_gt = out['smplx_mesh_cam_pseudo_gt']
            mesh_out = out['smplx_mesh_cam']

            # MPVPE from all vertices
            mesh_out_align = mesh_out - np.dot(smpl_x.J_regressor, mesh_out)[smpl_x.J_regressor_idx['pelvis'], None,
                                        :] + np.dot(smpl_x.J_regressor, mesh_gt)[smpl_x.J_regressor_idx['pelvis'], None,
                                             :]
            mpvpe_all = np.sqrt(np.sum((mesh_out_align - mesh_gt) ** 2, 1)).mean() * 1000
            eval_result['mpvpe_all'].append(mpvpe_all)
            mesh_out_align = rigid_align(mesh_out, mesh_gt)
            pa_mpvpe_all = np.sqrt(np.sum((mesh_out_align - mesh_gt) ** 2, 1)).mean() * 1000
            eval_result['pa_mpvpe_all'].append(pa_mpvpe_all)

            # MPVPE from hand vertices
            mesh_gt_lhand = mesh_gt[smpl_x.hand_vertex_idx['left_hand'], :]
            mesh_out_lhand = mesh_out[smpl_x.hand_vertex_idx['left_hand'], :]
            mesh_gt_rhand = mesh_gt[smpl_x.hand_vertex_idx['right_hand'], :]
            mesh_out_rhand = mesh_out[smpl_x.hand_vertex_idx['right_hand'], :]
            mesh_out_lhand_align = mesh_out_lhand - np.dot(smpl_x.J_regressor, mesh_out)[
                                                    smpl_x.J_regressor_idx['lwrist'], None, :] + np.dot(
                smpl_x.J_regressor, mesh_gt)[smpl_x.J_regressor_idx['lwrist'], None, :]
            mesh_out_rhand_align = mesh_out_rhand - np.dot(smpl_x.J_regressor, mesh_out)[
                                                    smpl_x.J_regressor_idx['rwrist'], None, :] + np.dot(
                smpl_x.J_regressor, mesh_gt)[smpl_x.J_regressor_idx['rwrist'], None, :]
            eval_result['mpvpe_l_hand'].append(np.sqrt(
                np.sum((mesh_out_lhand_align - mesh_gt_lhand) ** 2, 1)).mean() * 1000)
            eval_result['mpvpe_r_hand'].append(np.sqrt(
                np.sum((mesh_out_rhand_align - mesh_gt_rhand) ** 2, 1)).mean() * 1000)
            eval_result['mpvpe_hand'].append((np.sqrt(
                np.sum((mesh_out_lhand_align - mesh_gt_lhand) ** 2, 1)).mean() * 1000 + np.sqrt(
                np.sum((mesh_out_rhand_align - mesh_gt_rhand) ** 2, 1)).mean() * 1000) / 2.)
            mesh_out_lhand_align = rigid_align(mesh_out_lhand, mesh_gt_lhand)
            mesh_out_rhand_align = rigid_align(mesh_out_rhand, mesh_gt_rhand)
            eval_result['pa_mpvpe_l_hand'].append(np.sqrt(
                np.sum((mesh_out_lhand_align - mesh_gt_lhand) ** 2, 1)).mean() * 1000)
            eval_result['pa_mpvpe_r_hand'].append(np.sqrt(
                np.sum((mesh_out_rhand_align - mesh_gt_rhand) ** 2, 1)).mean() * 1000)
            eval_result['pa_mpvpe_hand'].append((np.sqrt(
                np.sum((mesh_out_lhand_align - mesh_gt_lhand) ** 2, 1)).mean() * 1000 + np.sqrt(
                np.sum((mesh_out_rhand_align - mesh_gt_rhand) ** 2, 1)).mean() * 1000) / 2.)

            # MPVPE from face vertices
            mesh_gt_face = mesh_gt[smpl_x.face_vertex_idx, :]
            mesh_out_face = mesh_out[smpl_x.face_vertex_idx, :]
            mesh_out_face_align = mesh_out_face - np.dot(smpl_x.J_regressor, mesh_out)[smpl_x.J_regressor_idx['neck'],
                                                  None, :] + np.dot(smpl_x.J_regressor, mesh_gt)[
                                                             smpl_x.J_regressor_idx['neck'], None, :]
            eval_result['mpvpe_face'].append(
                np.sqrt(np.sum((mesh_out_face_align - mesh_gt_face) ** 2, 1)).mean() * 1000)
            mesh_out_face_align = rigid_align(mesh_out_face, mesh_gt_face)
            eval_result['pa_mpvpe_face'].append(
                np.sqrt(np.sum((mesh_out_face_align - mesh_gt_face) ** 2, 1)).mean() * 1000)

            # MPJPE from body joints
            joint_gt_body = np.dot(smpl_x.j14_regressor, mesh_gt)
            joint_out_body = np.dot(smpl_x.j14_regressor, mesh_out)
            joint_out_body_align = rigid_align(joint_out_body, joint_gt_body)
            eval_result['pa_mpjpe_body'].append(
                np.sqrt(np.sum((joint_out_body_align - joint_gt_body) ** 2, 1)).mean() * 1000)

            # MPJPE from hand joints
            joint_gt_lhand = np.dot(smpl_x.orig_hand_regressor['left'], mesh_gt)
            joint_out_lhand = np.dot(smpl_x.orig_hand_regressor['left'], mesh_out)
            joint_out_lhand_align = rigid_align(joint_out_lhand, joint_gt_lhand)
            joint_gt_rhand = np.dot(smpl_x.orig_hand_regressor['right'], mesh_gt)
            joint_out_rhand = np.dot(smpl_x.orig_hand_regressor['right'], mesh_out)
            joint_out_rhand_align = rigid_align(joint_out_rhand, joint_gt_rhand)
            eval_result['pa_mpjpe_l_hand'].append(np.sqrt(
                np.sum((joint_out_lhand_align - joint_gt_lhand) ** 2, 1)).mean() * 1000)
            eval_result['pa_mpjpe_r_hand'].append(np.sqrt(
                np.sum((joint_out_rhand_align - joint_gt_rhand) ** 2, 1)).mean() * 1000)
            eval_result['pa_mpjpe_hand'].append((np.sqrt(
                np.sum((joint_out_lhand_align - joint_gt_lhand) ** 2, 1)).mean() * 1000 + np.sqrt(
                np.sum((joint_out_rhand_align - joint_gt_rhand) ** 2, 1)).mean() * 1000) / 2.)

            if getattr(cfg, 'vis', False):
                img_path = out['img_path']
                rel_img_path = img_path.split('..')[-1]
                smplx_pred = {}
                smplx_pred['global_orient'] = out['smplx_root_pose'].reshape(-1,3)
                smplx_pred['body_pose'] = out['smplx_body_pose'].reshape(-1,3)
                smplx_pred['left_hand_pose'] = out['smplx_lhand_pose'].reshape(-1,3)
                smplx_pred['right_hand_pose'] = out['smplx_rhand_pose'].reshape(-1,3)
                smplx_pred['jaw_pose'] = out['smplx_jaw_pose'].reshape(-1,3)
                smplx_pred['leye_pose'] = np.zeros((1, 3))
                smplx_pred['reye_pose'] = np.zeros((1, 3))
                smplx_pred['betas'] = out['smplx_shape'].reshape(-1,10)
                smplx_pred['expression'] = out['smplx_expr'].reshape(-1,10)
                smplx_pred['transl'] =  out['gt_smplx_transl'].reshape(-1,3)
                smplx_pred['img_path'] = rel_img_path
                
                npz_path = os.path.join(cfg.vis_dir, f'{self.save_idx}.npz')
                np.savez(npz_path, **smplx_pred)

                # save img path and error
                new_line = [self.save_idx, rel_img_path, mpvpe_all, pa_mpvpe_all]
                # Append the new line to the CSV file
                writer.writerow(new_line)
                self.save_idx += 1

        if getattr(cfg, 'vis', False):
            file.close()

        return eval_result
            
    def print_eval_result(self, eval_result):
        print(f'======{cfg.testset}======')
        print(f'{cfg.vis_dir}')
        print('PA MPVPE (All): %.2f mm' % np.mean(eval_result['pa_mpvpe_all']))
        print('PA MPVPE (L-Hands): %.2f mm' % np.mean(eval_result['pa_mpvpe_l_hand']))
        print('PA MPVPE (R-Hands): %.2f mm' % np.mean(eval_result['pa_mpvpe_r_hand']))
        print('PA MPVPE (Hands): %.2f mm' % np.mean(eval_result['pa_mpvpe_hand']))
        print('PA MPVPE (Face): %.2f mm' % np.mean(eval_result['pa_mpvpe_face']))
        print()

        print('MPVPE (All): %.2f mm' % np.mean(eval_result['mpvpe_all']))
        print('MPVPE (L-Hands): %.2f mm' % np.mean(eval_result['mpvpe_l_hand']))
        print('MPVPE (R-Hands): %.2f mm' % np.mean(eval_result['mpvpe_r_hand']))
        print('MPVPE (Hands): %.2f mm' % np.mean(eval_result['mpvpe_hand']))
        print('MPVPE (Face): %.2f mm' % np.mean(eval_result['mpvpe_face']))
        print()

        print('PA MPJPE (Body): %.2f mm' % np.mean(eval_result['pa_mpjpe_body']))
        print('PA MPJPE (L-Hands): %.2f mm' % np.mean(eval_result['pa_mpjpe_l_hand']))
        print('PA MPJPE (R-Hands): %.2f mm' % np.mean(eval_result['pa_mpjpe_r_hand']))
        print('PA MPJPE (Hands): %.2f mm' % np.mean(eval_result['pa_mpjpe_hand']))

        print()
        print(f"{np.mean(eval_result['pa_mpvpe_all'])},{np.mean(eval_result['pa_mpvpe_l_hand'])},{np.mean(eval_result['pa_mpvpe_r_hand'])},{np.mean(eval_result['pa_mpvpe_hand'])},{np.mean(eval_result['pa_mpvpe_face'])},"
        f"{np.mean(eval_result['mpvpe_all'])},{np.mean(eval_result['mpvpe_l_hand'])},{np.mean(eval_result['mpvpe_r_hand'])},{np.mean(eval_result['mpvpe_hand'])},{np.mean(eval_result['mpvpe_face'])},"
        f"{np.mean(eval_result['pa_mpjpe_body'])},{np.mean(eval_result['pa_mpjpe_l_hand'])},{np.mean(eval_result['pa_mpjpe_r_hand'])},{np.mean(eval_result['pa_mpjpe_hand'])}")
        print()


        f = open(os.path.join(cfg.result_dir, 'result.txt'), 'w')
        f.write(f'{cfg.testset} dataset \n')
        f.write('PA MPVPE (All): %.2f mm\n' % np.mean(eval_result['pa_mpvpe_all']))
        f.write('PA MPVPE (L-Hands): %.2f mm' % np.mean(eval_result['pa_mpvpe_l_hand']))
        f.write('PA MPVPE (R-Hands): %.2f mm' % np.mean(eval_result['pa_mpvpe_r_hand']))
        f.write('PA MPVPE (Hands): %.2f mm\n' % np.mean(eval_result['pa_mpvpe_hand']))
        f.write('PA MPVPE (Face): %.2f mm\n' % np.mean(eval_result['pa_mpvpe_face']))
        f.write('MPVPE (All): %.2f mm\n' % np.mean(eval_result['mpvpe_all']))
        f.write('MPVPE (L-Hands): %.2f mm' % np.mean(eval_result['mpvpe_l_hand']))
        f.write('MPVPE (R-Hands): %.2f mm' % np.mean(eval_result['mpvpe_r_hand']))
        f.write('MPVPE (Hands): %.2f mm' % np.mean(eval_result['mpvpe_hand']))
        f.write('MPVPE (Face): %.2f mm\n' % np.mean(eval_result['mpvpe_face']))
        f.write('PA MPJPE (Body): %.2f mm\n' % np.mean(eval_result['pa_mpjpe_body']))
        f.write('PA MPJPE (L-Hands): %.2f mm' % np.mean(eval_result['pa_mpjpe_l_hand']))
        f.write('PA MPJPE (R-Hands): %.2f mm' % np.mean(eval_result['pa_mpjpe_r_hand']))
        f.write('PA MPJPE (Hands): %.2f mm\n' % np.mean(eval_result['pa_mpjpe_hand']))
        f.write(f"{np.mean(eval_result['pa_mpvpe_all'])},{np.mean(eval_result['pa_mpvpe_l_hand'])},{np.mean(eval_result['pa_mpvpe_r_hand'])},{np.mean(eval_result['pa_mpvpe_hand'])},{np.mean(eval_result['pa_mpvpe_face'])},"
        f"{np.mean(eval_result['mpvpe_all'])},{np.mean(eval_result['mpvpe_l_hand'])},{np.mean(eval_result['mpvpe_r_hand'])},{np.mean(eval_result['mpvpe_hand'])},{np.mean(eval_result['mpvpe_face'])},"
        f"{np.mean(eval_result['pa_mpjpe_body'])},{np.mean(eval_result['pa_mpjpe_l_hand'])},{np.mean(eval_result['pa_mpjpe_r_hand'])},{np.mean(eval_result['pa_mpjpe_hand'])}")

        if getattr(cfg, 'eval_on_train', False):
            import csv
            csv_file = f'{cfg.root_dir}/output/{cfg.testset}_eval_on_train.csv'
            exp_id = cfg.exp_name.split('_')[1]
            new_line = [exp_id,np.mean(eval_result['pa_mpvpe_all']),np.mean(eval_result['pa_mpvpe_l_hand']),np.mean(eval_result['pa_mpvpe_r_hand']),np.mean(eval_result['pa_mpvpe_hand']),np.mean(eval_result['pa_mpvpe_face']),
                        np.mean(eval_result['mpvpe_all']),np.mean(eval_result['mpvpe_l_hand']),np.mean(eval_result['mpvpe_r_hand']),np.mean(eval_result['mpvpe_hand']),np.mean(eval_result['mpvpe_face']),
                        np.mean(eval_result['pa_mpjpe_body']),np.mean(eval_result['pa_mpjpe_l_hand']),np.mean(eval_result['pa_mpjpe_r_hand']),np.mean(eval_result['pa_mpjpe_hand'])]

            # Append the new line to the CSV file
            with open(csv_file, 'a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(new_line)

    def decompress_keypoints(self, humandata) -> None:
        """If a key contains 'keypoints', and f'{key}_mask' is in self.keys(),
        invalid zeros will be inserted to the right places and f'{key}_mask'
        will be unlocked.

        Raises:
            KeyError:
                A key contains 'keypoints' has been found
                but its corresponding mask is missing.
        """
        assert bool(humandata['__keypoints_compressed__']) is True
        key_pairs = []
        for key in humandata.files:
            if key not in KPS2D_KEYS + KPS3D_KEYS:
                continue
            mask_key = f'{key}_mask'
            if mask_key in humandata.files:
                print(f'Decompress {key}...')
                key_pairs.append([key, mask_key])
        decompressed_dict = {}
        for kpt_key, mask_key in key_pairs:
            mask_array = np.asarray(humandata[mask_key])
            compressed_kpt = humandata[kpt_key]
            kpt_array = \
                self.add_zero_pad(compressed_kpt, mask_array)
            decompressed_dict[kpt_key] = kpt_array
        del humandata
        return decompressed_dict

    def add_zero_pad(self, compressed_array: np.ndarray,
                         mask_array: np.ndarray) -> np.ndarray:
        """Pad zeros to a compressed keypoints array.

        Args:
            compressed_array (np.ndarray):
                A compressed keypoints array.
            mask_array (np.ndarray):
                The mask records compression relationship.

        Returns:
            np.ndarray:
                A keypoints array in full-size.
        """
        assert mask_array.sum() == compressed_array.shape[1]
        data_len, _, dim = compressed_array.shape
        mask_len = mask_array.shape[0]
        ret_value = np.zeros(
            shape=[data_len, mask_len, dim], dtype=compressed_array.dtype)
        valid_mask_index = np.where(mask_array == 1)[0]
        ret_value[:, valid_mask_index, :] = compressed_array
        return ret_value
