import os
import os.path as osp
from glob import glob
import numpy as np
from config import cfg
import copy
import json
import pickle
import cv2
import torch
from pycocotools.coco import COCO
from utils.human_models import smpl_x
from utils.preprocessing import load_img, sanitize_bbox, process_bbox, augmentation, process_db_coord, \
    process_human_model_output, load_ply, load_obj
from utils.transforms import rigid_align
import tqdm
import random
from humandata import Cache

class AGORA(torch.utils.data.Dataset):
    def __init__(self, transform, data_split):
        self.transform = transform
        self.data_split = data_split
        if getattr(cfg, 'eval_on_train', False):
            self.data_split = 'eval_train'
            print("Evaluate on train set.")
        self.data_path = osp.join(cfg.data_dir, 'AGORA', 'data')
        self.save_idx = 0
        self.resolution = (2160, 3840)  # height, width. one of (720, 1280) and (2160, 3840)
        if cfg.agora_benchmark == 'agora_model_test' or cfg.agora_benchmark == 'test_only':
            self.test_set = 'test'
        else:
            self.test_set = 'val'  # val, test

        # AGORA joint set
        self.joint_set = {
            'joint_num': 127,
            'joints_name': \
                ('Pelvis', 'L_Hip', 'R_Hip', 'Spine_1', 'L_Knee', 'R_Knee', 'Spine_2', 'L_Ankle', 'R_Ankle', 'Spine_3',
                 'L_Foot', 'R_Foot', 'Neck', 'L_Collar', 'R_Collar', 'Head', 'L_Shoulder', 'R_Shoulder', 'L_Elbow',
                 'R_Elbow', 'L_Wrist', 'R_Wrist',  # body
                 'Jaw', 'L_Eye_SMPLH', 'R_Eye_SMPLH',  # SMPLH
                 'L_Index_1', 'L_Index_2', 'L_Index_3', 'L_Middle_1', 'L_Middle_2', 'L_Middle_3', 'L_Pinky_1',
                 'L_Pinky_2', 'L_Pinky_3', 'L_Ring_1', 'L_Ring_2', 'L_Ring_3', 'L_Thumb_1', 'L_Thumb_2', 'L_Thumb_3',
                 # fingers
                 'R_Index_1', 'R_Index_2', 'R_Index_3', 'R_Middle_1', 'R_Middle_2', 'R_Middle_3', 'R_Pinky_1',
                 'R_Pinky_2', 'R_Pinky_3', 'R_Ring_1', 'R_Ring_2', 'R_Ring_3', 'R_Thumb_1', 'R_Thumb_2', 'R_Thumb_3',
                 # fingers
                 'Nose', 'R_Eye', 'L_Eye', 'R_Ear', 'L_Ear',  # face in body
                 'L_Big_toe', 'L_Small_toe', 'L_Heel', 'R_Big_toe', 'R_Small_toe', 'R_Heel',  # feet
                 'L_Thumb_4', 'L_Index_4', 'L_Middle_4', 'L_Ring_4', 'L_Pinky_4',  # finger tips
                 'R_Thumb_4', 'R_Index_4', 'R_Middle_4', 'R_Ring_4', 'R_Pinky_4',  # finger tips
                 *['Face_' + str(i) for i in range(5, 56)]  # face
                 ),
            'flip_pairs': \
                ((1, 2), (4, 5), (7, 8), (10, 11), (13, 14), (16, 17), (18, 19), (20, 21),  # body
                 (23, 24),  # SMPLH
                 (25, 40), (26, 41), (27, 42), (28, 43), (29, 44), (30, 45), (31, 46), (32, 47), (33, 48), (34, 49),
                 (35, 50), (36, 51), (37, 52), (38, 53), (39, 54),  # fingers
                 (56, 57), (58, 59),  # face in body
                 (60, 63), (61, 64), (62, 65),  # feet
                 (66, 71), (67, 72), (68, 73), (69, 74), (70, 75),  # fingertips
                 (76, 85), (77, 84), (78, 83), (79, 82), (80, 81),  # face eyebrow
                 (90, 94), (91, 93),  # face below nose
                 (95, 104), (96, 103), (97, 102), (98, 101), (99, 106), (100, 105),  # face eyes
                 (107, 113), (108, 112), (109, 111), (114, 118), (115, 117),  # face mouth
                 (119, 123), (120, 122), (124, 126)  # face lip
                 )

        }

        self.joint_set['joint_part'] = {
            'body': list(range(self.joint_set['joints_name'].index('Pelvis'),
                               self.joint_set['joints_name'].index('R_Eye_SMPLH') + 1)) + list(
                range(self.joint_set['joints_name'].index('Nose'), self.joint_set['joints_name'].index('R_Heel') + 1)),
            'lhand': list(range(self.joint_set['joints_name'].index('L_Index_1'),
                                self.joint_set['joints_name'].index('L_Thumb_3') + 1)) + list(
                range(self.joint_set['joints_name'].index('L_Thumb_4'),
                      self.joint_set['joints_name'].index('L_Pinky_4') + 1)),
            'rhand': list(range(self.joint_set['joints_name'].index('R_Index_1'),
                                self.joint_set['joints_name'].index('R_Thumb_3') + 1)) + list(
                range(self.joint_set['joints_name'].index('R_Thumb_4'),
                      self.joint_set['joints_name'].index('R_Pinky_4') + 1)),
            'face': list(range(self.joint_set['joints_name'].index('Face_5'),
                               self.joint_set['joints_name'].index('Face_55') + 1))}
        self.joint_set['root_joint_idx'] = self.joint_set['joints_name'].index('Pelvis')
        self.joint_set['lwrist_idx'] = self.joint_set['joints_name'].index('L_Wrist')
        self.joint_set['rwrist_idx'] = self.joint_set['joints_name'].index('R_Wrist')
        self.joint_set['neck_idx'] = self.joint_set['joints_name'].index('Neck')

        # self.datalist = self.load_data()

        # load data or cache
        self.use_cache = getattr(cfg, 'use_cache', False)
        if 'train'in self.data_split or (self.data_split == 'test' and self.test_set == 'val'):
            if 'train' in self.data_split:
                if getattr(cfg, 'agora_fix_betas', False):
                    assert getattr(cfg, 'agora_fix_global_orient_transl')
                    self.annot_path_cache = osp.join(cfg.data_dir, 'cache', f'AGORA_{self.data_split}_fix_betas.npz')
                elif getattr(cfg, 'agora_fix_global_orient_transl', False):
                    self.annot_path_cache = osp.join(cfg.data_dir, 'cache', f'AGORA_{self.data_split}_fix_global_orient_transl.npz')
                else:
                    self.annot_path_cache = osp.join(cfg.data_dir, 'cache', f'AGORA_{self.data_split}.npz')
            else:
                if getattr(cfg, 'agora_fix_betas', False):
                    assert getattr(cfg, 'agora_fix_global_orient_transl')
                    self.annot_path_cache = osp.join(cfg.data_dir, 'cache', 'AGORA_validation_fix_betas.npz')
                elif getattr(cfg, 'agora_fix_global_orient_transl', False):
                    self.annot_path_cache = osp.join(cfg.data_dir, 'cache', 'AGORA_validation_fix_global_orient_transl.npz')
                else:
                    self.annot_path_cache = osp.join(cfg.data_dir, 'cache', 'AGORA_validation.npz')

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

        else: # test
            self.datalist = self.load_data()

    def load_data(self):
        datalist = []
        if 'train' in self.data_split or (self.data_split == 'test' and self.test_set == 'val'):
            print('dataset settings:')
            print('agora_fix_betas', getattr(cfg, 'agora_fix_betas', False))
            print('agora_fix_global_orient_transl', getattr(cfg, 'agora_fix_global_orient_transl', False))
            print('agora_valid_root_pose', getattr(cfg, 'agora_valid_root_pose', False))

            if 'train' in self.data_split:
                if getattr(cfg, 'agora_fix_betas', False):
                    assert getattr(cfg, 'agora_fix_global_orient_transl')
                    db = COCO(osp.join(self.data_path, 'AGORA_train_fix_betas.json'))
                elif getattr(cfg, 'agora_fix_global_orient_transl', False):
                    db = COCO(osp.join(self.data_path, 'AGORA_train_fix_global_orient_transl.json'))
                else:
                    db = COCO(osp.join(self.data_path, 'AGORA_train.json'))
            else:
                if getattr(cfg, 'agora_fix_betas', False):
                    assert getattr(cfg, 'agora_fix_global_orient_transl')
                    db = COCO(osp.join(self.data_path, 'AGORA_validation_fix_betas.json'))
                elif getattr(cfg, 'agora_fix_global_orient_transl', False):
                    db = COCO(osp.join(self.data_path, 'AGORA_validation_fix_global_orient_transl.json'))
                else:
                    db = COCO(osp.join(self.data_path, 'AGORA_validation.json'))

            i = 0
            for aid in tqdm.tqdm(list(db.anns.keys())):
    
                i += 1
                if self.data_split == 'train' and i % getattr(cfg, 'AGORA_train_sample_interval', 1) != 0:
                    continue

                ann = db.anns[aid]
                image_id = ann['image_id']
                img = db.loadImgs(image_id)[0]
                if not ann['is_valid']:
                    continue

                joints_2d_path = osp.join(self.data_path, ann['smplx_joints_2d_path'])
                joints_3d_path = osp.join(self.data_path, ann['smplx_joints_3d_path'])
                verts_path = osp.join(self.data_path, ann['smplx_verts_path'])
                smplx_param_path = osp.join(self.data_path, ann['smplx_param_path'])
                kid = ann['kid']
                gender = ann['gender']
                if not osp.exists(smplx_param_path): print(smplx_param_path)

                if self.resolution == (720, 1280):
                    img_shape = self.resolution
                    img_path = osp.join(self.data_path, img['file_name_1280x720'])

                    # convert to current resolution
                    bbox = np.array(ann['bbox']).reshape(2, 2)
                    bbox[:, 0] = bbox[:, 0] / 3840 * 1280
                    bbox[:, 1] = bbox[:, 1] / 2160 * 720
                    bbox = bbox.reshape(4)
                    if hasattr(cfg, 'bbox_ratio'):
                        bbox_ratio = cfg.bbox_ratio * 0.833 # agora preprocess is giving 1.2 box padding
                    else:
                        bbox_ratio = 1.25
                    bbox = process_bbox(bbox, img_shape[1], img_shape[0], ratio=bbox_ratio)
                    if bbox is None:
                        continue

                    lhand_bbox = np.array(ann['lhand_bbox']).reshape(2, 2)
                    lhand_bbox[:, 0] = lhand_bbox[:, 0] / 3840 * 1280
                    lhand_bbox[:, 1] = lhand_bbox[:, 1] / 2160 * 720
                    lhand_bbox = lhand_bbox.reshape(4)
                    lhand_bbox = sanitize_bbox(lhand_bbox, img_shape[1], img_shape[0])
                    if lhand_bbox is not None:
                        lhand_bbox[2:] += lhand_bbox[:2]  # xywh -> xyxy

                    rhand_bbox = np.array(ann['rhand_bbox']).reshape(2, 2)
                    rhand_bbox[:, 0] = rhand_bbox[:, 0] / 3840 * 1280
                    rhand_bbox[:, 1] = rhand_bbox[:, 1] / 2160 * 720
                    rhand_bbox = rhand_bbox.reshape(4)
                    rhand_bbox = sanitize_bbox(rhand_bbox, img_shape[1], img_shape[0])
                    if rhand_bbox is not None:
                        rhand_bbox[2:] += rhand_bbox[:2]  # xywh -> xyxy

                    face_bbox = np.array(ann['face_bbox']).reshape(2, 2)
                    face_bbox[:, 0] = face_bbox[:, 0] / 3840 * 1280
                    face_bbox[:, 1] = face_bbox[:, 1] / 2160 * 720
                    face_bbox = face_bbox.reshape(4)
                    face_bbox = sanitize_bbox(face_bbox, img_shape[1], img_shape[0])
                    if face_bbox is not None:
                        face_bbox[2:] += face_bbox[:2]  # xywh -> xyxy

                    data_dict = {'img_path': img_path, 'img_shape': img_shape, 'bbox': bbox, 'lhand_bbox': lhand_bbox,
                                 'rhand_bbox': rhand_bbox, 'face_bbox': face_bbox, 'joints_2d_path': joints_2d_path,
                                 'joints_3d_path': joints_3d_path, 'verts_path': verts_path,
                                 'smplx_param_path': smplx_param_path, 'ann_id': str(aid), 'kid': kid, 'gender': gender}
                    datalist.append(data_dict)

                elif self.resolution == (2160,
                                         3840):  # use cropped and resized images. loading 4K images in pytorch dataloader takes too much time...
                    img_path = osp.join(self.data_path, '3840x2160',
                                        img['file_name_3840x2160'].split('/')[-2] + '_crop',
                                        img['file_name_3840x2160'].split('/')[-1][:-4] + '_ann_id_' + str(aid) + '.png')
                    json_path = osp.join(self.data_path, '3840x2160',
                                         img['file_name_3840x2160'].split('/')[-2] + '_crop',
                                         img['file_name_3840x2160'].split('/')[-1][:-4] + '_ann_id_' + str(
                                             aid) + '.json')
                    if not osp.isfile(json_path):
                        continue
                    with open(json_path) as f:
                        crop_resize_info = json.load(f)
                        img2bb_trans_from_orig = np.array(crop_resize_info['img2bb_trans'], dtype=np.float32)
                        resized_height, resized_width = crop_resize_info['resized_height'], crop_resize_info[
                            'resized_width']
                    img_shape = (resized_height, resized_width)
                    bbox = np.array([0, 0, resized_width, resized_height], dtype=np.float32)

                    # transform from original image to crop_and_resize image
                    lhand_bbox = np.array(ann['lhand_bbox']).reshape(2, 2)
                    lhand_bbox[1] += lhand_bbox[0]  # xywh -> xyxy
                    lhand_bbox = np.dot(img2bb_trans_from_orig,
                                        np.concatenate((lhand_bbox, np.ones_like(lhand_bbox[:, :1])), 1).transpose(1,
                                                                                                                   0)).transpose(
                        1, 0)
                    lhand_bbox[1] -= lhand_bbox[0]  # xyxy -> xywh
                    lhand_bbox = lhand_bbox.reshape(4)
                    lhand_bbox = sanitize_bbox(lhand_bbox, self.resolution[1], self.resolution[0])
                    if lhand_bbox is not None:
                        lhand_bbox[2:] += lhand_bbox[:2]  # xywh -> xyxy

                    # transform from original image to crop_and_resize image
                    rhand_bbox = np.array(ann['rhand_bbox']).reshape(2, 2)
                    rhand_bbox[1] += rhand_bbox[0]  # xywh -> xyxy
                    rhand_bbox = np.dot(img2bb_trans_from_orig,
                                        np.concatenate((rhand_bbox, np.ones_like(rhand_bbox[:, :1])), 1).transpose(1,
                                                                                                                   0)).transpose(
                        1, 0)
                    rhand_bbox[1] -= rhand_bbox[0]  # xyxy -> xywh
                    rhand_bbox = rhand_bbox.reshape(4)
                    rhand_bbox = sanitize_bbox(rhand_bbox, self.resolution[1], self.resolution[0])
                    if rhand_bbox is not None:
                        rhand_bbox[2:] += rhand_bbox[:2]  # xywh -> xyxy

                    # transform from original image to crop_and_resize image
                    face_bbox = np.array(ann['face_bbox']).reshape(2, 2)
                    face_bbox[1] += face_bbox[0]  # xywh -> xyxy
                    face_bbox = np.dot(img2bb_trans_from_orig,
                                       np.concatenate((face_bbox, np.ones_like(face_bbox[:, :1])), 1).transpose(1,
                                                                                                                0)).transpose(
                        1, 0)
                    face_bbox[1] -= face_bbox[0]  # xyxy -> xywh
                    face_bbox = face_bbox.reshape(4)
                    face_bbox = sanitize_bbox(face_bbox, self.resolution[1], self.resolution[0])
                    if face_bbox is not None:
                        face_bbox[2:] += face_bbox[:2]  # xywh -> xyxy

                    data_dict = {'img_path': img_path, 'img_shape': img_shape, 'bbox': bbox, 'lhand_bbox': lhand_bbox,
                                 'rhand_bbox': rhand_bbox, 'face_bbox': face_bbox,
                                 'img2bb_trans_from_orig': img2bb_trans_from_orig, 'joints_2d_path': joints_2d_path,
                                 'joints_3d_path': joints_3d_path, 'verts_path': verts_path,
                                 'smplx_param_path': smplx_param_path, 'ann_id': str(aid), 'kid': kid, 'gender': gender}
                    datalist.append(data_dict)

            print('[AGORA train] original size:', len(db.anns.keys()),
                  '. Sample interval:', getattr(cfg, 'AGORA_train_sample_interval', 1),
                  '. Sampled size:', len(datalist))

        elif self.data_split == 'test' and self.test_set == 'test':
            with open(osp.join(self.data_path, 'AGORA_test_bbox.json')) as f:
                bboxs = json.load(f)

            for filename in tqdm.tqdm(bboxs.keys()):
                if self.resolution == (720, 1280):
                    img_path = osp.join(self.data_path, 'test', filename)
                    img_shape = self.resolution
                    person_num = len(bboxs[filename])
                    for pid in range(person_num):
                        # change bbox from (2160,3840) to target resoution
                        bbox = np.array(bboxs[filename][pid]['bbox']).reshape(2, 2)
                        bbox[:, 0] = bbox[:, 0] / 3840 * 1280
                        bbox[:, 1] = bbox[:, 1] / 2160 * 720
                        bbox = bbox.reshape(4)
                        if hasattr(cfg, 'bbox_ratio'):
                            bbox_ratio = cfg.bbox_ratio * 0.833 # agora preprocess is giving 1.2 box padding
                        else:
                            bbox_ratio = 1.25
                        bbox = process_bbox(bbox, img_shape[1], img_shape[0], ratio=bbox_ratio)
                        if bbox is None:
                            continue
                        datalist.append({'img_path': img_path, 'img_shape': img_shape, 'bbox': bbox, 'person_idx': pid})

                elif self.resolution == (2160,
                                         3840):  # use cropped and resized images. loading 4K images in pytorch dataloader takes too much time...
                    person_num = len(bboxs[filename])
                    for pid in range(person_num):
                        img_path = osp.join(self.data_path, '3840x2160', 'test_crop',
                                            filename[:-4] + '_pid_' + str(pid) + '.png')
                        json_path = osp.join(self.data_path, '3840x2160', 'test_crop',
                                             filename[:-4] + '_pid_' + str(pid) + '.json')
                        if not osp.isfile(json_path):
                            continue
                        with open(json_path) as f:
                            crop_resize_info = json.load(f)
                            img2bb_trans_from_orig = np.array(crop_resize_info['img2bb_trans'], dtype=np.float32)
                            resized_height, resized_width = crop_resize_info['resized_height'], crop_resize_info[
                                'resized_width']
                        img_shape = (resized_height, resized_width)
                        bbox = np.array([0, 0, resized_width, resized_height], dtype=np.float32)
                        datalist.append({'img_path': img_path, 'img_shape': img_shape,
                                         'img2bb_trans_from_orig': img2bb_trans_from_orig, 'bbox': bbox,
                                         'person_idx': pid})
        
        if (getattr(cfg, 'data_strategy', None) == 'balance' and self.data_split == 'train') or \
                self.data_split == 'eval_train':
            print(f"[Agora] Using [balance] strategy with datalist shuffled...")
            random.seed(2023)
            random.shuffle(datalist)

            if self.data_split == "eval_train":
                return datalist[:10000]

        return datalist

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
            xmin = np.min(bbox[:, 0]);
            xmax = np.max(bbox[:, 0]);
            ymin = np.min(bbox[:, 1]);
            ymax = np.max(bbox[:, 1]);
            bbox = np.array([xmin, ymin, xmax, ymax], dtype=np.float32)

            bbox_valid = float(True)
            bbox = bbox.reshape(2, 2)

        return bbox, bbox_valid

    def __len__(self):
        return len(self.datalist)

    def __getitem__(self, idx):
        data = copy.deepcopy(self.datalist[idx])
        img_path, img_shape, bbox = data['img_path'], data['img_shape'], data['bbox']

        # image load
        img = load_img(img_path)

        # affine transform
        img, img2bb_trans, bb2img_trans, rot, do_flip = augmentation(img, bbox, self.data_split)
        img = self.transform(img.astype(np.float32)) / 255.

        if self.data_split == 'train':
            # gt load
            with open(data['joints_2d_path']) as f:
                joint_img = np.array(json.load(f)).reshape(-1, 2)
                if self.resolution == (2160, 3840):
                    joint_img[:, :2] = np.dot(data['img2bb_trans_from_orig'],
                                              np.concatenate((joint_img, np.ones_like(joint_img[:, :1])), 1).transpose(
                                                  1, 0)).transpose(1,
                                                                   0)  # transform from original image to crop_and_resize image
                joint_img[:, 0] = joint_img[:, 0] / 3840 * self.resolution[1]
                joint_img[:, 1] = joint_img[:, 1] / 2160 * self.resolution[0]
            with open(data['joints_3d_path']) as f:
                joint_cam = np.array(json.load(f)).reshape(-1, 3)
            with open(data['smplx_param_path'], 'rb') as f:
                smplx_param = pickle.load(f, encoding='latin1')

            # hand and face bbox transform
            lhand_bbox, rhand_bbox, face_bbox = data['lhand_bbox'], data['rhand_bbox'], data['face_bbox']
            lhand_bbox, lhand_bbox_valid = self.process_hand_face_bbox(lhand_bbox, do_flip, img_shape, img2bb_trans)
            rhand_bbox, rhand_bbox_valid = self.process_hand_face_bbox(rhand_bbox, do_flip, img_shape, img2bb_trans)
            face_bbox, face_bbox_valid = self.process_hand_face_bbox(face_bbox, do_flip, img_shape, img2bb_trans)
            if do_flip:
                lhand_bbox, rhand_bbox = rhand_bbox, lhand_bbox
                lhand_bbox_valid, rhand_bbox_valid = rhand_bbox_valid, lhand_bbox_valid
            lhand_bbox_center = (lhand_bbox[0] + lhand_bbox[1]) / 2.;
            rhand_bbox_center = (rhand_bbox[0] + rhand_bbox[1]) / 2.;
            face_bbox_center = (face_bbox[0] + face_bbox[1]) / 2.
            lhand_bbox_size = lhand_bbox[1] - lhand_bbox[0];
            rhand_bbox_size = rhand_bbox[1] - rhand_bbox[0];
            face_bbox_size = face_bbox[1] - face_bbox[0];

            # coordinates
            joint_cam = joint_cam - joint_cam[self.joint_set['root_joint_idx'], None, :]  # root-relative
            joint_cam[self.joint_set['joint_part']['lhand'], :] = joint_cam[self.joint_set['joint_part']['lhand'],
                                                                  :] - joint_cam[self.joint_set['lwrist_idx'], None,
                                                                       :]  # left hand root-relative
            joint_cam[self.joint_set['joint_part']['rhand'], :] = joint_cam[self.joint_set['joint_part']['rhand'],
                                                                  :] - joint_cam[self.joint_set['rwrist_idx'], None,
                                                                       :]  # right hand root-relative
            joint_cam[self.joint_set['joint_part']['face'], :] = joint_cam[self.joint_set['joint_part']['face'],
                                                                 :] - joint_cam[self.joint_set['neck_idx'], None,
                                                                      :]  # face root-relative
            joint_img = np.concatenate((joint_img[:, :2], joint_cam[:, 2:]), 1)  # x, y, depth
            joint_img[self.joint_set['joint_part']['body'], 2] = (joint_cam[self.joint_set['joint_part'][
                                                                                'body'], 2].copy() / (
                                                                              cfg.body_3d_size / 2) + 1) / 2. * \
                                                                 cfg.output_hm_shape[0]  # body depth discretize
            joint_img[self.joint_set['joint_part']['lhand'], 2] = (joint_cam[self.joint_set['joint_part'][
                                                                                 'lhand'], 2].copy() / (
                                                                               cfg.hand_3d_size / 2) + 1) / 2. * \
                                                                  cfg.output_hm_shape[0]  # left hand depth discretize
            joint_img[self.joint_set['joint_part']['rhand'], 2] = (joint_cam[self.joint_set['joint_part'][
                                                                                 'rhand'], 2].copy() / (
                                                                               cfg.hand_3d_size / 2) + 1) / 2. * \
                                                                  cfg.output_hm_shape[0]  # right hand depth discretize
            joint_img[self.joint_set['joint_part']['face'], 2] = (joint_cam[self.joint_set['joint_part'][
                                                                                'face'], 2].copy() / (
                                                                              cfg.face_3d_size / 2) + 1) / 2. * \
                                                                 cfg.output_hm_shape[0]  # face depth discretize
            joint_valid = np.ones_like(joint_img[:, :1])
            # alr ra when passed into this function
            joint_img, joint_cam_ra, _, joint_valid, joint_trunc = process_db_coord(joint_img, joint_cam, joint_valid,
                                                                              do_flip, img_shape,
                                                                              self.joint_set['flip_pairs'],
                                                                              img2bb_trans, rot,
                                                                              self.joint_set['joints_name'],
                                                                              smpl_x.joints_name)
            # reverse ra
            joint_cam_wo_ra = joint_cam_ra.copy()
            joint_cam_wo_ra[smpl_x.joint_part['lhand'], :] = joint_cam_wo_ra[smpl_x.joint_part['lhand'], :] \
                                                            + joint_cam_wo_ra[smpl_x.lwrist_idx, None, :]  # left hand root-relative
            joint_cam_wo_ra[smpl_x.joint_part['rhand'], :] = joint_cam_wo_ra[smpl_x.joint_part['rhand'], :] \
                                                            + joint_cam_wo_ra[smpl_x.rwrist_idx, None, :]  # right hand root-relative
            joint_cam_wo_ra[smpl_x.joint_part['face'], :] = joint_cam_wo_ra[smpl_x.joint_part['face'], :] \
                                                            + joint_cam_wo_ra[smpl_x.neck_idx, None,: ]  # face root-relative

            # smplx parameters
            root_pose = np.array(smplx_param['global_orient'], dtype=np.float32).reshape(
                -1)  # rotation to world coordinate
            body_pose = np.array(smplx_param['body_pose'], dtype=np.float32).reshape(-1)
            
            # use adapted shape for adults
            if getattr(cfg, 'agora_fix_betas', False) and not data['kid']:
                shape = np.array(smplx_param['betas_neutral'], dtype=np.float32).reshape(-1)[:10]
            else:
                shape = np.array(smplx_param['betas'], dtype=np.float32).reshape(-1)[:10]  # bug?
            
            lhand_pose = np.array(smplx_param['left_hand_pose'], dtype=np.float32).reshape(-1)
            rhand_pose = np.array(smplx_param['right_hand_pose'], dtype=np.float32).reshape(-1)
            jaw_pose = np.array(smplx_param['jaw_pose'], dtype=np.float32).reshape(-1)
            expr = np.array(smplx_param['expression'], dtype=np.float32).reshape(-1)
            trans = np.array(smplx_param['transl'], dtype=np.float32).reshape(-1)  # translation to world coordinate
            cam_param = {'focal': cfg.focal,
                         'princpt': cfg.princpt}  # put random camera paraemter as we do not use coordinates from smplx parameters
            smplx_param = {'root_pose': root_pose, 'body_pose': body_pose, 'shape': shape,
                           'lhand_pose': lhand_pose, 'lhand_valid': True,
                           'rhand_pose': rhand_pose, 'rhand_valid': True,
                           'jaw_pose': jaw_pose, 'expr': expr, 'face_valid': True,
                           'trans': trans}
            _, _, _, smplx_pose, smplx_shape, smplx_expr, smplx_pose_valid, _, smplx_expr_valid, _ = process_human_model_output(
                smplx_param, cam_param, do_flip, img_shape, img2bb_trans, rot, 'smplx')
            smplx_pose_valid = np.tile(smplx_pose_valid[:, None], (1, 3)).reshape(-1)
            
            if not getattr(cfg, 'agora_valid_root_pose', False):
                smplx_pose_valid[:3] = 0  # global orient of the provided parameter is a rotation to world coordinate system. I want camera coordinate system.
            smplx_shape_valid = True
            inputs = {'img': img}
            targets = {'joint_img': joint_img, 'joint_cam': joint_cam_wo_ra, #from annot
                       'smplx_joint_img': joint_img, 'smplx_joint_cam': joint_cam_ra, #_smplx_joint_cam, # from smplx param w/ ra
                       'smplx_pose': smplx_pose, 'smplx_shape': smplx_shape, 'smplx_expr': smplx_expr, 
                       'lhand_bbox_center': lhand_bbox_center, 'lhand_bbox_size': lhand_bbox_size, 
                       'rhand_bbox_center': rhand_bbox_center, 'rhand_bbox_size': rhand_bbox_size, 
                       'face_bbox_center': face_bbox_center, 'face_bbox_size': face_bbox_size}
            meta_info = {'joint_valid': joint_valid, 'joint_trunc': joint_trunc, 
                         'smplx_joint_valid': joint_valid, 'smplx_joint_trunc': joint_trunc, 
                         'smplx_pose_valid': smplx_pose_valid, 'smplx_shape_valid': float(smplx_shape_valid), 
                         'smplx_expr_valid': float(smplx_expr_valid), 'is_3D': float(True), 
                         'lhand_bbox_valid': lhand_bbox_valid, 'rhand_bbox_valid': rhand_bbox_valid, 'face_bbox_valid': face_bbox_valid}
            return inputs, targets, meta_info
        else:
            # load crop and resize information (for the 4K setting)
            if self.resolution == (2160, 3840):
                img2bb_trans = np.dot(
                    np.concatenate((img2bb_trans,
                                    np.array([0, 0, 1], dtype=np.float32).reshape(1, 3))),
                    np.concatenate((data['img2bb_trans_from_orig'],
                                    np.array([0, 0, 1], dtype=np.float32).reshape(1, 3)))
                )
                bb2img_trans = np.linalg.inv(img2bb_trans)[:2, :]
                img2bb_trans = img2bb_trans[:2, :]

            if self.test_set == 'val':
                # gt load
                with open(data['verts_path']) as f:
                    verts = np.array(json.load(f)).reshape(-1, 3)
                
                with open(data['smplx_param_path'], 'rb') as f:
                    smplx_param = pickle.load(f, encoding='latin1')
                transl = np.array(smplx_param['transl'], dtype=np.float32).reshape(-1) 

                inputs = {'img': img}
                targets = {'smplx_mesh_cam': verts}
                meta_info = {'bb2img_trans': bb2img_trans, 'img_path': img_path, 'gt_smplx_transl':transl}
            else:
                inputs = {'img': img}
                targets = {'smplx_mesh_cam': np.zeros((smpl_x.vertex_num, 3), dtype=np.float32)}  # dummy vertex
                meta_info = {'bb2img_trans': bb2img_trans, 'img_path': img_path}

            return inputs, targets, meta_info

    def evaluate(self, outs, cur_sample_idx):
        annots = self.datalist
        sample_num = len(outs)
        eval_result = {'pa_mpvpe_all': [], 'pa_mpvpe_l_hand': [], 'pa_mpvpe_r_hand': [], 'pa_mpvpe_hand': [], 'pa_mpvpe_face': [], 
                       'mpvpe_all': [], 'mpvpe_l_hand': [], 'mpvpe_r_hand': [], 'mpvpe_hand': [], 'mpvpe_face': []}

        vis = getattr(cfg, 'vis', False)
        vis_save_dir = cfg.vis_dir

        if getattr(cfg, 'vis', False):
            import csv
            csv_file = f'{cfg.vis_dir}/agora_smplx_error.csv'
            file = open(csv_file, 'a', newline='')
            writer = csv.writer(file)

        for n in range(sample_num):
            annot = annots[cur_sample_idx + n]
            out = outs[n]
            mesh_gt = out['smplx_mesh_cam_target']
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


            if vis:
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
                smplx_pred['transl'] = out['gt_smplx_transl'].reshape(-1,3)
                smplx_pred['img_path'] = rel_img_path

                npz_path = os.path.join(cfg.vis_dir, f'{self.save_idx}.npz')
                np.savez(npz_path, **smplx_pred)

                # save img path and error
                new_line = [self.save_idx, rel_img_path, mpvpe_all, pa_mpvpe_all]
                # Append the new line to the CSV file
                writer.writerow(new_line)
                self.save_idx += 1

                # save_obj(out['smplx_mesh_cam'], smpl_x.face, str(cur_sample_idx + n) + '.obj')
        
            # save results for the official evaluation codes/server
            save_name = annot['img_path'].split('/')[-1][:-4]
            if self.data_split == 'test' and self.test_set == 'test':
                if self.resolution == (2160, 3840):
                    save_name = save_name.split('_pid')[0]
            elif self.data_split == 'test' and self.test_set == 'val':
                if self.resolution == (2160, 3840):
                    save_name = save_name.split('_ann_id')[0]
                else:
                    save_name = save_name.split('_1280x720')[0]
            if 'person_idx' in annot:
                person_idx = annot['person_idx']
            else:
                exist_result_path = glob(osp.join(cfg.result_dir, 'AGORA', save_name + '*'))
                if len(exist_result_path) == 0:
                    person_idx = 0
                else:
                    last_person_idx = max(
                        [int(name.split('personId_')[1].split('.pkl')[0]) for name in exist_result_path])
                    person_idx = last_person_idx + 1
            save_name += '_personId_' + str(person_idx) + '.pkl'

            joint_proj = out['smplx_joint_proj']
            joint_proj[:, 0] = joint_proj[:, 0] / cfg.output_hm_shape[2] * cfg.input_img_shape[1]
            joint_proj[:, 1] = joint_proj[:, 1] / cfg.output_hm_shape[1] * cfg.input_img_shape[0]
            joint_proj = np.concatenate((joint_proj, np.ones_like(joint_proj[:, :1])), 1)
            joint_proj = np.dot(out['bb2img_trans'], joint_proj.transpose(1, 0)).transpose(1, 0)
            joint_proj[:, 0] = joint_proj[:, 0] / self.resolution[1] * 3840  # restore to original resolution
            joint_proj[:, 1] = joint_proj[:, 1] / self.resolution[0] * 2160  # restore to original resolution
            save_dict = {'params':
                             {'transl': out['cam_trans'].reshape(1, -1),
                              'global_orient': out['smplx_root_pose'].reshape(1, -1),
                              'body_pose': out['smplx_body_pose'].reshape(1, -1),
                              'left_hand_pose': out['smplx_lhand_pose'].reshape(1, -1),
                              'right_hand_pose': out['smplx_rhand_pose'].reshape(1, -1),
                              'reye_pose': np.zeros((1, 3)),
                              'leye_pose': np.zeros((1, 3)),
                              'jaw_pose': out['smplx_jaw_pose'].reshape(1, -1),
                              'expression': out['smplx_expr'].reshape(1, -1),
                              'betas': out['smplx_shape'].reshape(1, -1)},
                         'joints': joint_proj.reshape(1, -1, 2)
                         }
            os.makedirs(osp.join(cfg.result_dir, 'predictions'), exist_ok=True)
            with open(osp.join(cfg.result_dir, 'predictions', save_name), 'wb') as f:
                pickle.dump(save_dict, f)

        if getattr(cfg, 'vis', False):
            file.close()

        return eval_result

    def print_eval_result(self, eval_result):

        print('AGORA test results are dumped at: ' + osp.join(cfg.result_dir, 'predictions'))

        if self.data_split == 'test' and self.test_set == 'test':  # do not print. just submit the results to the official evaluation server
            return

        print('======AGORA-val======')
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
        
        print(f"{np.mean(eval_result['pa_mpvpe_all'])},{np.mean(eval_result['pa_mpvpe_l_hand'])},{np.mean(eval_result['pa_mpvpe_r_hand'])},{np.mean(eval_result['pa_mpvpe_hand'])},{np.mean(eval_result['pa_mpvpe_face'])},"
                f"{np.mean(eval_result['mpvpe_all'])},{np.mean(eval_result['mpvpe_l_hand'])},{np.mean(eval_result['mpvpe_r_hand'])},{np.mean(eval_result['mpvpe_hand'])},{np.mean(eval_result['mpvpe_face'])}")
        print()

        f = open(os.path.join(cfg.result_dir, 'result.txt'), 'w')
        f.write(f'AGORA-val dataset: \n')
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
        f.write(f"{np.mean(eval_result['pa_mpvpe_all'])},{np.mean(eval_result['pa_mpvpe_l_hand'])},{np.mean(eval_result['pa_mpvpe_r_hand'])},{np.mean(eval_result['pa_mpvpe_hand'])},{np.mean(eval_result['pa_mpvpe_face'])},"
                f"{np.mean(eval_result['mpvpe_all'])},{np.mean(eval_result['mpvpe_l_hand'])},{np.mean(eval_result['mpvpe_r_hand'])},{np.mean(eval_result['mpvpe_hand'])},{np.mean(eval_result['mpvpe_face'])}")

        if getattr(cfg, 'eval_on_train', False):
            import csv
            csv_file = f'{cfg.root_dir}/output/agora_eval_on_train.csv'
            exp_id = cfg.exp_name.split('_')[1]
            new_line = [exp_id,np.mean(eval_result['pa_mpvpe_all']),np.mean(eval_result['pa_mpvpe_l_hand']),np.mean(eval_result['pa_mpvpe_r_hand']),np.mean(eval_result['pa_mpvpe_hand']),np.mean(eval_result['pa_mpvpe_face']),
                        np.mean(eval_result['mpvpe_all']),np.mean(eval_result['mpvpe_l_hand']),np.mean(eval_result['mpvpe_r_hand']),np.mean(eval_result['mpvpe_hand']),np.mean(eval_result['mpvpe_face'])]

            # Append the new line to the CSV file
            with open(csv_file, 'a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(new_line)