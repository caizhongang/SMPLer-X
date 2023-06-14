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
from utils.preprocessing import load_img, process_bbox, augmentation, process_db_coord, process_human_model_output, get_fitting_error_3D
from utils.transforms import world2cam, cam2pixel, rigid_align
import random
from humandata import Cache

class Human36M(torch.utils.data.Dataset):
    def __init__(self, transform, data_split):
        self.transform = transform
        self.data_split = data_split
        self.img_dir = osp.join(cfg.data_dir, 'Human36M', 'images')
        self.annot_path = osp.join(cfg.data_dir, 'Human36M', 'annotations')
        self.action_name = ['Directions', 'Discussion', 'Eating', 'Greeting', 'Phoning', 'Posing', 'Purchases', 'Sitting', 'SittingDown', 'Smoking', 'Photo', 'Waiting', 'Walking', 'WalkDog', 'WalkTogether']
        # H36M joint set
        self.joint_set = {'joint_num': 17,
                        'joints_name': ('Pelvis', 'R_Hip', 'R_Knee', 'R_Ankle', 'L_Hip', 'L_Knee', 'L_Ankle', 'Torso', 'Neck', 'Head', 'Head_top', 'L_Shoulder', 'L_Elbow', 'L_Wrist', 'R_Shoulder', 'R_Elbow', 'R_Wrist'),
                        'flip_pairs': ( (1, 4), (2, 5), (3, 6), (14, 11), (15, 12), (16, 13) ),
                        'eval_joint': (1, 2, 3, 4, 5, 6, 8, 10, 11, 12, 13, 14, 15, 16),
                        'regressor': np.load(osp.join(cfg.data_dir, 'Human36M', 'J_regressor_h36m_smplx.npy'))
                        }
        self.joint_set['root_joint_idx'] = self.joint_set['joints_name'].index('Pelvis')

        # self.datalist = self.load_data()

        # load data or cache
        self.use_cache = getattr(cfg, 'use_cache', False)
        self.annot_path_cache = osp.join(cfg.data_dir, 'cache', f'Human36M_{data_split}.npz')
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

    def get_subsampling_ratio(self):
        if self.data_split == 'train':
            return 5
        elif self.data_split == 'test':
            return 64
        else:
            assert 0, print('Unknown subset')

    def get_subject(self):
        if self.data_split == 'train':
            subject = [1,5,6,7,8]
        elif self.data_split == 'test':
            subject = [9,11]
        else:
            assert 0, print("Unknown subset")

        return subject
    
    def load_data(self):
        subject_list = self.get_subject()
        sampling_ratio = self.get_subsampling_ratio()
        
        # aggregate annotations from each subject
        db = COCO()
        cameras = {}
        joints = {}
        smplx_params = {}
        for subject in subject_list:
            # data load
            with open(osp.join(self.annot_path, 'Human36M_subject' + str(subject) + '_data.json'),'r') as f:
                annot = json.load(f)
            if len(db.dataset) == 0:
                for k,v in annot.items():
                    db.dataset[k] = v
            else:
                for k,v in annot.items():
                    db.dataset[k] += v
            # camera load
            with open(osp.join(self.annot_path, 'Human36M_subject' + str(subject) + '_camera.json'),'r') as f:
                cameras[str(subject)] = json.load(f)
            # joint coordinate load
            with open(osp.join(self.annot_path, 'Human36M_subject' + str(subject) + '_joint_3d.json'),'r') as f:
                joints[str(subject)] = json.load(f)
            # smplx parameter load
            with open(osp.join(self.annot_path, 'Human36M_subject' + str(subject) + '_SMPLX_NeuralAnnot.json'),'r') as f:
                smplx_params[str(subject)] = json.load(f)

        db.createIndex()

        datalist = []
        i = 0
        for aid in db.anns.keys():

            i += 1
            if self.data_split == 'train' and i % getattr(cfg, 'Human36M_train_sample_interval', 1) != 0:
                continue

            ann = db.anns[aid]
            image_id = ann['image_id']
            img = db.loadImgs(image_id)[0]
            img_path = osp.join(self.img_dir, img['file_name'])
            img_shape = (img['height'], img['width'])
            
            # check subject and frame_idx
            frame_idx = img['frame_idx'];
            if frame_idx % sampling_ratio != 0:
                continue

            # smplx parameter
            subject = img['subject']; action_idx = img['action_idx']; subaction_idx = img['subaction_idx']; frame_idx = img['frame_idx']; cam_idx = img['cam_idx'];
            smplx_param = smplx_params[str(subject)][str(action_idx)][str(subaction_idx)][str(frame_idx)]

            # camera parameter
            cam_param = cameras[str(subject)][str(cam_idx)]
            R,t,f,c = np.array(cam_param['R'], dtype=np.float32), np.array(cam_param['t'], dtype=np.float32), np.array(cam_param['f'], dtype=np.float32), np.array(cam_param['c'], dtype=np.float32)
            cam_param = {'R': R, 't': t, 'focal': f, 'princpt': c}
            
            # only use frontal camera following previous works (HMR and SPIN)
            if self.data_split == 'test' and str(cam_idx) != '4':
                continue
                
            # project world coordinate to cam, image coordinate space
            joint_world = np.array(joints[str(subject)][str(action_idx)][str(subaction_idx)][str(frame_idx)], dtype=np.float32)
            joint_cam = world2cam(joint_world, R, t)
            joint_img = cam2pixel(joint_cam, f, c)[:,:2]
            joint_valid = np.ones((self.joint_set['joint_num'],1))
        
            bbox = process_bbox(np.array(ann['bbox']), img['width'], img['height'], ratio=getattr(cfg, 'bbox_ratio', 1.25))
            if bbox is None: continue
            
            datalist.append({
                'img_path': img_path,
                'img_shape': img_shape,
                'bbox': bbox,
                'joint_img': joint_img,
                'joint_cam': joint_cam,
                'joint_valid': joint_valid,
                'smplx_param': smplx_param,
                'cam_param': cam_param})

        if self.data_split == 'train':
            print('[Human36M train] original size:', len(db.anns.keys()),
                  '. Sample interval:', getattr(cfg, 'Human36M_train_sample_interval', 1),
                  '. Sampled size:', len(datalist))

        if getattr(cfg, 'data_strategy', None) == 'balance' and self.data_split == 'train':
            print(f'[Human36M] Using [balance] strategy with datalist shuffled...')
            random.shuffle(datalist)

        return datalist

    def __len__(self):
        return len(self.datalist)

    def __getitem__(self, idx):
        data = copy.deepcopy(self.datalist[idx])
        img_path, img_shape, bbox, cam_param = data['img_path'], data['img_shape'], data['bbox'], data['cam_param']
        
        # img
        img = load_img(img_path)
        img, img2bb_trans, bb2img_trans, rot, do_flip = augmentation(img, bbox, self.data_split)
        img = self.transform(img.astype(np.float32))/255.
        
        if self.data_split == 'train':
            # h36m gt
            joint_cam = data['joint_cam']
            joint_cam = (joint_cam - joint_cam[self.joint_set['root_joint_idx'],None,:]) / 1000 # root-relative. milimeter to meter.
            joint_img = data['joint_img']
            joint_img = np.concatenate((joint_img[:,:2], joint_cam[:,2:]),1) # x, y, depth
            joint_img[:,2] = (joint_img[:,2] / (cfg.body_3d_size / 2) + 1)/2. * cfg.output_hm_shape[0] # discretize depth
            joint_img, joint_cam, joint_cam_ra, joint_valid, joint_trunc = process_db_coord(joint_img, joint_cam, data['joint_valid'], do_flip, img_shape, self.joint_set['flip_pairs'], img2bb_trans, rot, self.joint_set['joints_name'], smpl_x.joints_name)
            
            # smplx coordinates and parameters
            smplx_param = data['smplx_param']
            cam_param['t'] /= 1000 # milimeter to meter
            smplx_joint_img, smplx_joint_cam, smplx_joint_trunc, smplx_pose, smplx_shape, smplx_expr, \
                smplx_pose_valid, smplx_joint_valid, smplx_expr_valid, smplx_mesh_cam_orig = \
                    process_human_model_output(smplx_param, cam_param, do_flip, img_shape, img2bb_trans, rot, 'smplx')

            # reverse ra
            smplx_joint_cam_wo_ra = smplx_joint_cam.copy()
            smplx_joint_cam_wo_ra[smpl_x.joint_part['lhand'], :] = smplx_joint_cam_wo_ra[smpl_x.joint_part['lhand'], :] \
                                                            + smplx_joint_cam_wo_ra[smpl_x.lwrist_idx, None, :]  # left hand root-relative
            smplx_joint_cam_wo_ra[smpl_x.joint_part['rhand'], :] = smplx_joint_cam_wo_ra[smpl_x.joint_part['rhand'], :] \
                                                            + smplx_joint_cam_wo_ra[smpl_x.rwrist_idx, None, :]  # right hand root-relative
            smplx_joint_cam_wo_ra[smpl_x.joint_part['face'], :] = smplx_joint_cam_wo_ra[smpl_x.joint_part['face'], :] \
                                                                + smplx_joint_cam_wo_ra[smpl_x.neck_idx, None,: ]  # face root-relative


            # SMPLX pose parameter validity
            for name in ('L_Ankle', 'R_Ankle', 'L_Wrist', 'R_Wrist'):
                smplx_pose_valid[smpl_x.orig_joints_name.index(name)] = 0
            smplx_pose_valid = np.tile(smplx_pose_valid[:,None], (1,3)).reshape(-1)
            # SMPLX joint coordinate validity
            for name in ('L_Big_toe', 'L_Small_toe', 'L_Heel', 'R_Big_toe', 'R_Small_toe', 'R_Heel'):
                smplx_joint_valid[smpl_x.joints_name.index(name)] = 0
            smplx_joint_valid = smplx_joint_valid[:,None]
            smplx_joint_trunc = smplx_joint_valid * smplx_joint_trunc
            smplx_shape_valid = True

            # dummy hand/face bbox
            dummy_center = np.zeros((2), dtype=np.float32)
            dummy_size = np.zeros((2), dtype=np.float32)

            inputs = {'img': img}
            targets = {'joint_img': smplx_joint_img, 'smplx_joint_img': smplx_joint_img, 
                       'joint_cam': smplx_joint_cam_wo_ra, 'smplx_joint_cam': smplx_joint_cam, 
                       'smplx_pose': smplx_pose, 'smplx_shape': smplx_shape, 'smplx_expr': smplx_expr, 
                       'lhand_bbox_center': dummy_center, 'lhand_bbox_size': dummy_size, 
                       'rhand_bbox_center': dummy_center, 'rhand_bbox_size': dummy_size, 
                       'face_bbox_center': dummy_center, 'face_bbox_size': dummy_size}
            meta_info = {'joint_valid': smplx_joint_valid, 'joint_trunc': smplx_joint_trunc, 
                         'smplx_joint_valid': smplx_joint_valid, 'smplx_joint_trunc': smplx_joint_trunc, 
                         'smplx_pose_valid': smplx_pose_valid, 'smplx_shape_valid': float(smplx_shape_valid), 
                         'smplx_expr_valid': float(smplx_expr_valid), 'is_3D': float(True), 
                         'lhand_bbox_valid': float(False), 'rhand_bbox_valid': float(False), 
                         'face_bbox_valid': float(False)}
            return inputs, targets, meta_info
        else:
            inputs = {'img': img}
            targets = {}
            meta_info = {}
            return inputs, targets, meta_info

    def evaluate(self, outs, cur_sample_idx):

        annots = self.datalist
        sample_num = len(outs)
        eval_result = {'mpjpe': [], 'pa_mpjpe': []}
        for n in range(sample_num):
            annot = annots[cur_sample_idx + n]
            out = outs[n]
            
            # h36m joint from gt mesh
            joint_gt = annot['joint_cam'] 
            joint_gt = joint_gt - joint_gt[self.joint_set['root_joint_idx'],None] # root-relative 
            joint_gt = joint_gt[self.joint_set['eval_joint'],:] 
            
            # h36m joint from param mesh
            mesh_out = out['smpl_mesh_cam'] * 1000 # meter to milimeter
            joint_out = np.dot(self.joint_set['regressor'], mesh_out) # meter to milimeter
            joint_out = joint_out - joint_out[self.joint_set['root_joint_idx'],None] # root-relative
            joint_out = joint_out[self.joint_set['eval_joint'],:]
            joint_out_aligned = rigid_align(joint_out, joint_gt)
            eval_result['mpjpe'].append(np.sqrt(np.sum((joint_out - joint_gt)**2,1)).mean())
            eval_result['pa_mpjpe'].append(np.sqrt(np.sum((joint_out_aligned - joint_gt)**2,1)).mean())

            vis = False
            if vis:
                from utils.vis import vis_keypoints, vis_mesh, save_obj
                filename = annot['img_path'].split('/')[-1][:-4]

                img = load_img(annot['img_path'])[:,:,::-1]
                img = vis_mesh(img, mesh_out_img, 0.5)
                cv2.imwrite(filename + '.jpg', img)
                save_obj(mesh_out, smpl_x.face, filename + '.obj')

        return eval_result

    def print_eval_result(self, eval_result):
        print('MPJPE: %.2f mm' % np.mean(eval_result['mpjpe']))
        print('PA MPJPE: %.2f mm' % np.mean(eval_result['pa_mpjpe']))
