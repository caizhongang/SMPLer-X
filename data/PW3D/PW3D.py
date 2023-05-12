import os
import os.path as osp
import numpy as np
import torch
import cv2
import json
import copy
from pycocotools.coco import COCO
from config import cfg
from utils.human_models import smpl_x, smpl
from utils.preprocessing import load_img, process_bbox, augmentation, process_human_model_output
from utils.transforms import rigid_align

class PW3D(torch.utils.data.Dataset):
    def __init__(self, transform, data_split):
        self.transform = transform
        self.data_split = data_split
        self.data_path = osp.join(cfg.data_dir, 'PW3D', 'data')
        self.datalist = self.load_data()

    def load_data(self):
        db = COCO(osp.join(self.data_path, '3DPW_' + self.data_split + '.json'))

        datalist = []
        for aid in db.anns.keys():
            ann = db.anns[aid]
            image_id = ann['image_id']
            img = db.loadImgs(image_id)[0]
            sequence_name = img['sequence']
            img_name = img['file_name']
            img_path = osp.join(self.data_path, 'imageFiles', sequence_name, img_name)
            cam_param = {k: np.array(v, dtype=np.float32) for k,v in img['cam_param'].items()}

            smpl_param = ann['smpl_param']
            bbox = process_bbox(np.array(ann['bbox']), img['width'], img['height'], ratio=getattr(cfg, 'bbox_ratio', 1.25))
            if bbox is None: continue
            data_dict = {'img_path': img_path, 'ann_id': aid, 'img_shape': (img['height'], img['width']), 'bbox': bbox, 'smpl_param': smpl_param, 'cam_param': cam_param}
            datalist.append(data_dict)

        return datalist

    def __len__(self):
        return len(self.datalist)

    def __getitem__(self, idx):
        data = copy.deepcopy(self.datalist[idx])
        img_path, img_shape = data['img_path'], data['img_shape']
        
        # img
        img = load_img(img_path)
        bbox, smpl_param, cam_param = data['bbox'], data['smpl_param'], data['cam_param']
        img, img2bb_trans, bb2img_trans, rot, do_flip = augmentation(img, bbox, self.data_split)
        img = self.transform(img.astype(np.float32))/255.

        # smpl coordinates
        smpl_joint_img, smpl_joint_cam, smpl_joint_trunc, smpl_pose, smpl_shape, smpl_mesh_cam_orig = process_human_model_output(smpl_param, cam_param, do_flip, img_shape, img2bb_trans, rot, 'smpl')

        inputs = {'img': img}
        targets = {'smpl_mesh_cam': smpl_mesh_cam_orig}
        meta_info = {}
        return inputs, targets, meta_info
        
    
    def evaluate(self, outs, cur_sample_idx):
        annots = self.datalist
        sample_num = len(outs)
        eval_result = {'mpjpe_body': [], 'pa_mpjpe_body': [], }
        
        ## smpl/smplx -> lsp
        #     ['left_hip', 'right_hip', 'left_knee', 'right_knee', 'left_ankle',
        #    'right_ankle', 'neck', 'head', 'left_shoulder', 'right_shoulder',
        #    'left_elbow', 'right_elbow', 'left_wrist', 'right_wrist']
        joint_mapper = [1, 2, 4, 5, 7, 8, 12, 15, 16, 17, 18, 19, 20, 21]

        ### Save vis for debug
        # joint_gt_body_to_save = np.zeros((sample_num, len(joint_mapper), 3))
        # joint_out_body_root_align_to_save = np.zeros((sample_num, len(joint_mapper), 3))
        # joint_out_body_pa_align_to_save = np.zeros((sample_num, len(joint_mapper), 3))
        
        for n in range(sample_num):

            out = outs[n]

            # MPVPE from all vertices
            mesh_gt = out['smpl_mesh_cam_target']
            mesh_out = out['smplx_mesh_cam']

            # MPJPE from body joints
            mesh_out_align = mesh_out - np.dot(smpl_x.J_regressor, mesh_out)[smpl_x.J_regressor_idx['pelvis'], None, :] \
                                      + np.dot(smpl.joint_regressor, mesh_gt)[smpl.root_joint_idx, None, :]

            # only eval point0-21 since only smpl gt is given
            # joint_gt_body = np.dot(smpl.joint_regressor, mesh_gt)[:22, :] 
            # joint_out_body = np.dot(smpl_x.J_regressor, mesh_out)[:22, :] 
            # joint_out_body_root_align = np.dot(smpl_x.J_regressor, mesh_out_align)[:22, :]

            # only test 14 keypoints
            joint_gt_body = np.dot(smpl.joint_regressor, mesh_gt)[joint_mapper, :] 
            joint_out_body = np.dot(smpl_x.J_regressor, mesh_out)[joint_mapper, :] 
            joint_out_body_root_align = np.dot(smpl_x.J_regressor, mesh_out_align)[joint_mapper, :]

            eval_result['mpjpe_body'].append(
                np.sqrt(np.sum((joint_out_body_root_align - joint_gt_body) ** 2, 1)).mean() * 1000)

            # PAMPJPE from body joints
            joint_out_body_pa_align = rigid_align(joint_out_body, joint_gt_body)
            eval_result['pa_mpjpe_body'].append(
                np.sqrt(np.sum((joint_out_body_pa_align - joint_gt_body) ** 2, 1)).mean() * 1000)
            
            ### Save vis for debug
            # joint_gt_body_to_save[n, ...] = joint_gt_body
            # joint_out_body_root_align_to_save[n, ...] = joint_out_body_root_align
            # joint_out_body_pa_align_to_save[n, ...] = joint_out_body_pa_align
        
        ### Save vis for debug
        # import numpy as np
        # np.save(f'./vis/val_0509_joint_gt_body.npy', joint_gt_body_to_save)
        # np.save(f'./vis/val_0509_joint_out_body_root_align.npy', joint_out_body_root_align_to_save)
        # np.save(f'./vis/val_0509_joint_out_body_pa_align.npy', joint_out_body_pa_align_to_save)
        # import pdb; pdb.set_trace()

        return eval_result

    def print_eval_result(self, eval_result):
        print('======3DPW-test======')
        print('MPJPE (Body): %.2f mm' % np.mean(eval_result['mpjpe_body']))
        print('PA MPJPE (Body): %.2f mm' % np.mean(eval_result['pa_mpjpe_body']))

        f = open(os.path.join(cfg.result_dir, 'result.txt'), 'w')
        f.write(f'3DPW-test dataset: \n')
        f.write('MPJPE (Body): %.2f mm\n' % np.mean(eval_result['mpjpe_body']))
        f.write('PA MPJPE (Body): %.2f mm\n' % np.mean(eval_result['pa_mpjpe_body']))




