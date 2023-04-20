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
        eval_result = {}
        for n in range(sample_num):
            annot = annots[cur_sample_idx + n]
            out = outs[n]

            vis = True
            if vis:
                from utils.vis import vis_keypoints, vis_mesh, save_obj
                """
                file_name = str(cur_sample_idx+n)
                img = (out['img'].transpose(1,2,0)[:,:,::-1] * 255).copy()
                cv2.imwrite(file_name + '.jpg', img)
                save_obj(out['smplx_mesh_cam'], smpl_x.face, file_name + '.obj')
                """
                
                """
                img_path = annot['img_path']
                img = load_img(img_path)[:,:,::-1]
                bbox = annot['bbox']
                focal = list(cfg.focal)
                princpt = list(cfg.princpt)
                focal[0] = focal[0] / cfg.input_body_shape[1] * bbox[2]
                focal[1] = focal[1] / cfg.input_body_shape[0] * bbox[3]
                princpt[0] = princpt[0] / cfg.input_body_shape[1] * bbox[2] + bbox[0]
                princpt[1] = princpt[1] / cfg.input_body_shape[0] * bbox[3] + bbox[1]
                img = render_mesh(img, out['smplx_mesh_cam'], smpl_x.face, {'focal': focal, 'princpt': princpt})
                #img = cv2.resize(img, (512,512))
                cv2.imwrite(img_id + '_' + str(ann_id) + '.jpg', img)
                """
                
                ann_id = annot['ann_id']
                bbox = annot['bbox']
                focal = list(cfg.focal)
                princpt = list(cfg.princpt)
                focal[0] = focal[0] / cfg.input_body_shape[1] * bbox[2]
                focal[1] = focal[1] / cfg.input_body_shape[0] * bbox[3]
                princpt[0] = princpt[0] / cfg.input_body_shape[1] * bbox[2] + bbox[0]
                princpt[1] = princpt[1] / cfg.input_body_shape[0] * bbox[3] + bbox[1]
                param_save = {'smplx_param': {'root_pose': out['smplx_root_pose'].tolist(), 'body_pose': out['smplx_body_pose'].tolist(), 'lhand_pose': out['smplx_lhand_pose'].tolist(), 'rhand_pose': out['smplx_rhand_pose'].tolist(), 'jaw_pose': out['smplx_jaw_pose'].tolist(), 'shape': out['smplx_shape'].tolist(), 'expr': out['smplx_expr'].tolist(), 'trans': out['cam_trans'].tolist()},
                        'cam_param': {'focal': focal, 'princpt': princpt}
                        }
                with open(str(ann_id) + '.json', 'w') as f:
                    json.dump(param_save, f)


        return eval_result

    def print_eval_result(self, eval_result):
        pass



