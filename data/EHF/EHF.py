import os
import os.path as osp
from glob import glob
import numpy as np
from config import cfg
import copy
import json
import cv2
import torch
from pycocotools.coco import COCO
from utils.human_models import smpl_x
from utils.preprocessing import load_img, process_bbox, augmentation, load_ply
from utils.transforms import rigid_align


class EHF(torch.utils.data.Dataset):
    def __init__(self, transform, data_split):
        self.transform = transform
        self.data_split = data_split
        # self.data_path = osp.join('..', 'data', 'EHF', 'data')
        self.data_path = osp.join(cfg.data_dir, 'EHF', 'data')
        self.datalist = self.load_data()
        self.cam_param = {'R': [-2.98747896, 0.01172457, -0.05704687]}
        self.cam_param['R'], _ = cv2.Rodrigues(np.array(self.cam_param['R']))
        self.save_idx = 0

    def load_data(self):
        datalist = []
        db = COCO(osp.join(self.data_path, 'EHF.json'))
        for aid in db.anns.keys():
            ann = db.anns[aid]
            image_id = ann['image_id']
            img = db.loadImgs(image_id)[0]
            img_shape = (img['height'], img['width'])
            img_path = osp.join(self.data_path, img['file_name'])

            bbox = ann['body_bbox']
            bbox = process_bbox(bbox, img['width'], img['height'], ratio=getattr(cfg, 'bbox_ratio', 1.25))
            if bbox is None:
                continue

            lhand_bbox = np.array(ann['lefthand_bbox']).reshape(4)
            if hasattr(cfg, 'bbox_ratio'):
                lhand_bbox = process_bbox(lhand_bbox, img['width'], img['height'], ratio=cfg.bbox_ratio)
            lhand_bbox[2:] += lhand_bbox[:2]  # xywh -> xyxy

            rhand_bbox = np.array(ann['righthand_bbox']).reshape(4)
            if hasattr(cfg, 'bbox_ratio'):
                rhand_bbox = process_bbox(rhand_bbox, img['width'], img['height'], ratio=cfg.bbox_ratio)
            rhand_bbox[2:] += rhand_bbox[:2]  # xywh -> xyxy

            face_bbox = np.array(ann['face_bbox']).reshape(4)
            if hasattr(cfg, 'bbox_ratio'):
                face_bbox = process_bbox(face_bbox, img['width'], img['height'], ratio=cfg.bbox_ratio)
            face_bbox[2:] += face_bbox[:2]  # xywh -> xyxy

            mesh_gt_path = osp.join(self.data_path, img['file_name'].split('_')[0] + '_align.ply')

            data_dict = {'img_path': img_path, 'img_shape': img_shape, 'bbox': bbox, 'lhand_bbox': lhand_bbox,
                         'rhand_bbox': rhand_bbox, 'face_bbox': face_bbox, 'mesh_gt_path': mesh_gt_path}
            datalist.append(data_dict)

        return datalist

    def process_hand_face_bbox(self, bbox, do_flip, img_shape, img2bb_trans):
        if bbox is None:
            bbox = np.zeros((2, 2), dtype=np.float32)  # dummy value
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
            bbox[:, 0] = bbox[:, 0] / cfg.input_img_shape[1] * cfg.output_hm_shape[1]
            bbox[:, 1] = bbox[:, 1] / cfg.input_img_shape[0] * cfg.output_hm_shape[0]

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
        img_path, img_shape, bbox, mesh_gt_path = data['img_path'], data['img_shape'], data['bbox'], data[
            'mesh_gt_path']

        # image load
        img = load_img(img_path)

        # affine transform
        img, img2bb_trans, bb2img_trans, rot, do_flip = augmentation(img, bbox, self.data_split)
        img = self.transform(img.astype(np.float32)) / 255.

        # hand and face bbox transform
        lhand_bbox, lhand_bbox_valid = self.process_hand_face_bbox(data['lhand_bbox'], do_flip, img_shape, img2bb_trans)
        rhand_bbox, rhand_bbox_valid = self.process_hand_face_bbox(data['rhand_bbox'], do_flip, img_shape, img2bb_trans)
        face_bbox, face_bbox_valid = self.process_hand_face_bbox(data['face_bbox'], do_flip, img_shape, img2bb_trans)
        if do_flip:
            lhand_bbox, rhand_bbox = rhand_bbox, lhand_bbox
            lhand_bbox_valid, rhand_bbox_valid = rhand_bbox_valid, lhand_bbox_valid
        lhand_bbox_center = (lhand_bbox[0] + lhand_bbox[1]) / 2.;
        rhand_bbox_center = (rhand_bbox[0] + rhand_bbox[1]) / 2.;
        face_bbox_center = (face_bbox[0] + face_bbox[1]) / 2.
        lhand_bbox_size = lhand_bbox[1] - lhand_bbox[0];
        rhand_bbox_size = rhand_bbox[1] - rhand_bbox[0];
        face_bbox_size = face_bbox[1] - face_bbox[0];

        # mesh gt load
        mesh_gt = load_ply(mesh_gt_path)

        inputs = {'img': img}
        targets = {'smplx_mesh_cam': mesh_gt, 'lhand_bbox_center': lhand_bbox_center,
                   'rhand_bbox_center': rhand_bbox_center, 'face_bbox_center': face_bbox_center,
                   'lhand_bbox_size': lhand_bbox_size, 'rhand_bbox_size': rhand_bbox_size,
                   'face_bbox_size': face_bbox_size}
        meta_info = {'bb2img_trans': bb2img_trans, 'lhand_bbox_valid': float(True), 'rhand_bbox_valid': float(True),
                     'face_bbox_valid': float(True),
                     'img_path': img_path}
        return inputs, targets, meta_info

    def evaluate(self, outs, cur_sample_idx):
        annots = self.datalist
        sample_num = len(outs)
        eval_result = {'pa_mpvpe_all': [], 'pa_mpvpe_l_hand': [], 'pa_mpvpe_r_hand': [], 'pa_mpvpe_hand': [], 'pa_mpvpe_face': [], 
                       'mpvpe_all': [], 'mpvpe_l_hand': [], 'mpvpe_r_hand': [], 'mpvpe_hand': [], 'mpvpe_face': [], 
                       'pa_mpjpe_body': [], 'pa_mpjpe_l_hand': [], 'pa_mpjpe_r_hand': [], 'pa_mpjpe_hand': []}
        
        if getattr(cfg, 'vis', False):
            import csv
            csv_file = f'{cfg.vis_dir}/ehf_smplx_error.csv'
            file = open(csv_file, 'a', newline='')
            writer = csv.writer(file)

        for n in range(sample_num):
            annot = annots[cur_sample_idx + n]
            ann_id = annot['img_path'].split('/')[-1].split('_')[0]
            # print(annot['img_path'])
            # ann_id = annot['ann_id']
            out = outs[n]

            # MPVPE from all vertices
            mesh_gt = np.dot(self.cam_param['R'], out['smplx_mesh_cam_target'].transpose(1, 0)).transpose(1, 0)
            mesh_out = out['smplx_mesh_cam']

            # mesh_gt_align = rigid_align(mesh_gt, mesh_out)

            # print(mesh_out.shape)
            mesh_out_align = rigid_align(mesh_out, mesh_gt)
            pa_mpvpe_all = np.sqrt(np.sum((mesh_out_align - mesh_gt) ** 2, 1)).mean() * 1000
            eval_result['pa_mpvpe_all'].append(pa_mpvpe_all)
            mesh_out_align = mesh_out - np.dot(smpl_x.J_regressor, mesh_out)[smpl_x.J_regressor_idx['pelvis'], None,
                                        :] + np.dot(smpl_x.J_regressor, mesh_gt)[smpl_x.J_regressor_idx['pelvis'], None,
                                             :]
            mpvpe_all = np.sqrt(np.sum((mesh_out_align - mesh_gt) ** 2, 1)).mean() * 1000
            eval_result['mpvpe_all'].append(mpvpe_all)

            # MPVPE from hand vertices
            mesh_gt_lhand = mesh_gt[smpl_x.hand_vertex_idx['left_hand'], :]
            mesh_out_lhand = mesh_out[smpl_x.hand_vertex_idx['left_hand'], :]
            mesh_out_lhand_align = rigid_align(mesh_out_lhand, mesh_gt_lhand)
            mesh_gt_rhand = mesh_gt[smpl_x.hand_vertex_idx['right_hand'], :]
            mesh_out_rhand = mesh_out[smpl_x.hand_vertex_idx['right_hand'], :]
            mesh_out_rhand_align = rigid_align(mesh_out_rhand, mesh_gt_rhand)
            eval_result['pa_mpvpe_l_hand'].append(np.sqrt(
                np.sum((mesh_out_lhand_align - mesh_gt_lhand) ** 2, 1)).mean() * 1000)
            eval_result['pa_mpvpe_r_hand'].append(np.sqrt(
                np.sum((mesh_out_rhand_align - mesh_gt_rhand) ** 2, 1)).mean() * 1000)
            eval_result['pa_mpvpe_hand'].append((np.sqrt(
                np.sum((mesh_out_lhand_align - mesh_gt_lhand) ** 2, 1)).mean() * 1000 + np.sqrt(
                np.sum((mesh_out_rhand_align - mesh_gt_rhand) ** 2, 1)).mean() * 1000) / 2.)

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

            # MPVPE from face vertices
            mesh_gt_face = mesh_gt[smpl_x.face_vertex_idx, :]
            mesh_out_face = mesh_out[smpl_x.face_vertex_idx, :]
            mesh_out_face_align = rigid_align(mesh_out_face, mesh_gt_face)
            eval_result['pa_mpvpe_face'].append(
                np.sqrt(np.sum((mesh_out_face_align - mesh_gt_face) ** 2, 1)).mean() * 1000)
            mesh_out_face_align = mesh_out_face - np.dot(smpl_x.J_regressor, mesh_out)[smpl_x.J_regressor_idx['neck'],
                                                  None, :] + np.dot(smpl_x.J_regressor, mesh_gt)[
                                                             smpl_x.J_regressor_idx['neck'], None, :]
            eval_result['mpvpe_face'].append(
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

            vis = cfg.vis
            if vis:
                # save_folder = cfg.vis_dir
                # kpt_save_folder = os.path.join(save_folder, 'KPT')
                # os.makedirs(kpt_save_folder, exist_ok=True)
                # mesh_save_folder = os.path.join(save_folder, 'mesh_origin')
                # os.makedirs(mesh_save_folder, exist_ok=True)
                # # from utils.vis import vis_keypoints, render_mesh, save_obj
                # img = (out['img'].transpose(1, 2, 0)[:, :, ::-1] * 255).copy()
                # joint_img = out['joint_img'].copy()
                # joint_img[:, 0] = joint_img[:, 0] / cfg.output_hm_shape[2] * cfg.input_img_shape[1]
                # joint_img[:, 1] = joint_img[:, 1] / cfg.output_hm_shape[1] * cfg.input_img_shape[0]
                # for j in range(len(joint_img)):
                #     cv2.circle(img, (int(joint_img[j][0]), int(joint_img[j][1])), 3, (0, 0, 255), -1)
                # lhand_bbox = out['lhand_bbox'].reshape(2, 2).copy()
                # cv2.rectangle(img, (int(lhand_bbox[0][0]), int(lhand_bbox[0][1])),
                #               (int(lhand_bbox[1][0]), int(lhand_bbox[1][1])), (255, 0, 0), 3)
                # rhand_bbox = out['rhand_bbox'].reshape(2, 2).copy()
                # cv2.rectangle(img, (int(rhand_bbox[0][0]), int(rhand_bbox[0][1])),
                #               (int(rhand_bbox[1][0]), int(rhand_bbox[1][1])), (255, 0, 0), 3)
                # face_bbox = out['face_bbox'].reshape(2, 2).copy()
                # cv2.rectangle(img, (int(face_bbox[0][0]), int(face_bbox[0][1])),
                #               (int(face_bbox[1][0]), int(face_bbox[1][1])), (255, 0, 0), 3)
                # cv2.imwrite(os.path.join(kpt_save_folder, str(cur_sample_idx + n) + '.jpg'), img)

                # vis_img = img.copy()
                # focal = [cfg.focal[0] / cfg.input_body_shape[1] * cfg.input_img_shape[1],
                #          cfg.focal[1] / cfg.input_body_shape[0] * cfg.input_img_shape[0]]
                # princpt = [cfg.princpt[0] / cfg.input_body_shape[1] * cfg.input_img_shape[1],
                #            cfg.princpt[1] / cfg.input_body_shape[0] * cfg.input_img_shape[0]]
                # rendered_img = render_mesh(vis_img, mesh_out, smpl_x.face, {'focal': focal, 'princpt': princpt})
                # vis_img = img.copy()
                # # rendered_img_gt = render_mesh(vis_img, mesh_gt_align, smpl_x.face, {'focal': focal, 'princpt': princpt})
                # cv2.imwrite(os.path.join(mesh_save_folder, f'{ann_id}_render.jpg'), rendered_img)
                # # cv2.imwrite(os.path.join(mesh_save_folder, f'{ann_id}_render_gt.jpg'), rendered_img_gt)
                # cv2.imwrite(os.path.join(mesh_save_folder, f'{ann_id}.jpg'), vis_img)
                # np.save(os.path.join(mesh_save_folder, f'{ann_id}.npy'), mesh_out)

                # save smplx param
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
                smplx_pred['transl'] = out['cam_trans'].reshape(-1,3)
                
                np.savez(os.path.join(cfg.vis_dir, f'{self.save_idx}.npz'), **smplx_pred)

                # save img path and error
                img_path = out['img_path']
                rel_img_path = img_path.split('..')[-1]
                new_line = [self.save_idx, rel_img_path, mpvpe_all, pa_mpvpe_all]
                # Append the new line to the CSV file
                writer.writerow(new_line)
                self.save_idx += 1

                # save_obj(out['smplx_mesh_cam'], smpl_x.face, str(cur_sample_idx + n) + '.obj')
        
        if getattr(cfg, 'vis', False):
            file.close()

            
        return eval_result

    def print_eval_result(self, eval_result):
        print('======EHF======')
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
        f"{np.mean(eval_result['mpvpe_all'])},{np.mean(eval_result['mpvpe_l_hand'])},{np.mean(eval_result['mpvpe_r_hand'])},{np.mean(eval_result['mpvpe_hand'])},{np.mean(eval_result['mpvpe_face'])}")
        print()


        f = open(os.path.join(cfg.result_dir, 'result.txt'), 'w')
        f.write(f'EHF dataset: \n')
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
        f"{np.mean(eval_result['mpvpe_all'])},{np.mean(eval_result['mpvpe_l_hand'])},{np.mean(eval_result['mpvpe_r_hand'])},{np.mean(eval_result['mpvpe_hand'])},{np.mean(eval_result['mpvpe_face'])}")


        
        # for i in range(len(eval_result['pa_mpvpe_all'])):
        #     f.write(f'{i+1:02d}.jpg\n')
        #     f.write('PA MPVPE (All): %.2f mm\n' % eval_result['pa_mpvpe_all'][i])
        #     f.write('PA MPVPE (Hands): %.2f mm\n' % eval_result['pa_mpvpe_hand'][i])
        #     f.write('PA MPVPE (Face): %.2f mm\n' % eval_result['pa_mpvpe_face'][i])
        #     f.write('MPVPE (All): %.2f mm\n' % eval_result['mpvpe_all'][i])
        #     f.write('MPVPE (Hands): %.2f mm\n' % eval_result['mpvpe_hand'][i])
        #     f.write('MPVPE (Face): %.2f mm\n' % eval_result['mpvpe_face'][i])
        #     f.write('PA MPJPE (Body): %.2f mm\n' % eval_result['pa_mpjpe_body'][i])
        #     f.write('PA MPJPE (Hands): %.2f mm\n' % eval_result['pa_mpjpe_hand'][i])


