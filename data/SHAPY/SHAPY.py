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
import pickle
from body_measurements import BodyMeasurements
import smplx
from test_submission_format import test_submission_file_format


def point_error(x, y, align=True):
    """ Ref: https://github.com/muelea/shapy/blob/master/regressor/hbw_evaluation/evaluate_hbw.py#LL44C1-L58C31 """
    t = 0.0
    if align:
        t = x.mean(0, keepdims=True) - y.mean(0, keepdims=True)
    x_hat = x - t
    error = np.sqrt(np.power(x_hat - y, 2).sum(axis=-1))
    return error.mean().item()


class SHAPY(HumanDataset):
    def __init__(self, transform, data_split):
        super(SHAPY, self).__init__(transform, data_split)

        self.eval_split = getattr(cfg, 'shapy_eval_split')
        if self.data_split == 'train':
            raise NotImplementedError('Shapy train not implemented yet. Need to consider invalid parameters')
        if self.data_split == 'test' and self.eval_split == 'test':
            filename = getattr(cfg, 'filename', 'shapy_test_230512_1631.npz')
        elif self.data_split == 'test' and self.eval_split == 'val':
            filename = getattr(cfg, 'filename', 'shapy_val_230512_705.npz')
        else:
            raise ValueError(f'Undefined. data split: {self.data_split}; eval_split: {self.test_set}')

        self.img_dir = osp.join(cfg.data_dir, 'SHAPY')
        self.annot_path = osp.join(cfg.data_dir, 'preprocessed_datasets', filename)
        self.v_shape_load_dir = osp.join(cfg.data_dir, 'SHAPY', 'HBW', 'smplx', 'val')
        self.img_shape = None  # variable img_shape
        self.cam_param = {}

        # load data
        self.datalist = self.load_data(
            train_sample_interval=getattr(cfg, f'{self.__class__.__name__}_train_sample_interval', 1))

        ### SHAPY utils
        ### ref: https://github.com/muelea/shapy/blob/master/regressor/hbw_evaluation/evaluate_hbw.py#L28

        # load body model
        # ref: common/utils/human_models.py
        self.layer_arg = {'create_global_orient': False, 'create_body_pose': False, 'create_left_hand_pose': False,
                          'create_right_hand_pose': False, 'create_jaw_pose': False, 'create_leye_pose': False,
                          'create_reye_pose': False, 'create_betas': False, 'create_expression': False,
                          'create_transl': False}
        self.smplx_layer = smplx.create(cfg.human_model_path,
                                        'smplx',
                                        gender='NEUTRAL',
                                        use_pca=False,
                                        use_face_contour=True,
                                        flat_hand_mean=True,  # critical!
                                        **self.layer_arg).cuda()
        # self.smplx_layer = copy.deepcopy(smpl_x.layer['neutral']).cuda()
        self.faces_tensor_smplx = self.smplx_layer.faces_tensor.detach().cpu().numpy()

        # load files to compute P2P-20K Error
        point_reg = osp.join(cfg.data_dir, 'SHAPY', 'utility_files', 'evaluation', 'eval_point_set', 'HD_SMPLX_from_SMPL.pkl')
        with open(point_reg, 'rb') as f:
            self.point_regressor = pickle.load(f)

        # load files to compute Measurements Error
        body_measurement_folder = osp.join(cfg.data_dir, 'SHAPY', 'utility_files', 'measurements')
        meas_def_path = osp.join(body_measurement_folder, 'measurement_defitions.yaml')
        meas_verts_path_gt = osp.join(body_measurement_folder, 'smplx_measurements.yaml')
        self.body_measurements = BodyMeasurements(
            {'meas_definition_path': meas_def_path,
                'meas_vertices_path': meas_verts_path_gt},
        ).to('cuda')

        self.v_shaped_gt = {}

        # to save preditions
        self.images_names = []
        self.v_shaped = []

    def evaluate(self, outs, cur_sample_idx):
        annots = self.datalist
        sample_num = len(outs)
        eval_result = {'v2v_t_errors': [], 'point_t_errors': [], 'height': [], 'chest': [], 'waist': [], 'hips': [], 'mass': []}

        for n in range(sample_num):
            annot = annots[cur_sample_idx + n]
            out = outs[n]
            
            betas_fit = out['smplx_shape']
            img_path = out['img_path']

            # compute v_shaped
            betas_fit = torch.tensor(betas_fit.reshape(-1, 10)).cuda()
            output = self.smplx_layer(
                betas=betas_fit, 
                body_pose=torch.zeros((1, 63)).to(betas_fit.device), 
                global_orient=torch.zeros((1, 3)).to(betas_fit.device), 
                right_hand_pose=torch.zeros((1, 45)).to(betas_fit.device),
                left_hand_pose=torch.zeros((1, 45)).to(betas_fit.device), 
                jaw_pose=torch.zeros((1, 3)).to(betas_fit.device), 
                leye_pose=torch.zeros((1, 3)).to(betas_fit.device),
                reye_pose=torch.zeros((1, 3)).to(betas_fit.device), 
                expression=torch.zeros((1, 10)).to(betas_fit.device),
                return_verts=True
                )
            v_shaped_fit = output.vertices.detach().cpu().numpy().squeeze()

            image_name = '/'.join(img_path.split('/')[-4:])
            self.images_names.append(image_name)
            self.v_shaped.append(v_shaped_fit)

            if self.eval_split == 'val':
                # load gt vertices
                subject = img_path.split('/')[-3]
                subject_id_npy = subject.split('_')[0] + '.npy'
                v_shaped_gt_path = osp.join(self.v_shape_load_dir, subject_id_npy)
                if v_shaped_gt_path not in self.v_shaped_gt:
                    v_shaped_gt = np.load(v_shaped_gt_path)
                    self.v_shaped_gt[v_shaped_gt_path] = v_shaped_gt
                else:
                    v_shaped_gt = self.v_shaped_gt[v_shaped_gt_path]

                # compute vertex-to-vertex error (SMPL-X only)
                # ref: https://github.com/muelea/shapy/blob/master/regressor/hbw_evaluation/evaluate_hbw.py#LL142C1-L171C48
                v2v_error = point_error(v_shaped_fit, v_shaped_gt, align=True)
                eval_result['v2v_t_errors'].append(v2v_error)

                # compute P2P-20k error
                points_gt = self.point_regressor.dot(v_shaped_gt)
                points_fit = self.point_regressor.dot(v_shaped_fit)
                p2p_error = point_error(points_gt, points_fit, align=True)
                eval_result['point_t_errors'].append(p2p_error)

                # compute height/chest/waist/hip error
                shaped_triangles_gt = v_shaped_gt[self.faces_tensor_smplx]
                shaped_triangles_gt = torch.from_numpy(shaped_triangles_gt).unsqueeze(0).to('cuda')
                measurements_gt = self.body_measurements(shaped_triangles_gt)['measurements']

                shaped_triangles_fit = v_shaped_fit[self.faces_tensor_smplx]
                shaped_triangles_fit = torch.from_numpy(shaped_triangles_fit).unsqueeze(0).to('cuda')
                measurements_fit = self.body_measurements(shaped_triangles_fit)['measurements']

                for k in ['height', 'chest', 'waist', 'hips', 'mass']:
                    error = abs(measurements_gt[k]['tensor'].item() - measurements_fit[k]['tensor'].item())
                    eval_result[k].append(error)


        return eval_result


    def print_eval_result(self, eval_result):

        # print('SHAPY results are dumped at: ' + osp.join(cfg.result_dir, 'predictions'))

        if self.data_split == 'test' and self.eval_split == 'test':  # do not print. just submit the results to the official evaluation server
            # save predictions in the format of HBW challenge
            # ref: https://github.com/muelea/shapy/blob/master/regressor/hbw_evaluation/README_HBW_EVAL.md#hbw-challenge
            save_dir = osp.join(cfg.result_dir, 'predictions')
            os.makedirs(save_dir, exist_ok=True)
            save_name = osp.join(save_dir, 'hbw_prediction')
            images_names = np.array(self.images_names).reshape(1631, )
            v_shaped = np.array(self.v_shaped).reshape(1631, 10475, 3)
            np.savez(save_name,
                     image_name=images_names,
                     v_shaped=v_shaped)
            print('predictions saved at: ' + save_name + '.npz')

            # run format test
            test_submission_file_format(save_name + '.npz')
            return

        v2v_t_errors = np.mean(eval_result['v2v_t_errors']) * 1000 
        point_t_errors = np.mean(eval_result['point_t_errors']) * 1000 
        chest = np.mean(eval_result['chest']) * 1000 
        waist = np.mean(eval_result['waist']) * 1000 
        hips = np.mean(eval_result['hips']) * 1000 
        height = np.mean(eval_result['height']) * 1000 
        mass = np.mean(eval_result['mass']) 

        print('======SHAPY-val======')
        print('Height Error: %.2f mm' % height)
        print('Chest Error: %.2f mm' % chest)
        print('Waist Error: %.2f mm' % waist)
        print('Hips Error: %.2f mm' % hips)
        print('P2P-20k Error: %.2f mm' % point_t_errors)
        print('V2V Error: %.2f mm' % v2v_t_errors)
        print('Mass Error: %.2f kg' % mass)

        f = open(os.path.join(cfg.result_dir, 'result.txt'), 'w')
        f.write(f'SHAPY-val dataset: \n')
        f.write('Height Error: %.2f mm\n' % height)
        f.write('Chest Error: %.2f mm' % chest)
        f.write('Waist Error: %.2f mm\n' % waist)
        f.write('Hips Error: %.2f mm\n' % hips)
        f.write('P2P-20k Error: %.2f mm' % point_t_errors)
        f.write('V2V Error: %.2f mm\n' % v2v_t_errors)
        f.write('Mass Error: %.2f kg\n' % mass)
        f.close()
