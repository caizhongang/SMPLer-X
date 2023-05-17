import torch
import torch.nn as nn
from torch.nn import functional as F
from nets.module import PositionNet, HandRotationNet, FaceRegressor, BoxNet, BoxSizeNet, HandRoI, FaceRoI, BodyRotationNet
from nets.loss import CoordLoss, ParamLoss, CELoss
from utils.human_models import smpl_x
from utils.transforms import rot6d_to_axis_angle, restore_bbox
from config import cfg
import math
import copy
from mmpose.models import build_posenet
from mmcv import Config

class Model(nn.Module):
    def __init__(self, encoder, body_position_net, body_rotation_net, box_net, hand_position_net, hand_roi_net, hand_decoder,
                 hand_rotation_net, face_position_net, face_roi_net, face_decoder, face_regressor):
        super(Model, self).__init__()
        # body
        self.encoder = encoder
        self.body_position_net = body_position_net
        self.body_regressor = body_rotation_net
        self.box_net = box_net

        # hand
        self.hand_roi_net = hand_roi_net
        self.hand_position_net = hand_position_net
        self.hand_decoder = hand_decoder
        self.hand_regressor = hand_rotation_net

        # face
        self.face_roi_net = face_roi_net
        self.face_position_net = face_position_net
        self.face_decoder = face_decoder
        self.face_regressor = face_regressor

        self.smplx_layer = copy.deepcopy(smpl_x.layer['neutral']).cuda()

        self.coord_loss = CoordLoss()
        self.param_loss = ParamLoss()
        self.ce_loss = CELoss()

        self.body_num_joints = len(smpl_x.pos_joint_part['body'])
        self.hand_num_joints = len(smpl_x.pos_joint_part['rhand'])

        self.trainable_modules = [self.encoder, self.body_position_net, self.body_regressor,
                                  self.box_net, self.hand_position_net, self.hand_roi_net, self.hand_regressor,
                                  self.face_regressor, self.face_roi_net, self.face_position_net]
        self.special_trainable_modules = [self.hand_decoder, self.face_decoder]

    def get_camera_trans(self, cam_param):
        # camera translation
        t_xy = cam_param[:, :2]
        gamma = torch.sigmoid(cam_param[:, 2])  # apply sigmoid to make it positive
        k_value = torch.FloatTensor([math.sqrt(cfg.focal[0] * cfg.focal[1] * cfg.camera_3d_size * cfg.camera_3d_size / (
                cfg.input_body_shape[0] * cfg.input_body_shape[1]))]).cuda().view(-1)
        t_z = k_value * gamma
        cam_trans = torch.cat((t_xy, t_z[:, None]), 1)
        return cam_trans

    def get_coord(self, root_pose, body_pose, lhand_pose, rhand_pose, jaw_pose, shape, expr, cam_trans, mode):
        batch_size = root_pose.shape[0]
        zero_pose = torch.zeros((1, 3)).float().cuda().repeat(batch_size, 1)  # eye poses
        output = self.smplx_layer(betas=shape, body_pose=body_pose, global_orient=root_pose, right_hand_pose=rhand_pose,
                                  left_hand_pose=lhand_pose, jaw_pose=jaw_pose, leye_pose=zero_pose,
                                  reye_pose=zero_pose, expression=expr)
        # camera-centered 3D coordinate
        mesh_cam = output.vertices
        if mode == 'test' and cfg.testset == 'AGORA':  # use 144 joints for AGORA evaluation
            joint_cam = output.joints
        else:
            joint_cam = output.joints[:, smpl_x.joint_idx, :]

        # project 3D coordinates to 2D space
        if mode == 'train' and len(cfg.trainset_3d) == 1 and cfg.trainset_3d[0] == 'AGORA' and len(
                cfg.trainset_2d) == 0:  # prevent gradients from backpropagating to SMPLX paraemter regression module
            x = (joint_cam[:, :, 0].detach() + cam_trans[:, None, 0]) / (
                    joint_cam[:, :, 2].detach() + cam_trans[:, None, 2] + 1e-4) * cfg.focal[0] + cfg.princpt[0]
            y = (joint_cam[:, :, 1].detach() + cam_trans[:, None, 1]) / (
                    joint_cam[:, :, 2].detach() + cam_trans[:, None, 2] + 1e-4) * cfg.focal[1] + cfg.princpt[1]
        else:
            x = (joint_cam[:, :, 0] + cam_trans[:, None, 0]) / (joint_cam[:, :, 2] + cam_trans[:, None, 2] + 1e-4) * \
                cfg.focal[0] + cfg.princpt[0]
            y = (joint_cam[:, :, 1] + cam_trans[:, None, 1]) / (joint_cam[:, :, 2] + cam_trans[:, None, 2] + 1e-4) * \
                cfg.focal[1] + cfg.princpt[1]
        x = x / cfg.input_body_shape[1] * cfg.output_hm_shape[2]
        y = y / cfg.input_body_shape[0] * cfg.output_hm_shape[1]
        joint_proj = torch.stack((x, y), 2)

        # root-relative 3D coordinates
        root_cam = joint_cam[:, smpl_x.root_joint_idx, None, :]
        joint_cam = joint_cam - root_cam
        mesh_cam = mesh_cam + cam_trans[:, None, :]  # for rendering

        # left hand root (left wrist)-relative 3D coordinatese
        lhand_idx = smpl_x.joint_part['lhand']
        lhand_cam = joint_cam[:, lhand_idx, :]
        lwrist_cam = joint_cam[:, smpl_x.lwrist_idx, None, :]
        lhand_cam = lhand_cam - lwrist_cam
        joint_cam = torch.cat((joint_cam[:, :lhand_idx[0], :], lhand_cam, joint_cam[:, lhand_idx[-1] + 1:, :]), 1)

        # right hand root (right wrist)-relative 3D coordinatese
        rhand_idx = smpl_x.joint_part['rhand']
        rhand_cam = joint_cam[:, rhand_idx, :]
        rwrist_cam = joint_cam[:, smpl_x.rwrist_idx, None, :]
        rhand_cam = rhand_cam - rwrist_cam
        joint_cam = torch.cat((joint_cam[:, :rhand_idx[0], :], rhand_cam, joint_cam[:, rhand_idx[-1] + 1:, :]), 1)

        # face root (neck)-relative 3D coordinates
        face_idx = smpl_x.joint_part['face']
        face_cam = joint_cam[:, face_idx, :]
        neck_cam = joint_cam[:, smpl_x.neck_idx, None, :]
        face_cam = face_cam - neck_cam
        joint_cam = torch.cat((joint_cam[:, :face_idx[0], :], face_cam, joint_cam[:, face_idx[-1] + 1:, :]), 1)

        return joint_proj, joint_cam, mesh_cam

    def generate_mesh_gt(self, targets, mode):
        if 'smplx_mesh_cam' in targets:
            return targets['smplx_mesh_cam']
        nums = [3, 63, 45, 45, 3]
        accu = []
        temp = 0
        for num in nums:
            temp += num
            accu.append(temp)
        pose = targets['smplx_pose']
        root_pose, body_pose, lhand_pose, rhand_pose, jaw_pose = \
            pose[:, :accu[0]], pose[:, accu[0]:accu[1]], pose[:, accu[1]:accu[2]], pose[:, accu[2]:accu[3]], pose[:,accu[3]:accu[4]]
        shape = targets['smplx_shape']
        expr = targets['smplx_expr']
        cam_trans = targets['smplx_cam_trans']

        # final output
        joint_proj, joint_cam, mesh_cam = self.get_coord(root_pose, body_pose, lhand_pose, rhand_pose, jaw_pose, shape,
                                                         expr, cam_trans, mode)

        return mesh_cam

    def norm2heatmap(self, input, hm_shape):
        assert input.shape[-1] in [2, 3, 4]
        if input.shape[-1] == 2:
            x, y = input[..., 0], input[..., 1]
            x = x * hm_shape[2]
            y = y * hm_shape[1]
            output = torch.stack((x, y), dim=-1)
        elif input.shape[-1] == 3:
            x, y, z = input[..., 0], input[..., 1], input[..., 2]
            x = x * hm_shape[2]
            y = y * hm_shape[1]
            z = z * hm_shape[0]
            output = torch.stack((x, y, z), dim=-1)
        elif input.shape[-1] == 4:
            x, y, w, h = input[..., 0], input[..., 1], input[..., 2], input[..., 3]
            x = x * hm_shape[2]
            y = y * hm_shape[1]
            w = w * hm_shape[2]
            h = h * hm_shape[1]
            output = torch.stack((x, y, w, h), dim=-1)
        return output

    def heatmap2norm(self, input, hm_shape):
        assert input.shape[-1] in [2, 3, 4]
        if input.shape[-1] == 2:
            x, y = input[..., 0], input[..., 1]
            x = x / hm_shape[2]
            y = y / hm_shape[1]
            output = torch.stack((x, y), dim=-1)
        elif input.shape[-1] == 3:
            x, y, z = input[..., 0], input[..., 1], input[..., 2]
            x = x / hm_shape[2]
            y = y / hm_shape[1]
            z = z / hm_shape[0]
            output = torch.stack((x, y, z), dim=-1)
        elif input.shape[-1] == 4:
            x, y, w, h = input[..., 0], input[..., 1], input[..., 2], input[..., 3]
            x = x / hm_shape[2]
            y = y / hm_shape[1]
            w = w / hm_shape[2]
            h = h / hm_shape[1]
            output = torch.stack((x, y, w, h), dim=-1)

        return output

    def bbox_split(self, bbox):
        # bbox:[bs, 3, 3]
        lhand_bbox_center, rhand_bbox_center, face_bbox_center = \
            bbox[:, 0, :2], bbox[:, 1, :2], bbox[:, 2, :2]
        return lhand_bbox_center, rhand_bbox_center, face_bbox_center

    def forward(self, inputs, targets, meta_info, mode):

        body_img = F.interpolate(inputs['img'], cfg.input_body_shape)

        # 1. Encoder
        img_feat, task_tokens = self.encoder(body_img)  # task_token:[bs, N, c]
        shape_token, cam_token, expr_token, jaw_pose_token, hand_token, body_pose_token = \
            task_tokens[:, 0], task_tokens[:, 1], task_tokens[:, 2], task_tokens[:, 3], task_tokens[:, 4:6], task_tokens[:, 6:]

        # 2. Body Regressor
        body_joint_hm, body_joint_img = self.body_position_net(img_feat)
        root_pose, body_pose, shape, cam_param, = self.body_regressor(body_pose_token, shape_token, cam_token, body_joint_img.detach())
        root_pose = rot6d_to_axis_angle(root_pose)
        body_pose = rot6d_to_axis_angle(body_pose.reshape(-1, 6)).reshape(body_pose.shape[0], -1)  # (N, J_R*3)
        cam_trans = self.get_camera_trans(cam_param)

        # 3. Hand and Face BBox Estimation
        lhand_bbox_center, lhand_bbox_size, rhand_bbox_center, rhand_bbox_size, face_bbox_center, face_bbox_size = self.box_net(img_feat, body_joint_hm.detach())
        lhand_bbox = restore_bbox(lhand_bbox_center, lhand_bbox_size, cfg.input_hand_shape[1] / cfg.input_hand_shape[0], 2.0).detach()  # xyxy in (cfg.input_body_shape[1], cfg.input_body_shape[0]) space
        rhand_bbox = restore_bbox(rhand_bbox_center, rhand_bbox_size, cfg.input_hand_shape[1] / cfg.input_hand_shape[0], 2.0).detach()  # xyxy in (cfg.input_body_shape[1], cfg.input_body_shape[0]) space
        face_bbox = restore_bbox(face_bbox_center, face_bbox_size, cfg.input_face_shape[1] / cfg.input_face_shape[0], 1.5).detach()  # xyxy in (cfg.input_body_shape[1], cfg.input_body_shape[0]) space

        # 4. Differentiable Feature-level Hand/Face Crop-Upsample
        # hand_feat: list, [bsx2, c, cfg.output_hand_hm_shape[1]*scale, cfg.output_hand_hm_shape[2]*scale]
        hand_feats = self.hand_roi_net(img_feat, lhand_bbox, rhand_bbox)  # list, hand_feat: flipped left hand + right hand
        # face_feat: list, [bs, c, cfg.output_face_hm_shape[1]*scale, cfg.output_face_hm_shape[2]*scale]
        face_feats = self.face_roi_net(img_feat, face_bbox)

        # 4. keypoint-guided deformable decoder
        # hand keypoint-guided deformable decoder
        _, hand_joint_img, hand_img_feat_joints = self.hand_position_net(hand_feats[-2])  # (2N, J_P, 3) in (hand_hm_shape[2], hand_hm_shape[1], hand_hm_shape[0]) space
        # [-2]: scale=2, because the roi size = (hand_hm_shape*scale//2)
        hand_coord_init = self.heatmap2norm(hand_joint_img, cfg.output_hand_hm_shape)
        hand_img_feat_joints = self.hand_decoder(hand_feats, coord_init=hand_coord_init.detach(), query_init=hand_img_feat_joints)
        # hand regression head
        hand_pose = self.hand_regressor(hand_img_feat_joints, hand_joint_img.detach())
        hand_pose = rot6d_to_axis_angle(hand_pose.reshape(-1, 6)).reshape(hand_img_feat_joints.shape[0], -1)  # (2N, J_R*3)
        # restore flipped left hand joint coordinates
        batch_size = hand_joint_img.shape[0] // 2
        lhand_joint_img = hand_joint_img[:batch_size, :, :]
        lhand_joint_img = torch.cat(
            (cfg.output_hand_hm_shape[2] - 1 - lhand_joint_img[:, :, 0:1], lhand_joint_img[:, :, 1:]), 2)
        rhand_joint_img = hand_joint_img[batch_size:, :, :]
        # restore flipped left hand joint rotations
        batch_size = hand_pose.shape[0] // 2
        lhand_pose = hand_pose[:batch_size, :].reshape(-1, len(smpl_x.orig_joint_part['lhand']), 3)
        lhand_pose = torch.cat((lhand_pose[:, :, 0:1], -lhand_pose[:, :, 1:3]), 2).view(batch_size, -1)
        rhand_pose = hand_pose[batch_size:, :]

        # face keypoint-guided deformable decoder
        _, face_joint_img, face_img_feat_joints = self.face_position_net(face_feats[-2])  # (N, J_P, 3) in (face_hm_shape[2], face_hm_shape[1], face_hm_shape[0]) space
        face_coord_init = self.heatmap2norm(face_joint_img, cfg.output_face_hm_shape)
        face_img_feat_joints = self.face_decoder(face_feats, coord_init=face_coord_init.detach(), query_init=face_img_feat_joints)
        # face regression head
        expr, jaw_pose = self.face_regressor(face_img_feat_joints, face_joint_img.detach(), face_feats[-1])
        jaw_pose = rot6d_to_axis_angle(jaw_pose)

        # final output
        joint_proj, joint_cam, mesh_cam = self.get_coord(root_pose, body_pose, lhand_pose, rhand_pose, jaw_pose, shape, expr, cam_trans, mode)
        pose = torch.cat((root_pose, body_pose, lhand_pose, rhand_pose, jaw_pose), 1)
        joint_img = torch.cat((body_joint_img, lhand_joint_img, rhand_joint_img), 1)

        if mode == 'test' and 'smplx_pose' in targets:
            mesh_pseudo_gt = self.generate_mesh_gt(targets, mode)

        if mode == 'train':
            # loss functions
            loss = {}

            smplx_kps_3d_weight = getattr(cfg, 'smplx_kps_3d_weight', 1.0)
            smplx_kps_3d_weight = getattr(cfg, 'smplx_kps_weight', smplx_kps_3d_weight) # old config

            smplx_kps_2d_weight = getattr(cfg, 'smplx_kps_2d_weight', 1.0)
            net_kps_2d_weight = getattr(cfg, 'net_kps_2d_weight', 1.0)

            smplx_pose_weight = getattr(cfg, 'smplx_pose_weight', 1.0)
            smplx_shape_weight = getattr(cfg, 'smplx_loss_weight', 1.0)
            # smplx_orient_weight = getattr(cfg, 'smplx_orient_weight', smplx_pose_weight) # if not specified, use the same weight as pose
    

            # do not supervise root pose if original agora json is used
            if getattr(cfg, 'agora_fix_global_orient_transl', False):
                # loss['smplx_pose'] = self.param_loss(pose, targets['smplx_pose'], meta_info['smplx_pose_valid'])[:, 3:] * smplx_pose_weight
                if hasattr(cfg, 'smplx_orient_weight'):
                    smplx_orient_weight = getattr(cfg, 'smplx_orient_weight')
                    loss['smplx_orient'] = self.param_loss(pose, targets['smplx_pose'], meta_info['smplx_pose_valid'])[:, :3] * smplx_orient_weight

                loss['smplx_pose'] = self.param_loss(pose, targets['smplx_pose'], meta_info['smplx_pose_valid']) * smplx_pose_weight
            else:
                loss['smplx_pose'] = self.param_loss(pose, targets['smplx_pose'], meta_info['smplx_pose_valid'])[:, 3:] * smplx_pose_weight

            loss['smplx_shape'] = self.param_loss(shape, targets['smplx_shape'],
                                                  meta_info['smplx_shape_valid'][:, None]) * smplx_shape_weight 
            loss['smplx_expr'] = self.param_loss(expr, targets['smplx_expr'], meta_info['smplx_expr_valid'][:, None])

            loss['joint_cam'] = self.coord_loss(joint_cam, targets['joint_cam'], meta_info['joint_valid'] * meta_info['is_3D'][:, None, None]) * smplx_kps_3d_weight
            loss['smplx_joint_cam'] = self.coord_loss(joint_cam, targets['smplx_joint_cam'], meta_info['smplx_joint_valid']) * smplx_kps_3d_weight

            loss['lhand_bbox'] = (self.coord_loss(lhand_bbox_center, targets['lhand_bbox_center'], meta_info['lhand_bbox_valid'][:, None]) +
                                  self.coord_loss(lhand_bbox_size, targets['lhand_bbox_size'], meta_info['lhand_bbox_valid'][:, None]))
            loss['rhand_bbox'] = (self.coord_loss(rhand_bbox_center, targets['rhand_bbox_center'], meta_info['rhand_bbox_valid'][:, None]) +
                                  self.coord_loss(rhand_bbox_size, targets['rhand_bbox_size'], meta_info['rhand_bbox_valid'][:, None]))
            loss['face_bbox'] = (self.coord_loss(face_bbox_center, targets['face_bbox_center'], meta_info['face_bbox_valid'][:, None]) +
                                 self.coord_loss(face_bbox_size, targets['face_bbox_size'], meta_info['face_bbox_valid'][:, None]))
            
            if getattr(cfg, 'save_vis', False):
                out = {}
                targets['original_joint_img'] = targets['joint_img'].clone()
                out['original_joint_proj'] = joint_proj.clone()

            # change hand target joint_img and joint_trunc according to hand bbox (cfg.output_hm_shape -> downsampled hand bbox space)
            for part_name, bbox in (('lhand', lhand_bbox), ('rhand', rhand_bbox)):
                for coord_name, trunc_name in (('joint_img', 'joint_trunc'), ('smplx_joint_img', 'smplx_joint_trunc')):
                    x = targets[coord_name][:, smpl_x.joint_part[part_name], 0]
                    y = targets[coord_name][:, smpl_x.joint_part[part_name], 1]
                    z = targets[coord_name][:, smpl_x.joint_part[part_name], 2]
                    trunc = meta_info[trunc_name][:, smpl_x.joint_part[part_name], 0]

                    x -= (bbox[:, None, 0] / cfg.input_body_shape[1] * cfg.output_hm_shape[2])
                    x *= (cfg.output_hand_hm_shape[2] / (
                            (bbox[:, None, 2] - bbox[:, None, 0]) / cfg.input_body_shape[1] * cfg.output_hm_shape[
                        2]))
                    y -= (bbox[:, None, 1] / cfg.input_body_shape[0] * cfg.output_hm_shape[1])
                    y *= (cfg.output_hand_hm_shape[1] / (
                            (bbox[:, None, 3] - bbox[:, None, 1]) / cfg.input_body_shape[0] * cfg.output_hm_shape[
                        1]))
                    z *= cfg.output_hand_hm_shape[0] / cfg.output_hm_shape[0]
                    trunc *= ((x >= 0) * (x < cfg.output_hand_hm_shape[2]) * (y >= 0) * (
                            y < cfg.output_hand_hm_shape[1]))

                    coord = torch.stack((x, y, z), 2)
                    trunc = trunc[:, :, None]
                    targets[coord_name] = torch.cat((targets[coord_name][:, :smpl_x.joint_part[part_name][0], :], coord,
                                                     targets[coord_name][:, smpl_x.joint_part[part_name][-1] + 1:, :]),
                                                    1)
                    meta_info[trunc_name] = torch.cat((meta_info[trunc_name][:, :smpl_x.joint_part[part_name][0], :],
                                                       trunc,
                                                       meta_info[trunc_name][:, smpl_x.joint_part[part_name][-1] + 1:,
                                                       :]), 1)

            # change hand projected joint coordinates according to hand bbox (cfg.output_hm_shape -> hand bbox space)
            for part_name, bbox in (('lhand', lhand_bbox), ('rhand', rhand_bbox)):
                x = joint_proj[:, smpl_x.joint_part[part_name], 0]
                y = joint_proj[:, smpl_x.joint_part[part_name], 1]

                x -= (bbox[:, None, 0] / cfg.input_body_shape[1] * cfg.output_hm_shape[2])
                x *= (cfg.output_hand_hm_shape[2] / (
                        (bbox[:, None, 2] - bbox[:, None, 0]) / cfg.input_body_shape[1] * cfg.output_hm_shape[2]))
                y -= (bbox[:, None, 1] / cfg.input_body_shape[0] * cfg.output_hm_shape[1])
                y *= (cfg.output_hand_hm_shape[1] / (
                        (bbox[:, None, 3] - bbox[:, None, 1]) / cfg.input_body_shape[0] * cfg.output_hm_shape[1]))

                coord = torch.stack((x, y), 2)
                trans = []
                for bid in range(coord.shape[0]):
                    mask = meta_info['joint_trunc'][bid, smpl_x.joint_part[part_name], 0] == 1
                    if torch.sum(mask) == 0:
                        trans.append(torch.zeros((2)).float().cuda())
                    else:
                        trans.append((-coord[bid, mask, :2] + targets['joint_img'][:, smpl_x.joint_part[part_name], :][
                                                              bid, mask, :2]).mean(0))
                trans = torch.stack(trans)[:, None, :]
                coord = coord + trans  # global translation alignment
                joint_proj = torch.cat((joint_proj[:, :smpl_x.joint_part[part_name][0], :], coord,
                                        joint_proj[:, smpl_x.joint_part[part_name][-1] + 1:, :]), 1)

            # change face projected joint coordinates according to face bbox (cfg.output_hm_shape -> face bbox space)
            coord = joint_proj[:, smpl_x.joint_part['face'], :]
            trans = []
            for bid in range(coord.shape[0]):
                mask = meta_info['joint_trunc'][bid, smpl_x.joint_part['face'], 0] == 1
                if torch.sum(mask) == 0:
                    trans.append(torch.zeros((2)).float().cuda())
                else:
                    trans.append((-coord[bid, mask, :2] + targets['joint_img'][:, smpl_x.joint_part['face'], :][bid,
                                                          mask, :2]).mean(0))
            trans = torch.stack(trans)[:, None, :]
            coord = coord + trans  # global translation alignment
            joint_proj = torch.cat((joint_proj[:, :smpl_x.joint_part['face'][0], :], coord,
                                    joint_proj[:, smpl_x.joint_part['face'][-1] + 1:, :]), 1)

            loss['joint_proj'] = self.coord_loss(joint_proj, targets['joint_img'][:, :, :2], meta_info['joint_trunc']) * smplx_kps_2d_weight
            
            loss['joint_img'] = self.coord_loss(joint_img, smpl_x.reduce_joint_set(targets['joint_img']),
                                                smpl_x.reduce_joint_set(meta_info['joint_trunc']), meta_info['is_3D']) * net_kps_2d_weight
            loss['joint_img_face'] = self.coord_loss(face_joint_img, targets['joint_img'][:, smpl_x.joint_part['face']],
                                                meta_info['joint_trunc'][:, smpl_x.joint_part['face']], meta_info['is_3D']) * net_kps_2d_weight
            loss['smplx_joint_img'] = self.coord_loss(joint_img, smpl_x.reduce_joint_set(targets['smplx_joint_img']), 
                                                    smpl_x.reduce_joint_set(meta_info['smplx_joint_trunc'])) * net_kps_2d_weight
            ### HARDCODE vis for debug
            if getattr(cfg, 'save_vis', False):
                
                out['img'] = inputs['img']
                out['joint_img'] = joint_img
                out['smplx_joint_proj'] = joint_proj
                out['smplx_mesh_cam'] = mesh_cam
                out['smplx_root_pose'] = root_pose
                out['smplx_body_pose'] = body_pose
                out['smplx_lhand_pose'] = lhand_pose
                out['smplx_rhand_pose'] = rhand_pose
                out['smplx_jaw_pose'] = jaw_pose
                out['smplx_shape'] = shape
                out['smplx_expr'] = expr
                out['cam_trans'] = cam_trans
                out['lhand_bbox'] = lhand_bbox
                out['rhand_bbox'] = rhand_bbox
                out['face_bbox'] = face_bbox
                
                import numpy as np
                # np.save('./vis/train_output_18.npy', out)
                # np.save('./vis/train_target_18.npy', targets)
                # np.save('./vis/train_input_18.npy', inputs)
                # np.save('./vis/train_meta_info_18.npy', meta_info)
                for key in ['joint_cam_', 'mesh_rot_', 'joint_cam', 'mesh_orig', 'joint_cam_orig_', 'joint_cam_orig']:
                    to_save = targets[key].cpu().detach().numpy()
                    np.save(f'./vis/train_{key}.npy', to_save)

                import pdb; pdb.set_trace()

            return loss
        else:
            # change hand output joint_img according to hand bbox
            for part_name, bbox in (('lhand', lhand_bbox), ('rhand', rhand_bbox)):
                joint_img[:, smpl_x.pos_joint_part[part_name], 0] *= (
                        ((bbox[:, None, 2] - bbox[:, None, 0]) / cfg.input_body_shape[1] * cfg.output_hm_shape[2]) /
                        cfg.output_hand_hm_shape[2])
                joint_img[:, smpl_x.pos_joint_part[part_name], 0] += (
                        bbox[:, None, 0] / cfg.input_body_shape[1] * cfg.output_hm_shape[2])
                joint_img[:, smpl_x.pos_joint_part[part_name], 1] *= (
                        ((bbox[:, None, 3] - bbox[:, None, 1]) / cfg.input_body_shape[0] * cfg.output_hm_shape[1]) /
                        cfg.output_hand_hm_shape[1])
                joint_img[:, smpl_x.pos_joint_part[part_name], 1] += (
                        bbox[:, None, 1] / cfg.input_body_shape[0] * cfg.output_hm_shape[1])

            # change input_body_shape to input_img_shape
            for bbox in (lhand_bbox, rhand_bbox, face_bbox):
                bbox[:, 0] *= cfg.input_img_shape[1] / cfg.input_body_shape[1]
                bbox[:, 1] *= cfg.input_img_shape[0] / cfg.input_body_shape[0]
                bbox[:, 2] *= cfg.input_img_shape[1] / cfg.input_body_shape[1]
                bbox[:, 3] *= cfg.input_img_shape[0] / cfg.input_body_shape[0]

            # test output
            out = {}
            out['img'] = inputs['img']
            out['img_path'] = meta_info['img_path']
            out['joint_img'] = joint_img
            out['smplx_joint_proj'] = joint_proj
            out['smplx_mesh_cam'] = mesh_cam
            out['smplx_root_pose'] = root_pose
            out['smplx_body_pose'] = body_pose
            out['smplx_lhand_pose'] = lhand_pose
            out['smplx_rhand_pose'] = rhand_pose
            out['smplx_jaw_pose'] = jaw_pose
            out['smplx_shape'] = shape
            out['smplx_expr'] = expr
            out['cam_trans'] = cam_trans
            out['lhand_bbox'] = lhand_bbox
            out['rhand_bbox'] = rhand_bbox
            out['face_bbox'] = face_bbox
            if 'smplx_pose' in targets:
                out['smplx_mesh_cam_pseudo_gt'] = mesh_pseudo_gt
            if 'smplx_shape' in targets:
                out['smplx_shape_target'] = targets['smplx_shape']
            if 'smplx_mesh_cam' in targets:
                out['smplx_mesh_cam_target'] = targets['smplx_mesh_cam']
            if 'smpl_mesh_cam' in targets:
                out['smpl_mesh_cam_target'] = targets['smpl_mesh_cam']
            if 'bb2img_trans' in meta_info:
                out['bb2img_trans'] = meta_info['bb2img_trans']
            
            ### HARDCODE vis for debug
            # import numpy as np
            # np.save('./vis/val_0509_out.npy', out)
            # for key in ['smpl_mesh_cam_target', 'smplx_mesh_cam']:
            #         to_save = out[key].cpu().detach().numpy()
            #         np.save(f'./vis/val_0509_{key}.npy', to_save)
            
            return out

def init_weights(m):
    try:
        if type(m) == nn.ConvTranspose2d:
            nn.init.normal_(m.weight, std=0.001)
        elif type(m) == nn.Conv2d:
            nn.init.normal_(m.weight, std=0.001)
            nn.init.constant_(m.bias, 0)
        elif type(m) == nn.BatchNorm2d:
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif type(m) == nn.Linear:
            nn.init.normal_(m.weight, std=0.01)
            nn.init.constant_(m.bias, 0)
    except AttributeError:
        pass


def get_model(mode):
    third_party_encoder = getattr(cfg, 'third_party_encoder', None)
    if third_party_encoder is None:

        vit_cfg = Config.fromfile(cfg.encoder_config_file)
        vit = build_posenet(vit_cfg.model)
        if mode == 'train':
            encoder_pretrained_model = torch.load(cfg.encoder_pretrained_model_path)['state_dict']
            vit.load_state_dict(encoder_pretrained_model, strict=False)
            print(f"Initialize backbone from {cfg.encoder_pretrained_model_path}")
        encoder = vit.backbone

        # apply adapters
        # currently, adapters have only been tested for ViTPose
        adapter_name = getattr(cfg, 'adapter_name', None)
        if adapter_name == 'lora':
            from lora_utils import apply_adapter
            encoder = apply_adapter(encoder)
            print(f"Apply adapter {adapter_name}.")
        elif adapter_name == 'vit_adapter':
            from vit_adapter_utils import apply_adapter
            encoder = apply_adapter(encoder, cfg.model_type)
            print(f"Apply adapter {adapter_name}.")
        else:
            raise NotImplementedError('Undefined adapter: {}'.format(adapter_name))

    elif third_party_encoder == 'humanbench':
        # ref: https://github.com/OpenGVLab/HumanBench/blob/6478b659773de5de8ac6f2dd8e35d47cedc54877/PATH/core/models/backbones/vitdet_for_ladder_attention_share_pos_embed.py
        from humanbench_utils import get_backbone, load_checkpoint

        backbone = get_backbone(cfg.third_party_encoder_type)
        if mode == 'train' and hasattr(cfg, 'encoder_pretrained_model_path'):
            checkpoint = torch.load(cfg.encoder_pretrained_model_path)['model']
            load_checkpoint(backbone, checkpoint, load_pos_embed=True, strict=False)
            print(f"Initialize backbone from {cfg.encoder_pretrained_model_path}")
        else:
            print(f"Initialize backbone from random weights")
        encoder = backbone
    
    elif third_party_encoder == 'dinov2':
        from dinov2_utils import get_backbone, load_checkpoint

        backbone = get_backbone(cfg.third_party_encoder_type)
        if mode == 'train':
            load_checkpoint(backbone, cfg.encoder_pretrained_model_path)
            print(f"Initialize backbone from {cfg.encoder_pretrained_model_path}")
        encoder = backbone

    elif third_party_encoder == 'motionbert':
        raise NotImplementedError('MotionBert not supported yet.')
        from motionbert_utils import get_backbone

        backbone = get_backbone(cfg.third_party_encoder_type)
        if mode == 'train':
            checkpoint = torch.load(cfg.encoder_pretrained_model_path, map_location=lambda storage, loc: storage)
            backbone.load_state_dict(checkpoint['model_pos'], strict=True)
            print(f"Initialize backbone from {cfg.encoder_pretrained_model_path}")
        encoder = backbone

    else:
        raise NotImplementedError('Undefined third party encoder: {}'.format(third_party_encoder))

    # body
    body_position_net = PositionNet('body', feat_dim=cfg.feat_dim)
    body_rotation_net = BodyRotationNet(feat_dim=cfg.feat_dim)
    box_net = BoxNet(feat_dim=cfg.feat_dim)

    # hand
    hand_roi_net = HandRoI(feat_dim=cfg.feat_dim, upscale=cfg.upscale)
    hand_position_net = PositionNet('hand', feat_dim=cfg.feat_dim // 2)
    hand_rotation_net = HandRotationNet('hand', feat_dim=256)
    decoder_cfg = Config.fromfile('transformer_utils/configs/osx/decoder/hand_decoder.py')
    hand_decoder = build_posenet(decoder_cfg.model)

    # face
    face_roi_net = FaceRoI(feat_dim=cfg.feat_dim, upscale=cfg.upscale)
    face_position_net = PositionNet('face', feat_dim=cfg.feat_dim // 2)
    face_regressor = FaceRegressor(feat_dim=cfg.feat_dim, joint_feat_dim=256)
    decoder_cfg = Config.fromfile('transformer_utils/configs/osx/decoder/face_decoder.py')
    face_decoder = build_posenet(decoder_cfg.model)

    if mode == 'train':
        body_position_net.apply(init_weights)
        body_rotation_net.apply(init_weights)
        box_net.apply(init_weights)

        # hand
        hand_position_net.apply(init_weights)
        hand_roi_net.apply(init_weights)
        hand_rotation_net.apply(init_weights)
        hand_decoder.apply(init_weights)

        # face
        face_position_net.apply(init_weights)
        face_roi_net.apply(init_weights)
        face_decoder.apply(init_weights)
        face_regressor.apply(init_weights)

    model = Model(encoder, body_position_net, body_rotation_net, box_net, hand_position_net, hand_roi_net, hand_decoder,
                  hand_rotation_net,
                  face_position_net, face_roi_net, face_decoder, face_regressor)

    return model
