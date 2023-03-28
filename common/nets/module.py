import torch
import torch.nn as nn
from torch.nn import functional as F
from nets.layer import make_conv_layers, make_linear_layers, make_deconv_layers
from utils.transforms import sample_joint_features, soft_argmax_2d, soft_argmax_3d
from utils.human_models import smpl_x
from config import cfg
import math
from mmcv.ops.roi_align import roi_align

class PositionNet(nn.Module):
    def __init__(self, part, feat_dim=768):
        super(PositionNet, self).__init__()
        if part == 'body':
            self.joint_num = len(smpl_x.pos_joint_part['body'])
            self.hm_shape = cfg.output_hm_shape
        elif part == 'hand':
            self.joint_num = cfg.hand_pos_joint_num
            self.hm_shape = cfg.output_hand_hm_shape
            self.hand_conv = make_conv_layers([feat_dim, 256], kernel=1, stride=1, padding=0)
        elif part == 'face':
            self.joint_num = cfg.face_pos_joint_num
            self.hm_shape = cfg.output_face_hm_shape
            self.face_conv = make_conv_layers([feat_dim, 256], kernel=1, stride=1, padding=0)

        self.conv = make_conv_layers([feat_dim, self.joint_num * self.hm_shape[0]], kernel=1, stride=1, padding=0,
                                     bnrelu_final=False)
        self.part = part

    def forward(self, img_feat):
        assert (img_feat.shape[2], img_feat.shape[3]) == (self.hm_shape[1], self.hm_shape[2])
        joint_hm = self.conv(img_feat).view(-1, self.joint_num, self.hm_shape[0], self.hm_shape[1], self.hm_shape[2])
        joint_coord = soft_argmax_3d(joint_hm)
        joint_hm = F.softmax(joint_hm.view(-1, self.joint_num, self.hm_shape[0] * self.hm_shape[1] * self.hm_shape[2]),
                             2)
        joint_hm = joint_hm.view(-1, self.joint_num, self.hm_shape[0], self.hm_shape[1], self.hm_shape[2])
        if self.part=='hand':
            img_feat = self.hand_conv(img_feat)
            img_feat_joints = sample_joint_features(img_feat, joint_coord.detach()[:, :, :2])
            return joint_hm, joint_coord, img_feat_joints
        elif self.part=='face':
            img_feat = self.face_conv(img_feat)
            img_feat_joints = sample_joint_features(img_feat, joint_coord.detach()[:, :, :2])
            return joint_hm, joint_coord, img_feat_joints
        return joint_hm, joint_coord

class HandRotationNet(nn.Module):
    def __init__(self, part, feat_dim = 768):
        super(HandRotationNet, self).__init__()
        self.part = part
        self.joint_num = cfg.hand_pos_joint_num

        self.hand_conv = make_linear_layers([feat_dim, 512], relu_final=False)
        self.hand_pose_out = make_linear_layers([self.joint_num * 515, len(smpl_x.orig_joint_part['rhand']) * 6],
                                                    relu_final=False)
        self.feat_dim = feat_dim

    def forward(self, img_feat_joints, joint_coord_img):
        batch_size = img_feat_joints.shape[0]
        # hand pose parameter
        img_feat_joints = self.hand_conv(img_feat_joints)
        feat = torch.cat((img_feat_joints, joint_coord_img), 2)  # batch_size, joint_num, 512+3
        hand_pose = self.hand_pose_out(feat.view(batch_size, -1))
        return hand_pose

class BodyRotationNet(nn.Module):
    def __init__(self, feat_dim = 768):
        super(BodyRotationNet, self).__init__()
        self.joint_num = len(smpl_x.pos_joint_part['body'])
        self.body_conv = make_linear_layers([feat_dim, 512], relu_final=False)
        self.root_pose_out = make_linear_layers([self.joint_num * (512+3), 6], relu_final=False)
        self.body_pose_out = make_linear_layers(
            [self.joint_num * (512+3), (len(smpl_x.orig_joint_part['body']) - 1) * 6], relu_final=False)  # without root
        self.shape_out = make_linear_layers([feat_dim, smpl_x.shape_param_dim], relu_final=False)
        self.cam_out = make_linear_layers([feat_dim, 3], relu_final=False)
        self.feat_dim = feat_dim

    def forward(self, body_pose_token, shape_token, cam_token, body_joint_img):
        batch_size = body_pose_token.shape[0]

        # shape parameter
        shape_param = self.shape_out(shape_token)

        # camera parameter
        cam_param = self.cam_out(cam_token)

        # body pose parameter
        body_pose_token = self.body_conv(body_pose_token)
        body_feat = torch.cat((body_pose_token, body_joint_img), 2)
        # forward to fc
        root_pose = self.root_pose_out(body_feat.view(batch_size, -1))
        body_pose = self.body_pose_out(body_feat.view(batch_size, -1))

        return root_pose, body_pose, shape_param, cam_param

class FaceRegressor(nn.Module):
    def __init__(self, feat_dim=768, joint_feat_dim=256):
        super(FaceRegressor, self).__init__()
        self.joint_num = cfg.face_pos_joint_num
        self.conv = make_linear_layers([feat_dim, 512], relu_final=False)
        self.joint_conv = make_linear_layers([joint_feat_dim, 512], relu_final=False)
        self.expr_out = make_linear_layers([self.joint_num * 515+512, smpl_x.expr_code_dim], relu_final=False)
        self.jaw_pose_out = make_linear_layers([self.joint_num * 515+512, 6], relu_final=False)

    def forward(self, face_img_feat_joints, joint_coord_img, face_feat):
        batch_size = face_feat.shape[0]
        img_feat = self.conv(face_feat.mean((2, 3)))
        img_feat_joints = self.joint_conv(face_img_feat_joints)
        img_feat_joints = torch.cat((img_feat_joints, joint_coord_img), 2)  # batch_size, joint_num, 512+3
        img_feat_joints = img_feat_joints.view(batch_size, -1)
        feat = torch.cat([img_feat, img_feat_joints], dim=1)
        expr_param = self.expr_out(feat)  # expression parameter
        jaw_pose = self.jaw_pose_out(feat)  # jaw pose parameter
        return expr_param, jaw_pose

class BoxNet(nn.Module):
    def __init__(self, feat_dim=768):
        super(BoxNet, self).__init__()
        self.joint_num = len(smpl_x.pos_joint_part['body'])
        self.deconv = make_deconv_layers([feat_dim + self.joint_num * cfg.output_hm_shape[0], 256, 256, 256])
        self.bbox_center = make_conv_layers([256, 3], kernel=1, stride=1, padding=0, bnrelu_final=False)
        self.lhand_size = make_linear_layers([256, 256, 2], relu_final=False)
        self.rhand_size = make_linear_layers([256, 256, 2], relu_final=False)
        self.face_size = make_linear_layers([256, 256, 2], relu_final=False)

    def forward(self, img_feat, joint_hm, joint_img=None):
        joint_hm = joint_hm.view(joint_hm.shape[0], joint_hm.shape[1] * cfg.output_hm_shape[0], cfg.output_hm_shape[1],
                                 cfg.output_hm_shape[2])
        img_feat = torch.cat((img_feat, joint_hm), 1)
        img_feat = self.deconv(img_feat)

        # bbox center
        bbox_center_hm = self.bbox_center(img_feat)
        bbox_center = soft_argmax_2d(bbox_center_hm)
        lhand_center, rhand_center, face_center = bbox_center[:, 0, :], bbox_center[:, 1, :], bbox_center[:, 2, :]

        # bbox size
        lhand_feat = sample_joint_features(img_feat, lhand_center[:, None, :].detach())[:, 0, :]
        lhand_size = self.lhand_size(lhand_feat)
        rhand_feat = sample_joint_features(img_feat, rhand_center[:, None, :].detach())[:, 0, :]
        rhand_size = self.rhand_size(rhand_feat)
        face_feat = sample_joint_features(img_feat, face_center[:, None, :].detach())[:, 0, :]
        face_size = self.face_size(face_feat)

        lhand_center = lhand_center / 8
        rhand_center = rhand_center / 8
        face_center = face_center / 8
        return lhand_center, lhand_size, rhand_center, rhand_size, face_center, face_size


class BoxSizeNet(nn.Module):
    def __init__(self):
        super(BoxSizeNet, self).__init__()
        self.lhand_size = make_linear_layers([256, 256, 2], relu_final=False)
        self.rhand_size = make_linear_layers([256, 256, 2], relu_final=False)
        self.face_size = make_linear_layers([256, 256, 2], relu_final=False)

    def forward(self, box_fea):
        # box_fea: [bs, 3, C]
        lhand_size = self.lhand_size(box_fea[:, 0])
        rhand_size = self.rhand_size(box_fea[:, 1])
        face_size = self.face_size(box_fea[:, 2])
        return lhand_size, rhand_size, face_size


class HandRoI(nn.Module):
    def __init__(self, feat_dim=768, upscale=4):
        super(HandRoI, self).__init__()
        self.deconv = nn.ModuleList([])
        for i in range(int(math.log2(upscale))+1):
            if i==0:
                self.deconv.append(make_conv_layers([feat_dim, feat_dim], kernel=1, stride=1, padding=0, bnrelu_final=False))
            elif i==1:
                self.deconv.append(make_deconv_layers([feat_dim, feat_dim//2]))
            elif i==2:
                self.deconv.append(make_deconv_layers([feat_dim, feat_dim//2, feat_dim//4]))
            elif i==3:
                self.deconv.append(make_deconv_layers([feat_dim, feat_dim//2, feat_dim//4, feat_dim//8]))

    def forward(self, img_feat, lhand_bbox, rhand_bbox):
        lhand_bbox = torch.cat((torch.arange(lhand_bbox.shape[0]).float().cuda()[:, None], lhand_bbox),
                               1)  # batch_idx, xmin, ymin, xmax, ymax
        rhand_bbox = torch.cat((torch.arange(rhand_bbox.shape[0]).float().cuda()[:, None], rhand_bbox),
                               1)  # batch_idx, xmin, ymin, xmax, ymax
        hand_img_feats = []
        for i, deconv in enumerate(self.deconv):
            scale = 2**i
            img_feat_i = deconv(img_feat)
            lhand_bbox_roi = lhand_bbox.clone()
            lhand_bbox_roi[:, 1] = lhand_bbox_roi[:, 1] / cfg.input_body_shape[1] * cfg.output_hm_shape[2] * scale
            lhand_bbox_roi[:, 2] = lhand_bbox_roi[:, 2] / cfg.input_body_shape[0] * cfg.output_hm_shape[1] * scale
            lhand_bbox_roi[:, 3] = lhand_bbox_roi[:, 3] / cfg.input_body_shape[1] * cfg.output_hm_shape[2] * scale
            lhand_bbox_roi[:, 4] = lhand_bbox_roi[:, 4] / cfg.input_body_shape[0] * cfg.output_hm_shape[1] * scale
            assert (cfg.output_hm_shape[1]*scale, cfg.output_hm_shape[2]*scale) == (img_feat_i.shape[2], img_feat_i.shape[3])
            lhand_img_feat = roi_align(img_feat_i, lhand_bbox_roi,
                                                       (cfg.output_hand_hm_shape[1]*scale//2,
                                                        cfg.output_hand_hm_shape[2]*scale//2),
                                                       1.0, 0, 'avg', False)
            lhand_img_feat = torch.flip(lhand_img_feat, [3])  # flip to the right hand

            rhand_bbox_roi = rhand_bbox.clone()
            rhand_bbox_roi[:, 1] = rhand_bbox_roi[:, 1] / cfg.input_body_shape[1] * cfg.output_hm_shape[2] * scale
            rhand_bbox_roi[:, 2] = rhand_bbox_roi[:, 2] / cfg.input_body_shape[0] * cfg.output_hm_shape[1] * scale
            rhand_bbox_roi[:, 3] = rhand_bbox_roi[:, 3] / cfg.input_body_shape[1] * cfg.output_hm_shape[2] * scale
            rhand_bbox_roi[:, 4] = rhand_bbox_roi[:, 4] / cfg.input_body_shape[0] * cfg.output_hm_shape[1] * scale
            rhand_img_feat = roi_align(img_feat_i, rhand_bbox_roi,
                                                       (cfg.output_hand_hm_shape[1]*scale//2,
                                                        cfg.output_hand_hm_shape[2]*scale//2),
                                                       1.0, 0, 'avg', False)
            hand_img_feat = torch.cat((lhand_img_feat, rhand_img_feat))  # [bs, c, cfg.output_hand_hm_shape[2]*scale, cfg.output_hand_hm_shape[1]*scale]
            hand_img_feats.append(hand_img_feat)
        return hand_img_feats[::-1]   # high resolution -> low resolution

class FaceRoI(nn.Module):
    def __init__(self, feat_dim=768, upscale=4):
        super(FaceRoI, self).__init__()
        self.deconv = nn.ModuleList([])
        for i in range(int(math.log2(upscale))+1):
            if i==0:
                self.deconv.append(make_conv_layers([feat_dim, feat_dim], kernel=1, stride=1, padding=0, bnrelu_final=False))
            elif i==1:
                self.deconv.append(make_deconv_layers([feat_dim, feat_dim//2]))
            elif i==2:
                self.deconv.append(make_deconv_layers([feat_dim, feat_dim//2, feat_dim//4]))
            elif i==3:
                self.deconv.append(make_deconv_layers([feat_dim, feat_dim//2, feat_dim//4, feat_dim//8]))

    def forward(self, img_feat, face_bbox):
        face_bbox = torch.cat((torch.arange(face_bbox.shape[0]).float().cuda()[:, None], face_bbox),
                               1)  # batch_idx, xmin, ymin, xmax, ymax
        face_img_feats = []
        for i, deconv in enumerate(self.deconv):
            scale = 2**i
            img_feat_i = deconv(img_feat)
            face_bbox_roi = face_bbox.clone()
            face_bbox_roi[:, 1] = face_bbox_roi[:, 1] / cfg.input_body_shape[1] * cfg.output_hm_shape[2] * scale
            face_bbox_roi[:, 2] = face_bbox_roi[:, 2] / cfg.input_body_shape[0] * cfg.output_hm_shape[1] * scale
            face_bbox_roi[:, 3] = face_bbox_roi[:, 3] / cfg.input_body_shape[1] * cfg.output_hm_shape[2] * scale
            face_bbox_roi[:, 4] = face_bbox_roi[:, 4] / cfg.input_body_shape[0] * cfg.output_hm_shape[1] * scale
            assert (cfg.output_hm_shape[1]*scale, cfg.output_hm_shape[2]*scale) == (img_feat_i.shape[2], img_feat_i.shape[3])
            face_img_feat = roi_align(img_feat_i, face_bbox_roi,
                                                       (cfg.output_face_hm_shape[1]*scale//2,
                                                        cfg.output_face_hm_shape[2]*scale//2),
                                                       1.0, 0, 'avg', False)
            face_img_feats.append(face_img_feat)
        return face_img_feats[::-1]   # high resolution -> low resolution
