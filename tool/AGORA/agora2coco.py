import json
import torch
import numpy as np
import os.path as osp
from glob import glob
from tqdm import tqdm
import cv2
import pickle
import smplx
import os
import pathlib
import argparse

def get_bbox(joint_img, joint_valid):
    x_img, y_img = joint_img[:,0], joint_img[:,1]
    x_img = x_img[joint_valid==1]; y_img = y_img[joint_valid==1];
    xmin = min(x_img); ymin = min(y_img); xmax = max(x_img); ymax = max(y_img);

    x_center = (xmin+xmax)/2.; width = xmax-xmin;
    xmin = x_center - 0.5*width*1.2
    xmax = x_center + 0.5*width*1.2
    
    y_center = (ymin+ymax)/2.; height = ymax-ymin;
    ymin = y_center - 0.5*height*1.2
    ymax = y_center + 0.5*height*1.2

    bbox = np.array([xmin, ymin, xmax - xmin, ymax - ymin]).astype(np.float32)
    return bbox

def load_obj(file_name):
    v = []
    obj_file = open(file_name)
    for line in obj_file:
        words = line.split(' ')
        if words[0] == 'v':
            x,y,z = float(words[1]), float(words[2]), float(words[3])
            v.append(np.array([x,y,z]))
    return np.stack(v)

def save_obj(v, f, file_name='output.obj'):
    obj_file = open(file_name, 'w')
    for i in range(len(v)):
        obj_file.write('v ' + str(v[i][0]) + ' ' + str(v[i][1]) + ' ' + str(v[i][2]) + '\n')
    for i in range(len(f)):
        obj_file.write('f ' + str(f[i][0]+1) + '/' + str(f[i][0]+1) + ' ' + str(f[i][1]+1) + '/' + str(f[i][1]+1) + ' ' + str(f[i][2]+1) + '/' + str(f[i][2]+1) + '\n')
    obj_file.close()

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str, dest='dataset_path')
    parser.add_argument('--human_model_path', type=str, dest='human_model_path')
    args = parser.parse_args()

    if not args.dataset_path:
        assert 0, "Please set dataset_path"
    if not args.human_model_path:
        assert 0, "Please set human_model_path"

    return args

args = parse_args()
root_path = args.root_path
smplx_layer = smplx.create(args.human_model_path, 'smplx', use_pca=False)

image_id = 0
ann_id = 0
gt_joints_2d_path = './gt_joints_2d'
gt_joints_3d_path = './gt_joints_3d'
gt_verts_path = './gt_verts'

smplx_joints_name= \
    ('Pelvis', 'L_Hip', 'R_Hip', 'Spine_1', 'L_Knee', 'R_Knee', 'Spine_2', 'L_Ankle', 'R_Ankle', 'Spine_3', 'L_Foot', 'R_Foot', 'Neck', 'L_Collar', 'R_Collar', 'Head', 'L_Shoulder', 'R_Shoulder', 'L_Elbow', 'R_Elbow', 'L_Wrist', 'R_Wrist',  # body
    'Jaw', 'L_Eye_SMPLH', 'R_Eye_SMPLH',  # SMPLH
    'L_Index_1', 'L_Index_2', 'L_Index_3', 'L_Middle_1', 'L_Middle_2', 'L_Middle_3', 'L_Pinky_1', 'L_Pinky_2', 'L_Pinky_3', 'L_Ring_1', 'L_Ring_2', 'L_Ring_3', 'L_Thumb_1', 'L_Thumb_2', 'L_Thumb_3',  # fingers
    'R_Index_1', 'R_Index_2', 'R_Index_3', 'R_Middle_1', 'R_Middle_2', 'R_Middle_3', 'R_Pinky_1', 'R_Pinky_2', 'R_Pinky_3', 'R_Ring_1', 'R_Ring_2', 'R_Ring_3', 'R_Thumb_1', 'R_Thumb_2', 'R_Thumb_3',  # fingers
    'Nose', 'R_Eye', 'L_Eye', 'R_Ear', 'L_Ear',  # face in body
    'L_Big_toe', 'L_Small_toe', 'L_Heel', 'R_Big_toe', 'R_Small_toe', 'R_Heel',  # feet
    'L_Thumb_4', 'L_Index_4', 'L_Middle_4', 'L_Ring_4', 'L_Pinky_4',  # finger tips
    'R_Thumb_4', 'R_Index_4', 'R_Middle_4', 'R_Ring_4', 'R_Pinky_4', # finger tips
    *['Face_' + str(i) for i in range(5,56)] # face
    )
smplx_joint_part = {
            'body': list(range(smplx_joints_name.index('Pelvis'), smplx_joints_name.index('R_Eye_SMPLH')+1)) + list(range(smplx_joints_name.index('Nose'), smplx_joints_name.index('R_Heel')+1)),
            'lhand': list(range(smplx_joints_name.index('L_Index_1'), smplx_joints_name.index('L_Thumb_3')+1)) + list(range(smplx_joints_name.index('L_Thumb_4'), smplx_joints_name.index('L_Pinky_4')+1)),
            'rhand': list(range(smplx_joints_name.index('R_Index_1'), smplx_joints_name.index('R_Thumb_3')+1)) + list(range(smplx_joints_name.index('R_Thumb_4'), smplx_joints_name.index('R_Pinky_4')+1)),
            'face': list(range(smplx_joints_name.index('Face_5'), smplx_joints_name.index('Face_55')+1))}
smpl_joints_name = ('Pelvis', 'L_Hip', 'R_Hip', 'Spine_1', 'L_Knee', 'R_Knee', 'Spine_2', 'L_Ankle', 'R_Ankle', 'Spine_3', 'L_Foot', 'R_Foot', 'Neck', 'L_Collar', 'R_Collar', 'Head', 'L_Shoulder', 'R_Shoulder', 'L_Elbow', 'R_E     lbow', 'L_Wrist', 'R_Wrist', 'L_Hand', 'R_Hand', 'Nose', 'R_Eye', 'L_Eye', 'R_Ear', 'L_Ear', 'L_Big_toe', 'L_Small_toe', 'L_Heel', 'R_Big_toe', 'R_Small_toe', 'R_Heel', 'L_Thumb_4', 'L_Index_4', 'L_Middle_4', 'L_Ring_4', 'L_Pinky_4', 'R_Thumb_4', 'R_Index_4', 'R_Middle_4', 'R_Ring_4', 'R_Pinky_4')

pathlib.Path(osp.join(root_path, gt_joints_2d_path, 'smplx')).mkdir(parents=True, exist_ok=True)
pathlib.Path(osp.join(root_path, gt_joints_3d_path, 'smplx')).mkdir(parents=True, exist_ok=True)
pathlib.Path(osp.join(root_path, gt_verts_path, 'smplx')).mkdir(parents=True, exist_ok=True)
pathlib.Path(osp.join(root_path, gt_joints_2d_path, 'smpl')).mkdir(parents=True, exist_ok=True)
pathlib.Path(osp.join(root_path, gt_joints_3d_path, 'smpl')).mkdir(parents=True, exist_ok=True)
pathlib.Path(osp.join(root_path, gt_verts_path, 'smpl')).mkdir(parents=True, exist_ok=True)

for split in ('train', 'validation'):
    images = []
    annotations = []
    data_path_list = glob(osp.join(root_path, split + '_SMPLX', 'SMPLX', '*.pkl')) + glob(osp.join(root_path, split + '_SMPL', 'SMPL', '*.pkl'))
    data_path_list = sorted(data_path_list)

    for data_path in tqdm(data_path_list):
        with open(data_path, 'rb') as f:
            data_smplx = pickle.load(f, encoding='latin1')
            data_smplx = {k: list(v) for k,v in data_smplx.items()}

        with open(osp.join(root_path, split + '_SMPL', 'SMPL', data_path.split('/')[-1]), 'rb') as f:
            data_smpl = pickle.load(f, encoding='latin1')
            data_smpl = {k: list(v) for k,v in data_smpl.items()}
       
        if split == 'train':
            img_folder_name = data_path.split('/')[-1].split('_withjv')[0] # e.g., train_0
        else:
            img_folder_name = 'validation'
        img_num = len(data_smplx['imgPath'])
        
        for i in range(img_num):
            img_dict = {}
            img_dict['id'] = image_id
            img_dict['file_name_3840x2160'] = osp.join('3840x2160', img_folder_name, data_smplx['imgPath'][i])
            img_dict['file_name_1280x720'] = osp.join('1280x720', img_folder_name, data_smplx['imgPath'][i][:-4] + '_1280x720.png')
            images.append(img_dict)

            person_num = len(data_smplx['gt_path_smplx'][i])
            for j in range(person_num):
                ann_dict = {}
                ann_dict['id'] = ann_id
                ann_dict['image_id'] = image_id
                ann_dict['smplx_joints_2d_path'] = osp.join(gt_joints_2d_path, 'smplx', str(ann_id) + '.json')
                ann_dict['smplx_joints_3d_path'] = osp.join(gt_joints_3d_path, 'smplx', str(ann_id) + '.json')
                ann_dict['smplx_verts_path'] = osp.join(gt_verts_path, 'smplx', str(ann_id) + '.json')
                ann_dict['smplx_param_path'] = data_smplx['gt_path_smplx'][i][j][:-4] + '.pkl'
                ann_dict['smpl_joints_2d_path'] = osp.join(gt_joints_2d_path, 'smpl', str(ann_id) + '.json')
                ann_dict['smpl_joints_3d_path'] = osp.join(gt_joints_3d_path, 'smpl', str(ann_id) + '.json')
                ann_dict['smpl_verts_path'] = osp.join(gt_verts_path, 'smpl', str(ann_id) + '.json')
                ann_dict['smpl_param_path'] = data_smpl['gt_path_smpl'][i][j][:-4] + '.pkl'
                ann_dict['gender'] = data_smplx['gender'][i][j]
                ann_dict['kid'] = data_smplx['kid'][i][j]
                ann_dict['occlusion'] = data_smplx['occlusion'][i][j]
                ann_dict['is_valid'] = data_smplx['isValid'][i][j]
                ann_dict['age'] = data_smplx['age'][i][j]
                ann_dict['ethnicity'] = data_smplx['ethnicity'][i][j]
                
                # bbox
                joints_2d = np.array(data_smplx['gt_joints_2d'][i][j]).reshape(-1,2)
                bbox = get_bbox(joints_2d, np.ones_like(joints_2d[:,0])).reshape(4)
                ann_dict['bbox'] = bbox.tolist()
                
                joints_2d_lhand = joints_2d[smplx_joint_part['lhand'],:]
                lhand_bbox = get_bbox(joints_2d_lhand, np.ones_like(joints_2d_lhand[:,0])).reshape(4)
                ann_dict['lhand_bbox'] = lhand_bbox.tolist()

                joints_2d_rhand = joints_2d[smplx_joint_part['rhand'],:]
                rhand_bbox = get_bbox(joints_2d_rhand, np.ones_like(joints_2d_rhand[:,0])).reshape(4)
                ann_dict['rhand_bbox'] = rhand_bbox.tolist()

                joints_2d_face = joints_2d[smplx_joint_part['face'],:]
                face_bbox = get_bbox(joints_2d_face, np.ones_like(joints_2d_face[:,0])).reshape(4)
                ann_dict['face_bbox'] = face_bbox.tolist()
                annotations.append(ann_dict)

                # save smplx gts
                joints_2d = np.array(data_smplx['gt_joints_2d'][i][j]).reshape(-1,2)
                with open(osp.join(root_path, gt_joints_2d_path, 'smplx', str(ann_id) + '.json'), 'w') as f:
                    json.dump(joints_2d.tolist(), f)
                joints_3d = np.array(data_smplx['gt_joints_3d'][i][j]).reshape(-1,3)
                with open(osp.join(root_path, gt_joints_3d_path, 'smplx', str(ann_id) + '.json'), 'w') as f:
                    json.dump(joints_3d.tolist(), f)
                verts = np.array(data_smplx['gt_verts'][i][j]).reshape(-1,3)
                with open(osp.join(root_path, gt_verts_path, 'smplx', str(ann_id) + '.json'), 'w') as f:
                    json.dump(verts.tolist(), f)
                
                # save smpl gts
                joints_2d = np.array(data_smpl['gt_joints_2d'][i][j]).reshape(-1,2)
                with open(osp.join(root_path, gt_joints_2d_path, 'smpl', str(ann_id) + '.json'), 'w') as f:
                    json.dump(joints_2d.tolist(), f)
                joints_3d = np.array(data_smpl['gt_joints_3d'][i][j]).reshape(-1,3)
                with open(osp.join(root_path, gt_joints_3d_path, 'smpl', str(ann_id) + '.json'), 'w') as f:
                    json.dump(joints_3d.tolist(), f)
                verts = np.array(data_smpl['gt_verts'][i][j]).reshape(-1,3)
                with open(osp.join(root_path, gt_verts_path, 'smpl', str(ann_id) + '.json'), 'w') as f:
                    json.dump(verts.tolist(), f)

                ann_id += 1
            image_id += 1

    with open(osp.join(root_path, 'AGORA_' + split + '.json'), 'w') as f:
        json.dump({'images': images, 'annotations': annotations}, f)
        

    
