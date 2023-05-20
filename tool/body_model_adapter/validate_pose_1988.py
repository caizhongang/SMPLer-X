import torch
try:
    import open3d as o3d
except:
    print('open3d not installed.')
import glob
import pickle
import numpy as np
import tqdm
import os
import os.path as osp
import smplx
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model_path = '/mnt/cache/caizhongang/body_models'
smplx_neutral = smplx.create(model_path ,
                          model_type='smplx',
                          gender='neutral',
                          flat_hand_mean=False,
                          use_pca=False,
                          num_betas=10,
                          use_compressed=False).to(device)

smplx_male = smplx.create(model_path ,
                          model_type='smplx',
                          gender='male',
                          flat_hand_mean=False,
                          use_pca=False,
                          num_betas=10,
                          use_compressed=False).to(device)

smplx_female = smplx.create(model_path ,
                          model_type='smplx',
                          gender='female',
                          flat_hand_mean=False,
                          use_pca=False,
                          num_betas=10,
                          use_compressed=False).to(device)


def per_vertex_loss(P, G):
    dist = torch.norm(P - G, dim=-1)  # Shape: (B, N, V)
    loss = torch.mean(dist)
    return loss


def body_model_forward(body_model, betas=None, body_pose=None, global_orient=None, transl=None, expression=None,
                          jaw_pose=None, leye_pose=None, reye_pose=None, left_hand_pose=None, right_hand_pose=None,
                            return_verts=True, return_joints=True):

    B = betas.shape[0]
    output = body_model(
        betas=betas if betas is not None else torch.zeros((B, 10), device=device),
        body_pose=body_pose if body_pose is not None else torch.zeros((B, 21, 3), device=device),
        global_orient=global_orient if global_orient is not None else torch.zeros((B, 3), device=device),
        transl=transl if transl is not None else torch.zeros((B, 3), device=device),
        expression=expression if expression is not None else torch.zeros((B, 10), device=device),
        jaw_pose=jaw_pose if jaw_pose is not None else torch.zeros((B, 3), device=device),
        leye_pose=leye_pose if leye_pose is not None else torch.zeros((B, 3), device=device),
        reye_pose=reye_pose if reye_pose is not None else torch.zeros((B, 3), device=device),
        left_hand_pose=left_hand_pose if left_hand_pose is not None else torch.zeros((B, 15, 3), device=device),
        right_hand_pose=right_hand_pose if right_hand_pose is not None else torch.zeros((B, 15, 3), device=device),
        return_verts=return_verts,
        return_joints=return_joints)

    return output


def validate_humandata(load_path):
    """ on 1988 """
    print('load_path:', load_path)
    ann = np.load(load_path, allow_pickle=True)
    smplx = ann['smplx'].item()
    smplx = {k: torch.from_numpy(v[:1000]).float().to(device) for k, v in smplx.items()}

    meta = ann['meta'].item()
    gender = meta['gender']
    female_idx = gender == 'female'
    male_idx = gender == 'male'

    neutral_output = body_model_forward(smplx_neutral, betas=smplx['betas_neutral'], body_pose=smplx['body_pose'],
                                        global_orient=smplx['global_orient'], transl=smplx['transl'],
                                        expression=smplx['expression'], jaw_pose=smplx['jaw_pose'],
                                        leye_pose=smplx['leye_pose'], reye_pose=smplx['reye_pose'],
                                        left_hand_pose=smplx['left_hand_pose'], right_hand_pose=smplx['right_hand_pose'],
                                        return_verts=True, return_joints=True)

    male_output = body_model_forward(smplx_male, betas=smplx['betas'], body_pose=smplx['body_pose'],
                                        global_orient=smplx['global_orient'], transl=smplx['transl'],
                                        expression=smplx['expression'], jaw_pose=smplx['jaw_pose'],
                                        leye_pose=smplx['leye_pose'], reye_pose=smplx['reye_pose'],
                                        left_hand_pose=smplx['left_hand_pose'], right_hand_pose=smplx['right_hand_pose'],
                                        return_verts=True, return_joints=True)

    female_output = body_model_forward(smplx_male, betas=smplx['betas'], body_pose=smplx['body_pose'],
                                     global_orient=smplx['global_orient'], transl=smplx['transl'],
                                     expression=smplx['expression'], jaw_pose=smplx['jaw_pose'],
                                     leye_pose=smplx['leye_pose'], reye_pose=smplx['reye_pose'],
                                     left_hand_pose=smplx['left_hand_pose'], right_hand_pose=smplx['right_hand_pose'],
                                     return_verts=True, return_joints=True)

    neutral_verts = neutral_output.vertices
    male_verts = male_output.vertices
    female_verts = female_output.vertices
    gender_verts = male_verts.clone()
    gender_verts[female_idx] = female_verts[female_idx]
    pve = per_vertex_loss(neutral_verts, gender_verts)
    print('pve:', pve.item())

    neutral_joints = neutral_output.joints
    male_joints = male_output.joints
    female_joints = female_output.joints
    gender_joints = male_joints.clone()
    gender_joints[female_idx] = female_joints[female_idx]
    mpjpe = per_vertex_loss(neutral_joints, gender_joints)
    print('mpjpe:', mpjpe.item())


if __name__ == '__main__':
    validate_humandata('/mnt/cache/share_data/caizhongang/data/preprocessed_datasets/egobody_egocentric_train_230425_065_fix_betas.npz')
    validate_humandata('/mnt/cache/share_data/caizhongang/data/preprocessed_datasets/egobody_egocentric_train_230425_065_fix_betas.npz')
