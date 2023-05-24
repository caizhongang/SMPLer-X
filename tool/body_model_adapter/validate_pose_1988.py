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
import matplotlib.pyplot as plt
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
    gender = np.array(meta['gender'])[:1000]
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


def analyse_humandata_betas_distribution(load_path, save_path):
    """ Ref: train_body_shape_adapter.py """
    print('load_path:', load_path)
    ann = np.load(load_path, allow_pickle=True)
    smplx = ann['smplx'].item()
    smplx = {k: torch.from_numpy(v).float().to(device) for k, v in smplx.items()}

    meta = ann['meta'].item()
    gender = np.array(meta['gender'])
    female_idx = gender == 'female'
    male_idx = gender == 'male'

    betas = smplx['betas']
    betas_all_male = betas[male_idx]
    betas_all_female = betas[female_idx]

    fig, axs = plt.subplots(2, 5, figsize=(10, 10))
    for i in range(2):
        for j in range(5):
            idx = i * 5 + j
            beta_male, beta_female = betas_all_male[:, idx], betas_all_female[:, idx]
            male_min, male_max = beta_male.min(), beta_male.max()
            female_min, female_max = beta_female.min(), beta_female.max()

            bins = np.histogram(np.hstack((beta_male, beta_female)), bins=100)[1]  # get the bin edges
            axs[i, j].hist(beta_male, bins=bins, alpha=0.5, density=False, label='male')
            axs[i, j].hist(beta_female, bins=bins, alpha=0.5, density=False, label='female')

            print(f'smplx betas', idx, 'male:[', male_min, ',', male_max,']; female:[', female_min, ',', female_max, ']')
            axs[i, j].set_title(f'beta {idx}')
            axs[i, j].legend()

    plt.savefig(save_path)


if __name__ == '__main__':
    egobody_load_path = '/mnt/cache/share_data/caizhongang/data/preprocessed_datasets/egobody_egocentric_train_230425_065_fix_betas.npz'
    egobody_save_path = '/mnt/cache/caizhongang/osx/tool/body_model_adapter/vis_betas_distribution/egobody.png'
    validate_humandata(egobody_load_path)
    analyse_humandata_betas_distribution(egobody_load_path, egobody_save_path)