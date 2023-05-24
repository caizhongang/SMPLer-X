""" Evaluate if body pose parameters can be shared, given body shape has been adapted """
""" Need to run pip install -v -e . in vposer/ first"""

from human_body_prior.tools.model_loader import load_model
from human_body_prior.models.vposer_model import VPoser
from train_body_shape_adapter import (
    create_body_model,
    compute_smplx_vertices,
    compute_smpl_vertices,
    BetasAdapter,
    per_vertex_loss,
    smplx_range,
    smpl_range,
    apply_deformation_transfer,
    read_deformation_transfer,
    model_path
)
import torch
import smplx
try:
    import open3d as o3d
except:
    print('open3d not installed.')
import glob
import pickle
import numpy as np
import tqdm
import os.path as osp

torch.manual_seed(666)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

expr_dir = '/home/alex/github/OSX/tool/body_model_adapter/V02_05'
smplx_female_to_smplx_neutral_path = '/home/alex/github/OSX/tool/body_model_adapter/smplx_female_to_smplx_neutral.pth'
smplx_male_to_smplx_neutral_path = '/home/alex/github/OSX/tool/body_model_adapter/smplx_male_to_smplx_neutral.pth'
load_dir = '/home/alex/github/OSX/tool/body_model_adapter/'
num_poses = 1000


def validate_random_smplx():
    vp, ps = load_model(expr_dir, model_code=VPoser,
                        remove_words_in_model_weights='vp_model.',
                        disable_grad=True)
    vp = vp.to(device)

    # random body_pose
    random_body_pose = vp.sample_poses(num_poses=num_poses)['pose_body'].contiguous().view(num_poses, -1)  # (N, 63)
    random_betas = (torch.rand(num_poses, 10, device=device) - 0.5) \
                   * torch.tensor(smplx_range, device=device).reshape(1, 10) \
                   * 1.2  # 1.2: scale up a cover a bit of unseen range

    for gender in ('male', 'female'):

        smplx_gender = create_body_model(('smplx', gender)).to(device)
        smplx_neutral = create_body_model(('smplx', 'neutral')).to(device)

        type_gender_to_smplx_neutral = BetasAdapter()
        type_gender_to_smplx_neutral.load_state_dict(torch.load(osp.join(load_dir, f'smplx_{gender}_to_smplx_neutral.pth')))
        type_gender_to_smplx_neutral.to(device)

        target_betas = type_gender_to_smplx_neutral(random_betas).to(device)

        vertices_source, joints_source = compute_smplx_vertices(smplx_gender, random_betas, random_body_pose)
        vertices_target, joints_target = compute_smplx_vertices(smplx_neutral, target_betas, random_body_pose)
        vertices_target_wo_adapt, joints_target_wo_adapt = compute_smplx_vertices(smplx_neutral, random_betas, random_body_pose)

        vertices_source = vertices_source - joints_source[:, [0], :]
        vertices_target = vertices_target - joints_target[:, [0], :]
        vertices_target_wo_adapt = vertices_target_wo_adapt - joints_target_wo_adapt[:, [0], :]

        pve = per_vertex_loss(vertices_target, vertices_source)
        pve_wo_adapt = per_vertex_loss(vertices_target_wo_adapt, vertices_source)
        print('smplx', gender, 'to smplx neutral, pve =', pve.item(), 'wo adapt, pve =', pve_wo_adapt.item())

        for vert_gender, vert_neutral, vert_neutral_wo_adapt in zip(vertices_source[:10], vertices_target[:10], vertices_target_wo_adapt[:10]):
            pcd_source = o3d.geometry.PointCloud()
            pcd_source.points = o3d.utility.Vector3dVector(vert_gender.cpu().detach().numpy())
            pcd_source.paint_uniform_color([0, 1, 0])
            pcd_target = o3d.geometry.PointCloud()
            pcd_target.points = o3d.utility.Vector3dVector(vert_neutral.cpu().detach().numpy())
            pcd_target.paint_uniform_color([1, 0, 0])
            pcd_wo_adapt = o3d.geometry.PointCloud()
            pcd_wo_adapt.points = o3d.utility.Vector3dVector(vert_neutral_wo_adapt.cpu().detach().numpy())
            pcd_wo_adapt.paint_uniform_color([0, 0, 1])
            axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1, origin=[0, 0, 0])
            o3d.visualization.draw_geometries([pcd_source, pcd_target, pcd_wo_adapt, axis])


def validate_agora_smplx():

    smplx_paths = sorted(glob.glob('/home/alex/github/OSX/tool/body_model_adapter/smplx_gt/*/*.pkl'))

    smplx_female = create_body_model(('smplx', 'female')).to(device)
    smplx_male = create_body_model(('smplx', 'male')).to(device)
    smplx_neutral = create_body_model(('smplx', 'neutral')).to(device)

    smplx_male_to_smplx_neutral = BetasAdapter()
    smplx_male_to_smplx_neutral.load_state_dict(torch.load(osp.join(load_dir, f'smplx_male_to_smplx_neutral.pth')))
    smplx_male_to_smplx_neutral.to(device)

    smplx_female_to_smplx_neutral = BetasAdapter()
    smplx_female_to_smplx_neutral.load_state_dict(torch.load(osp.join(load_dir, f'smplx_female_to_smplx_neutral.pth')))
    smplx_female_to_smplx_neutral.to(device)

    results = {
        'male': [],
        'male_wo_adapt': [],
        'female': [],
        'female_wo_adapt': [],
    }
    num_kid = 0
    num_male = 0
    num_female = 0
    num_no_gender = 0

    betas_min = np.inf
    betas_max = -np.inf

    for path in tqdm.tqdm(smplx_paths):

        with open(path, 'rb') as f:
            content = pickle.load(f)

        if content['betas'].shape[1] == 11:
            num_kid += 1
            continue

        betas = torch.Tensor(content['betas'].reshape((1, 10))).to(device)
        betas_min = min(betas.min().item(), betas_min)
        betas_max = max(betas.max().item(), betas_max)

        body_pose = torch.Tensor(content['body_pose'].reshape((1, 21, 3))).to(device)
        global_orient = torch.Tensor(content['global_orient'].reshape((1, 3))).to(device)
        transl = torch.Tensor(content['transl'].reshape((1, 3))).to(device)
        expression = torch.Tensor(content['expression'].reshape((1, 10))).to(device)
        jaw_pose = torch.Tensor(content['jaw_pose'].reshape((1, 3))).to(device)
        leye_pose = torch.Tensor(content['leye_pose'].reshape((1, 3))).to(device)
        reye_pose = torch.Tensor(content['reye_pose'].reshape((1, 3))).to(device)
        left_hand_pose = torch.Tensor(content['left_hand_pose'].reshape((1, 15, 3))).to(device)
        right_hand_pose = torch.Tensor(content['right_hand_pose'].reshape((1, 15, 3))).to(device)

        try:
            gender = content['gender']
        except:
            num_no_gender += 1
            continue

        assert gender in ('male', 'female')

        if gender == 'male':
            num_male += 1
            smplx_gender = smplx_male
            smplx_gender_to_smplx_neutral = smplx_male_to_smplx_neutral
        else:
            num_female += 1
            smplx_gender = smplx_female
            smplx_gender_to_smplx_neutral = smplx_female_to_smplx_neutral

        neutral_betas_from_gender = smplx_gender_to_smplx_neutral(betas).to(device)

        vertices_gender, joints_gender = compute_smplx_vertices(smplx_gender,
                                           betas=betas,
                                           body_pose=body_pose,
                                           global_orient=global_orient,
                                           transl=transl,
                                           expression=expression,
                                           jaw_pose=jaw_pose,
                                           leye_pose=leye_pose,
                                           reye_pose=reye_pose,
                                           left_hand_pose=left_hand_pose,
                                           right_hand_pose=right_hand_pose,
                                           )
        vertices_neutral, joints_neutral = compute_smplx_vertices(smplx_neutral,
                                            betas=neutral_betas_from_gender,
                                            body_pose=body_pose,
                                            global_orient=global_orient,
                                            transl=transl,
                                            expression=expression,
                                            jaw_pose=jaw_pose,
                                            leye_pose=leye_pose,
                                            reye_pose=reye_pose,
                                            left_hand_pose=left_hand_pose,
                                            right_hand_pose=right_hand_pose,
                                            )
        vertices_neutral_wo_adapt, joints_neutral_wo_adapt = compute_smplx_vertices(smplx_neutral,
                                                     betas=betas,
                                                     body_pose=body_pose,
                                                     global_orient=global_orient,
                                                     transl=transl,
                                                     expression=expression,
                                                     jaw_pose=jaw_pose,
                                                     leye_pose=leye_pose,
                                                     reye_pose=reye_pose,
                                                     left_hand_pose=left_hand_pose,
                                                     right_hand_pose=right_hand_pose,
                                                     )

        # offset
        vertices_gender = vertices_gender - joints_gender[:, [0], :]
        vertices_neutral = vertices_neutral - joints_neutral[:, [0], :]
        vertices_neutral_wo_adapt = vertices_neutral_wo_adapt - joints_neutral_wo_adapt[:, [0], :]

        pve = per_vertex_loss(vertices_neutral, vertices_gender)
        pve_wo_adapt = per_vertex_loss(vertices_neutral_wo_adapt, vertices_gender)

        results[gender].append(pve.item())
        results[f'{gender}_wo_adapt'].append(pve_wo_adapt.item())

    print('betas min =', betas_min, 'betas max =', betas_max)
    print('num_kid =', num_kid, 'which is', num_kid / len(smplx_paths) * 100, '% of the dataset')
    print('num_male =', num_male, 'which is', num_male / len(smplx_paths) * 100, '% of the dataset')
    print('num_female =', num_female, 'which is', num_female / len(smplx_paths) * 100, '% of the dataset')
    print('num_no_gender =', num_no_gender, 'which is', num_no_gender / len(smplx_paths) * 100, '% of the dataset')

    for gender in ('male', 'female'):
        res = results[gender]
        res_wo_adapt = results[f'{gender}_wo_adapt']
        print(gender, 'to neutral, pve =', np.array(res).mean(), '; wo adapt, pve =', np.array(res_wo_adapt).mean())


def validate_random_smpl():
    vp, ps = load_model(expr_dir, model_code=VPoser,
                        remove_words_in_model_weights='vp_model.',
                        disable_grad=True)
    vp = vp.to(device)

    smplx_mask_ids_load_path = '/home/alex/github/OSX/tool/body_model_adapter/model_transfer/smplx_mask_ids.npy'
    deformation_transfer_path = '/home/alex/github/OSX/tool/body_model_adapter/model_transfer/smpl2smplx_deftrafo_setup.pkl'

    mask_ids = np.load(smplx_mask_ids_load_path)
    mask_ids = torch.from_numpy(mask_ids).to(device=device)
    def_matrix = read_deformation_transfer(deformation_transfer_path, device=device)

    # random body_pose
    random_body_pose_smplx = vp.sample_poses(num_poses=num_poses)['pose_body'].contiguous().view(num_poses, -1)  # (N, 63)
    random_body_pose = torch.concat([random_body_pose_smplx, torch.zeros((num_poses, 6), device=device)], dim=1)  # (N, 69)
    random_betas = (torch.rand(num_poses, 10, device=device) - 0.5) \
                   * torch.tensor(smpl_range, device=device).reshape(1, 10) \
                   * 1.2  # 1.2: scale up a cover a bit of unseen range

    for gender in ('neutral', 'male', 'female'):

        source_model = create_body_model(('smpl', gender)).to(device)
        # smplx_neutral = create_body_model(('smplx', 'neutral')).to(device)
        smplx_neutral = smplx.create(model_path ,
                                  model_type='smplx',
                                  gender='neutral',
                                  flat_hand_mean=True,  # important: set to True
                                  use_pca=False,
                                  num_betas=10,
                                  use_compressed=False).to(device)

        type_gender_to_smplx_neutral = BetasAdapter()
        type_gender_to_smplx_neutral.load_state_dict(torch.load(osp.join(load_dir, f'smpl_{gender}_to_smplx_neutral.pth')))
        type_gender_to_smplx_neutral.to(device)

        target_betas = type_gender_to_smplx_neutral(random_betas).to(device)

        vertices_source, joints_source = compute_smpl_vertices(source_model, random_betas, random_body_pose)
        vertices_target, joints_target = compute_smplx_vertices(smplx_neutral, target_betas, random_body_pose_smplx)
        vertices_target_wo_adapt, joints_target_wo_adapt = compute_smplx_vertices(smplx_neutral, random_betas, random_body_pose_smplx)

        vertices_source = vertices_source - joints_source[:, [0], :]
        vertices_target = vertices_target - joints_target[:, [0], :]
        vertices_target_wo_adapt = vertices_target_wo_adapt - joints_target_wo_adapt[:, [0], :]

        vertices_source = apply_deformation_transfer(def_matrix, vertices_source, None, use_normals=False)
        vertices_source = vertices_source[:, mask_ids]
        vertices_target = vertices_target[:, mask_ids]
        vertices_target_wo_adapt = vertices_target_wo_adapt[:, mask_ids]

        pve = per_vertex_loss(vertices_target, vertices_source)
        pve_wo_adapt = per_vertex_loss(vertices_target_wo_adapt, vertices_source)
        print('smpl', gender, 'to smplx neutral, pve =', pve.item(), 'wo adapt, pve =', pve_wo_adapt.item())

        for vert_gender, vert_neutral, vert_neutral_wo_adapt in zip(vertices_source[:10], vertices_target[:10], vertices_target_wo_adapt[:10]):
            pcd_source = o3d.geometry.PointCloud()
            pcd_source.points = o3d.utility.Vector3dVector(vert_gender.cpu().detach().numpy())
            pcd_source.paint_uniform_color([0, 1, 0])
            pcd_target = o3d.geometry.PointCloud()
            pcd_target.points = o3d.utility.Vector3dVector(vert_neutral.cpu().detach().numpy())
            pcd_target.paint_uniform_color([1, 0, 0])
            pcd_wo_adapt = o3d.geometry.PointCloud()
            pcd_wo_adapt.points = o3d.utility.Vector3dVector(vert_neutral_wo_adapt.cpu().detach().numpy())
            pcd_wo_adapt.paint_uniform_color([0, 0, 1])
            axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1, origin=[0, 0, 0])
            o3d.visualization.draw_geometries([pcd_source, pcd_target, pcd_wo_adapt, axis])


def validate_agora_smpl():
    """ Note: not only betas, body_pose is also adapted from smpl (69) -> smplx (21,3)=(63) """

    smpl_paths = sorted(glob.glob('/home/alex/github/OSX/tool/body_model_adapter/smpl_gt/*/*.pkl'))

    smplx_mask_ids_load_path = '/home/alex/github/OSX/tool/body_model_adapter/model_transfer/smplx_mask_ids.npy'
    deformation_transfer_path = '/home/alex/github/OSX/tool/body_model_adapter/model_transfer/smpl2smplx_deftrafo_setup.pkl'

    mask_ids = np.load(smplx_mask_ids_load_path)
    mask_ids = torch.from_numpy(mask_ids).to(device=device)
    def_matrix = read_deformation_transfer(deformation_transfer_path, device=device)

    smpl_neutral = create_body_model(('smpl', 'neutral')).to(device)
    smplx_neutral = smplx.create(model_path,
                                 model_type='smplx',
                                 gender='neutral',
                                 flat_hand_mean=True,  # important: set to True
                                 use_pca=False,
                                 num_betas=10,
                                 use_compressed=False).to(device)

    smpl_neutral_to_smplx_neutral = BetasAdapter()
    smpl_neutral_to_smplx_neutral.load_state_dict(torch.load(osp.join(load_dir, f'smpl_neutral_to_smplx_neutral.pth')))
    smpl_neutral_to_smplx_neutral.to(device)

    results = {
        'pve': [],
        'pve_wo_adapt': [],
    }
    num_kid = 0
    num_male = 0
    num_female = 0
    num_no_gender = 0

    for path in tqdm.tqdm(smpl_paths):

        with open(path, 'rb') as f:
            content = pickle.load(f)

        if content['betas'].shape[1] == 11:
            num_kid += 1
            continue

        betas = torch.Tensor(content['betas'].reshape((1, 10))).to(device)
        body_pose = torch.Tensor(content['body_pose'].reshape((1, 69))).to(device)
        body_pose_smplx = body_pose[:, :63]

        global_orient = torch.Tensor(content['root_pose'].reshape((1, 3))).to(device)
        transl = torch.Tensor(content['translation'].reshape((1, 3))).to(device)

        expression = torch.zeros((1, 10), device=device)
        jaw_pose = torch.zeros((1, 3), device=device)
        leye_pose = torch.zeros((1, 3), device=device)
        reye_pose = torch.zeros((1, 3), device=device)
        left_hand_pose = torch.zeros((1, 15, 3), device=device)
        right_hand_pose = torch.zeros((1, 15, 3), device=device)

        adapted_smplx_betas = smpl_neutral_to_smplx_neutral(betas).to(device)

        vertices_smpl, joints_smpl = compute_smpl_vertices(smpl_neutral,
                                           betas=betas,
                                           body_pose=body_pose,
                                           global_orient=global_orient,
                                           transl=transl
                                           )
        vertices_neutral, joints_neutral = compute_smplx_vertices(smplx_neutral,
                                            betas=adapted_smplx_betas,
                                            body_pose=body_pose_smplx,
                                            global_orient=global_orient,
                                            transl=transl,
                                            expression=expression,
                                            jaw_pose=jaw_pose,
                                            leye_pose=leye_pose,
                                            reye_pose=reye_pose,
                                            left_hand_pose=left_hand_pose,
                                            right_hand_pose=right_hand_pose,
                                            )
        vertices_neutral_wo_adapt, joints_neutral_wo_adapt = compute_smplx_vertices(smplx_neutral,
                                                     betas=betas,
                                                     body_pose=body_pose_smplx,
                                                     global_orient=global_orient,
                                                     transl=transl,
                                                     expression=expression,
                                                     jaw_pose=jaw_pose,
                                                     leye_pose=leye_pose,
                                                     reye_pose=reye_pose,
                                                     left_hand_pose=left_hand_pose,
                                                     right_hand_pose=right_hand_pose,
                                                     )

        # offset
        vertices_smpl = vertices_smpl - joints_smpl[:, [0], :]
        vertices_neutral = vertices_neutral - joints_neutral[:, [0], :]
        vertices_neutral_wo_adapt = vertices_neutral_wo_adapt - joints_neutral_wo_adapt[:, [0], :]

        vertices_smpl = apply_deformation_transfer(def_matrix, vertices_smpl, None, use_normals=False)
        vertices_smpl = vertices_smpl[:, mask_ids]
        vertices_neutral = vertices_neutral[:, mask_ids]
        vertices_neutral_wo_adapt = vertices_neutral_wo_adapt[:, mask_ids]

        pve = per_vertex_loss(vertices_neutral, vertices_smpl)
        pve_wo_adapt = per_vertex_loss(vertices_neutral_wo_adapt, vertices_smpl)

        results['pve'].append(pve.item())
        results[f'pve_wo_adapt'].append(pve_wo_adapt.item())

    res = results['pve']
    res_wo_adapt = results[f'pve_wo_adapt']
    print('smpl neutral to smplx neutral, pve =', np.array(res).mean(), '; wo adapt, pve =', np.array(res_wo_adapt).mean())


if __name__ == '__main__':
    # validate_random_smplx()
    validate_random_smpl()
    # validate_agora_smplx()
    # validate_agora_smpl()
