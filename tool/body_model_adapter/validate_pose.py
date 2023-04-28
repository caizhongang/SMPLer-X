""" Evaluate if body pose parameters can be shared, given body shape has been adapted """
""" Need to run pip install -v -e . in vposer/ first"""

from human_body_prior.tools.model_loader import load_model
from human_body_prior.models.vposer_model import VPoser
from train_body_shape_adapter import create_body_model, compute_vertices, BetasAdapter, per_vertex_loss
import torch
import open3d as o3d
import glob
import pickle
import numpy as np
import tqdm

torch.manual_seed(2023)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

expr_dir = '/home/alex/github/OSX/tool/body_model_adapter/V02_05'
smplx_female_to_smplx_neutral_path = '/home/alex/github/OSX/tool/body_model_adapter/smplx_female_to_smplx_neutral.pth'
smplx_male_to_smplx_neutral_path = '/home/alex/github/OSX/tool/body_model_adapter/smplx_male_to_smplx_neutral.pth'
num_poses = 100


def validate_random():
    vp, ps = load_model(expr_dir, model_code=VPoser,
                        remove_words_in_model_weights='vp_model.',
                        disable_grad=True)
    vp = vp.to(device)

    # random body_pose
    random_body_pose = vp.sample_poses(num_poses=num_poses)['pose_body'].contiguous().view(num_poses, -1)

    # random betas
    random_betas = (torch.randn(num_poses, 10, device=device) - 0.5) * 3

    for gender in ('male', 'female'):
        smplx_gender = create_body_model(('smplx', gender)).to(device)
        smplx_neutral = create_body_model(('smplx', 'neutral')).to(device)

        smplx_gender_to_smplx_neutral = BetasAdapter()
        smplx_gender_to_smplx_neutral.load_state_dict(torch.load(eval(f'smplx_{gender}_to_smplx_neutral_path')))
        smplx_gender_to_smplx_neutral.to(device)

        neutral_betas_from_gender = smplx_gender_to_smplx_neutral(random_betas).to(device)

        vertices_gender = compute_vertices(smplx_gender, random_betas, random_body_pose)
        vertices_neutral = compute_vertices(smplx_neutral, neutral_betas_from_gender, random_body_pose)
        vertices_neutral_wo_adapt = compute_vertices(smplx_neutral, random_betas, random_body_pose)

        pve = per_vertex_loss(vertices_neutral, vertices_gender)
        pve_wo_adapt = per_vertex_loss(vertices_neutral, vertices_neutral_wo_adapt)
        print(gender, 'to neutral, pve =', pve.item(), 'wo adapt, pve =', pve_wo_adapt.item())

        for vert_gender, vert_neutral, vert_neutral_wo_adapt in zip(vertices_gender[:10], vertices_neutral[:10], vertices_neutral_wo_adapt[:10]):
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


def validate_agora():

    smplx_paths = glob.glob('/home/alex/github/OSX/tool/body_model_adapter/smplx_gt/*/*.pkl')

    smplx_female = create_body_model(('smplx', 'female')).to(device)
    smplx_male = create_body_model(('smplx', 'male')).to(device)
    smplx_neutral = create_body_model(('smplx', 'neutral')).to(device)

    smplx_male_to_smplx_neutral = BetasAdapter()
    smplx_male_to_smplx_neutral.load_state_dict(torch.load(eval(f'smplx_male_to_smplx_neutral_path')))
    smplx_male_to_smplx_neutral.to(device)

    smplx_female_to_smplx_neutral = BetasAdapter()
    smplx_female_to_smplx_neutral.load_state_dict(torch.load(eval(f'smplx_female_to_smplx_neutral_path')))
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

    for path in tqdm.tqdm(smplx_paths):

        with open(path, 'rb') as f:
            content = pickle.load(f)

        if content['betas'].shape[1] == 11:
            num_kid += 1
            continue

        betas = torch.Tensor(content['betas'].reshape((1, 10))).to(device)
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

        vertices_gender = compute_vertices(smplx_gender,
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
        vertices_neutral = compute_vertices(smplx_neutral,
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
        vertices_neutral_wo_adapt = compute_vertices(smplx_neutral,
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

        pve = per_vertex_loss(vertices_neutral, vertices_gender)
        pve_wo_adapt = per_vertex_loss(vertices_neutral, vertices_neutral_wo_adapt)

        results[gender].append(pve.item())
        results[f'{gender}_wo_adapt'].append(pve_wo_adapt.item())

    print('num_kid =', num_kid, 'which is', num_kid / len(smplx_paths) * 100, '% of the dataset')
    print('num_male =', num_male, 'which is', num_male / len(smplx_paths) * 100, '% of the dataset')
    print('num_female =', num_female, 'which is', num_female / len(smplx_paths) * 100, '% of the dataset')
    print('num_no_gender =', num_no_gender, 'which is', num_no_gender / len(smplx_paths) * 100, '% of the dataset')

    for gender in ('male', 'female'):
        res = results[gender]
        res_wo_adapt = results[f'{gender}_wo_adapt']
        print(gender, 'to neutral, pve =', np.array(res).mean(), '; wo adapt, pve =', np.array(res_wo_adapt).mean())


if __name__ == '__main__':
    # validate_random()
    validate_agora()
