""" Train a simple network to map body shape parameters from one body model to another."""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset
import smplx
import os.path as osp
import numpy as np
import pickle
import tqdm
import glob
import matplotlib.pyplot as plt
try:
    import open3d as o3d
except:
    print('open3d not installed.')
torch.manual_seed(2023)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model_path = '/home/alex/github/zoehuman/mmhuman3d/data/body_models'
num_samples = 10000
num_epochs = 50
batch_size = 512

smplx_range = [12, 6, 6, 6, 3, 4, 5, 2, 4, 4]
smpl_range = [8, 7, 7, 7, 7, 7, 5, 6, 7, 8]


def per_vertex_loss(P, G):
    dist = torch.norm(P - G, dim=-1)  # Shape: (B, N, V)
    loss = torch.mean(dist)
    return loss


def analyse_agora_betas_distribution():
    # draw smplx
    betas_all_male = []
    betas_all_female = []
    smplx_paths = sorted(glob.glob(f'/home/alex/github/OSX/tool/body_model_adapter/smplx_gt/*/*.pkl'))
    for path in tqdm.tqdm(smplx_paths):

        with open(path, 'rb') as f:
            content = pickle.load(f)

        if content['betas'].shape[1] == 11:
            continue

        betas = content['betas'].reshape((1, 10))

        if  'gender' not in content:
            print(path, 'has no gender.')
            continue
        gender = content['gender']
        assert gender in ('male', 'female')
        if gender == 'male':
            betas_all_male.append(betas)
        else:
            betas_all_female.append(betas)

    betas_all_male = np.concatenate(betas_all_male, axis=0)
    betas_all_female = np.concatenate(betas_all_female, axis=0)

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

    plt.show()

    # draw smpl
    betas_all = []
    smplx_paths = sorted(glob.glob(f'/home/alex/github/OSX/tool/body_model_adapter/smpl_gt/*/*.pkl'))
    for path in tqdm.tqdm(smplx_paths):

        with open(path, 'rb') as f:
            content = pickle.load(f)

        if content['betas'].shape[1] == 11:
            continue
        betas = content['betas'].reshape((1, 10)).detach().cpu().numpy()
        betas_all.append(betas)

    betas_all = np.concatenate(betas_all, axis=0)
    fig, axs = plt.subplots(2, 5, figsize=(10, 10))
    for i in range(2):
        for j in range(5):
            idx = i * 5 + j
            beta = betas_all[:, idx]
            mi, ma = beta.min(), betas.max()
            axs[i, j].hist(beta, bins=100, density=False, label='neutral')

            print(f'smpl betas', idx, '[', mi, ',', ma, ']')
            axs[i, j].set_title(f'beta {idx}')
            axs[i, j].legend()

    plt.show()


def compute_smplx_vertices(body_model, betas=None, body_pose=None, global_orient=None, transl=None, expression=None,
                           jaw_pose=None, leye_pose=None, reye_pose=None, left_hand_pose=None, right_hand_pose=None):
    B = betas.shape[0]
    output = body_model(betas=betas if betas is not None else torch.zeros((B, 10), device=device),
                        body_pose=body_pose if body_pose is not None else torch.zeros((B, 21, 3), device=betas.device),
                        global_orient=global_orient if global_orient is not None else torch.zeros((B, 3), device=betas.device),
                        transl=transl if transl is not None else torch.zeros((B, 3), device=betas.device),
                        expression=expression if expression is not None else torch.zeros((B, 10), device=betas.device),
                        jaw_pose=jaw_pose if jaw_pose is not None else torch.zeros((B, 3), device=betas.device),
                        leye_pose=leye_pose if leye_pose is not None else torch.zeros((B, 3), device=betas.device),
                        reye_pose=reye_pose if reye_pose is not None else torch.zeros((B, 3), device=betas.device),
                        left_hand_pose=left_hand_pose if left_hand_pose is not None else torch.zeros((B, 15, 3), device=betas.device),
                        right_hand_pose=right_hand_pose if right_hand_pose is not None else torch.zeros((B, 15, 3), device=betas.device),
                        return_verts=True,
                        return_joints=True)
    return output.vertices, output.joints


def compute_smpl_vertices(body_model, betas=None, body_pose=None, global_orient=None, transl=None):
    B = betas.shape[0]
    output = body_model(betas=betas if betas is not None else torch.zeros((B, 10), device=device),
                        body_pose=body_pose if body_pose is not None else torch.zeros((B, 69), device=betas.device),
                        global_orient=global_orient if global_orient is not None else torch.zeros((B, 3), device=betas.device),
                        transl=transl if transl is not None else torch.zeros((B, 3), device=betas.device),
                        return_verts=True,
                        return_joints=True)
    return output.vertices, output.joints


def create_body_model(config):
    type, gender = config

    if type == 'smplx':
        body_model = smplx.create(model_path ,
                                  model_type='smplx',
                                  gender=gender,
                                  flat_hand_mean=False,
                                  use_pca=False,
                                  num_betas=10,
                                  use_compressed=False)
    elif type == 'smpl':
        body_model = smplx.create(model_path,
                      model_type='smpl',
                      gender=gender,
                      num_betas=10)

    return body_model


class BetasDataset(Dataset):
    def __init__(self, num_samples=10000, range=6):
        super(BetasDataset, self).__init__()
        self.num_samples = num_samples
        if isinstance(range, list):
            self.range = torch.Tensor(range)
        else:
            self.range = range

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        x = (torch.rand(10) - 0.5) * 2.  # [-1, 1]
        x = x * self.range / 2.
        return x


class BetasAdapter(nn.Module):
    def __init__(self):
        super(BetasAdapter, self).__init__()
        self.fc1 = nn.Linear(10, 2048)
        self.fc2 = nn.Linear(2048, 2048)
        self.fc3 = nn.Linear(2048, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.view(-1, 10)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def fit_smplx():
    """ Train model to map smplx male/female shape to smplx neutral shape """

    for gender in ('male', 'female'):
        source = ('smplx', gender)
        target = ('smplx', 'neutral')

        train_dataset = BetasDataset(num_samples=num_samples, range=smplx_range)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        # Initialize the neural network and optimizer
        net = BetasAdapter().to(device)
        optimizer = optim.Adam(net.parameters(), lr=0.0001)

        # Initialize the body models
        body_model_source = create_body_model(source).to(device)
        print('source model:', body_model_source)
        body_model_target = create_body_model(target).to(device)
        print('target model:', body_model_target)

        # Train the neural network
        for epoch in range(num_epochs):
            running_loss = 0.0
            for i, inputs in enumerate(train_loader):
                inputs = inputs.to(device)
                optimizer.zero_grad()
                outputs = net(inputs)

                # compute vertices
                input_vertices, input_joints = compute_smplx_vertices(body_model_source, betas=inputs)
                output_vertices, output_joints = compute_smplx_vertices(body_model_target, betas=outputs)

                # offset
                input_vertices = input_vertices - input_joints[:, [0], :]
                output_vertices = output_vertices - output_joints[:, [0], :]

                loss = per_vertex_loss(input_vertices, output_vertices)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

            # pcd_source = o3d.geometry.PointCloud()
            # pcd_source.points = o3d.utility.Vector3dVector(input_vertices[0].cpu().detach().numpy())
            # pcd_source.paint_uniform_color([0, 1, 0])
            # pcd_target = o3d.geometry.PointCloud()
            # pcd_target.points = o3d.utility.Vector3dVector(output_vertices[0].cpu().detach().numpy())
            # pcd_target.paint_uniform_color([1, 0, 0])
            # axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1, origin=[0, 0, 0])
            # o3d.visualization.draw_geometries([pcd_source, pcd_target])

            print('Epoch %d loss: %.8f' % (epoch + 1, running_loss / len(train_loader)))

        # save
        save_path = f'{source[0]}_{source[1]}_to_{target[0]}_{target[1]}.pth'
        torch.save(net.state_dict(), save_path)
        print(save_path, 'saved.')


def read_deformation_transfer(deformation_transfer_path, device=None, use_normal=False):
    ''' Ref: https://github.com/vchoutas/smplx/blob/bb595ae115c82eee630c9090ad2d9c9691167eee/transfer_model/utils/def_transfer.py#L28-L60
        Reads a deformation transfer
    '''
    if device is None:
        device = torch.device('cpu')
    assert osp.exists(deformation_transfer_path), (
        'Deformation transfer path does not exist:'
        f' {deformation_transfer_path}')
    print(f'Loading deformation transfer from: {deformation_transfer_path}')
    # Read the deformation transfer matrix
    with open(deformation_transfer_path, 'rb') as f:
        def_transfer_setup = pickle.load(f, encoding='latin1')
    if 'mtx' in def_transfer_setup:
        def_matrix = def_transfer_setup['mtx']
        if hasattr(def_matrix, 'todense'):
            def_matrix = def_matrix.todense()
        def_matrix = np.array(def_matrix, dtype=np.float32)
        if not use_normal:
            num_verts = def_matrix.shape[1] // 2
            def_matrix = def_matrix[:, :num_verts]
    elif 'matrix' in def_transfer_setup:
        def_matrix = def_transfer_setup['matrix']
    else:
        valid_keys = ['mtx', 'matrix']
        raise KeyError(f'Deformation transfer setup must contain {valid_keys}')

    def_matrix = torch.tensor(def_matrix, device=device, dtype=torch.float32)
    return def_matrix


def apply_deformation_transfer(def_matrix, vertices, faces, use_normals=False):
    ''' Ref: https://github.com/vchoutas/smplx/blob/bb595ae115c82eee630c9090ad2d9c9691167eee/transfer_model/utils/def_transfer.py#L63-L75
        Applies the deformation transfer on the given meshes
    '''
    if use_normals:
        raise NotImplementedError
    else:
        def_vertices = torch.einsum('mn,bni->bmi', [def_matrix, vertices])
        return def_vertices


def fit_smpl():
    """
    Train model to map smpl neutral/male/female shape to smplx neutral shape
    Ref:
        https://github.com/vchoutas/smplx/blob/bb595ae115c82eee630c9090ad2d9c9691167eee/transfer_model/__main__.py#L36
        https://github.com/vchoutas/smplx/blob/bb595ae115c82eee630c9090ad2d9c9691167eee/transfer_model/transfer_model.py#L257
    """

    smplx_mask_ids_load_path = '/home/alex/github/OSX/tool/body_model_adapter/model_transfer/smplx_mask_ids.npy'
    deformation_transfer_path = '/home/alex/github/OSX/tool/body_model_adapter/model_transfer/smpl2smplx_deftrafo_setup.pkl'

    mask_ids = np.load(smplx_mask_ids_load_path)
    mask_ids = torch.from_numpy(mask_ids).to(device=device)
    def_matrix = read_deformation_transfer(deformation_transfer_path, device=device)

    for gender in ('neutral', 'female', 'male'):
        source = ('smpl', gender)
        target = ('smplx', 'neutral')

        train_dataset = BetasDataset(num_samples=num_samples, range=12)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        # Initialize the neural network and optimizer
        net = BetasAdapter().to(device)
        optimizer = optim.Adam(net.parameters(), lr=0.0001)

        # Initialize the body models
        body_model_source = create_body_model(source).to(device)
        print('source model:', body_model_source)

        body_model_target = smplx.create(model_path ,
                                  model_type='smplx',
                                  gender='neutral',
                                  flat_hand_mean=True,  # important: set to True
                                  use_pca=False,
                                  num_betas=10,
                                  use_compressed=False).to(device)
        # body_model_target = create_body_model(target).to(device)
        print('target model:', body_model_target)

        # Train the neural network
        for epoch in range(num_epochs):
            running_loss = 0.0
            for i, inputs in enumerate(train_loader):
                inputs = inputs.to(device)
                optimizer.zero_grad()
                outputs = net(inputs)

                # compute smpl vertices
                input_vertices, input_joints = compute_smpl_vertices(body_model_source, betas=inputs)
                input_vertices = apply_deformation_transfer(def_matrix, input_vertices, None, use_normals=False)

                # compute smplx vertices
                output_vertices, output_joints = compute_smplx_vertices(body_model_target, betas=outputs)

                # mask
                input_vertices = input_vertices[:, mask_ids]
                output_vertices = output_vertices[:, mask_ids]

                # offset
                input_vertices = input_vertices - input_joints[:, [0], :]
                output_vertices = output_vertices - output_joints[:, [0], :]

                loss = per_vertex_loss(input_vertices, output_vertices)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

            # # import pdb; pdb.set_trace()
            # pcd_source = o3d.geometry.PointCloud()
            # pcd_source.points = o3d.utility.Vector3dVector(input_vertices[0].cpu().detach().numpy())
            # pcd_source.paint_uniform_color([0, 1, 0])
            # pcd_target = o3d.geometry.PointCloud()
            # pcd_target.points = o3d.utility.Vector3dVector(output_vertices[0].cpu().detach().numpy())
            # pcd_target.paint_uniform_color([1, 0, 0])
            #
            # # draw correspondences
            # qs, idx, color = [], [], []
            # for pred, gt in zip(pcd_source.points, pcd_target.points):
            #     qs.append(pred)
            #     qs.append(gt)
            #     idx.append([len(qs) - 2, len(qs) - 1])
            #     color.append([0.0, 0, 1.0])
            # lineset = o3d.geometry.LineSet(
            #     points=o3d.utility.Vector3dVector(qs),
            #     lines=o3d.utility.Vector2iVector(idx))
            # lineset.colors = o3d.utility.Vector3dVector(color)
            #
            # axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1, origin=[0, 0, 0])
            # o3d.visualization.draw_geometries([pcd_source, pcd_target, lineset])

            print('Epoch %d loss: %.8f' % (epoch + 1, running_loss / len(train_loader)))

        # save
        save_path = f'{source[0]}_{source[1]}_to_{target[0]}_{target[1]}.pth'
        torch.save(net.state_dict(), save_path)
        print(save_path, 'saved.')


if __name__ == '__main__':
    analyse_agora_betas_distribution()
    # fit_smplx()
    # fit_smpl()