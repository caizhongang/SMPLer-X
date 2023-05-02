""" Train a simple network to map body shape parameters from one body model to another."""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset
import smplx
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

source = ('smplx', 'male')
target = ('smplx', 'neutral')


def per_vertex_loss(P, G):
    dist = torch.norm(P - G, dim=-1)  # Shape: (B, N, V)
    loss = torch.mean(dist)
    return loss


def compute_vertices(body_model, betas=None, body_pose=None, global_orient=None, transl=None, expression=None, jaw_pose=None, leye_pose=None, reye_pose=None, left_hand_pose=None, right_hand_pose=None):
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
    return output.vertices


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
    def __init__(self, num_samples=10000):
        super(BetasDataset, self).__init__()
        self.num_samples = num_samples

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        x = (torch.randn(10) - 0.5) * 3  # [-1.5, 1.5]

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


def main():
    train_dataset = BetasDataset(num_samples=num_samples)
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
        correct = 0
        total = 0
        for i, inputs in enumerate(train_loader):
            inputs = inputs.to(device)
            optimizer.zero_grad()
            outputs = net(inputs)

            # compute vertices
            input_vertices = compute_vertices(body_model_source, inputs)
            output_vertices = compute_vertices(body_model_target, outputs)

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


if __name__ == '__main__':
    main()