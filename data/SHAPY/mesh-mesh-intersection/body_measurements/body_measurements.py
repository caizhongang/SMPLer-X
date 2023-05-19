from typing import NewType, Dict, Tuple
import os.path as osp
import yaml
import numpy as np

from mesh_mesh_intersection import MeshMeshIntersection
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
from scipy.spatial import ConvexHull
# from loguru import logger

Tensor = NewType('Tensor', torch.Tensor)


class BodyMeasurements(nn.Module):

    # The density of the human body is 985 kg / m^3
    DENSITY = 985

    def __init__(self, cfg, **kwargs):
        ''' Loss that penalizes deviations in weight and height
        '''
        super(BodyMeasurements, self).__init__()

        meas_definition_path = cfg.get('meas_definition_path', '')
        meas_definition_path = osp.expanduser(
            osp.expandvars(meas_definition_path))
        meas_vertices_path = cfg.get('meas_vertices_path', '')
        meas_vertices_path = osp.expanduser(
            osp.expandvars(meas_vertices_path))

        with open(meas_definition_path, 'r') as f:
            measurements_definitions = yaml.safe_load(f, )

        with open(meas_vertices_path, 'r') as f:
            meas_vertices = yaml.safe_load(f)

        head_top = meas_vertices['HeadTop']
        left_heel = meas_vertices['HeelLeft']

        left_heel_bc = left_heel['bc']
        self.left_heel_face_idx = left_heel['face_idx']

        left_heel_bc = torch.tensor(left_heel['bc'], dtype=torch.float32)
        self.register_buffer('left_heel_bc', left_heel_bc)

        head_top_bc = torch.tensor(head_top['bc'], dtype=torch.float32)
        self.register_buffer('head_top_bc', head_top_bc)

        self.head_top_face_idx = head_top['face_idx']

        action = measurements_definitions['CW_p']
        chest_periphery_data = meas_vertices[action[0]]

        self.chest_face_index = chest_periphery_data['face_idx']
        chest_bcs = torch.tensor(
            chest_periphery_data['bc'], dtype=torch.float32)
        self.register_buffer('chest_bcs', chest_bcs)

        action = measurements_definitions['BW_p']
        belly_periphery_data = meas_vertices[action[0]]

        self.belly_face_index = belly_periphery_data['face_idx']
        belly_bcs = torch.tensor(
            belly_periphery_data['bc'], dtype=torch.float32)
        self.register_buffer('belly_bcs', belly_bcs)

        action = measurements_definitions['IW_p']
        hips_periphery_data = meas_vertices[action[0]]

        self.hips_face_index = hips_periphery_data['face_idx']
        hips_bcs = torch.tensor(
            hips_periphery_data['bc'], dtype=torch.float32)
        self.register_buffer('hips_bcs', hips_bcs)

        max_collisions = cfg.get('max_collisions', 256)
        self.isect_module = MeshMeshIntersection(max_collisions=max_collisions)

    def extra_repr(self) -> str:
        msg = []
        msg.append(f'Human Body Density: {self.DENSITY}')
        return '\n'.join(msg)

    def _get_plane_at_heights(self, height: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        device = height.device
        batch_size = height.shape[0]

        verts = torch.tensor(
            [[-1., 0, -1], [1, 0, -1], [1, 0, 1], [-1, 0, 1]],
            device=device).unsqueeze(dim=0).expand(batch_size, -1, -1).clone()
        verts[:, :, 1] = height.reshape(batch_size, -1)
        faces = torch.tensor([[0, 1, 2], [0, 2, 3]], device=device,
                             dtype=torch.long)

        return verts, faces, verts[:, faces]

    def compute_peripheries(
        self,
        triangles: Tensor,
        compute_chest: bool = True,
        compute_waist: bool = True,
        compute_hips: bool = True,
    ) -> Dict[str, Tensor]:
        '''
            Parameters
            ----------
                triangles: BxFx3x3 torch.Tensor
                Contains the triangle coordinates for a batch of meshes with
                the same topology
        '''

        batch_size, num_triangles = triangles.shape[:2]
        device = triangles.device

        batch_indices = torch.arange(
            batch_size, dtype=torch.long,
            device=device).reshape(-1, 1) * num_triangles

        meas_data = {}
        if compute_chest:
            meas_data['chest'] = (self.chest_face_index, self.chest_bcs)
        if compute_waist:
            meas_data['waist'] = (self.belly_face_index, self.belly_bcs)
        if compute_hips:
            meas_data['hips'] = (self.hips_face_index, self.hips_bcs)

        output = {}
        for name, (face_index, bcs) in meas_data.items():

            vertex = (
                triangles[:, face_index] * bcs.reshape(1, 3, 1)).sum(axis=1)

            _, _, plane_tris = self._get_plane_at_heights(vertex[:, 1])

            with torch.no_grad():
                collision_faces, collision_bcs = self.isect_module(
                    plane_tris, triangles)

            selected_triangles = triangles.view(-1, 3, 3)[
                (collision_faces + batch_indices).view(-1)].reshape(
                    batch_size, -1, 3, 3)
            points = (
                selected_triangles[:, :, None] *
                collision_bcs[:, :, :, :, None]).sum(
                axis=-2).reshape(batch_size, -1, 2, 3)

            np_points = points.detach().cpu().numpy()
            collision_faces = collision_faces.detach().cpu().numpy()
            collision_bcs = collision_bcs.detach().cpu().numpy()

            output[name] = {
                'points': [],
                'valid_points': [],
                'value': [],
                'plane_height': vertex[:, 1],
            }

            for ii in range(batch_size):
                valid_face_idxs = np.where(collision_faces[ii] > 0)[0]
                points_in_plane = np_points[
                    ii, valid_face_idxs, :, ][:, :, [0, 2]].reshape(
                        -1, 2)
                hull = ConvexHull(points_in_plane)
                point_indices = hull.simplices.reshape(-1)

                hull_points = points[ii][valid_face_idxs].view(
                    -1, 3)[point_indices].reshape(-1, 2, 3)

                meas_value = (
                    hull_points[:, 1] - hull_points[:, 0]).pow(2).sum(
                    dim=-1).sqrt().sum()

                output[name]['valid_points'].append(
                    np_points[ii, valid_face_idxs])
                output[name]['points'].append(hull_points)
                output[name]['value'].append(meas_value)
            output[name]['tensor'] = torch.stack(output[name]['value'])
        return output

    def compute_height(self, shaped_triangles: Tensor) -> Tuple[Tensor, Tensor]:
        ''' Compute the height using the heel and the top of the head
        '''
        head_top_tri = shaped_triangles[:, self.head_top_face_idx]
        head_top = (head_top_tri[:, 0, :] * self.head_top_bc[0] +
                    head_top_tri[:, 1, :] * self.head_top_bc[1] +
                    head_top_tri[:, 2, :] * self.head_top_bc[2])
        head_top = (
            head_top_tri * self.head_top_bc.reshape(1, 3, 1)
        ).sum(dim=1)
        left_heel_tri = shaped_triangles[:, self.left_heel_face_idx]
        left_heel = (
            left_heel_tri * self.left_heel_bc.reshape(1, 3, 1)
        ).sum(dim=1)

        return (torch.abs(head_top[:, 1] - left_heel[:, 1]),
                torch.stack([head_top, left_heel], axis=0)
                )

    def compute_mass(self, tris: Tensor) -> Tensor:
        ''' Computes the mass from volume and average body density
        '''
        x = tris[:, :, :, 0]
        y = tris[:, :, :, 1]
        z = tris[:, :, :, 2]
        volume = (
            -x[:, :, 2] * y[:, :, 1] * z[:, :, 0] +
            x[:, :, 1] * y[:, :, 2] * z[:, :, 0] +
            x[:, :, 2] * y[:, :, 0] * z[:, :, 1] -
            x[:, :, 0] * y[:, :, 2] * z[:, :, 1] -
            x[:, :, 1] * y[:, :, 0] * z[:, :, 2] +
            x[:, :, 0] * y[:, :, 1] * z[:, :, 2]
        ).sum(dim=1).abs() / 6.0
        return volume * self.DENSITY

    def forward(
        self,
        triangles: Tensor,
        compute_mass: bool = True,
        compute_height: bool = True,
        compute_chest: bool = True,
        compute_waist: bool = True,
        compute_hips: bool = True,
        **kwargs
    ):
        measurements = {}
        if compute_mass:
            measurements['mass'] = {}
            mesh_mass = self.compute_mass(triangles)
            measurements['mass']['tensor'] = mesh_mass

        if compute_height:
            measurements['height'] = {}
            mesh_height, points = self.compute_height(triangles)
            measurements['height']['tensor'] = mesh_height
            measurements['height']['points'] = points

        output = self.compute_peripheries(triangles,
                                          compute_chest=compute_chest,
                                          compute_waist=compute_waist,
                                          compute_hips=compute_hips,
                                          )
        measurements.update(output)

        return {'measurements': measurements}
