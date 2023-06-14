
# -*- coding: utf-8 -*-

# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
# holder of all proprietary rights on this computer program.
# You can only use this computer program if you have closed
# a license agreement with MPG or you get the right to use the computer
# program from someone who is authorized to grant you that right.
# Any use of the computer program without a valid license is prohibited and
# liable to prosecution.
#
# Copyright©2019 Max-Planck-Gesellschaft zur Förderung
# der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
# for Intelligent Systems. All rights reserved.
#
# Contact: ps-license@tuebingen.mpg.de

import sys
import os
import os.path as osp

from typing import NewType, Dict
import time

import yaml
import numpy as np
import torch
import torch.nn as nn
import torch.autograd as autograd
# from loguru import logger

from mesh_mesh_intersection import MeshMeshIntersection
from scipy.spatial import ConvexHull

Tensor = NewType('Tensor', torch.Tensor)


class ChestWaistHipsMeasurements(nn.Module):
    def __init__(
        self, meas_definition_path: str, meas_vertices_path: str,
        max_collisions=256,
        *args, **kwargs
    ) -> None:
        super(ChestWaistHipsMeasurements, self).__init__()
        meas_definition_path = osp.expanduser(
            osp.expandvars(meas_definition_path))
        meas_vertices_path = osp.expanduser(
            osp.expandvars(meas_vertices_path))

        assert osp.exists(meas_definition_path), (
            'Measurement definition path does not exist:'
            f' {meas_definition_path}'
        )
        assert osp.exists(meas_definition_path), (
            'Measurement vertex path does not exist:'
            f' {meas_vertices_path}'
        )

        with open(meas_definition_path, 'r') as f:
            measurements_definitions = yaml.load(f)

        with open(meas_vertices_path, 'r') as f:
            meas_vertices = yaml.load(f)

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

        self.isect_module = MeshMeshIntersection(max_collisions=max_collisions)

    def _get_plane_at_heights(self, height: Tensor):
        device = height.device
        batch_size = height.shape[0]

        verts = torch.tensor(
            [[-1., 0, -1], [1, 0, -1], [1, 0, 1], [-1, 0, 1]],
            device=device).unsqueeze(dim=0).expand(batch_size, -1, -1).clone()
        verts[:, :, 1] = height.reshape(batch_size, -1)
        faces = torch.tensor([[0, 1, 2], [0, 2, 3]], device=device,
                             dtype=torch.long)

        return verts, faces, verts[:, faces]

    def forward(
        self,
        triangles: Tensor
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

        meas_data = {
            'chest': (self.chest_face_index, self.chest_bcs),
            'belly': (self.belly_face_index, self.belly_bcs),
            'hips': (self.hips_face_index, self.hips_bcs),
        }

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
                    -1, 3)[point_indices]

                meas_value = (
                    hull_points[1::2] - hull_points[:-1:2]).pow(2).sum(
                    dim=-1).sqrt().sum()
                # logger.info(f'{ii}: {name}, {meas_value}')

                output[name]['valid_points'].append(
                    np_points[ii, valid_face_idxs])
                output[name]['points'].append(hull_points)
                output[name]['value'].append(meas_value)
                #  values.append(
                #  )
        return output
