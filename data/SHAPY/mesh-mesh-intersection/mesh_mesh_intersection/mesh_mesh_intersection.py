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

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import sys

import torch
import torch.nn as nn
import torch.autograd as autograd
# from loguru import logger

import mesh_mesh_intersection
import mesh_mesh_intersect_cuda


class MeshMeshIntersectionFunction(autograd.Function):

    @staticmethod
    @torch.no_grad()
    def forward(ctx, query_triangles, target_triangles, print_timings=False,
                max_collisions=32,
                *args, **kwargs):
        outputs = mesh_mesh_intersect_cuda.mesh_to_mesh_forward(
            query_triangles, target_triangles, print_timings=print_timings,
            max_collisions=max_collisions)
        #  ctx.save_for_backward(query_triangles, outputs)
        collision_faces, collision_bcs = outputs
        return collision_faces, collision_bcs

    @staticmethod
    def backward(ctx, grad_output, *args, **kwargs):
        raise NotImplementedError


class MeshMeshIntersection(nn.Module):

    def __init__(self, max_collisions=32):
        super(MeshMeshIntersection, self).__init__()
        self.max_collisions = max_collisions
        #  MeshMeshIntersectionFunction.max_collisions = self.max_collisions

    def forward(self, query_triangles, target_triangles,
                print_timings=False):
        return MeshMeshIntersectionFunction.apply(
            query_triangles, target_triangles, print_timings,
            self.max_collisions)
