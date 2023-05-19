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
# for Intelligent Systems and the Max Planck Institute for Biological
# Cybernetics. All rights reserved.
#
# Contact: ps-license@tuebingen.mpg.de

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import sys

import time

import torch
import torch.nn as nn
import torch.nn.functional as F


def calc_circumcircle(triangles, edge_cross_prod, idx=None):
    ''' Calculate the circumscribed circle for the given triangles

        Args:
            - triangles (torch.tensor BxTx3x3): The tensor that contains the
              coordinates of the triangle vertices
            - edge_cross_prod (torch.tensor BxCx3): Contains the unnormalized
              perpendicular vector to the surface of the triangle.
        Returns:
            - circumradius (torch.tensor BxTx1): The radius of the
              circumscribed circle
            - circumcenter (torch.tensor BxTx3): The center of the
              circumscribed circel
    '''

    alpha = triangles[:, :, 0] - triangles[:, :, 2]
    beta = triangles[:, :, 1] - triangles[:, :, 2]

    # Calculate the radius of the circumscribed circle
    # Should be BxF
    circumradius = (torch.norm(alpha - beta, dim=2, keepdim=True) /
                    (2 * torch.norm(edge_cross_prod, dim=2, keepdim=True)) *
                    torch.norm(alpha, dim=2, keepdim=True) *
                    torch.norm(beta, dim=2, keepdim=True))

    # Calculate the coordinates of the circumcenter of each triangle
    # Should BxFx3
    circumcenter = torch.cross(
        torch.sum(alpha ** 2, dim=2, keepdim=True) * beta -
        torch.sum(beta ** 2, dim=2, keepdim=True) * alpha,
        torch.cross(alpha, beta, dim=-1), dim=2)
    circumcenter /= (2 * torch.sum(edge_cross_prod ** 2, dim=2, keepdim=True))

    return circumradius, circumcenter + triangles[:, :, 2]


def repulsion_intensity(x, sigma=0.5, penalize_outside=True, linear_max=1000):
    ''' Penalizer function '''
    quad_penalty = (-(1.0 - 2.0 * sigma) / (4.0 * sigma ** 2) *
                    x ** 2 - 1 / (2.0 * sigma) * x +
                    0.25 * (3 - 2 * sigma))
    linear_region_mask = (x.le(-sigma) * x.gt(-linear_max)).to(dtype=x.dtype)
    if penalize_outside:
        quad_region_mask = (x.gt(-sigma) * x.lt(sigma)).to(dtype=x.dtype)
    else:
        quad_region_mask = (x.gt(-sigma) * x.lt(0)).to(dtype=x.dtype)

    return (linear_region_mask * (-x + 1 - sigma) +
            quad_region_mask * quad_penalty)


def dist_to_cone_axis(points_rel, dot_prod, cone_axis, cone_radius,
                      sigma=0.5, epsilon=1e-6, vectorized=True):
    ''' Computes the distance of each point to the axis

        This function projects the points on the plane of the base of the cone
        and computes the distance to the axis. This is subsequently normalized
        by the radius of the cone at the height level of the point, so that
        points with distance < 1 are in the code, distance == 1 means that the
        point is on the surface and distance > 1 means that the point is
        outside the cone.

        Args:
            - points_rel (torch.tensor BxCxNx3): The coordinates of the points
              relative to the center of the cone
            - dot_prod (torch.tensor BxCxN): The dot product of the points (in
              relative coordinates with respect to the cone center) with the
              axis of the cone
            - cone_axis (torch.tensor BxCx3): The axis of the cone
            - cone_radius (torch.tensor BxCx1): The radius of the cone
        Keyword args:
            - sigma (float = 0.5): The height of the cone
            - epsilon (float = 1e-6): Numerical stability constant for the
              float division
            - vectorized (bool = True): Whether to use an iterative or a
              vectorized version of the function
    '''

    if vectorized:
        batch_size, num_collisions = cone_radius.shape[:2]
        numerator = torch.norm(points_rel - dot_prod.unsqueeze(dim=-1) *
                               cone_axis.unsqueeze(dim=-2),
                               p=2, dim=-1)
        denominator = -cone_radius / sigma * dot_prod + cone_radius
    else:
        batch_size, num_collisions = cone_radius.shape[:2]
        numerator = torch.norm(points_rel - dot_prod.unsqueeze(-1) * cone_axis,
                               p=2, dim=-1)
        denominator = -cone_radius.view(batch_size, num_collisions) / sigma * \
            dot_prod + cone_radius.view(batch_size, num_collisions)

    return numerator / (denominator + epsilon)


def conical_distance_field(triangle_points, cone_center, cone_radius,
                           cone_axis, sigma=0.5, vectorized=True,
                           penalize_outside=True, linear_max=1000):
    ''' Distance field calculation for a cone

        Args:
            - triangle_points (torch.tensor (BxCxNx3): Contains
            the points whose distance from the cone we want to calculate.
            - cone_center (torch.tensor (BxCx3)): The coordinates of the center
              of the cone
            - cone_radius (torch.tensor (BxC)): The radius of the base of the
              cone
            - cone_axis (torch.tensor(BxCx3)): The unit vector that represents
              the axis of the cone
        Keyword Arguments
            - sigma (float = 0.5): The float value of the height of the cone
            - vectorized (bool = True): Whether to use an iterative or a
              vectorized version of the function
        Returns:
            - (torch.tensor BxCxN): The distance field values at the N points
              for the cone
    '''

    if vectorized:
        # Calculate the coordinates of the points relative to the center of
        # the cone
        points_rel = triangle_points - cone_center.unsqueeze(dim=-2)
        # Calculate the dot product between the relative point coordinates and
        # the axis (normal) of the cone. Essentially, it is the length of the
        # projection of the relative vector on the axis of the cone
        dot_prod = torch.sum(points_rel * cone_axis.unsqueeze(dim=-2), dim=-1)

        # Calculate the distance of the projections of the points on the cone
        # base plane to the center of cone, normalized by the height
        axis_dist = dist_to_cone_axis(points_rel, dot_prod,
                                      cone_axis, cone_radius,
                                      sigma=sigma, vectorized=True)

        circumcenter_dist = repulsion_intensity(
            dot_prod, sigma=sigma, penalize_outside=penalize_outside,
            linear_max=linear_max)

        # Ignore the points with axis_dist > 1, since they are out of the cone
        mask = axis_dist.lt(1).to(dtype=triangle_points.dtype)

        distance_field = mask * ((1 - axis_dist) * circumcenter_dist).pow(2)
    else:
        batch_size, num_collisions, num_points = triangle_points.shape[:3]
        distance_field = torch.zeros([batch_size, num_collisions, 3],
                                     dtype=triangle_points.dtype,
                                     device=triangle_points.device)
        for idx in range(num_points):
            # The relative coordinates of each point to the center of the cone
            # BxCx3
            points_rel = triangle_points[:, :, idx, :] - cone_center

            # Calculate the dot product between the relative point coordinates
            # and the axis (normal) of the cone. Essentially, it is the length
            # of the projection of the relative vector on the axis of the cone
            dot_prod = torch.sum(points_rel * cone_axis, dim=-1)

            axis_dist = dist_to_cone_axis(points_rel, dot_prod,
                                          cone_axis, cone_radius,
                                          sigma=sigma,
                                          vectorized=False)

            circumcenter_dist = repulsion_intensity(
                dot_prod, sigma=sigma, penalize_outside=penalize_outside)
            mask = (axis_dist < 1).to(dtype=triangle_points.dtype)

            distance_field[:, :, idx] = (1 - axis_dist) * mask * \
                circumcenter_dist

    return torch.pow(distance_field, 2)


class DistanceFieldPenetrationLoss(nn.Module):
    def __init__(self, sigma=0.5, point2plane=False, vectorized=True,
                 penalize_outside=True, linear_max=1000):
        super(DistanceFieldPenetrationLoss, self).__init__()
        self.sigma = sigma
        self.point2plane = point2plane
        self.vectorized = vectorized
        self.penalize_outside = penalize_outside
        self.linear_max = linear_max

    def forward(self, triangles, collision_idxs):
        '''
        Args:
            - triangles: A torch tensor of size BxFx3x3 that contains the
                coordinates of the triangle vertices
            - collision_idxs: A torch tensor of size Bx(-1)x2 that contains the
              indices of the colliding pairs
        Returns:
            A tensor with size B that contains the self penetration loss for
            each mesh in the batch
        '''

        coll_idxs = collision_idxs[:, :, 0].ge(0).nonzero()
        if len(coll_idxs) < 1:
            return torch.zeros([triangles.shape[0]],
                               dtype=triangles.dtype,
                               device=triangles.device,
                               requires_grad=triangles.requires_grad)

        receiver_faces = collision_idxs[coll_idxs[:, 0], coll_idxs[:, 1], 0]
        intruder_faces = collision_idxs[coll_idxs[:, 0], coll_idxs[:, 1], 1]

        batch_idxs = coll_idxs[:, 0]
        num_collisions = coll_idxs.shape[0]

        batch_size = triangles.shape[0]

        if len(intruder_faces) < 1:
            return torch.tensor(0.0, dtype=triangles.dtype,
                                device=triangles.device,
                                requires_grad=triangles.requires_grad)
        # Calculate the edges of the triangles
        # Size: BxFx3
        edge0 = triangles[:, :, 1] - triangles[:, :, 0]
        edge1 = triangles[:, :, 2] - triangles[:, :, 0]
        # Compute the cross product of the edges to find the normal vector of
        # the triangle
        aCrossb = torch.cross(edge0, edge1, dim=2)

        circumradius, circumcenter = calc_circumcircle(triangles, aCrossb)

        # Normalize the result to get a unit vector
        normals = aCrossb / torch.norm(aCrossb, 2, dim=2, keepdim=True)

        recv_triangles = triangles[batch_idxs, receiver_faces]
        intr_triangles = triangles[batch_idxs, intruder_faces]
        
        recv_normals = normals[batch_idxs, receiver_faces]
        recv_circumradius = circumradius[batch_idxs, receiver_faces]
        recv_circumcenter = circumcenter[batch_idxs, receiver_faces]

        intr_normals = normals[batch_idxs, intruder_faces]
        intr_circumradius = circumradius[batch_idxs, intruder_faces]
        intr_circumcenter = circumcenter[batch_idxs, intruder_faces]

        # Compute the distance field for the intruding triangles
        # B x NUM_COLLISIONS x 3
        # For each batch element, for each collision pair, 3 distance values
        # for the vertices of the intruding triangle
        phi_receivers = conical_distance_field(
            intr_triangles,
            recv_circumcenter, recv_circumradius,
            recv_normals,
            sigma=self.sigma,
            vectorized=self.vectorized,
            penalize_outside=self.penalize_outside,
            linear_max=self.linear_max)

        # Compute the distance field for the intruding triangles
        # B x NUM_COLLISIONS x 3
        # For each batch element, for each collision pair, 3 distance values
        # for the vertices of the intruding triangle
        # Same as above, but now the receiver is the "intruder".
        phi_intruders = conical_distance_field(
            recv_triangles,
            intr_circumcenter,
            intr_circumradius,
            intr_normals,
            sigma=self.sigma,
            vectorized=self.vectorized,
            penalize_outside=self.penalize_outside,
            linear_max=self.linear_max)

        receiver_loss = torch.tensor(0, device=triangles.device,
                                     dtype=torch.float32)
        intruder_loss = torch.tensor(0, device=triangles.device,
                                     dtype=torch.float32)

        if self.point2plane:
            receiver_loss = (-phi_receivers).pow(2).sum(dim=-1)
            intruder_loss = (-phi_intruders).pow(2).sum(dim=-1)
        else:
            receiver_loss = torch.norm(-phi_receivers.unsqueeze(dim=-1) *
                                       intr_normals.unsqueeze(dim=-2), p=2,
                                       dim=-1).pow(2).sum(dim=-1)
            intruder_loss = torch.norm(-phi_intruders.unsqueeze(dim=-1) *
                                       recv_normals.unsqueeze(dim=-2), p=2,
                                       dim=-1).pow(2).sum(dim=-1)

        batch_ind = torch.arange(0, batch_size, dtype=batch_idxs.dtype,
                                 device=triangles.device).unsqueeze(dim=1)
        batch_mask = batch_ind.repeat([1, num_collisions]).eq(batch_idxs)\
            .to(receiver_loss.dtype)

        loss = torch.matmul(batch_mask, receiver_loss) + \
            torch.matmul(batch_mask, intruder_loss)
        return loss
