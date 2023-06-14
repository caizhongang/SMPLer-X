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
import os.path as osp
import argparse

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

import smplx
import open3d as o3d
import time
import cv2
from tqdm import tqdm

import trimesh
from loguru import logger
from star.pytorch.star import STAR
from star.config import cfg as star_cfg

from body_measurements import BodyMeasurements
from torchtrustncg import TrustRegion


def get_plane_at_height(h):
    verts = np.array([[-1., h, -1], [1, h, -1], [1, h, 1], [-1, h, 1]])
    faces = np.array([[0, 1, 2], [0, 2, 3]])

    normal = np.array([0.0, 1.0, 0.0])
    return verts, faces, (verts[0], normal)


def main(
    model_folder,
    height: float = 1.76,
    mass: float = -1,
    chest: float = 1.12,
    waist: float = 0.93,
    hips: float = 1.14,
    model_type='smplx',
    ext='npz',
    gender='neutral',
    num_betas=10,
    meas_definition_path: str = 'data/measurement_defitions.yaml',
    meas_vertices_path: str = 'data/smpl_measurement_vertices.yaml',
    summary_steps: int = 50,
    num_iterations: int = 500,
    betas_weight: float = 0.0,
):

    device = torch.device('cuda')
    dtype = torch.float32

    cfg = {
        'meas_definition_path': meas_definition_path,
        'meas_vertices_path': meas_vertices_path,
    }
    meas_module = BodyMeasurements(cfg)
    meas_module = meas_module.to(device=device)

    num_samples = 1

    trans, pose = None, None
    logger.info(f'Model type: {model_type}')
    if 'star' in model_type:
        star_cfg.path_male_star = osp.expandvars(
            osp.join(model_folder, 'star', 'STAR_MALE.npz'))
        star_cfg.path_female_star = osp.expandvars(
            osp.join(model_folder, 'star', 'STAR_FEMALE.npz'))
        model = STAR(gender=gender, num_betas=num_betas)
        trans = torch.zeros([num_samples, 3], dtype=dtype, device=device)
        pose = torch.zeros([num_samples, 72], dtype=dtype, device=device)
    else:
        model = smplx.build_layer(
            model_folder, model_type=model_type,
            gender=gender,
            num_betas=num_betas,
            ext=ext)

    logger.info(model)
    model = model.to(device=device)

    betas = torch.zeros(
        [num_samples, model.num_betas],
        requires_grad=True, dtype=torch.float32, device=device)

    dtype = torch.float32
    gt = {
        'height': torch.tensor(height, dtype=dtype, device=device),
        'mass': torch.tensor(mass, dtype=dtype, device=device),
        'chest': torch.tensor(chest, dtype=dtype, device=device),
        'waist': torch.tensor(waist, dtype=dtype, device=device),
        'hips': torch.tensor(hips, dtype=dtype, device=device),
    }
    weights = {
        'height': 100.0 if height > 0 else 0.0,
        'mass': 1.0 if mass > 0 else 0.0,
        'chest': 2000.0 if chest > 0 else 0.0,
        'waist': 1000.0 if waist > 0 else 0.0,
        'hips': 1000.0 if hips > 0 else 0.0,
    }

    optimizer = TrustRegion([betas])

    def compute_loss(gt, output, weights):
        losses = {}
        for key, gt_val in gt.items():
            if weights[key] <= 1e-3 or gt_val.item() < 0:
                continue
            est_val = output[key]['tensor']
            if isinstance(est_val, (tuple, list)):
                est_val = torch.stack(output[key]['value'])
            curr_loss = (gt_val - est_val).pow(2).sum() * weights[key]
            losses[key] = curr_loss

        losses['betas'] = betas_weight * betas.pow(2).sum()
        return losses

    def closure(backward=True):
        if backward:
            optimizer.zero_grad()

        if model_type == 'star':
            vertices = model(pose=pose, trans=trans, betas=betas)
            model_tris = vertices[:, model.faces]
        else:
            output = model(betas=betas, return_verts=True)
            model_tris = output.vertices[:, model.faces_tensor]

        output = meas_module(model_tris)['measurements']

        losses = compute_loss(gt, output, weights)

        loss = sum(losses.values())
        if backward:
            loss.backward(create_graph=True)

        return loss

    Y_OFFSET = -1.10

    for n in tqdm(range(num_iterations)):
        loss = optimizer.step(closure)

        if n % summary_steps == 0:
            if model_type == 'star':
                vertices = model(pose=pose, trans=trans, betas=betas)
                model_tris = vertices[:, model.faces]
                vertices = vertices.detach().cpu().numpy().squeeze()
                faces = model.faces.detach().cpu().numpy()
            else:
                output = model(betas=betas, return_verts=True)
                vertices = output.vertices.detach().cpu().numpy().squeeze()
                faces = model.faces
                model_tris = output.vertices[:, model.faces_tensor]

            y_offset = - vertices[:, 1].min() + Y_OFFSET
            vertices[:, 1] = vertices[:, 1] + y_offset

            #  for key, val in losses.items():
            mesh = o3d.geometry.TriangleMesh()
            mesh.vertices = o3d.utility.Vector3dVector(vertices)
            mesh.triangles = o3d.utility.Vector3iVector(faces)
            mesh.compute_vertex_normals()

            colors = np.ones_like(vertices) * [0.3, 0.3, 0.3]
            mesh.vertex_colors = o3d.utility.Vector3dVector(colors)

            geometry = []
            geometry.append(mesh)

            output = meas_module(model_tris)['measurements']
            for key, val in gt.items():
                est_val = output[key]["tensor"][0].item()
                logger.info(
                    f'[{n:04d}]: {key}: est = {est_val}, gt = {val}')

            losses = compute_loss(gt, output, weights)
            for key, val in losses.items():
                logger.info(f'[{n:04d}]: {key} loss = {val:.3f}')

            for meas_name in output:
                pcl = o3d.geometry.PointCloud()
                if 'points' not in output[meas_name]:
                    continue

                points = output[meas_name]['points']
                if isinstance(points, (tuple, list)):
                    points = torch.stack(points)
                if torch.is_tensor(points):
                    points = points.detach().cpu().numpy()
                points = points.reshape(-1, 3)

                points[:, 1] = points[:, 1] + y_offset

                pcl.points = o3d.utility.Vector3dVector(points)
                pcl.paint_uniform_color([1.0, 0.0, 0.0])
                geometry.append(pcl)

                lineset = o3d.geometry.LineSet()
                line_ids = np.arange(len(points)).reshape(-1, 2)
                lineset.points = o3d.utility.Vector3dVector(points)
                lineset.lines = o3d.utility.Vector2iVector(line_ids)
                lineset.paint_uniform_color([0.0, 0.0, 0.0])
                geometry.append(lineset)

            o3d.visualization.draw_geometries(
                geometry,
                lookat=np.array([0.0, 0.0, 0.0]).reshape(3, 1),
                up=np.array([0.0, 1.0, 0.0]).reshape(3, 1),
                front=np.array([0.0, 0.0, 1.0]).reshape(3, 1),
                zoom=1.0,
            )


if __name__ == '__main__':
    logger.remove()
    logger.add(lambda x: tqdm.write(x, end=''), colorize=True)
    parser = argparse.ArgumentParser(description='SMPL-X Demo')

    parser.add_argument('--model-folder', required=True, type=str,
                        help='The path to the model folder')
    parser.add_argument('--model-type', default='smpl', type=str,
                        choices=['smpl', 'smplh', 'smplx', 'mano', 'flame',
                                 'star', ],
                        help='The type of model to load')
    parser.add_argument('--gender', type=str, default='neutral',
                        help='The gender of the model')
    parser.add_argument('--num-betas', default=10, type=int,
                        dest='num_betas',
                        help='Number of shape coefficients.')
    parser.add_argument('--ext', type=str, default='npz',
                        help='Which extension to use for loading')
    parser.add_argument('--height', type=float, default=1.80,
                        help='Height of the subject in meters')
    parser.add_argument('--mass', type=float, default=-1,
                        help='Mass of the subject in kilograms')
    parser.add_argument('--chest', type=float, default=-1,
                        help='Chest circumference in meters')
    parser.add_argument('--waist', type=float, default=-1,
                        help='Waist circumference in meters')
    parser.add_argument('--hips', type=float, default=-1,
                        help='Hips circumference in meters')
    parser.add_argument('--meas-definition-path',
                        dest='meas_definition_path',
                        default='data/measurement_defitions.yaml',
                        type=str,
                        help='The definitions of the measurements')
    parser.add_argument('--meas-vertices-path', dest='meas_vertices_path',
                        type=str,
                        default='data/smpl_measurement_vertices.yaml',
                        help='The indices of the vertices used for the'
                        ' the measurements')
    parser.add_argument('--betas-weight', dest='betas_weight', default=0.0,
                        type=float,
                        help='The weight of the shape prior term.')

    args = parser.parse_args()

    model_folder = osp.expanduser(osp.expandvars(args.model_folder))
    model_type = args.model_type
    gender = args.gender
    ext = args.ext
    num_betas = args.num_betas

    height = args.height
    mass = args.mass
    chest = args.chest
    waist = args.waist
    hips = args.hips
    meas_definition_path = args.meas_definition_path
    meas_vertices_path = args.meas_vertices_path
    betas_weight = args.betas_weight

    main(model_folder,
         height=height,
         mass=mass,
         chest=chest,
         waist=waist,
         hips=hips,
         model_type=model_type,
         ext=ext,
         gender=gender,
         num_betas=num_betas,
         meas_definition_path=meas_definition_path,
         meas_vertices_path=meas_vertices_path,
         betas_weight=betas_weight,
         )
