import open3d as o3d
import numpy as np
import os
import os.path as osp
import glob


def vis_shapy():
    load_dir = '/home/alex/github/OSX/tool/SHAPY/vis_shapy/smplx_shape_target'
    load_paths = glob.glob(osp.join(load_dir, '*.npz'))
    for load_path in load_paths:
        content = np.load(load_path)
        v_shaped_gt = content['v_shaped_gt']
        v_shaped_fit = content['v_shaped_fit']
        for k in ['height', 'chest', 'waist', 'hips', 'mass']:
            print(k, content[k])
        pcd_source = o3d.geometry.PointCloud()
        pcd_source.points = o3d.utility.Vector3dVector(v_shaped_gt)
        pcd_source.paint_uniform_color([0, 1, 0])
        pcd_target = o3d.geometry.PointCloud()
        pcd_target.points = o3d.utility.Vector3dVector(v_shaped_fit)
        pcd_target.paint_uniform_color([1, 0, 0])
        pcd_wo_adapt = o3d.geometry.PointCloud()
        axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1, origin=[0, 0, 0])
        o3d.visualization.draw_geometries([pcd_source, pcd_target, pcd_wo_adapt, axis])

if __name__ == '__main__':
    vis_shapy()