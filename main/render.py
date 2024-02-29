import numpy as np
import glob
import random
import cv2
import os
import argparse
import torch
import pyrender
import trimesh
import pandas as pd
import json

from tqdm import tqdm
from multiprocessing import Pool

# from mmhuman3d.models.body_models.builder import build_body_model
import smplx
import pdb

smpl_shape = {'betas': (-1, 10), 'transl': (-1, 3), 'global_orient': (-1, 3), 'body_pose': (-1, 69)}
smplx_shape = {'betas': (-1, 10), 'transl': (-1, 3), 'global_orient': (-1, 3), 
        'body_pose': (-1, 21, 3), 'left_hand_pose': (-1, 15, 3), 'right_hand_pose': (-1, 15, 3), 
        'leye_pose': (-1, 3), 'reye_pose': (-1, 3), 'jaw_pose': (-1, 3), 'expression': (-1, 10)}
smplx_shape_except_expression = {'betas': (-1, 10), 'transl': (-1, 3), 'global_orient': (-1, 3), 
        'body_pose': (-1, 21, 3), 'left_hand_pose': (-1, 15, 3), 'right_hand_pose': (-1, 15, 3), 
        'leye_pose': (-1, 3), 'reye_pose': (-1, 3), 'jaw_pose': (-1, 3)}
# smplx_shape = smplx_shape_except_expression

def render_pose(img, body_model_param, body_model, camera, return_mask=False):

    # the inverse is same
    pyrender2opencv = np.array([[1.0, 0, 0, 0],
                                [0, -1, 0, 0],
                                [0, 0, -1, 0],
                                [0, 0, 0, 1]])
    
    output = body_model(**body_model_param, return_verts=True)
    
    vertices = output['vertices'].detach().cpu().numpy().squeeze()
    faces = body_model.faces

    # render material
    base_color = (1.0, 193/255, 193/255, 1.0)
    material = pyrender.MetallicRoughnessMaterial(
            metallicFactor=0,
            alphaMode='OPAQUE',
            baseColorFactor=base_color)
    
    material_new = pyrender.MetallicRoughnessMaterial(
            metallicFactor=0.1,
            roughnessFactor=0.4,
            alphaMode='OPAQUE',
            emissiveFactor=(0.2, 0.2, 0.2),
            baseColorFactor=(0.7, 0.7, 0.7, 1))  
    material = material_new
    
    # get body mesh
    body_trimesh = trimesh.Trimesh(vertices, faces, process=False)
    body_mesh = pyrender.Mesh.from_trimesh(body_trimesh, material=material)

    # prepare camera and light
    light = pyrender.DirectionalLight(color=np.ones(3), intensity=2.0)
    cam_pose = pyrender2opencv @ np.eye(4)
    
    # build scene
    scene = pyrender.Scene(bg_color=[0.0, 0.0, 0.0, 0.0],
                                    ambient_light=(0.3, 0.3, 0.3))
    scene.add(camera, pose=cam_pose)
    scene.add(light, pose=cam_pose)
    scene.add(body_mesh, 'mesh')

    # render scene
    os.environ["PYOPENGL_PLATFORM"] = "osmesa" # include this line if use in vscode
    r = pyrender.OffscreenRenderer(viewport_width=img.shape[1],
                                    viewport_height=img.shape[0],
                                    point_size=1.0)
    
    color, _ = r.render(scene, flags=pyrender.RenderFlags.RGBA)
    color = color.astype(np.float32) / 255.0
    # alpha = 1.0  # set transparency in [0.0, 1.0]
    # color[:, :, -1] = color[:, :, -1] * alpha
    valid_mask = (color[:, :, -1] > 0)[:, :, np.newaxis]
    img = img / 255
    # output_img = (color[:, :, :-1] * valid_mask + (1 - valid_mask) * img)
    color = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
    output_img = (color[:, :, :] * valid_mask + (1 - valid_mask) * img)

    # output_img = color

    img = (output_img * 255).astype(np.uint8)
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    if return_mask:
        return img, valid_mask, (color * 255).astype(np.uint8)

    return img


def render_multi_pose(img,
                      body_model_params,
                      body_model,
                      cameras):

    masks, colors = [], []

    # calculate distance based on transl
    dists, valid_idx = [], []
    for i, body_model_param in enumerate(body_model_params):
        dist = np.linalg.norm(body_model_param['transl'].detach().cpu()) * 2/ (cameras[i].fx + cameras[i].fy)
        if dist not in dists:
            valid_idx.append(i)
            dists.append(dist)

    # pdb.set_trace()

    # select by valid idx
    body_model_params = [body_model_params[i] for i in valid_idx]
    cameras = [cameras[i] for i in valid_idx]

    # sort by dist

    body_model_params = [x for _, x in sorted(zip(dists, body_model_params), reverse=True)]
    cameras = [x for _, x in sorted(zip(dists, cameras), reverse=True)]


    # render separate masks
    for i, body_model_param in enumerate(body_model_params):

        _, mask, color = render_pose(
            img=img,
            body_model_param=body_model_param,
            body_model=body_model,
            camera=cameras[i],
            return_mask=True,
        )
        masks.append(mask)
        colors.append(color)
    # sum masks
    mask_sum = np.sum(masks, axis=0)
    mask_all = (mask_sum > 0)

    # pp_occ = 1 - np.sum(mask_all) / np.sum(mask_sum)

    # overlay colors to img
    for i, color in enumerate(colors):
        mask = masks[i]
        img = img * (1 - mask) + color * mask

    img = img.astype(np.uint8)
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    return img

def render_frame(framestamp, anno_ps, image_base_path, seq, smplx_model, args):
    annos = [p for p in anno_ps if framestamp in os.path.basename(p)]
    annos = [p for p in annos if 'person' not in os.path.basename(p)]

    body_model_params = []
    cameras = []
    bbox_sizes = []
    try:
        # image_path = os.path.join(seq, f'0{framestamp}.jpg').replace(args.data_path, args.image_path)
        image_path = os.path.join(image_base_path, f'0{framestamp}.jpg')
        # pdb.set_trace()
        image = cv2.imread(image_path)
    except:

        pass
    # pdb.set_trace()
    for anno_p in annos:

        anno = dict(np.load(anno_p, allow_pickle=True))

        meta = json.load(open(os.path.join(seq, 'meta', 
                                        os.path.basename(anno_p).replace('.npz', '.json')
                                        )))

        bbox_size = meta['bbox'][2] * meta['bbox'][3]
        focal_length = meta['focal']
        principal_point = meta['princpt']
        camera = pyrender.camera.IntrinsicsCamera(
                fx=focal_length[0], fy=focal_length[1],
                cx=principal_point[0], cy=principal_point[1],)

        # prepare body model params
        intersect_key = list(set(anno.keys()) & set(smplx_shape.keys()))
        body_model_param_tensor = {key: torch.tensor(
                np.array(anno[key]).reshape(smplx_shape[key]), device=args.device, dtype=torch.float32)
                        for key in intersect_key if len(anno[key]) > 0}
        
        cameras.append(camera)
        body_model_params.append(body_model_param_tensor)
        bbox_sizes.append(bbox_size)

    # render pose
    if args.render_biggest_person == 'True':
        bid = bbox_sizes.index(max(bbox_sizes))
        rendered_image = render_pose(img=image,
                        body_model_param=body_model_params[bid],
                        body_model=smplx_model,
                        camera=cameras[bid])
    else:
        rendered_image = render_multi_pose(img=image, 
                        body_model_params=body_model_params, 
                        body_model=smplx_model,
                        cameras=cameras)

    sp = seq.replace(f'{args.data_path}{os.path.sep}', '')
    save_path = os.path.join(args.data_path, 'output', sp)
    os.makedirs(save_path, exist_ok=True)

    save_name = os.path.join(save_path, framestamp+'.jpg')
    cv2.imwrite(save_name, rendered_image)


def call_frame_render(args):
    return render_frame(*args)

def visualize_seqs(args):

    if args.seq == 'default':
        seqs = glob.glob(os.path.join(args.data_path, '**/smplx'), recursive=True)
        seqs = [os.path.dirname(p) for p in seqs]
    else:
        seqs = glob.glob(os.path.join(args.data_path, args.seq), recursive=True)    
    
    kwargs = dict(gender='neutral',
        num_betas=10,
        use_face_contour=True,
        flat_hand_mean=args.flat_hand_mean,
        use_pca=False,
        batch_size=1)

    smplx_model = smplx.create(
        '../common/utils/human_model_files', 'smplx', 
        **kwargs).to(args.device)

    # seqs = [p for p in seqs if 'dance' not in os.path.basename(p)]

    # pdb.set_trace()
    
    for i, seq in enumerate(seqs):
        
        # prepare image path
        if args.load_mode == 'smplerx':
            image_base_path = os.path.join(seq, 'orig_img')
        else:
            image_base_path = os.path.join(seq, 'frames')

        assert os.path.exists(image_base_path)

        smplx_path = os.path.join(seq, 'smplx')
        anno_ps = sorted(glob.glob(os.path.join(smplx_path, '*.npz')))

        # group by framestamps
        framestamps = sorted(list(set([os.path.basename(p)[:5] for p in anno_ps 
                                       if 'person' not in os.path.basename(p)]
                                       )))

        for framestamp in tqdm(framestamps, leave=False, desc=f'Seqs {os.path.basename(seq)}'
                               f' : {i}/{len(seqs)}'):
                
            annos = [p for p in anno_ps if framestamp in os.path.basename(p)]
            annos = [p for p in annos if 'person' not in os.path.basename(p)]

            body_model_params = []
            cameras = []
            bbox_sizes = []
            # pdb.set_trace()
            try:
                if args.load_mode == 'smplerx':
                    image_path = os.path.join(image_base_path, f'{int(framestamp):06d}.jpg')
                    image = cv2.imread(image_path)
                else:
                    image_path = os.path.join(image_base_path, f'{int(framestamp):04d}.png')
                    image = cv2.imread(image_path)
                    image = cv2.resize(image, (1280, 720), interpolation = cv2.INTER_AREA)
                # image = cv2.imread(image_path)
                
            except:
                pass

            for anno_p in annos:

                anno = dict(np.load(anno_p, allow_pickle=True))

                meta = json.load(open(os.path.join(seq, 'meta', 
                                                os.path.basename(anno_p).replace('.npz', '.json')
                                                )))

                bbox_size = meta['bbox'][2] * meta['bbox'][3]
                focal_length = meta['focal']
                principal_point = meta['princpt']
                camera = pyrender.camera.IntrinsicsCamera(
                        fx=focal_length[0], fy=focal_length[1],
                        cx=principal_point[0], cy=principal_point[1],)

                # prepare body model params
                intersect_key = list(set(anno.keys()) & set(smplx_shape.keys()))
                body_model_param_tensor = {key: torch.tensor(
                        np.array(anno[key]).reshape(smplx_shape[key]), device=args.device, dtype=torch.float32)
                                for key in intersect_key if len(anno[key]) > 0}
                
                cameras.append(camera)
                body_model_params.append(body_model_param_tensor)
                bbox_sizes.append(bbox_size)

            # render pose
            if args.render_biggest_person == 'True':
                bid = bbox_sizes.index(max(bbox_sizes))
                rendered_image = render_pose(img=image,
                                body_model_param=body_model_params[bid],
                                body_model=smplx_model,
                                camera=cameras[bid])
            else:
                rendered_image = render_multi_pose(img=image, 
                                body_model_params=body_model_params, 
                                body_model=smplx_model,
                                cameras=cameras)
            # save image
            sp = seq.replace(f'{args.data_path}{os.path.sep}', '')
            save_path = os.path.join(args.data_path, sp, f'{args.load_mode}_overlay_img')
            os.makedirs(save_path, exist_ok=True)

            save_name = os.path.join(save_path, f'{int(framestamp):06d}.jpg')
            cv2.imwrite(save_name, rendered_image)
            

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_path', type=str, required=False,
                        help='path to the data folder')
    parser.add_argument('--load_mode', type=str, required=False,
                        default='smplerx',
                        help='load mode: smplerx or other test mode, please select smplerx')
    parser.add_argument('--seq', type=str, required=False,
                        help='seq name or seq pattern',
                        default='default')
    parser.add_argument('--image_path', type=str, required=False,
                        help='path to the image folder')

    # optional args
    parser.add_argument('--flat_hand_mean', type=bool, required=False,
                        help='use flat hand mean for smplx',
                        default=False)
    parser.add_argument('--render_biggest_person', type=str, required=False,
                        help='render biggest person in the frame',
                        default='True')
    args = parser.parse_args()

    args.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    visualize_seqs(args)