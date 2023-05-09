""" Modify transl such that parametric model can be projected properly on cropped images
    HOWEVER, it is discovered that this is not trivial, if not impossible for the following reasons:
        1. there is perspective distortion, so cropping is not exactly the same as translating
        2. there is scaling as the bbox is resized to 384,512
    Hence, the alterntive solution is to estimate on crop image and then project back to original image for supervisions
 """

import os
import os.path as osp
import json
import pickle
import tqdm
import numpy as np
import cv2
import glob
from pycocotools.coco import COCO
from fix_global_orient_transl import (
    vis_2d,
    get_2d,
    get_smplx_vertices,
    load_model,
    model_folder,
    kid_template_path,
    focalLength_mm2px
)
from affine_transform import (
    process_bbox,
    gen_trans_from_patch_cv
)

work_dir = '/mnt/lustrenew/share_data/caizhongang/data/datasets/agora'
imgHeight, imgWidth = (2160, 3840)
aspect_ratio = 384 / 512  # width / height


def get_intrinsics(imgPath):
    """ Ref:
            https://github.com/pixelite1201/agora_evaluation/blob/3b9de80193137b66224b0df18de72374877d3b31/agora_evaluation/projection.py#L84-L146
            https://github.com/pixelite1201/agora_evaluation/blob/ed739e0f5496dedb4a2f45f9a45aac4c479adea8/agora_evaluation/projection.py#L187-L188
     """
    if 'hdri' in imgPath:
        focalLength = 50
    elif 'cam00' in imgPath:
        focalLength = 18
    elif 'cam01' in imgPath:
        focalLength = 18
    elif 'cam02' in imgPath:
        focalLength = 18
    elif 'cam03' in imgPath:
        focalLength = 18
    elif 'ag2' in imgPath:
        focalLength = 28
    else:
        focalLength = 28

    dslr_sens_width = 36
    dslr_sens_height = 20.25

    cy= imgHeight / 2
    cx= imgWidth / 2

    focalLength_x = focalLength_mm2px(focalLength, dslr_sens_width, cx)
    focalLength_y = focalLength_mm2px(focalLength, dslr_sens_height, cy)

    return focalLength_x, focalLength_y, cx, cy


def compute_crop_transl_offset(u, v, w, h, z, fx, fy, width, height):
    """ Compute transl offset in 3D that gives equavalent pixel offset in 2D """
    # pixel offset
    pixel_offset_x = u + w / 2 - width / 2
    pixel_offset_y = v + h / 2 - height / 2

    # 3D offset
    offset_x = pixel_offset_x * z / fx
    offset_y = pixel_offset_y * z / fy

    return offset_x, offset_y, 0.0


def fix_crop_transl():
    """ Not working, see above """

    for split in ('train', 'validation'):
        data_load_path = osp.join(work_dir, f'AGORA_{split}_fix_betas.json')
        new_data_save_path = osp.join(work_dir, f'AGORA_{split}_fix_crop_transl.json')

        with open(data_load_path, 'r') as f:
            data = json.load(f)
        annotations = data['annotations']
        images = data['images']
        id_to_image_idx = {image['id']: i for i, image in enumerate(images)}

        new_annotations = []
        for ann in tqdm.tqdm(annotations[:1000]):

            # get intrinsics
            image_path = images[id_to_image_idx[ann['image_id']]]['file_name_3840x2160']
            fx, fy, cx, cy = get_intrinsics(image_path)

            # get crop box offset
            box = ann['bbox']
            bbox = process_bbox(box, imgWidth, imgHeight, aspect_ratio=aspect_ratio)
            if bbox is None:
                continue
            u, v, w, h = bbox

            # get current transl
            smplx_param_path = osp.join(work_dir, ann['smplx_param_path'])
            with open(smplx_param_path, 'rb') as f:
                smplx_param = pickle.load(f, encoding='latin1')
            transl = smplx_param['transl'].squeeze()
            z = transl[2]

            # compute offset
            crop_transl_offset = compute_crop_transl_offset(u=u, v=v, w=w, h=h, z=z, fx=fx, fy=fy, width=imgWidth, height=imgHeight)

            new_ann = {k: v for k, v in ann.items()}
            new_ann['crop_transl_offset'] = crop_transl_offset
            new_annotations.append(new_ann)

        new_data = {
            'annotations': new_annotations,
            'images': images,
        }
        with open(new_data_save_path, 'w') as f:
            json.dump(new_data, f)


def visualize():
    """ Compare solutions """

    vis_save_dir = '/mnt/cache/caizhongang/osx/tool/AGORA/vis_fix_crop_transl'
    os.makedirs(vis_save_dir, exist_ok=True)

    model_male, model_male_kid, model_female, model_female_kid, model_neutral, model_neutral_kid = load_model(
        modeltype='SMPLX',
        modelFolder=model_folder,
        numBetas=10,
        kid_template_path=kid_template_path,
        pose2rot=True
    )

    new_data_load_path = osp.join(work_dir, 'AGORA_train_fix_crop_transl.json')
    with open(new_data_load_path, 'r') as f:
        data = json.load(f)
    annotations = data['annotations']
    images = data['images']
    id_to_image_idx = {image['id']: i for i, image in enumerate(images)}

    np.random.seed(0)
    for i in np.random.permutation(len(annotations))[:10]:
        ann = annotations[i]
        image_path = images[id_to_image_idx[ann['image_id']]]['file_name_3840x2160']

        ann_id = ann['id']
        kid_flag = ann['kid']
        gender = ann['gender']
        box = ann['bbox']
        crop_transl_offset = ann['crop_transl_offset']
        u, v, w, h = process_bbox(box, imgWidth, imgHeight, aspect_ratio=aspect_ratio)  # ensure aspect ratio
        num_betas = 10

        smplx_param_path = osp.join(work_dir, ann['smplx_param_path'])
        with open(smplx_param_path, 'rb') as f:
            smplx_params = pickle.load(f, encoding='latin1')

        # Solution 1: naive translating and scaling
        smplx_params_with_offset = {k: v for k, v in smplx_params.items() if k != 'transl'}
        transl = smplx_params['transl']
        transl = transl - np.array(crop_transl_offset)
        smplx_params_with_offset['transl'] = transl

        joints, vertices = get_smplx_vertices(num_betas, kid_flag, smplx_params_with_offset, gender,
                                              smplx_male_kid_gt=model_male_kid,
                                              smplx_female_kid_gt=model_female_kid,
                                              smplx_neutral_kid=model_neutral_kid,
                                              smplx_male_gt=model_male,
                                              smplx_female_gt=model_female,
                                              smplx_neutral=model_neutral)

        path_components = image_path.split('/')
        path_components[1] = path_components[1] + '_crop'
        stem, ext = osp.splitext(path_components[-1])
        path_components[-1] = stem + f'_ann_id_{ann_id}' + ext
        img_load_path = osp.join(work_dir, '/'.join(path_components))
        img = cv2.imread(img_load_path)

        # load img2bb_trans
        img2bb_trans_path = img_load_path.replace('.png', '.json')
        with open(img2bb_trans_path, 'r') as f:
            img2bb_trans = np.array(json.load(f)['img2bb_trans'])

        fx, fy, _, _ = get_intrinsics(image_path)
        cx = w / 2  # use bbox crop as the image
        cy = h / 2
        camMat = np.array([[fx, 0, cx],
                           [0, fy, cy],
                           [0, 0, 1]])
        vertices_2d = (camMat @ vertices.T).T
        vertices_2d = vertices_2d[:, 0:2] / vertices_2d[:, [2]]
        scale = 384 / w  # box is always resized to 512x384
        vertices_2d_naive = vertices_2d * scale

        # Solution 2: use img2bb_trans
        joints, vertices = get_smplx_vertices(num_betas, kid_flag, smplx_params, gender,
                                              smplx_male_kid_gt=model_male_kid,
                                              smplx_female_kid_gt=model_female_kid,
                                              smplx_neutral_kid=model_neutral_kid,
                                              smplx_male_gt=model_male,
                                              smplx_female_gt=model_female,
                                              smplx_neutral=model_neutral)

        fx, fy, cx, cy = get_intrinsics(image_path)
        camMat = np.array([[fx, 0, cx],
                           [0, fy, cy],
                           [0, 0, 1]])
        vertices_2d_trans = (camMat @ vertices.T).T
        vertices_2d_trans = vertices_2d_trans / vertices_2d_trans[:, [2]]  # (N, 3)
        vertices_2d_trans = (img2bb_trans @ vertices_2d_trans.T).T

        # visualize
        img = vis_2d(img, vertices_2d_trans, color=(0, 255, 0))
        img = vis_2d(img, vertices_2d_naive, color=(0, 0, 255))

        basename = osp.basename(image_path)
        img_save_path = osp.join(vis_save_dir, basename)
        cv2.imwrite(img_save_path, img)
        print(img_save_path, 'saved.')


def compute_bb2img_from_json(path):

    with open(path, 'r') as f:
        data = json.load(f)
    bbox = data['bbox']
    img2bb_trans = data['img2bb_trans']
    resized_height = data['resized_height']
    resized_width = data['resized_width']

    bb_x, bb_y, bb_width, bb_height = bbox
    bb_c_x = bb_x + bb_width / 2
    bb_c_y = bb_y + bb_height / 2

    bb2img_trans = gen_trans_from_patch_cv(bb_c_x, bb_c_y, bb_width, bb_height, resized_width, resized_height, inv=True)

    return img2bb_trans, bb2img_trans


def validate_img2bb():
    load_dir = '/mnt/lustrenew/share_data/caizhongang/data/datasets/agora/3840x2160/'
    load_paths = sorted(glob.glob(osp.join(load_dir, '*/*.json')))
    num_points = 100

    for load_path in tqdm.tqdm(load_paths):
        img2bb_trans, bb2img_trans = compute_bb2img_from_json(load_path)
        random_x = np.random.rand(num_points, 1) * 3840
        random_y = np.random.rand(num_points, 1) * 2160
        random_keypoints = np.concatenate([random_x, random_y], axis=1)

        img_keypoints = np.concatenate([random_keypoints, np.ones((num_points, 1))], axis=1)
        bb_keypoints = (img2bb_trans @ img_keypoints.T).T
        bb_keypoints = np.concatenate([bb_keypoints, np.ones((num_points, 1))], axis=1)
        reproj_img_keypoints = (bb2img_trans @ bb_keypoints.T).T

        assert not np.allclose(img_keypoints, bb_keypoints)
        assert np.allclose(reproj_img_keypoints, img_keypoints[:, 0:2], atol=1e-3)

    print('All passed!')


if __name__ == '__main__':
    """ Not working, see above """
    # main()

    """ Alternative solution """
    validate_img2bb()

    """ Compare solutions """
    # visualize()
