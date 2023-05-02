""" Fix global orientation and transl orientation so they are in camera frame.
    From official train_SMPLX/SMPLX/train_*_withjv.pkl
        - find image path to obtain scene info
        - find smpl/smplx param paths, convert global orient and transl, save new ones
        - validate with gt_joints_3d
"""

import smplx
from mmhuman3d.models.body_models.utils import transform_to_camera_frame, batch_transform_to_camera_frame
import os
import os.path as osp
import numpy as np
import torch
import json
import cv2
import glob
import tqdm
import pickle

model_folder = '/mnt/cache/share_data/zoetrope/body_models'
kid_template_path = '/mnt/cache/share_data/zoetrope/body_models/smplx/smplx_kid_template.npy'

def load_model(modeltype, modelFolder, numBetas, kid_template_path, pose2rot=False):
    """ Ref: https://github.com/pixelite1201/agora_evaluation/blob/ed739e0f5496dedb4a2f45f9a45aac4c479adea8/agora_evaluation/utils.py#L192 """

    if modeltype == 'SMPLX' and pose2rot:
        model_male = smplx.create(modelFolder, model_type='smplx',
                                  gender='male',
                                  ext='npz',
                                  num_betas=numBetas, use_pca=False)
        model_male_kid = smplx.create(modelFolder, model_type='smplx',
                                      gender='male',
                                      age='kid',
                                      kid_template_path=kid_template_path,
                                      ext='npz', use_pca=False)

        model_female = smplx.create(modelFolder, model_type='smplx',
                                    gender='female',
                                    ext='npz',
                                    num_betas=numBetas,
                                    use_pca=False)

        model_female_kid = smplx.create(
            modelFolder,
            model_type='smplx',
            gender='female',
            age='kid',
            kid_template_path=kid_template_path,
            ext='npz',
            use_pca=False)

        model_neutral = smplx.create(modelFolder, model_type='smplx',
                                     gender='neutral',
                                     ext='npz',
                                     num_betas=numBetas,
                                     use_pca=False)

        model_neutral_kid = smplx.create(
            modelFolder,
            model_type='smplx',
            gender='neutral',
            age='kid',
            kid_template_path=kid_template_path,
            ext='npz',
            use_pca=False)

    elif modeltype == 'SMPLX' and not pose2rot:
        # If params are in rotation matrix format then we need to use SMPLXLayer class
        model_male = smplx.build_layer(modelFolder, model_type='smplx',
                                  gender='male',
                                  ext='npz',
                                  num_betas=numBetas, use_pca=False)
        model_male_kid = smplx.build_layer(modelFolder, model_type='smplx',
                                      gender='male',
                                      age='kid',
                                      kid_template_path=kid_template_path,
                                      ext='npz', use_pca=False)

        model_female = smplx.build_layer(modelFolder, model_type='smplx',
                                    gender='female',
                                    ext='npz',
                                    num_betas=numBetas,
                                    use_pca=False)

        model_female_kid = smplx.build_layer(
            modelFolder,
            model_type='smplx',
            gender='female',
            age='kid',
            kid_template_path=kid_template_path,
            ext='npz',
            use_pca=False)

        model_neutral = smplx.build_layer(modelFolder, model_type='smplx',
                                     gender='neutral',
                                     ext='npz',
                                     num_betas=numBetas,
                                     use_pca=False)

        model_neutral_kid = smplx.build_layer(
            modelFolder,
            model_type='smplx',
            gender='neutral',
            age='kid',
            kid_template_path=kid_template_path,
            ext='npz',
            use_pca=False)

    elif modeltype == 'SMPL':
        model_male = smplx.create(modelFolder, model_type='smpl',
                                  gender='male',
                                  ext='npz')
        model_male_kid = smplx.create(modelFolder, model_type='smpl',
                                      gender='male', age='kid',
                                      kid_template_path=kid_template_path,
                                      ext='npz')
        model_female = smplx.create(modelFolder, model_type='smpl',
                                    gender='female',
                                    ext='npz')
        model_female_kid = smplx.create(
            modelFolder,
            model_type='smpl',
            gender='female',
            age='kid',
            kid_template_path=kid_template_path,
            ext='npz')
        model_neutral = smplx.create(modelFolder, model_type='smpl',
                                     gender='neutral',
                                     ext='npz')
        model_neutral_kid = smplx.create(
            modelFolder,
            model_type='smpl',
            gender='neutral',
            age='kid',
            kid_template_path=kid_template_path,
            ext='npz')
    else:
        raise ValueError('Provide correct modeltype smpl/smplx')
    return model_male, model_male_kid, model_female, model_female_kid, model_neutral, model_neutral_kid


def get_scene_info(imgPath, df, i, pNum, meanPose=False):
    """ Ref: https://github.com/pixelite1201/agora_evaluation/blob/3b9de80193137b66224b0df18de72374877d3b31/agora_evaluation/projection.py#L84-L146 """

    if 'hdri' in imgPath:
        ground_plane = [0, 0, 0]
        scene3d = False
        focalLength = 50
        camPosWorld = [0, 0, 170]
        camYaw = 0
        camPitch = 0

    elif 'cam00' in imgPath:
        ground_plane = [0, 0, 0]
        scene3d = True
        focalLength = 18
        camPosWorld = [400, -275, 265]
        camYaw = 135
        camPitch = 30
    elif 'cam01' in imgPath:
        ground_plane = [0, 0, 0]
        scene3d = True
        focalLength = 18
        camPosWorld = [400, 225, 265]
        camYaw = -135
        camPitch = 30
    elif 'cam02' in imgPath:
        ground_plane = [0, 0, 0]
        scene3d = True
        focalLength = 18
        camPosWorld = [-490, 170, 265]
        camYaw = -45
        camPitch = 30
    elif 'cam03' in imgPath:
        ground_plane = [0, 0, 0]
        scene3d = True
        focalLength = 18
        camPosWorld = [-490, -275, 265]
        camYaw = 45
        camPitch = 30
    elif 'ag2' in imgPath:
        ground_plane = [0, 0, 0]
        scene3d = False
        focalLength = 28
        camPosWorld = [0, 0, 170]
        camYaw = 0
        camPitch = 15
    else:
        ground_plane = [0, -1.7, 0]
        scene3d = True
        focalLength = 28
        # camPosWorld = [
        #     df.iloc[i]['camX'],
        #     df.iloc[i]['camY'],
        #     df.iloc[i]['camZ']]
        camPosWorld = [
            df['camX'][i],
            df['camY'][i],
            df['camZ'][i]]
        # camYaw = df.iloc[i]['camYaw']
        camYaw = df['camYaw'][i]
        camPitch = 0

    if meanPose:
        yawSMPL = 0
        trans3d = [0, 0, 0]
    else:
        # yawSMPL = df.iloc[i]['Yaw'][pNum]
        # trans3d = [df.iloc[i]['X'][pNum],
        #            df.iloc[i]['Y'][pNum],
        #            df.iloc[i]['Z'][pNum]]
        yawSMPL = df['Yaw'][i][pNum]
        trans3d = [df['X'][i][pNum],
                   df['Y'][i][pNum],
                   df['Z'][i][pNum]]

    return ground_plane, scene3d, focalLength, camPosWorld, camYaw, camPitch, yawSMPL, trans3d


def unreal2cv2(points):
    """ Ref: https://github.com/pixelite1201/agora_evaluation/blob/3b9de80193137b66224b0df18de72374877d3b31/agora_evaluation/projection.py#L41 """
    # x --> y, y --> z, z --> x
    points = np.roll(points, 2, 1)
    # change direction of y
    points = points * np.array([1, -1, 1])
    return points


def smpl2opencv(j3d):
    """ https://github.com/pixelite1201/agora_evaluation/blob/3b9de80193137b66224b0df18de72374877d3b31/agora_evaluation/projection.py#L49 """
    # change sign of axis 1 and axis 2
    j3d = j3d * np.array([1, -1, -1])
    return j3d


def compute_world2cam(trans3d, camPosWorld, yawSMPL, camYaw, camPitch, scene3d, ground_plane, meanPose=False):
    """ Modified from Ref: https://github.com/pixelite1201/agora_evaluation/blob/3b9de80193137b66224b0df18de72374877d3b31/agora_evaluation/projection.py#L194-L226
        Combine all steps from ref into one world2cam transformation matrix
    """

    # camPosWorld and trans3d are in cm. Transform to meter
    trans3d = np.array(trans3d) / 100
    trans3d = unreal2cv2(np.reshape(trans3d, (1, 3)))
    camPosWorld = np.array(camPosWorld) / 100
    if scene3d:
        camPosWorld = unreal2cv2(
            np.reshape(
                camPosWorld, (1, 3))) + np.array(ground_plane)
    else:
        camPosWorld = unreal2cv2(np.reshape(camPosWorld, (1, 3)))

    # step 1: get points in camera coordinate system
    T_smpl2opencv = np.diag([1, -1, -1, 1])

    # step 2:
    # scans have a 90deg rotation, but for mean pose from vposer there is no
    # such rotation
    if meanPose:
        rotMat, _ = cv2.Rodrigues(np.array([[0, (yawSMPL) / 180 * np.pi, 0]], dtype=float))
    else:
        rotMat, _ = cv2.Rodrigues(np.array([[0, ((yawSMPL - 90) / 180) * np.pi, 0]], dtype=float))
    T_rotMat_trans3d = np.eye(4)
    T_rotMat_trans3d[:3, :3] = rotMat
    T_rotMat_trans3d[:3, 3] = trans3d

    # step 3:
    T_camPosWorld = np.eye(4)
    T_camPosWorld[:3, 3] = - camPosWorld

    # step 4:
    camera_rotationMatrix, _ = cv2.Rodrigues(
        np.array([0, ((-camYaw) / 180) * np.pi, 0]).reshape(3, 1))
    camera_rotationMatrix2, _ = cv2.Rodrigues(
        np.array([camPitch / 180 * np.pi, 0, 0]).reshape(3, 1))
    T_camera_rotationMatrix = np.eye(4)
    T_camera_rotationMatrix[:3, :3] = camera_rotationMatrix2 @ camera_rotationMatrix

    # combine all
    world2cam = T_camera_rotationMatrix @ T_camPosWorld @ T_rotMat_trans3d @ T_smpl2opencv

    return world2cam


def check_compute_world2cam():

    trans3d = np.random.rand(3)
    camPosWorld = np.random.rand(3)
    yawSMPL = np.random.rand()
    camYaw = np.random.rand()
    camPitch = np.random.rand()
    scene3d = False
    ground_plane = np.random.rand(3)
    meanPose = False
    j3d = np.random.rand(100, 3)

    ### method 1
    world2cam = compute_world2cam(trans3d, camPosWorld, yawSMPL, camYaw, camPitch, scene3d, ground_plane, meanPose=meanPose)
    j3d_new_1 = world2cam @ np.concatenate([j3d, np.ones((j3d.shape[0], 1))], axis=1).T
    j3d_new_1 = j3d_new_1[:3, :].T

    ### method 2
    # camPosWorld and trans3d are in cm. Transform to meter
    trans3d = np.array(trans3d) / 100
    trans3d = unreal2cv2(np.reshape(trans3d, (1, 3)))
    camPosWorld = np.array(camPosWorld) / 100
    if scene3d:
        camPosWorld = unreal2cv2(
            np.reshape(
                camPosWorld, (1, 3))) + np.array(ground_plane)
    else:
        camPosWorld = unreal2cv2(np.reshape(camPosWorld, (1, 3)))

    # get points in camera coordinate system
    j3d = smpl2opencv(j3d)

    # scans have a 90deg rotation, but for mean pose from vposer there is no
    # such rotation
    if meanPose:
        rotMat, _ = cv2.Rodrigues(
            np.array([[0, (yawSMPL) / 180 * np.pi, 0]], dtype=float))
    else:
        rotMat, _ = cv2.Rodrigues(
            np.array([[0, ((yawSMPL - 90) / 180) * np.pi, 0]], dtype=float))

    j3d = np.matmul(rotMat, j3d.T).T
    j3d = j3d + trans3d

    camera_rotationMatrix, _ = cv2.Rodrigues(
        np.array([0, ((-camYaw) / 180) * np.pi, 0]).reshape(3, 1))
    camera_rotationMatrix2, _ = cv2.Rodrigues(
        np.array([camPitch / 180 * np.pi, 0, 0]).reshape(3, 1))

    j3d_new = np.matmul(camera_rotationMatrix, j3d.T - camPosWorld.T).T
    j3d_new_2 = np.matmul(camera_rotationMatrix2, j3d_new.T).T

    ### validate
    np.allclose(j3d_new_1, j3d_new_2, atol=1e-5)


def get_smplx_vertices(
        num_betas,
        kid_flag,
        gt,
        gender,
        smplx_male_kid_gt,
        smplx_female_kid_gt,
        smplx_neutral_kid,
        smplx_male_gt,
        smplx_female_gt,
        smplx_neutral,
        pose2rot=True):
    """ https://github.com/pixelite1201/agora_evaluation/blob/ed739e0f5496dedb4a2f45f9a45aac4c479adea8/agora_evaluation/get_joints_verts_from_dataframe.py#L70 """

    if kid_flag:
        num_betas = 11
        if gender == 'female':
            model_gt = smplx_neutral_kid
        elif gender == 'male':
            model_gt = smplx_male_kid_gt
        elif gender == 'neutral':
            model_gt = smplx_neutral_kid
        else:
            raise KeyError(
                'Kid: Got gender {}, what gender is it?'.format(gender))

    else:
        if gender == 'female':
            model_gt = smplx_female_gt
        elif gender == 'male':
            model_gt = smplx_male_gt
        elif gender == 'neutral':
            model_gt = smplx_neutral
        else:
            raise KeyError('Got gender {}, what gender is it?'.format(gender))

    smplx_gt = model_gt(
        betas=torch.tensor(gt['betas'][:, :num_betas], dtype=torch.float),
        global_orient=torch.tensor(gt['global_orient'], dtype=torch.float),
        body_pose=torch.tensor(gt['body_pose'], dtype=torch.float),
        left_hand_pose=torch.tensor(gt['left_hand_pose'], dtype=torch.float),
        right_hand_pose=torch.tensor(gt['right_hand_pose'], dtype=torch.float),
        transl=torch.tensor(gt['transl'], dtype=torch.float),
        expression=torch.tensor(gt['expression'], dtype=torch.float),
        jaw_pose=torch.tensor(gt['jaw_pose'], dtype=torch.float),
        leye_pose=torch.tensor(gt['leye_pose'], dtype=torch.float),
        reye_pose=torch.tensor(gt['reye_pose'], dtype=torch.float), pose2rot=pose2rot)

    return smplx_gt.joints.detach().cpu().numpy().squeeze(
    ), smplx_gt.vertices.detach().cpu().numpy().squeeze()


def vis_2d(img, keypoints, color):
    for kp in keypoints:
        cv2.circle(img, (int(kp[0]), int(kp[1])), 1, color, -1)
    return img


def main():
    load_dir = '/mnt/lustrenew/share_data/caizhongang/data/datasets/agora'
    image_id = 0
    ann_id = 0
    val_id_offset = 0

    # load model
    model_male, model_male_kid, model_female, model_female_kid, model_neutral, model_neutral_kid = load_model(
        modeltype='SMPLX',
        modelFolder=model_folder,
        numBetas=10,
        kid_template_path=kid_template_path,
        pose2rot=True
    )

    used_new_smplx_save_paths = set()  # check for duplicates
    for split in ('train', 'validation'):
        data_path_list = glob.glob(osp.join(load_dir, split + '_SMPLX', 'SMPLX', '*.pkl'))
        data_path_list = sorted(data_path_list)

        if split == 'validation':
            val_id_offset = ann_id

        # load existing annotations
        ann_load_path = osp.join(load_dir, 'AGORA_' + split + '.json')
        new_ann_save_path = osp.join(load_dir, 'AGORA_' + split + '_fix_global_orient_transl.json')
        with open(ann_load_path, 'r') as f:
            data = json.load(f)
        annotations = data['annotations']
        new_annotations = []

        for data_path in tqdm.tqdm(data_path_list):
            if split == 'train':
                img_folder_name = data_path.split('/')[-1].split('_withjv')[0] # e.g., train_0
            else:
                img_folder_name = 'validation'

            with open(data_path, 'rb') as f:
                data_smplx = pickle.load(f, encoding='latin1')
                data_smplx = {k: list(v) for k, v in data_smplx.items()}

            img_num = len(data_smplx['imgPath'])
            for i in tqdm.tqdm(range(img_num)):
                imgPath = data_smplx['imgPath'][i]

                person_num = len(data_smplx['gt_path_smplx'][i])
                for j in range(person_num):

                    if split == 'validation' and imgPath in (
                            'ag_validationset_renderpeople_bfh_flowers_5_15_00000.png',
                            'ag_validationset_renderpeople_bfh_flowers_5_15_00001.png',
                            'ag_validationset_renderpeople_bfh_flowers_5_15_00002.png',
                            'ag_validationset_renderpeople_bfh_flowers_5_15_00003.png',
                    ):  # TODO: debug only, to skip train. Note to comment all dump!

                        # set paths
                        smplx_param_path = data_smplx['gt_path_smplx'][i][j][:-4] + '.pkl'
                        smplx_load_path = osp.join(load_dir, smplx_param_path)

                        stem, _ = osp.splitext(osp.basename(smplx_load_path))
                        new_stem = stem + f'_{ann_id:08d}'
                        new_smplx_save_path = smplx_load_path.replace('smplx_gt', 'smplx_gt_fix_global_orient_transl')  # change save dir
                        new_smplx_save_path = new_smplx_save_path.replace(stem, new_stem)  # to avoid duplicates

                        # print('smplx', smplx_load_path, '->', new_smplx_save_path)
                        os.makedirs(osp.dirname(new_smplx_save_path), exist_ok=True)
                        # if osp.exists(new_smplx_save_path):  # skip existed
                        #     continue

                        # load smplx params (world frame)
                        with open(smplx_load_path, 'rb') as f:
                            smplx_params = pickle.load(f, encoding='latin1')

                        ground_plane, scene3d, focalLength, camPosWorld, camYaw, camPitch, yawSMPL, trans3d = \
                            get_scene_info(imgPath, df=data_smplx, i=i, pNum=j)

                        # get world2cam transformation
                        T_world2cam = compute_world2cam(trans3d, camPosWorld, yawSMPL, camYaw, camPitch, scene3d, ground_plane)

                        # compute pelvis position
                        kid_flag = data_smplx['kid'][i][j]
                        gender = data_smplx['gender'][i][j]
                        num_betas = 10

                        joints, vertices = get_smplx_vertices(num_betas, kid_flag, smplx_params, gender,
                                                                    smplx_male_kid_gt=model_male_kid,
                                                                    smplx_female_kid_gt=model_female_kid,
                                                                    smplx_neutral_kid=model_neutral_kid,
                                                                    smplx_male_gt=model_male,
                                                                    smplx_female_gt=model_female,
                                                                    smplx_neutral=model_neutral)
                        pelvis = joints[0]

                        # compute new global orient and transl
                        global_orient = smplx_params['global_orient']
                        transl = smplx_params['transl']
                        new_global_orient, new_transl = transform_to_camera_frame(global_orient, transl, pelvis, extrinsic=T_world2cam)

                        # create new params
                        new_smplx_params = {
                            k: v for k, v in smplx_params.items() if k not in ('global_orient', 'transl', 'keypoints_3d', 'pose_embedding', 'v')
                        }
                        new_smplx_params['global_orient'] = new_global_orient.reshape(1, 3)
                        new_smplx_params['transl'] = new_transl.reshape(1, 3)

                        # validate
                        new_joints, new_vertices = get_smplx_vertices(num_betas, kid_flag, new_smplx_params, gender,
                                        smplx_male_kid_gt=model_male_kid,
                                        smplx_female_kid_gt=model_female_kid,
                                        smplx_neutral_kid=model_neutral_kid,
                                        smplx_male_gt=model_male,
                                        smplx_female_gt=model_female,
                                        smplx_neutral=model_neutral)
                        gt_joints_3d = data_smplx['gt_joints_3d'][i][j]
                        gt_verts = data_smplx['gt_verts'][i][j]
                        try:
                            assert np.allclose(gt_joints_3d, new_joints, atol=1e-5)
                            assert np.allclose(gt_verts, new_vertices, atol=1e-5)
                        except:
                            print('error')
                            import pdb; pdb.set_trace()

                        # validate 2D
                        imgHeight, imgWidth = (2160, 3840)
                        new_joints_2d = get_2d(new_joints, focalLength, imgHeight, imgWidth)
                        new_verts_2d = get_2d(new_vertices, focalLength, imgHeight, imgWidth)

                        gt_joints_2d = data_smplx['gt_joints_2d'][i][j]
                        try:
                            assert np.allclose(gt_joints_2d, new_joints_2d, atol=1e-3)
                        except:
                            print('error: 2d joints')
                            import pdb; pdb.set_trace()
                        # visualize
                        img_load_dir = '/mnt/cache/caizhongang/osx/dataset/AGORA/data/3840x2160'
                        img_load_path = osp.join(img_load_dir, img_folder_name, imgPath)
                        img_gt_joints_2d_path = osp.basename(img_load_path.replace('.png', '_gt.png'))
                        img_new_joints_2d_path = osp.basename(img_load_path.replace('.png', '_new_joints.png'))
                        img_new_verts_2d_path = osp.basename(img_load_path.replace('.png', '_new_verts.png'))

                        if osp.isfile(img_gt_joints_2d_path):
                            img_gt_joints_2d = cv2.imread(img_gt_joints_2d_path)
                            img_new_joints_2d = cv2.imread(img_new_joints_2d_path)
                            img_new_verts_2d = cv2.imread(img_new_verts_2d_path)

                            img_gt_joints_2d = vis_2d(img_gt_joints_2d, gt_joints_2d, color=(0, 255, 0))
                            img_new_joints_2d = vis_2d(img_new_joints_2d, new_joints_2d, color=(0, 0, 255))
                            img_new_verts_2d = vis_2d(img_new_verts_2d, new_verts_2d, color=(0, 0, 255))
                        else:
                            img = cv2.imread(img_load_path)
                            img_gt_joints_2d = vis_2d(img.copy(), gt_joints_2d, color=(0, 255, 0))
                            img_new_joints_2d = vis_2d(img.copy(), new_joints_2d, color=(0, 0, 255))
                            img_new_verts_2d = vis_2d(img.copy(), new_verts_2d, color=(0, 0, 255))

                        cv2.imwrite(img_gt_joints_2d_path, img_gt_joints_2d)
                        cv2.imwrite(img_new_joints_2d_path, img_new_joints_2d)
                        cv2.imwrite(img_new_verts_2d_path, img_new_verts_2d)
                        continue

                        # update annotations
                        ann = annotations[ann_id - val_id_offset]
                        new_ann = {}
                        for k, v in ann.items():
                            if k in ('smplx_param_path', 'smpl_param_path'):
                                continue
                            new_ann[k] = v
                        new_ann['smplx_param_path'] = new_smplx_save_path

                        # check paths
                        ann_smplx_param_path = ann['smplx_param_path']
                        try:
                            ann_smplx_param_path_stem, _ = osp.splitext(osp.basename(ann_smplx_param_path))
                            new_ann_smplx_param_path_stem, _ = osp.splitext(osp.basename(new_smplx_save_path))
                            assert '_'.join(new_ann_smplx_param_path_stem.split('_')[:-1]) == ann_smplx_param_path_stem   # make sure param is aligned

                            gt_joints_3d_path = osp.join(load_dir, 'gt_joints_3d', 'smplx', str(ann_id) + '.json')
                            assert new_ann['smplx_joints_3d_path'].split('/')[-3:] == gt_joints_3d_path.split('/')[-3:]

                            gt_verts_path = osp.join(load_dir, 'gt_verts', 'smplx', str(ann_id) + '.json')
                            assert new_ann['smplx_verts_path'].split('/')[-3:] == gt_verts_path.split('/')[-3:]
                        except:
                            print('error: unmatch paths')
                            import pdb; pdb.set_trace()

                        new_annotations.append(new_ann)

                        # save new params
                        try:
                            assert new_smplx_save_path not in used_new_smplx_save_paths
                        except:
                            print('error: used name')
                            import pdb; pdb.set_trace()
                        # with open(new_smplx_save_path, 'wb') as f:
                        #     pickle.dump(new_smplx_params, f)
                        # print(new_smplx_save_path, 'saved.')

                        used_new_smplx_save_paths.add(new_smplx_save_path)

                    ann_id += 1
                image_id += 1
                
        new_data = {
            'images': data['images'],
            'annotations': new_annotations,
        }

        # print(ann_load_path, '->', new_ann_save_path)
        # with open(new_ann_save_path, 'w') as f:
        #     json.dump(new_data, f)
        # print(new_ann_save_path, 'saved.')


def generate_new_json():
    """" Deprecated! now use main() will also update the json. But still can use. """

    work_dir = '/mnt/lustrenew/share_data/caizhongang/data/datasets/agora/'
    for split in ('train', 'validation'):
        load_path = osp.join(work_dir, 'AGORA_' + split + '.json')
        save_path = osp.join(work_dir, 'AGORA_' + split + '_fix_global_orient_transl.json')

        with open(load_path, 'r') as f:
            data = json.load(f)
        annotations = data['annotations']

        new_annotations = []
        for ann in tqdm.tqdm(annotations):
            new_ann = {k: v for k, v in ann.items() if k not in ('smplx_param_path')}

            ann_id = ann['id']
            smplx_param_path = ann['smplx_param_path']
            stem, _ = osp.splitext(osp.basename(smplx_param_path))
            new_stem = stem + f'_{ann_id:08d}'
            new_smplx_param_path = smplx_param_path.replace('smplx_gt', 'smplx_gt_fix_global_orient_transl')  # change save dir
            new_smplx_param_path = new_smplx_param_path.replace(stem, new_stem)  # to avoid duplicates
            assert osp.isfile(osp.join(work_dir, new_smplx_param_path)), new_smplx_param_path

            new_ann['smplx_param_path'] = new_smplx_param_path
            new_annotations.append(new_ann)

        new_data = {
            'images': data['images'],
            'annotations': new_annotations,
        }

        print(load_path, '->', save_path)
        with open(save_path, 'w') as f:
            json.dump(new_data, f)


def validate_fix_global_orient_transl():
    """ Validate generated results """

    load_dir = '/mnt/lustrenew/share_data/caizhongang/data/datasets/agora'

    for split in ('train', 'validation'):
        load_path = osp.join(load_dir, f'AGORA_{split}_fix_global_orient_transl.json')
        with open(load_path, 'r') as f:
            content = json.load(f)

        # load model
        model_male, model_male_kid, model_female, model_female_kid, model_neutral, model_neutral_kid = load_model(
            modeltype='SMPLX',
            modelFolder=model_folder,
            numBetas=10,
            kid_template_path=kid_template_path,
            pose2rot=True
        )

        annotations = content['annotations']
        for ann in tqdm.tqdm(annotations):
            smplx_param_path = osp.join(load_dir, ann['smplx_param_path'])
            with open(smplx_param_path, 'rb') as f:
                smplx_param = pickle.load(f)
            
            kid_flag = ann['kid']
            gender = ann['gender']
            num_betas = 10

            joints, vertices = get_smplx_vertices(num_betas, kid_flag, smplx_param, gender,
                        smplx_male_kid_gt=model_male_kid,
                        smplx_female_kid_gt=model_female_kid,
                        smplx_neutral_kid=model_neutral_kid,
                        smplx_male_gt=model_male,
                        smplx_female_gt=model_female,
                        smplx_neutral=model_neutral)

            # load gt joints 3d
            smplx_joints_3d_path = osp.join(load_dir, ann['smplx_joints_3d_path'])
            with open(smplx_joints_3d_path, 'r') as f:
                smplx_joints_3d = np.array(json.load(f))

            # load gt vertices
            smplx_verts_path = osp.join(load_dir, ann['smplx_verts_path'])
            with open(smplx_verts_path, 'r') as f:
                smplx_verts = np.array(json.load(f))

            # print('gt_joints_3d_path', smplx_joints_3d_path)
            # print('gt_verts_path', smplx_verts_path)
            # print('new_smplx_save_path', smplx_param_path)
            # print('transl', smplx_param['transl'])

            # assert
            try:
                assert np.allclose(joints, smplx_joints_3d, atol=1e-5)
                assert np.allclose(vertices, smplx_verts, atol=1e-5)
            except:
                import pdb; pdb.set_trace()

            # print(ann['id'], 'is ok.')
            # import pdb; pdb.set_trace()


def check_duplicate_smplx_paths():
    load_dir = '/mnt/lustrenew/share_data/caizhongang/data/datasets/agora'

    for split in ('train', 'validation'):
        data_path_list = glob.glob(osp.join(load_dir, split + '_SMPLX', 'SMPLX', '*.pkl'))
        data_path_list = sorted(data_path_list)
        num_duplicates = 0
        num_total = 0

        for data_path in tqdm.tqdm(data_path_list):
            with open(data_path, 'rb') as f:
                data_smplx = pickle.load(f, encoding='latin1')
                data_smplx = {k: list(v) for k, v in data_smplx.items()}

            used_name = set()
            img_num = len(data_smplx['imgPath'])
            for i in tqdm.tqdm(range(img_num)):
                imgPath = data_smplx['imgPath'][i]

                person_num = len(data_smplx['gt_path_smplx'][i])
                for j in range(person_num):

                    # set paths
                    smplx_param_path = data_smplx['gt_path_smplx'][i][j][:-4] + '.pkl'
                    smplx_load_path = osp.join(load_dir, smplx_param_path)
                    new_smplx_save_path = smplx_load_path.replace('smplx_gt', 'smplx_gt_fix_global_orient_transl')
                    if new_smplx_save_path in used_name:
                        num_duplicates += 1
                    used_name.add(new_smplx_save_path)
                    num_total += 1

        print(split, '#duplicates =', num_duplicates, '/', num_total, '\n')


def focalLength_mm2px(focalLength, dslr_sens, focalPoint):
    focal_pixel = (focalLength / dslr_sens) * focalPoint * 2
    return focal_pixel


def get_2d(points3d, focalLength, imgHeight, imgWidth):
    """ Ref: https://github.com/pixelite1201/agora_evaluation/blob/ed739e0f5496dedb4a2f45f9a45aac4c479adea8/agora_evaluation/projection.py#L64
        Ref: https://github.com/pixelite1201/agora_evaluation/blob/ed739e0f5496dedb4a2f45f9a45aac4c479adea8/agora_evaluation/projection.py#L166
        Ref: https://github.com/pixelite1201/agora_evaluation/blob/ed739e0f5496dedb4a2f45f9a45aac4c479adea8/agora_evaluation/projection.py#L55
    """
    
    dslr_sens_width = 36
    dslr_sens_height = 20.25

    cy= imgHeight / 2
    cx= imgWidth / 2

    focalLength_x = focalLength_mm2px(focalLength, dslr_sens_width, cx)
    focalLength_y = focalLength_mm2px(focalLength, dslr_sens_height, cy)

    camMat = np.array([[focalLength_x, 0, cx],
                       [0, focalLength_y, cy],
                       [0, 0, 1]])

    points2d = (camMat @ points3d.T).T
    points2d = points2d[:, 0:2] / points2d[:, [2]]
    return points2d


if __name__ == '__main__':
    # main()
    # check_compute_world2cam()
    # generate_new_json()
    validate_fix_global_orient_transl()
    # check_duplicate_smplx_paths()
    # validate_2d_verts_joints()