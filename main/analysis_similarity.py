""" Investigate data distribution similarities """
import argparse
import tqdm
from collections import defaultdict
import numpy as np
import sys
import os.path as osp
import os
cur_dir = osp.dirname(os.path.abspath(__file__))
root_dir = osp.join(cur_dir, '..')
sys.path.insert(0, osp.join(root_dir, 'common'))

from config import cfg

from data.dataset import MultipleDatasets
from mmcv import Config
from pycocotools.coco import COCO

import smplx
import torch
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from common.utils.distribute_utils import set_seed
import umap
import matplotlib.pyplot as plt


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

data_dir = '/mnt/cache/caizhongang/osx/dataset'


def build_encoder(encoder_type):
    """ Ref: OSX.py get_model() """
    from mmpose.models import build_posenet

    # Load OSX
    if encoder_type == 'osx':
        encoder_config_file = 'transformer_utils/configs/osx/encoder/body_encoder_base.py'
        pretrained_model_path = '/mnt/lustrenew/share_data/zoetrope/osx/output_wanqi/train_exp11_1_20230411_155840/' \
                                'model_dump/snapshot_13.pth.tar'
        vit_cfg = Config.fromfile(encoder_config_file)
        vit = build_posenet(vit_cfg.model)
        network = torch.load(pretrained_model_path)['network']
        encoder_pretrained_model = {}
        for k, v in network.items():
            k = k.replace('module.', '')
            encoder_pretrained_model[k] = v
        vit.load_state_dict(encoder_pretrained_model, strict=False)
        encoder = vit.backbone

    # Load ViTPose
    elif encoder_type == 'vitpose':
        encoder_config_file = 'transformer_utils/configs/osx/encoder/body_encoder_base.py'
        encoder_pretrained_model_path = '../pretrained_models/osx_vit_b.pth'
        vit_cfg = Config.fromfile(encoder_config_file)
        vit = build_posenet(vit_cfg.model)
        encoder_pretrained_model = torch.load(encoder_pretrained_model_path)['state_dict']
        vit.load_state_dict(encoder_pretrained_model, strict=False)
        encoder = vit.backbone

    # Load humanbench
    elif encoder_type == 'humanbench':
        from humanbench_utils import get_backbone, load_checkpoint
        encoder_type = 'vit_base_patch16_ladder_attention_share_pos_embed'
        encoder_pretrained_model_path = '../pretrained_models/humanbench/v100_32g_vitbase_size224_lr1e3_stepLRx3_' \
                                        'bmp1_adafactor_wd01_clip05_layerdecay075_lpe_peddet_citypersons_LSA_' \
                                        'reduct8_tbn1_heads2_gate1_peddetShareDecoder_exp3_setting_SharePosEmbed.pth'
        backbone = get_backbone(encoder_type)
        checkpoint = torch.load(encoder_pretrained_model_path)['model']
        load_checkpoint(backbone, checkpoint, load_pos_embed=True, strict=False)
        encoder = backbone

    # Load dinov2
    elif encoder_type == 'dinov2':
        from dinov2_utils import get_backbone, load_checkpoint
        encoder_type = 'vitb'
        encoder_pretrained_model_path = '../pretrained_models/dinov2/dinov2_vitb14_pretrain.pth'
        backbone = get_backbone(encoder_type)
        load_checkpoint(backbone, encoder_pretrained_model_path)
        encoder = backbone

    else:
        raise ValueError('Undefined encoder type: {}'.format(encoder_type))

    return encoder


def build_dataloaders():
    for i in range(len(cfg.dataset_list)):
        exec('from ' + cfg.dataset_list[i] + ' import ' + cfg.dataset_list[i])

    dataloaders = []
    for i in range(len(cfg.dataset_list)):
        dataset = eval(cfg.dataset_list[i])(transforms.ToTensor(), "train")
        dataset = MultipleDatasets([dataset], make_same_len=False)
        dataloader = DataLoader(dataset=dataset, batch_size = cfg.num_gpus * cfg.train_batch_size,
                                shuffle=False, num_workers=cfg.num_thread, pin_memory=True, drop_last=False)
        dataloaders.append([
            cfg.dataset_list[i],
            dataloader
        ])

    return dataloaders


def parse_args():
    parser = argparse.ArgumentParser()
    # parser.add_argument('--gpu', type=str, dest='gpu_ids')
    parser.add_argument('--num_gpus', type=int, dest='num_gpus', default=1)
    # parser.add_argument('--master_port', type=int, dest='master_port', )
    parser.add_argument('--exp_name', type=str, default='output/analysis_similarity')
    parser.add_argument('--config', type=str, default='config_analysis_similarity.py')
    args = parser.parse_args()

    return args


def process():
    # dataloaders
    dataloaders = build_dataloaders()

    # save parameters and img
    for dataloader_name, dataloader in dataloaders:

        img_save_path = osp.join(cfg.root_dir, args.exp_name, f'img_{dataloader_name}.npy')
        smplx_params_save_path = osp.join(cfg.root_dir, args.exp_name, f'smplx_params_{dataloader_name}.npz')
        if osp.isfile(img_save_path):
            continue

        print(f'Parameters: processing {dataloader_name} ...')
        smplx_params = defaultdict(list)
        img_all = []
        for inputs, targets, meta_info in tqdm.tqdm(dataloader):

            # img
            img_all.append(inputs['img'].detach().cpu().numpy())

            # parameters
            if 'smplx_pose' in targets:
                smplx_params['global_orient'].append(targets['smplx_pose'][:, 0:3].detach().cpu().numpy())
                smplx_params['body_pose'].append(targets['smplx_pose'][:, 3:66])
                smplx_params['left_hand_pose'].append(targets['smplx_pose'][:, 66:111])
                smplx_params['right_hand_pose'].append(targets['smplx_pose'][:, 111:156])
                smplx_params['jaw_pose'].append(targets['smplx_pose'][:, 156:159])
                smplx_params['betas'].append(targets['smplx_shape'])
                smplx_params['expression'].append(targets['smplx_expr'])

        # save img
        img_all = np.concatenate(img_all, axis=0)
        np.save(img_save_path, img_all)
        print(f'Images saved to {img_save_path}.')

        # save parameters
        if smplx_params:
            for k in smplx_params.keys():
                smplx_params[k] = np.concatenate(smplx_params[k], axis=0)

            np.savez(smplx_params_save_path, **smplx_params)
            print(f'SMPL-X parameters saved to {smplx_params_save_path}.')

    # save appearance
    for encoder_type in ['osx', 'vitpose', 'humanbench', 'dinov2']:
        encoder = build_encoder(encoder_type)
        encoder.to(device)
        print(f'Encoder type {encoder_type} loaded successfully.')

        for dataloader_name, dataloader in dataloaders:

            img_feat_save_path = osp.join(cfg.root_dir, args.exp_name, f'img_feat_{dataloader_name}_{encoder_type}.npy')
            task_tokens_save_path = osp.join(cfg.root_dir, args.exp_name, f'task_tokens_{dataloader_name}_{encoder_type}.npy')
            if osp.isfile(img_feat_save_path) and osp.isfile(task_tokens_save_path):
                continue

            print(f'Appearance: processing {dataloader_name} with {encoder_type} ...')

            img_feat_all = []
            task_tokens_all = []

            for inputs, targets, meta_info in tqdm.tqdm(dataloader):

                body_img = F.interpolate(inputs['img'].to(device), cfg.input_body_shape)
                with torch.no_grad():
                    img_feat, task_tokens = encoder(body_img)

                img_feat = img_feat.detach().cpu().numpy()
                img_feat_all.append(img_feat)

                task_tokens = task_tokens.detach().cpu().numpy()
                task_tokens_all.append(task_tokens)

            img_feat_all = np.concatenate(img_feat_all, axis=0)
            np.save(img_feat_save_path, img_feat_all)
            print(f'Image features saved to {img_feat_save_path}.')

            task_tokens_all = np.concatenate(task_tokens_all, axis=0)
            np.save(task_tokens_save_path, task_tokens_all)
            print(f'Task tokens saved to {task_tokens_save_path}.')


def analyse_appearance():

    for encoder_type in ['osx', 'vitpose', 'humanbench', 'dinov2']:

        img_feat_all = []
        idxs = []  # dataset_name, start, end
        for dataset_name in tqdm.tqdm(cfg.dataset_list):

            # load feature
            load_path = osp.join(load_dir, f'img_feat_{dataset_name}_{encoder_type}.npy')
            img_feat = np.load(load_path)

            idxs.append([dataset_name, len(img_feat_all), len(img_feat_all) + len(img_feat)])
            img_feat_all.append(img_feat)

        img_feat_all = np.concatenate(img_feat_all, axis=0)
        img_feat_all = img_feat_all.reshape(img_feat_all.shape[0], -1)
        img_embedding = umap.UMAP(n_neighbors=15, min_dist=0.1, metric='correlation').fit_transform(img_feat_all)

        # plot
        fig, axs = plt.subplots(1, 1, figsize=(20, 20))
        axs.set_title(f'{encoder_type}')
        for dataset_name, start, end in idxs:
            xs = img_embedding[start:end, 0]
            ys = img_embedding[start:end, 1]
            axs.scatter(xs, ys, label=dataset_name)
        axs.legend(loc='upper right')
        save_path = f'vis_analysis_similarity/img_feat_{encoder_type}.png'
        plt.savefig(save_path)  # show must be placed before show
        print(f'{save_path} saved.')
        # plt.show()


def analyse_parameters():
    
    body_model = smplx.create(body_model_dir,
                        model_type='smplx',
                        gender='neutral',
                        flat_hand_mean=False,
                        use_pca=False,
                        use_face_contour=True,
                        batch_size=1,
                        num_betas=10,
                        use_compressed=False)

    factors_all = defaultdict(list)
    idxs = []
    for dataset_name in tqdm.tqdm(cfg.dataset_list):

        # load smplx params
        load_path = osp.join(load_dir, f'smplx_params_{dataset_name}.npz')
        if not osp.isfile(load_path):
            print(load_path, 'not found.')
            continue
        smplx_params = np.load(load_path)       

        global_orient = smplx_params['global_orient']
        B = smplx_params['global_orient'].shape[0]
        body_pose = smplx_params['body_pose']
        transl = smplx_params['transl'] if 'transl' in smplx_params else torch.zeros((B, 3))
        betas = smplx_params['betas']
        expression = smplx_params['expression']
        jaw_pose = smplx_params['jaw_pose']
        leye_pose = smplx_params['leye_pose'] if 'leye_pose' in smplx_params else torch.zeros((B, 3))
        reye_pose = smplx_params['reye_pose'] if 'reye_pose' in smplx_params else torch.zeros((B, 3))
        left_hand_pose = smplx_params['left_hand_pose']
        right_hand_pose = smplx_params['right_hand_pose']

        model_output = body_model(
            global_orient=torch.Tensor(global_orient.reshape(B, 3)),
            body_pose=torch.Tensor(body_pose.reshape(B, 21, 3)),
            transl=torch.Tensor(transl.reshape(B, 3)),
            betas=torch.Tensor(betas.reshape(B, 10)),
            expression=torch.Tensor(expression.reshape(B, 10)),
            jaw_pose=torch.Tensor(jaw_pose.reshape(B, 3)),
            leye_pose=torch.Tensor(leye_pose.reshape(B, 3)),
            reye_pose=torch.Tensor(reye_pose.reshape(B, 3)),
            left_hand_pose=torch.Tensor(left_hand_pose.reshape(B, 15, 3)),
            right_hand_pose=torch.Tensor(right_hand_pose.reshape(B, 15, 3)),
            return_joints=True
        )
        joints = model_output.joints.detach().cpu().numpy().squeeze()

        factors_all['global_orient'].append(global_orient)
        factors_all['body_pose'].append(body_pose)
        factors_all['joints'].append(joints)
        factors_all['betas'].append(betas)
        factors_all['expression'].append(expression)
        factors_all['left_hand_pose'].append(left_hand_pose)
        factors_all['right_hand_pose'].append(right_hand_pose)
        
        idxs.append([dataset_name, len(factors_all['global_orient']), len(factors_all['global_orient']) + len(global_orient)])

    # position-based
    for factor in ['joints']:

        feat_all = factors_all[factor]
        feat_all = np.concatenate(feat_all, axis=0)
        feat_all = feat_all.reshape(feat_all.shape[0], -1)
        embedding = umap.UMAP(n_neighbors=15, min_dist=0.1, metric='correlation').fit_transform(feat_all)

        # plot
        fig, axs = plt.subplots(1, 1, figsize=(20, 20))
        axs.set_title(f'{factor}')
        for dataset_name, start, end in idxs:
            xs = embedding[start:end, 0]
            ys = embedding[start:end, 1]
            axs.scatter(xs, ys, label=dataset_name)
        axs.legend(loc='upper right')
        save_path = f'vis_analysis_similarity/factor_{factor}.png'
        plt.savefig(save_path)
        print(f'{save_path} saved.')

    # PCA-based
    for factor in ['betas', 'expression']:

        feat_all = factors_all[factor]
        feat_all = np.concatenate(feat_all, axis=0)
        feat_all = feat_all.reshape(feat_all.shape[0], -1)
        embedding = umap.UMAP(n_neighbors=15, min_dist=0.1, metric='correlation').fit_transform(feat_all)

        # plot
        fig, axs = plt.subplots(1, 1, figsize=(20, 20))
        axs.set_title(f'{factor}')
        for dataset_name, start, end in idxs:
            xs = embedding[start:end, 0]
            ys = embedding[start:end, 1]
            axs.scatter(xs, ys, label=dataset_name)
        axs.legend(loc='upper right')
        save_path = f'vis_analysis_similarity/factor_{factor}.png'
        plt.savefig(save_path)
        print(f'{save_path} saved.')

    # rotation-based
    for factor in ['global_orient', 'body_pose', 'left_hand_pose', 'right_hand_pose']:

        feat_all = factors_all[factor]
        feat_all = np.concatenate(feat_all, axis=0)
        feat_all = feat_all.reshape(feat_all.shape[0], -1)
        embedding = umap.UMAP(n_neighbors=15, min_dist=0.1, metric='correlation').fit_transform(feat_all)

        # plot
        fig, axs = plt.subplots(1, 1, figsize=(20, 20))
        axs.set_title(f'{factor}')
        for dataset_name, start, end in idxs:
            xs = embedding[start:end, 0]
            ys = embedding[start:end, 1]
            axs.scatter(xs, ys, label=dataset_name)
        axs.legend(loc='upper right')
        save_path = f'vis_analysis_similarity/factor_{factor}.png'
        plt.savefig(save_path)
        print(f'{save_path} saved.')

    # COMMENTS:
    # leftover problem: betas has gender, how to compare fairly?


if __name__ == '__main__':
    args = parse_args()
    config_path = osp.join('./config', args.config)
    cfg.get_config_fromfile(config_path)
    cfg.update_config(args.num_gpus, args.exp_name)
    set_seed(1234)

    load_dir = osp.join(cfg.root_dir, args.exp_name)
    body_model_dir = '/mnt/cache/share_data/zoetrope/body_models'

    # process()

    # analyse_appearance()
    analyse_parameters()