""" Use a trained betas adapter to fix params. Must fix global orient transl first. """
import os.path as osp
import os
import glob
import pickle
from train_body_shape_adapter import BetasAdapter
import tqdm
import json
import torch
import numpy as np
device = torch.device('cpu' if torch.cuda.is_available() else 'cpu')

smplx_female_to_smplx_neutral_path = 'smplx_female_to_smplx_neutral.pth'
smplx_male_to_smplx_neutral_path = 'smplx_male_to_smplx_neutral.pth'

smplx_male_to_smplx_neutral = BetasAdapter()
smplx_male_to_smplx_neutral.load_state_dict(torch.load(smplx_male_to_smplx_neutral_path, map_location=device))
smplx_male_to_smplx_neutral.to(device)

smplx_female_to_smplx_neutral = BetasAdapter()
smplx_female_to_smplx_neutral.load_state_dict(torch.load(smplx_female_to_smplx_neutral_path, map_location=device))
smplx_female_to_smplx_neutral.to(device)


def fix_agora():
    """ Fix AGORA. Adults only. Kids not changed. 
        Do two things:
            - save new smplx params in smplx_gt_fix_betas/
            - save new ann file AGORA_{split}_fix_betas.json
    """

    work_dir = '/mnt/lustrenew/share_data/caizhongang/data/datasets/agora'

    for split in ['train', 'validation']:
        # load ann
        ann_load_path = osp.join(work_dir, f'AGORA_{split}_fix_global_orient_transl.json')   
        with open(ann_load_path, 'r') as f:
            data = json.load(f)
        annotations = data['annotations']
        
        new_annotations = []
        for ann in tqdm.tqdm(annotations):
            smplx_param_path = ann['smplx_param_path']
            smplx_param_load_path = osp.join(work_dir, smplx_param_path)

            with open(smplx_param_load_path, 'rb') as f:
                smplx_params = pickle.load(f, encoding='latin1')
            betas = smplx_params['betas']            

            is_kid = ann['kid']
            gender = ann['gender']
            assert gender in ('male', 'female')
            if not is_kid:
                with torch.no_grad():
                    if gender == 'male':
                        new_betas = smplx_male_to_smplx_neutral(torch.tensor(betas, device=device))
                    else:
                        new_betas = smplx_female_to_smplx_neutral(torch.tensor(betas, device=device))
                new_betas = new_betas.detach().cpu().numpy().reshape(1, 10)
                assert not np.allclose(betas, new_betas)
            else:  # change adults' betas only
                new_betas = betas

            # update smplx params
            new_smplx_params = {k: v for k, v in smplx_params.items()}
            new_smplx_params['betas_neutral'] = new_betas

            # update annotation
            new_smplx_param_path = smplx_param_path.replace('smplx_gt_fix_global_orient_transl', 'smplx_gt_fix_betas')
            new_ann = {k: v for k, v in ann.items() if k not in ('smplx_param_path')}
            new_ann['smplx_param_path'] = new_smplx_param_path
            new_annotations.append(new_ann)

            # save new smplx params
            new_smplx_save_path = osp.join(work_dir, new_smplx_param_path)
            os.makedirs(osp.dirname(new_smplx_save_path), exist_ok=True)
            with open(new_smplx_save_path, 'wb') as f:
                pickle.dump(new_smplx_params, f)

        new_ann_save_path = osp.join(work_dir, f'AGORA_{split}_fix_betas.json')
        new_data = {
            'images': data['images'],
            'annotations': new_annotations,
        }
        with open(new_ann_save_path, 'w') as f:
            json.dump(new_data, f)


def fix_egobody():
    work_dir = '/mnt/cache/share_data/caizhongang/data/preprocessed_datasets'
    load_paths = sorted(glob.glob(osp.join(work_dir, 'egobody_*.npz')))
    
    for load_path in load_paths:
        human_data = np.load(load_path, allow_pickle=True)
        gender = human_data['meta'].item()['gender']
        smplx = human_data['smplx'].item()
        betas = smplx['betas']

        new_betas = []
        assert len(gender) == len(betas)
        for gen, bet in tqdm.tqdm(zip(gender, betas), total=len(gender)):
            assert gen in ('male', 'female')

            with torch.no_grad():
                if gender == 'male':
                    new_bet = smplx_male_to_smplx_neutral(torch.tensor(bet.reshape(1, 10), device=device))
                else:
                    new_bet = smplx_female_to_smplx_neutral(torch.tensor(bet.reshape(1, 10), device=device))
            new_bet = new_bet.detach().cpu().numpy().reshape(10)
            assert not np.allclose(bet, new_bet)
            
            new_betas.append(new_bet)

        new_betas = np.stack(new_betas, axis=0)
        assert new_betas.shape == betas.shape

        new_smplx = { k: v for k, v in smplx.items() }
        new_smplx['betas_neutral'] = new_betas

        new_human_data = {}
        for k, v in human_data.items():
            if len(v.shape) == 0:
                new_human_data[k] = v.item()
            else:
                new_human_data[k] = v
        new_human_data['smplx'] = new_smplx

        stem, _ = osp.splitext(osp.basename(load_path))
        save_stem = stem + '_fix_betas'
        save_path = load_path.replace(stem, save_stem)
        np.savez_compressed(save_path, **new_human_data)
        print(load_path, '->', save_path)


if __name__ == '__main__':
    fix_agora()
    fix_egobody()