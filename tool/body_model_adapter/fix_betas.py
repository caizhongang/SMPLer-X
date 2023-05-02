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

            # with open(smplx_param_load_path, 'rb') as f:
            #     smplx_params = pickle.load(f, encoding='latin1')
            # betas = smplx_params['betas']            

            # is_kid = ann['kid']
            # gender = ann['gender']
            # assert gender in ('male', 'female')
            # if not is_kid:
            #     with torch.no_grad():
            #         if gender == 'male':
            #             new_betas = smplx_male_to_smplx_neutral(torch.tensor(betas, device=device))
            #         else:
            #             new_betas = smplx_female_to_smplx_neutral(torch.tensor(betas, device=device))
            #     new_betas = new_betas.detach().cpu().numpy().reshape(1, 10)
            #     assert not np.allclose(betas, new_betas)
            # else:  # change adults' betas only
            #     new_betas = betas

            # # update smplx params
            # new_smplx_params = {k: v for k, v in smplx_params.items()}
            # new_smplx_params['betas_neutral'] = new_betas

            # update annotation
            new_smplx_param_path = smplx_param_path.replace('smplx_gt_fix_global_orient_transl', 'smplx_gt_fix_betas')
            new_ann = {k: v for k, v in ann.items() if k not in ('smplx_param_path')}
            new_ann['smplx_param_path'] = new_smplx_param_path
            new_annotations.append(new_ann)

            # # save new smplx params
            # new_smplx_save_path = osp.join(work_dir, new_smplx_param_path)
            # os.makedirs(osp.dirname(new_smplx_save_path), exist_ok=True)
            # with open(new_smplx_save_path, 'wb') as f:
            #     pickle.dump(new_smplx_params, f)

        new_ann_save_path = osp.join(work_dir, f'AGORA_{split}_fix_betas.json') 
        new_data = {
            'images': data['images'],
            'annotations': new_annotations,
        }
        with open(new_ann_save_path, 'w') as f:
            json.dump(new_data, f)


if __name__ == '__main__':
    fix_agora()