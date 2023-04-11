from glob import glob
import torch
import os.path as osp
import pickle
from tqdm import tqdm
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str, dest='dataset_path')
    args = parser.parse_args()

    if not args.dataset_path:
        assert 0, "Please set dataset_path"
    return args

args = parse_args()
root_path = osp.join(args.dataset_path, 'smpl_gt')
folder_path_list = glob(osp.join(root_path, '*'))
for folder_path in folder_path_list:
    pkl_path_list = glob(osp.join(folder_path, '*.pkl'))
    for pkl_path in tqdm(pkl_path_list):
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f, encoding='latin1')

        for k in data.keys():
            if type(data[k]) is torch.Tensor:
                data[k] = data[k].cpu().detach().numpy()

        with open(pkl_path, 'wb') as f:
            pickle.dump(data, f)

