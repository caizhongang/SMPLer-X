import json
import numpy as np
import glob
import pickle
import os.path as osp


def extract_predictions():
    load_dir = '/mnt/lustrenew/share_data/zoetrope/osx/output_wanqi'
    save_dir = '/mnt/cache/caizhongang/osx/output/data_analysis'
    exps = ['test_exp18_20230417_124958', 'test_exp13_70_20230413_070152']

    for exp in exps:

        paths = sorted(glob.glob(osp.join(load_dir, exp, 'result', 'predictions', '*.pkl')))
        np.random.seed(2023)
        np.random.shuffle(paths)
        paths = paths[:1024]

        global_orient = []
        for path in tqdm.tqdm(paths):
            with open(path, 'rb') as f:
                data = pickle.load(f, encoding='latin1')
            global_orient.append(data['params']['global_orient'].squeeze())

        global_orient = np.array(global_orient).tolist()
        save_path = osp.join(save_dir, exp + '.json')
        with open(save_path, 'w') as f:
            json.dump(global_orient, f)
        print(save_path, 'saved.')


if __name__ == '__main__':
    extract_predictions()
