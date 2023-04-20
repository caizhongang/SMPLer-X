import json
import shutil
import os
import os.path as osp
import tqdm
load_path = '/mnt/lustrenew/share_data/zoetrope/osx/data/AGORA/data/AGORA_train.json'
image_dir = '/mnt/lustrenew/share_data/zoetrope/osx/data/AGORA/data/1280x720'

with open(load_path, 'r') as f:
    content = json.load(f)

images = content['images']
for image in tqdm.tqdm(images):
    file_name_1280x720 = image['file_name_1280x720']
    base = osp.basename(file_name_1280x720)
    dirname = osp.dirname(file_name_1280x720)
    part = dirname.split('/')[-1]
    src = osp.join(image_dir, 'train', base)
    dst_dir = osp.join(image_dir, part)
    os.makedirs(dst_dir, exist_ok=True)
    dst = osp.join(dst_dir, base)
    # print(src, '->', dst)
    shutil.move(src, dst)