import os
import sys
import collections
import torch
import numpy as np

sys.path.append('/mnt/cache/chencheng1/cc_proj/vitruvian/development/devL2/vitruvian-multitask')

mae_pretrain_path = '/mnt/cache/chencheng1/cc_proj/vitruvian/development/devL2/vitruvian-multitask/core/models/backbones/pretrain_weights/mae_pretrain_vit_base.pth'
mae_model = torch.load(mae_pretrain_path)

save_root = '/mnt/lustre/chencheng1/expr_files/vitruvian/devL2/transed_ckpt_for_pretrain/devL2_small_setting'
trained_ckpt_root = '/mnt/lustrenew/chencheng1/expr_files/vitruvian/devL2/LSA_devL2_small_setting/checkpoints/vitbase_lr1e3_StepLRx3_backboneclip_bmp08_ld75_pose_dpr03_dcLN_par_dpr03_dcBN_attr_dpr01_reid_clstoken_dpr0_LSA_10p_small_setting6_add_posetrack_DGMarket_deepfashion'
expr_name = trained_ckpt_root.split('/')[-1]

attr_index = 8  # pa100k
pose_index = 0  # coco
parsing_index = 4  # lip
reid_index = 9  # 5set

# attr
attr_train_model_path = os.path.join(trained_ckpt_root, 'ckpt_task{}_iter_newest.pth.tar'.format(attr_index))
attr_transed_ckpt_save_path = os.path.join(save_root, 'attr', expr_name+'.pth')

attr_model = torch.load(attr_train_model_path, map_location=torch.device('cpu'))

cnt = 0
traned_ckpt = collections.OrderedDict()
for name, param in mae_model['model'].items():
    trained_model_name = 'module.backbone_module.' + name
    if trained_model_name in attr_model['state_dict']:
        cnt += 1
        traned_ckpt[name] = attr_model['state_dict'][trained_model_name]
    else:
        traned_ckpt[name] = mae_model['model'][name]

torch.save({'model': traned_ckpt}, attr_transed_ckpt_save_path)
print('done! transed ckpt saved at: {}'.format(attr_transed_ckpt_save_path))

# pose
pose_train_model_path = os.path.join(trained_ckpt_root, 'ckpt_task{}_iter_newest.pth.tar'.format(pose_index))
pose_transed_ckpt_save_path = os.path.join(save_root, 'pose', expr_name+'.pth')

pose_model = torch.load(pose_train_model_path, map_location=torch.device('cpu'))

cnt = 0
traned_ckpt = collections.OrderedDict()
for name, param in mae_model['model'].items():
    trained_model_name = 'module.backbone_module.' + name
    if trained_model_name in pose_model['state_dict']:
        cnt += 1
        traned_ckpt[name] = pose_model['state_dict'][trained_model_name]
    else:
        traned_ckpt[name] = mae_model['model'][name]

torch.save({'model': traned_ckpt}, pose_transed_ckpt_save_path)
print('done! transed ckpt saved at: {}'.format(pose_transed_ckpt_save_path))

# reid
reid_train_model_path = os.path.join(trained_ckpt_root, 'ckpt_task{}_iter_newest.pth.tar'.format(reid_index))
reid_transed_ckpt_save_path = os.path.join(save_root, 'reid', expr_name+'.pth')

reid_model = torch.load(reid_train_model_path, map_location=torch.device('cpu'))

cnt = 0
traned_ckpt = collections.OrderedDict()
for name, param in mae_model['model'].items():
    trained_model_name = 'module.backbone_module.' + name
    if trained_model_name in reid_model['state_dict']:
        cnt += 1
        traned_ckpt[name] = reid_model['state_dict'][trained_model_name]
    else:
        traned_ckpt[name] = mae_model['model'][name]

torch.save({'model': traned_ckpt}, reid_transed_ckpt_save_path)
print('done! transed ckpt saved at: {}'.format(reid_transed_ckpt_save_path))

# parsing
parsing_train_model_path = os.path.join(trained_ckpt_root, 'ckpt_task{}_iter_newest.pth.tar'.format(parsing_index))
parsing_transed_ckpt_save_path = os.path.join(save_root, 'parsing', expr_name+'.pth')

parsing_model = torch.load(parsing_train_model_path, map_location=torch.device('cpu'))

cnt = 0
traned_ckpt = collections.OrderedDict()
for name, param in mae_model['model'].items():
    trained_model_name = 'module.backbone_module.' + name
    if trained_model_name in parsing_model['state_dict']:
        cnt += 1
        traned_ckpt[name] = parsing_model['state_dict'][trained_model_name]
    else:
        traned_ckpt[name] = mae_model['model'][name]

torch.save({'model': traned_ckpt}, parsing_transed_ckpt_save_path)
print('done! transed ckpt saved at: {}'.format(parsing_transed_ckpt_save_path))