import os
import sys
import collections
import torch
import numpy as np

sys.path.append('/mnt/cache/chencheng1/vitruvian/vitruvian-multitask')

mae_pretrain_path = '/mnt/cache/chencheng1/vitruvian/vitruvian-multitask/core/models/backbones/pretrain_weights/mae_pretrain_vit_base.pth'
mae_model = torch.load(mae_pretrain_path)

save_root = '/mnt/lustre/share_data/chencheng1/vitruvian/L2_final_base'

root = '/mnt/lustre/chencheng1/expr_files/vitruvian/L2_full_setting_joint/checkpoints'
config_lists = [
    'v100_32g_vitbase_size224_lr1e3_stepLRx3_bmp1_adafactor_wd01_clip05_layerdecay075_lpe_peddet_citypersons_LSA_reduct8_tbn1_heads2_gate1_peddetShareDecoder_exp3_setting_SharePosEmbed'
]

for config in config_lists:
    trained_ckpt_root = os.path.join(root, config)
    expr_name = trained_ckpt_root.split('/')[-1]

    wo_cls_token_index = 0  # coco
    with_cls_token_index = 20  # reid_4set


    # with_cls_token-reid
    with_cls_token_train_model_path = os.path.join(trained_ckpt_root, 'ckpt_task{}_iter_newest.pth.tar'.format(with_cls_token_index))
    with_cls_token_transed_ckpt_save_path = os.path.join(save_root, 'with_cls_token', expr_name+'.pth')
    with_cls_token_train_model = torch.load(with_cls_token_train_model_path, map_location=torch.device('cpu'))

    cnt = 0
    traned_ckpt = collections.OrderedDict()
    for name, param in mae_model['model'].items():
        trained_model_name = 'module.backbone_module.' + name
        if trained_model_name in with_cls_token_train_model['state_dict']:
            if name == 'pos_embed':
                cnt += 1
                traned_ckpt[name] = torch.cat([with_cls_token_train_model['state_dict']['module.backbone_module.cls_token_pos_embed'], with_cls_token_train_model['state_dict'][trained_model_name]], dim=1)
            else:
                cnt += 1
                traned_ckpt[name] = with_cls_token_train_model['state_dict'][trained_model_name]
        else:
            traned_ckpt[name] = mae_model['model'][name]

    torch.save({'model': traned_ckpt}, with_cls_token_transed_ckpt_save_path)
    print('done! transed ckpt saved at: {}'.format(with_cls_token_transed_ckpt_save_path))


    # wo_cls_token-reid
    wo_cls_token_train_model_path = os.path.join(trained_ckpt_root, 'ckpt_task{}_iter_newest.pth.tar'.format(wo_cls_token_index))
    wo_cls_token_transed_ckpt_save_path = os.path.join(save_root, 'wo_cls_token', expr_name+'.pth')
    wo_cls_token_train_model = torch.load(wo_cls_token_train_model_path, map_location=torch.device('cpu'))

    cnt = 0
    traned_ckpt = collections.OrderedDict()
    for name, param in mae_model['model'].items():
        trained_model_name = 'module.backbone_module.' + name
        if trained_model_name in wo_cls_token_train_model['state_dict']:
            cnt += 1
            traned_ckpt[name] = wo_cls_token_train_model['state_dict'][trained_model_name]
        else:
            traned_ckpt[name] = mae_model['model'][name]

    torch.save({'model': traned_ckpt}, wo_cls_token_transed_ckpt_save_path)
    print('done! transed ckpt saved at: {}'.format(wo_cls_token_transed_ckpt_save_path))