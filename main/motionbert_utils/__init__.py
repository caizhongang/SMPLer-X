from MotionBERT.lib.utils.learning import load_backbone
from easydict import EasyDict as edict


def get_backbone(type):

    if type == 'MB_pretrain':
        args = {
            'maxlen': 243,
            'dim_feat': 512,
            'mlp_ratio': 2,
            'depth': 5,
            'dim_rep': 512,
            'num_heads': 8,
            'att_fuse': True,
            # 'hidden_dim': 1024
        }
    elif type == 'MB_lite':
        args = {
            'maxlen': 243,
            'dim_feat': 256,
            'mlp_ratio': 4,
            'depth': 5,
            'dim_rep': 512,
            'num_heads': 8,
            'att_fuse': True,
        }
    elif type == 'MB_ft_h36m':
        args = {
            'maxlen': 243,
            'dim_feat': 512,
            'mlp_ratio': 2,
            'depth': 5,
            'dim_rep': 512,
            'num_heads': 8,
            'att_fuse': True,
        }
    elif type == 'MB_ft_pw3d':
        args = {
            'maxlen': 243,
            'dim_feat': 512,
            'mlp_ratio': 2,
            'depth': 5,
            'dim_rep': 512,
            'num_heads': 8,
            'att_fuse': True,
            # 'hidden_dim': 1024,
        }
    else:
        raise Exception("Undefined backbone type: {}".format(type))

    args = edict(args)
    model_backbone = load_backbone(args)

    return model_backbone


