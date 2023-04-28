# from PATH.core.models.backbones.vitdet import vit_base_patch16
# from PATH.core.models.backbones.vitdet_for_ladder_attention import vit_base_patch16_ladder_attention, vit_large_patch16_ladder_attention
from PATH.core.models.backbones.vitdet_for_ladder_attention_share_pos_embed import vit_base_patch16_ladder_attention_share_pos_embed, vit_large_patch16_ladder_attention_share_pos_embed, load_checkpoint

def backbone_entry(config):
    return globals()[config['type']](**config['kwargs'])

def get_backbone(type):

    if type == 'vit_large_patch16_ladder_attention_share_pos_embed':
          config = {
            'type': 'vit_large_patch16_ladder_attention_share_pos_embed',
            'kwargs': {
                'task_sp_list': ['rel_pos_h', 'rel_pos_w'],  # wrong list would somehow cause .cuda() stuck without error
                # 'pretrained': True,
                'pretrained': False,  # TODO: fix for our study
                'img_size': [480, 480],  # deprecated by simple interpolate
                'lms_checkpoint_train': "fairscale",
                'window': False,
                'test_pos_mode': "learnable_simple_interpolate",
                'pad_attn_mask': False,
                'round_padding': True,
                'learnable_pos': True,
                'drop_path_rate': 0.3
            }
        }
    elif type == 'vit_base_patch16_ladder_attention_share_pos_embed':
        config = {
            'type': 'vit_base_patch16_ladder_attention_share_pos_embed',
            'kwargs': {
                'task_sp_list': ['rel_pos_h', 'rel_pos_w'],  # wrong list would somehow cause .cuda() stuck without error
                # 'pretrained': True,
                'pretrained': False,  # TODO: fix for our study
                'img_size': [224, 224],  # deprecated by simple interpolate
                # 'lms_checkpoint_train': "fairscale",
                'lms_checkpoint_train': False,  # TODO: fix for our study
                'window': False,
                'test_pos_mode': "learnable_simple_interpolate",
                'pad_attn_mask': False,
                'round_padding': True,
                'learnable_pos': True,
                'drop_path_rate': 0.3
            }
        }
    else:
        raise NotImplementedError('backbone type not implemented: {}'.format(type))

    return backbone_entry(config)
