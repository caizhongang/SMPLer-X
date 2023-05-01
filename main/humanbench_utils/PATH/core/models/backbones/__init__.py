from .vitdet import vit_base_patch16
from .vitdet_for_ladder_attention import vit_base_patch16_ladder_attention, vit_large_patch16_ladder_attention
from .vitdet_for_ladder_attention_share_pos_embed import vit_base_patch16_ladder_attention_share_pos_embed, vit_large_patch16_ladder_attention_share_pos_embed

def backbone_entry(config):
    return globals()[config['type']](**config['kwargs'])
