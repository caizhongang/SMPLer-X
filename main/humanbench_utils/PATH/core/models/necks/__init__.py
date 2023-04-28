from .DoNothing import *
# from .simple_fpn import (SimpleFPN, MoreSimpleFPN, PoseSimpleFPN, SimpleNeck, PedDetMoreSimpleFPN)
from .ladder_side_attention_fpn import LadderSideAttentionFPN

def neck_entry(config):
    return globals()[config['type']](**config['kwargs'])
