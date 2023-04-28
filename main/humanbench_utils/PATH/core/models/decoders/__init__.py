from .classification_decoders.reid_decoders import *
from .pose_decodes.pose_decoder import TopDownSimpleHead
from .seg_decoders.seg_decoders import HRT_OCR_V2, ViT_OCR_V2, ViT_SimpleUpSampling
from .peddet_decoders import AIOHead
from .classification_decoders.pedattr_decoders import *
def decoder_entry(config):
    return globals()[config['type']](**config['kwargs'])
