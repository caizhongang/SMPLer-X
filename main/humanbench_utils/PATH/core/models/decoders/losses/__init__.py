from .classification_losses import MarginCosineProductLoss
from .classification_losses import ArcFaceLoss, Softmax_TripletLoss, Softmax_TripletLoss_wBN
from .seg_losses import FSAuxCELoss, FocalDiceLoss, FocalDiceLoss_bce_cls_emb, FSCELoss, FocalDiceLoss_bce_cls_emb_sample_weight
from .pos_losses import BasePosLoss
from .peddet_losses import DetFocalDiceLoss
from .pedattr_losses import CEL_Sigmoid

def loss_entry(config):
    return globals()[config['type']](**config['kwargs'])
