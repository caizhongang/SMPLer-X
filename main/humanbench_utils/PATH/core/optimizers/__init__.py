from torch.optim import SGD, RMSprop, Adadelta, Adagrad, Adam, AdamW # noqa F401
from .lars import LARS  # noqa F401
from .adam_clip import AdamWithClip, AdamWWithClip, AdamWWithClipDev, AdamWWithBackboneClipDev  # noqa F401
from .adafactor import Adafactor_dev



def optim_entry(config):
    return globals()[config['type']](**config['kwargs'])
