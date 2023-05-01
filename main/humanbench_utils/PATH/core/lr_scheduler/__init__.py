from .base import StepLRScheduler, CosineLRScheduler, WarmupCosineLRScheduler, WarmupPolyLRScheduler

def lr_scheduler_entry(config):
    return globals()[config['type']+'LRScheduler'](**config['kwargs'])
