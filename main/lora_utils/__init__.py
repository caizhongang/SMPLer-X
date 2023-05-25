""" Ref: https://github.com/JamesQFreeman/LoRA-ViT """

from .lora import LoRA_ViT_timm

def apply_adapter(model):
    model = LoRA_ViT_timm(model, r=4)
    return model
