from base_vit import ViT
from lora import LoRA_ViT
from torchvision.models.segmentation.deeplabv3 import DeepLabHead
import torch
import torch.nn as nn
import torch.nn.functional as F


def as_tuple(x):
    return x if isinstance(x, tuple) else (x, x)


class SegWrapForViT(nn.Module):
    def __init__(self,
                 vit_model: ViT,
                 image_size: int,
                 patches: int,
                 dim: int,
                 n_classes: int,
                 ) -> None:
        super().__init__()
        self.vit = vit_model
        if isinstance(self.vit, ViT):
            del self.vit.fc  # so hasattr(self, 'fc') == False
        elif isinstance(self.vit, LoRA_ViT):
            del self.vit.lora_vit.fc  # so hasattr(self, 'fc') == False
        self.deep_lab_head = DeepLabHead(dim, n_classes)

        h, w = as_tuple(image_size)  # image sizes
        fh, fw = as_tuple(patches)  # patch sizes
        self.gh, self.gw = h // fh, w // fw  # number of patches
        self.h, self.w = h, w

    def forward(self, x):
        x = self.vit(x)  # b,gh*gw+1,d
        b, gh_gw, d = x.shape
        x = x[:, :-1, :]  # b,gh*gw,d, remove the class token
        x = x.transpose(1, 2)
        x = x.reshape(b, d, self.gh, self.gw)
        x = self.deep_lab_head(x)
        x = F.interpolate(x,
                          size=(self.h, self.w),
                          mode='bilinear',
                          align_corners=False)
        return x


if __name__ == "__main__":  # Debug
    img = torch.randn(2, 3, 384, 384)
    model = ViT('B_16_imagenet1k')
    model.load_state_dict(torch.load('B_16_imagenet1k.pth'))
    seg_vit = SegWrapForViT(vit_model=model, image_size=384,
                            patches=16, dim=768, n_classes=10)
    mask = seg_vit(img)
    print(mask.shape)
