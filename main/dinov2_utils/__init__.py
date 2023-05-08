""" Ref: https://github.com/facebookresearch/dinov2/blob/fc49f49d734c767272a4ea0e18ff2ab8e60fc92d/dinov2/eval/setup.py#L63-L68 """

from functools import partial
from dinov2.models.vision_transformer import DinoVisionTransformerForSMPLX, Block, MemEffAttention
import torch

def build_model(args, only_teacher=False, img_size=224):
    args.arch = args.arch.removesuffix("_memeff")
    if "vit" in args.arch:
        vit_kwargs = dict(
            img_size=img_size,
            patch_size=args.patch_size,
            init_values=args.layerscale,
            ffn_layer=args.ffn_layer,
            block_chunks=args.block_chunks,
            qkv_bias=args.qkv_bias,
            proj_bias=args.proj_bias,
            ffn_bias=args.ffn_bias,
        )
        teacher = vits.__dict__[args.arch](**vit_kwargs)
        if only_teacher:
            return teacher, teacher.embed_dim
        student = vits.__dict__[args.arch](
            **vit_kwargs,
            drop_path_rate=args.drop_path_rate,
            drop_path_uniform=args.drop_path_uniform,
        )
        embed_dim = student.embed_dim
    return student, teacher, embed_dim



def vit_small(patch_size=16, **kwargs):
    model = DinoVisionTransformer(
        patch_size=patch_size,
        embed_dim=384,
        depth=12,
        num_heads=6,
        mlp_ratio=4,
        block_fn=partial(Block, attn_class=MemEffAttention),
        **kwargs,
    )
    return model


def vit_base(patch_size=16, **kwargs):
    model = DinoVisionTransformer(
        patch_size=patch_size,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        block_fn=partial(Block, attn_class=MemEffAttention),
        **kwargs,
    )
    return model


def vit_large(patch_size=16, **kwargs):
    model = DinoVisionTransformer(
        patch_size=patch_size,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4,
        block_fn=partial(Block, attn_class=MemEffAttention),
        **kwargs,
    )
    return model


def vit_giant2(patch_size=16, **kwargs):
    """
    Close to ViT-giant, with embed-dim 1536 and 24 heads => embed-dim per head 64
    """
    model = DinoVisionTransformer(
        patch_size=patch_size,
        embed_dim=1536,
        depth=40,
        num_heads=24,
        mlp_ratio=4,
        block_fn=partial(Block, attn_class=MemEffAttention),
        **kwargs,
    )
    return model





def get_backbone(type):
    """ Ref: 
        https://github.com/facebookresearch/dinov2/blob/c3c2683a13cde94d4d99f523cf4170384b00c34c/dinov2/utils/utils.py#L21 
        https://github.com/facebookresearch/dinov2/blob/fc49f49d734c767272a4ea0e18ff2ab8e60fc92d/dinov2/models/__init__.py#L15
        https://github.com/facebookresearch/dinov2/blob/main/dinov2/configs/train/vitl14.yaml
        https://github.com/facebookresearch/dinov2/blob/fc49f49d734c767272a4ea0e18ff2ab8e60fc92d/dinov2/configs/eval/vitl14_pretrain.yaml
    """
    patch_size = 14  # added self.resize to allow minimum changes to pretrain model and architecture, basically resizing (256, 192) for 16 patch -> (224, 168) for (16, 12) patch
    global_crops_size = 518  # to allow pretrained model to load, but actual image size is (256, 192). interpolate_pos_encoding() handles the mismatch
    kwargs = {
        'drop_path_rate': 0.4,
        'ffn_layer': 'swiglufused',
        'block_chunks': 4,
        'img_size': global_crops_size
    }
  
    if type == 'vits':
        model = DinoVisionTransformerForSMPLX(
            patch_size=patch_size,
            embed_dim=384,
            depth=12,
            num_heads=6,
            mlp_ratio=4,
            block_fn=partial(Block, attn_class=MemEffAttention),
            **kwargs,
        )

    elif type == 'vitb':
        model = DinoVisionTransformerForSMPLX(
            patch_size=patch_size,
            embed_dim=768,
            depth=12,
            num_heads=12,
            mlp_ratio=4,
            block_fn=partial(Block, attn_class=MemEffAttention),
            **kwargs,
        )

    elif type == 'vitl':
        model = DinoVisionTransformerForSMPLX(
            patch_size=patch_size,
            embed_dim=1024,
            depth=24,
            num_heads=16,
            mlp_ratio=4,
            block_fn=partial(Block, attn_class=MemEffAttention),
            **kwargs,
        )

    elif type == 'vitg':
        model = DinoVisionTransformerForSMPLX(
            patch_size=patch_size,
            embed_dim=1536,
            depth=40,
            num_heads=24,
            mlp_ratio=4,
            block_fn=partial(Block, attn_class=MemEffAttention),
            **kwargs,
        )

    else:
        raise NotImplementedError('backbone type not implemented: {}'.format(type))

    return model



# def load_pretrained_weights(model, pretrained_weights, checkpoint_key='teacher'):
#     if urlparse(pretrained_weights).scheme:  # If it looks like an URL
#         state_dict = torch.hub.load_state_dict_from_url(pretrained_weights, map_location="cpu")
#     else:
#         state_dict = torch.load(pretrained_weights, map_location="cpu")
#     if checkpoint_key is not None and checkpoint_key in state_dict:
#         logger.info(f"Take key {checkpoint_key} in provided checkpoint dict")
#         state_dict = state_dict[checkpoint_key]
#     # remove `module.` prefix
#     state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
#     # remove `backbone.` prefix induced by multicrop wrapper
#     state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}
#     msg = model.load_state_dict(state_dict, strict=False)
#     logger.info("Pretrained weights found at {} and loaded with msg: {}".format(pretrained_weights, msg))


def load_checkpoint(model, pretrained_weights_path, checkpoint_key='teacher'):
    """ Ref: https://github.com/facebookresearch/dinov2/blob/c3c2683a13cde94d4d99f523cf4170384b00c34c/dinov2/utils/utils.py#L21 """
    state_dict = torch.load(pretrained_weights_path, map_location="cpu")
    if checkpoint_key is not None and checkpoint_key in state_dict:
        logger.info(f"Take key {checkpoint_key} in provided checkpoint dict")
        state_dict = state_dict[checkpoint_key]
    # remove `module.` prefix
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    # remove `backbone.` prefix induced by multicrop wrapper
    state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict, strict=False)