import itertools
from torch.nn.utils import clip_grad_norm_
from torch.optim import Adam, AdamW


class AdamWithClip(Adam):

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0, amsgrad=False, max_norm=None, norm_type=2):
        super(AdamWithClip, self).__init__(params, lr, betas, eps, weight_decay, amsgrad)
        self.max_norm = max_norm
        self.norm_type = norm_type

    def step(self, closure=None):
        if self.max_norm is not None:
            for group in self.param_groups:
                clip_grad_norm_(group['params'], self.max_norm, self.norm_type)
        super(AdamWithClip, self).step(closure)


class AdamWWithClip(AdamW):

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0, amsgrad=False, max_norm=None, norm_type=2):
        super(AdamWWithClip, self).__init__(params, lr, betas, eps, weight_decay, amsgrad)
        self.max_norm = max_norm
        self.norm_type = norm_type

    def step(self, closure=None):

        if self.max_norm is not None:
            for group in self.param_groups:
                clip_grad_norm_(group['params'], self.max_norm, self.norm_type)
        super(AdamWWithClip, self).step(closure)


# class AdamWWithClipDev(AdamW):
#     def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
#                  weight_decay=1e-2, amsgrad=False, clip_norm=None, norm_type=2):
#         super(AdamWWithClipDev, self).__init__(params, lr, betas, eps, weight_decay, amsgrad)
#         self.clip_norm = clip_norm
#         self.norm_type = norm_type
#
#     def step(self, closure=None):
#         if self.clip_norm is not None:
#             all_params = itertools.chain(*[x["params"] for x in self.param_groups])
#             clip_grad_norm_(all_params, self.clip_norm, self.norm_type)
#
#         super(AdamWWithClipDev, self).step(closure)

class AdamWWithBackboneClipDev(AdamW):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=1e-2, amsgrad=False, clip_norm=None, norm_type=2):
        super(AdamWWithBackboneClipDev, self).__init__(params, lr, betas, eps, weight_decay, amsgrad)
        self.clip_norm = clip_norm
        self.norm_type = norm_type

    def step(self, closure=None):
        if self.clip_norm is not None:
            all_params = itertools.chain(*[x["params"] for x in self.param_groups if x['params'][0].backbone_specific ])
            clip_grad_norm_(all_params, self.clip_norm, self.norm_type)

        super(AdamWWithBackboneClipDev, self).step(closure)

class AdamWWithClipDev(AdamW):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=1e-2, amsgrad=False, clip_norm=None, norm_type=2):
        super(AdamWWithClipDev, self).__init__(params, lr, betas, eps, weight_decay, amsgrad)
        self.clip_norm = clip_norm
        self.norm_type = norm_type

        self._split_param_groups = None
        self.reset_split_param_groups()

    def reset_split_param_groups(self):
        if self.clip_norm is not None:
            backbone_param, neck_param, decoder_param, task_param = [], [], [], []
            for x in self.param_groups:
                if x["params"][0].backbone_specific:
                    backbone_param.append(x["params"])
                elif x["params"][0].neck_specific:
                    neck_param.append(x["params"])
                elif x["params"][0].decoder_specific:
                    decoder_param.append(x["params"])
                elif x["params"][0].task_specific:
                    task_param.append(x["params"])
            self._split_param_groups = [_g for _g in [backbone_param,
                                                      neck_param,
                                                      decoder_param,
                                                      task_param] if len(_g) > 0]
            print(f">>> reset_split_param_groups, backbone_param: {len(backbone_param)}"
                  f", neck_param: {len(neck_param)}, decoder_param: {len(decoder_param)}"
                  f", task_param: {len(task_param)}")

    def step(self, closure=None):
        if self.clip_norm is not None:
            for _g in self._split_param_groups:
                all_params = itertools.chain(*_g)
                clip_grad_norm_(all_params, self.clip_norm, self.norm_type)

        super(AdamWWithClipDev, self).step(closure)

class AdamWWithBackboneClipDev(AdamW):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=1e-2, amsgrad=False, clip_norm=None, norm_type=2):
        super(AdamWWithBackboneClipDev, self).__init__(params, lr, betas, eps, weight_decay, amsgrad)
        self.clip_norm = clip_norm
        self.norm_type = norm_type

    def step(self, closure=None):
        if self.clip_norm is not None:
            all_params = itertools.chain(*[x["params"] for x in self.param_groups if x["params"][0].backbone_specific])
            clip_grad_norm_(all_params, self.clip_norm, self.norm_type)
        # import pdb; pdb.set_trace()
        super(AdamWWithBackboneClipDev, self).step(closure)
