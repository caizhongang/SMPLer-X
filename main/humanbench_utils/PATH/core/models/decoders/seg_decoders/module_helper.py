import functools
import os
import pdb
import math

import torch
import torch.nn as nn

try:
    from urllib import urlretrieve
except ImportError:
    from urllib.request import urlretrieve

try:
    import spring.linklink as link
    from spring.linklink.nn import SyncBatchNorm2d
except:
    import linklink as link
    from linklink.nn import SyncBatchNorm2d


class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first. 
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with 
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs 
    with shape (batch_size, channels, height, width).
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError 
        self.normalized_shape = (normalized_shape, )
    
    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x
            

class ModuleHelper(object):
    @staticmethod
    def BNReLU(num_features, bn_type=None, bn_group=None, **kwargs):
        if bn_type == "torchbn":
            return nn.Sequential(nn.BatchNorm2d(num_features, **kwargs), nn.ReLU())
        elif bn_type == "syncBN":
            return nn.Sequential(SyncBatchNorm2d(num_features=num_features, group=bn_group, sync_stats=False), nn.ReLU())
        elif bn_type == "LN":
            return nn.Sequential(LayerNorm(num_features, data_format="channels_first"), nn.ReLU())
        elif bn_type == "gn":
            return nn.Sequential(
                nn.GroupNorm(num_groups=8, num_channels=num_features, **kwargs),
                nn.ReLU(),
            )
        else:
            raise ValueError("Not support BN type: {}.".format(bn_type))
            exit(1)

    @staticmethod
    def BatchNorm2d(bn_type="torch", ret_cls=False):
        if bn_type == "torchbn":
            return nn.BatchNorm2d

        elif bn_type == "torchsyncbn":
            return nn.SyncBatchNorm

        elif bn_type == "syncbn":
            from lib.extensions.syncbn.module import BatchNorm2d

            return BatchNorm2d

        elif bn_type == "sn":
            from lib.extensions.switchablenorms.switchable_norm import SwitchNorm2d

            return SwitchNorm2d

        elif bn_type == "gn":
            return functools.partial(nn.GroupNorm, num_groups=32)

        elif bn_type == "inplace_abn":
            torch_ver = torch.__version__[:3]
            if torch_ver == "0.4":
                from lib.extensions.inplace_abn.bn import InPlaceABNSync

                if ret_cls:
                    return InPlaceABNSync
                return functools.partial(InPlaceABNSync, activation="none")

            elif torch_ver in ("1.0", "1.1"):
                from lib.extensions.inplace_abn_1.bn import InPlaceABNSync

                if ret_cls:
                    return InPlaceABNSync
                return functools.partial(InPlaceABNSync, activation="none")

            elif torch_ver == "1.2":
                from inplace_abn import InPlaceABNSync

                if ret_cls:
                    return InPlaceABNSync
                return functools.partial(InPlaceABNSync, activation="identity")

        else:
            raise ValueError("Not support BN type: {}.".format(bn_type))
            exit(1)

    @staticmethod
    def load_model(model, pretrained=None, all_match=True, network="resnet101"):
        if pretrained is None:
            return model

        if all_match:
            # Log.info("Loading pretrained model:{}".format(pretrained))
            pretrained_dict = torch.load(pretrained)
            model_dict = model.state_dict()
            load_dict = dict()
            for k, v in pretrained_dict.items():
                if "resinit.{}".format(k) in model_dict:
                    load_dict["resinit.{}".format(k)] = v
                else:
                    load_dict[k] = v
            model.load_state_dict(load_dict)

        else:
            print("Loading pretrained model:{}".format(pretrained))
            pretrained_dict = torch.load(pretrained)

            # settings for "wide_resnet38"  or network == "resnet152"
            if network == "wide_resnet":
                pretrained_dict = pretrained_dict["state_dict"]

            model_dict = model.state_dict()

            if network == "hrnet_plus":
                # pretrained_dict['conv1_full_res.weight'] = pretrained_dict['conv1.weight']
                # pretrained_dict['conv2_full_res.weight'] = pretrained_dict['conv2.weight']
                load_dict = {
                    k: v for k, v in pretrained_dict.items() if k in model_dict.keys()
                }

            elif network == "hrt_window":
                pretrained_dict = pretrained_dict["model"]
                for name, m in model.named_parameters():
                    if "relative_position_bias_table" in name and "embed" not in name:
                        target_size = int(math.sqrt(m.shape[0]))
                        head_num = m.shape[-1]
                        ckpt_size = int(math.sqrt(pretrained_dict[name].shape[0]))
                        if target_size != ckpt_size:
                            # Log.info(
                            #     f"Interpolate from size {pretrained_dict[name ].shape} to {m.shape}."
                            # )
                            reshape_ckpt = (
                                pretrained_dict[name]
                                .permute(1, 0)
                                .reshape(1, head_num, ckpt_size, ckpt_size)
                            )
                            inter_ckpt = (
                                torch.nn.functional.interpolate(
                                    reshape_ckpt,
                                    size=(target_size, target_size),
                                    mode="bilinear",
                                )
                                .reshape(head_num, -1)
                                .permute(1, 0)
                            )
                            scale = 1
                            inter_ckpt *= scale
                            pretrained_dict[name] = inter_ckpt
                for name, m in list(pretrained_dict.items()):
                    if "relative_position_index" in name:
                        print(f"Remove {name}.")
                        # Log.info(f"Remove {name}.")
                        pretrained_dict.pop(name)
                load_dict = {
                    k: v for k, v in pretrained_dict.items() if k in model_dict.keys()
                }
                print("Missing keys: {}".format(list(set(model_dict) - set(load_dict))))
                # Log.info(
                #     "Missing keys: {}".format(list(set(model_dict) - set(load_dict)))
                # )

            elif network == "hrt":
                pretrained_dict = pretrained_dict["model"]
                load_dict = {
                    k: v for k, v in pretrained_dict.items() if k in model_dict.keys()
                }
                print("Missing keys: {}".format(list(set(model_dict) - set(load_dict))))
                # Log.info(
                #     "Missing keys: {}".format(list(set(model_dict) - set(load_dict)))
                # )

            elif network == "swin":
                pretrained_dict = pretrained_dict["model"]
                # TODO fix the mis-match between the dict keys and the checkpoint keys.
                pretrained_dict = {
                    k.replace(".attn.", ".attn.attn."): v
                    for k, v in pretrained_dict.items()
                }
                load_dict = {
                    k: v for k, v in pretrained_dict.items() if k in model_dict.keys()
                }
                print("Missing keys: {}".format(list(set(model_dict) - set(load_dict))))
                # Log.info(
                #     "Missing keys: {}".format(list(set(model_dict) - set(load_dict)))
                # )

            elif network == "hrnet" or network == "xception" or network == "resnest":
                load_dict = {
                    k: v for k, v in pretrained_dict.items() if k in model_dict.keys()
                }
                print("Missing keys: {}".format(list(set(model_dict) - set(load_dict))))
                
                # Log.info(
                #     "Missing keys: {}".format(list(set(model_dict) - set(load_dict)))
                # )

            elif network == "dcnet" or network == "resnext":
                load_dict = dict()
                for k, v in pretrained_dict.items():
                    if "resinit.{}".format(k) in model_dict:
                        load_dict["resinit.{}".format(k)] = v
                    else:
                        if k in model_dict:
                            load_dict[k] = v
                        else:
                            pass

            elif network == "wide_resnet":
                load_dict = {
                    ".".join(k.split(".")[1:]): v
                    for k, v in pretrained_dict.items()
                    if ".".join(k.split(".")[1:]) in model_dict
                }
            else:
                load_dict = {
                    ".".join(k.split(".")[1:]): v
                    for k, v in pretrained_dict.items()
                    if ".".join(k.split(".")[1:]) in model_dict
                }

            # used to debug
            # if int(os.environ.get("debug_load_model", 0)):
            #     Log.info("Matched Keys List:")
            #     for key in load_dict.keys():
            #         Log.info("{}".format(key))
            model_dict.update(load_dict)
            model.load_state_dict(model_dict)

        return model

    @staticmethod
    def load_url(url, map_location=None):
        model_dir = os.path.join("~", ".PyTorchCV", "models")
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

        filename = url.split("/")[-1]
        cached_file = os.path.join(model_dir, filename)
        if not os.path.exists(cached_file):
            # Log.info('Downloading: "{}" to {}\n'.format(url, cached_file))
            urlretrieve(url, cached_file)
        
        print("Loading pretrained model:{}".format(cached_file))
        # Log.info("Loading pretrained model:{}".format(cached_file))
        return torch.load(cached_file, map_location=map_location)

    @staticmethod
    def constant_init(module, val, bias=0):
        nn.init.constant_(module.weight, val)
        if hasattr(module, "bias") and module.bias is not None:
            nn.init.constant_(module.bias, bias)

    @staticmethod
    def xavier_init(module, gain=1, bias=0, distribution="normal"):
        assert distribution in ["uniform", "normal"]
        if distribution == "uniform":
            nn.init.xavier_uniform_(module.weight, gain=gain)
        else:
            nn.init.xavier_normal_(module.weight, gain=gain)
        if hasattr(module, "bias") and module.bias is not None:
            nn.init.constant_(module.bias, bias)

    @staticmethod
    def normal_init(module, mean=0, std=1, bias=0):
        nn.init.normal_(module.weight, mean, std)
        if hasattr(module, "bias") and module.bias is not None:
            nn.init.constant_(module.bias, bias)

    @staticmethod
    def uniform_init(module, a=0, b=1, bias=0):
        nn.init.uniform_(module.weight, a, b)
        if hasattr(module, "bias") and module.bias is not None:
            nn.init.constant_(module.bias, bias)

    @staticmethod
    def kaiming_init(
        module, mode="fan_in", nonlinearity="leaky_relu", bias=0, distribution="normal"
    ):
        assert distribution in ["uniform", "normal"]
        if distribution == "uniform":
            nn.init.kaiming_uniform_(
                module.weight, mode=mode, nonlinearity=nonlinearity
            )
        else:
            nn.init.kaiming_normal_(module.weight, mode=mode, nonlinearity=nonlinearity)
        if hasattr(module, "bias") and module.bias is not None:
            nn.init.constant_(module.bias, bias)
