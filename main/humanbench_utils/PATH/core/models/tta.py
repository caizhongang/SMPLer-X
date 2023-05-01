# Copyright (c) Facebook, Inc. and its affiliates.
import copy

import numpy as np
import torch
# from fvcore.transforms import HFlipTransform
from torch import nn
from torch.nn.parallel import DistributedDataParallel

from core.distributed_utils import DistModule
from core.data.transforms.seg_aug_dev import (
    RandomFlip,
    ResizeShortestEdge,
    ResizeTransform,
    HFlipTransform,
    NoOpTransform
)
from core.data.transforms.seg_transforms_dev import apply_augmentations

__all__ = [
    "SemanticSegmentorWithTTA",
]


class SemanticSegmentorWithTTA(nn.Module):
    """
    A SemanticSegmentor with test-time augmentation enabled.
    Its :meth:`__call__` method has the same interface as :meth:`SemanticSegmentor.forward`.

    combined with customized augmentation for original image
    """

    def __init__(self, cfg, model, batch_size=1):
        """
        Args:
            cfg (CfgNode):
            model (SemanticSegmentor): a SemanticSegmentor to apply TTA on.
            tta_mapper (callable): takes a dataset dict and returns a list of
                augmented versions of the dataset dict. Defaults to
                `DatasetMapperTTA(cfg)`.
            batch_size (int): batch the augmented images into this batch size for inference.
        """
        super().__init__()
        if isinstance(model, DistributedDataParallel) or isinstance(model, DistModule):
            model = model.module
        self.cfg = cfg

        self.min_sizes = cfg.min_sizes
        self.max_size = cfg.max_size
        self.flip = cfg.flip

        self.model = model

        # if tta_mapper is None:
        #     tta_mapper = DatasetMapperTTA(cfg)
        # self.tta_mapper = tta_mapper
        assert batch_size == 1
        self.batch_size = batch_size

    def tta_mapper(self, dataset_dict):
        """
        Args:
            dict: a dict in standard model input format. See tutorials for details.

        Returns:
            list[dict]:
                a list of dicts, which contain augmented version of the input image.
                The total number of dicts is ``len(min_sizes) * (2 if flip else 1)``.
                Each dict has field "transforms" which is a TransformList,
                containing the transforms that are used to generate this image.
        """
        assert len(dataset_dict["image"].shape) == 4
        numpy_image = dataset_dict["image"].squeeze().permute(1, 2, 0).cpu().numpy()
        shape = numpy_image.shape
        orig_shape = (dataset_dict["height"], dataset_dict["width"])
        if shape[:2] != orig_shape:
            # It transforms the "original" image in the dataset to the input image
            pre_tfm = ResizeTransform(orig_shape[0], orig_shape[1], shape[0], shape[1])
        else:
            pre_tfm = NoOpTransform()

        # Create all combinations of augmentations to use
        aug_candidates = []  # each element is a list[Augmentation]
        for min_size in self.min_sizes:
            resize = ResizeShortestEdge(min_size, self.max_size)
            aug_candidates.append([resize])  # resize only
            if self.flip:
                flip = RandomFlip(prob=1.0)
                aug_candidates.append([resize, flip])  # resize + flip

        # Apply all the augmentations
        ret = []
        for aug in aug_candidates:
            new_image, tfms = apply_augmentations(aug, np.copy(numpy_image))
            torch_image = torch.from_numpy(np.ascontiguousarray(new_image.transpose(2, 0, 1)))

            torch_image = torch_image.unsqueeze(0)

            dic = copy.deepcopy(dataset_dict)
            dic["transforms"] = pre_tfm + tfms
            dic["image"] = torch_image.cuda()
            ret.append(dic)
        return ret

    def __call__(self, batched_inputs, current_step):
        """
        Same input/output format as :meth:`SemanticSegmentor.forward`
        """
        self.current_step = current_step  # redundant param for api compliance
        def _maybe_read_image(dataset_dict):
            ret = copy.copy(dataset_dict)
            if "image" not in ret:
                raise
            if "height" not in ret and "width" not in ret:  # TODO: BUG HERE
                raise
                # ret["height"] = ret["ori_image"].shape[1]#ret["image"].shape[1]
                # ret["width"] = ret["ori_image"].shape[2]
            return ret

        processed_results = []
        # for x in batched_inputs:
        result = self._inference_one_image(_maybe_read_image(batched_inputs))
        processed_results.append(result)
        return processed_results

    def _inference_one_image(self, input):
        """
        Args:
            input (dict): one dataset dict with "image" field being a CHW tensor
        Returns:
            dict: one output dict
        """
        augmented_inputs, tfms = self._get_augmented_inputs(input)

        final_predictions = None
        count_predictions = 0
        for input, tfm in zip(augmented_inputs, tfms):
            count_predictions += 1
            with torch.no_grad():
                if final_predictions is None:
                    if any(isinstance(t, HFlipTransform) for t in tfm.transforms):
                        final_predictions = self.model(input, self.current_step)[0].pop("sem_seg").flip(dims=[2])  # should be [input] originally
                    else:
                        final_predictions = self.model(input, self.current_step)[0].pop("sem_seg")
                else:
                    if any(isinstance(t, HFlipTransform) for t in tfm.transforms):
                        final_predictions += self.model(input, self.current_step)[0].pop("sem_seg").flip(dims=[2])
                    else:
                        final_predictions += self.model(input, self.current_step)[0].pop("sem_seg")

        final_predictions = final_predictions / count_predictions
        return {"sem_seg": final_predictions}

    def _get_augmented_inputs(self, input):
        augmented_inputs = self.tta_mapper(input)
        tfms = [x.pop("transforms") for x in augmented_inputs]
        return augmented_inputs, tfms
