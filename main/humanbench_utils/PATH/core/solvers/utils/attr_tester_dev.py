import argparse
import math
import os, sys
import random
import datetime
import time
from typing import List
import json
import logging
import numpy as np
from copy import deepcopy
from .seg_tester_dev import DatasetEvaluator

import torch
import torch.nn as nn
import torch.nn.parallel
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
try:
    import spring.linklink as link
except:
    import linklink as link

import torch.optim
import torch.multiprocessing as mp
import torch.utils.data

class PedAttrEvaluator(DatasetEvaluator):

    def __init__(
        self,
        dataset_name,
        config,
        distributed=True,
        output_dir=None,
    ):

        self._logger = logging.getLogger(__name__)

        self._dataset_name = dataset_name
        self._distributed = distributed
        self._output_dir = output_dir

        self._cpu_device = torch.device("cpu")
        self.threshold = 0.5

    def reset(self):
        self.gt_label = []
        self.preds_probs = []

    def process(self, inputs, outputs):
        gt_label = inputs['label']
        gt_label[gt_label == -1] = 0
        preds_probs = outputs['pred_logits'].squeeze().sigmoid()
        self.gt_label.append(gt_label)
        self.preds_probs.append(preds_probs)

    @staticmethod
    def all_gather(data, group=0):
        assert link.get_world_size() == 1, f"distributed eval unsupported yet, uncertain if we can use torch.dist with link jointly"
        if link.get_world_size() == 1:
            return [data]

        world_size = link.get_world_size()
        tensors_gather = [torch.ones_like(data) for _ in range(world_size)]
        link.allgather(tensors_gather, data, group=group)
        return tensors_gather

    def evaluate(self):
        gt_label = torch.cat(self.gt_label, dim=0)
        preds_probs = torch.cat(self.preds_probs, dim=0)

        if self._distributed:
            link.synchronize()

            gt_label = self.all_gather(gt_label)
            preds_probs = self.all_gather(preds_probs)

            if link.get_rank() != 0:
                return

        gt_label = torch.cat(gt_label, dim=0)
        preds_probs = torch.cat(preds_probs, dim=0)
        preds_probs = preds_probs.cpu().numpy()
        gt_label = gt_label.cpu().numpy()

        pred_label = preds_probs > self.threshold

        eps = 1e-20
        result = {}

        ###############################
        # label metrics
        # TP + FN
        gt_pos = np.sum((gt_label == 1), axis=0).astype(float)
        # TN + FP
        gt_neg = np.sum((gt_label == 0), axis=0).astype(float)
        # TP
        true_pos = np.sum((gt_label == 1) * (pred_label == 1), axis=0).astype(float)
        # TN
        true_neg = np.sum((gt_label == 0) * (pred_label == 0), axis=0).astype(float)
        # FP
        false_pos = np.sum(((gt_label == 0) * (pred_label == 1)), axis=0).astype(float)
        # FN
        false_neg = np.sum(((gt_label == 1) * (pred_label == 0)), axis=0).astype(float)

        label_pos_recall = 1.0 * true_pos / (gt_pos + eps)  # true positive
        label_neg_recall = 1.0 * true_neg / (gt_neg + eps)  # true negative
        # mean accuracy
        label_ma = (label_pos_recall + label_neg_recall) / 2

        result['label_pos_recall'] = label_pos_recall
        result['label_neg_recall'] = label_neg_recall
        result['label_prec'] = true_pos / (true_pos + false_pos + eps)
        result['label_acc'] = true_pos / (true_pos + false_pos + false_neg + eps)
        result['label_f1'] = 2 * result['label_prec'] * result['label_pos_recall'] / (
                result['label_prec'] + result['label_pos_recall'] + eps)

        result['label_ma'] = label_ma
        result['ma'] = np.mean(label_ma)

        ################
        # instance metrics
        gt_pos = np.sum((gt_label == 1), axis=1).astype(float)
        true_pos = np.sum((pred_label == 1), axis=1).astype(float)
        # true positive
        intersect_pos = np.sum((gt_label == 1) * (pred_label == 1), axis=1).astype(float)
        # IOU
        union_pos = np.sum(((gt_label == 1) + (pred_label == 1)), axis=1).astype(float)

        instance_acc = intersect_pos / (union_pos + eps)
        instance_prec = intersect_pos / (true_pos + eps)
        instance_recall = intersect_pos / (gt_pos + eps)
        instance_f1 = 2 * instance_prec * instance_recall / (instance_prec + instance_recall + eps)

        instance_acc = np.mean(instance_acc)
        instance_prec = np.mean(instance_prec)
        instance_recall = np.mean(instance_recall)
        instance_f1 = np.mean(instance_f1)

        result['instance_acc'] = instance_acc
        result['instance_prec'] = instance_prec
        result['instance_recall'] = instance_recall
        result['instance_f1'] = instance_f1

        result['error_num'], result['fn_num'], result['fp_num'] = false_pos + false_neg, false_neg, false_pos

        result['pos_recall'] = np.mean(label_pos_recall)
        result['neg_recall'] = np.mean(label_neg_recall)
        return result
