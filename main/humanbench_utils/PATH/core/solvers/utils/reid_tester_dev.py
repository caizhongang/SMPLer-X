# Copyright (c) Facebook, Inc. and its affiliates.
import itertools
import json
import logging
import os
from collections import OrderedDict

import numpy as np
import torch
from .seg_tester_dev import DatasetEvaluator

try:
    import spring.linklink as link
except:
    import linklink as link

def euclidean_distance(qf, gf):
    m = qf.shape[0]
    n = gf.shape[0]
    dist_mat = torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, n) + \
               torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    dist_mat.addmm_(1, -2, qf, gf.t())
    return dist_mat.cpu().numpy()


def cosine_similarity(qf, gf):
    epsilon = 0.00001
    dist_mat = qf.mm(gf.t())
    qf_norm = torch.norm(qf, p=2, dim=1, keepdim=True)  # mx1
    gf_norm = torch.norm(gf, p=2, dim=1, keepdim=True)  # nx1
    qg_normdot = qf_norm.mm(gf_norm.t())

    dist_mat = dist_mat.mul(1 / qg_normdot).cpu().numpy()
    dist_mat = np.clip(dist_mat, -1 + epsilon, 1 - epsilon)
    dist_mat = np.arccos(dist_mat)
    return dist_mat


def eval_func(distmat, q_pids, g_pids, q_camids, g_camids, max_rank=50):
    """Evaluation with market1501 metric
        Key: for each query identity, its gallery images from the same camera view are discarded.
        """
    num_q, num_g = distmat.shape
    # distmat g
    #    q    1 3 2 4
    #         4 1 2 3
    if num_g < max_rank:
        max_rank = num_g
        print("Note: number of gallery samples is quite small, got {}".format(num_g))
    indices = np.argsort(distmat, axis=1)
    #  0 2 1 3
    #  1 2 3 0
    matches = (g_pids[indices] == q_pids[:, np.newaxis]).astype(np.int32)
    # compute cmc curve for each query
    all_cmc = []
    all_AP = []
    num_valid_q = 0.  # number of valid query
    for q_idx in range(num_q):
        # get query pid and camid
        q_pid = q_pids[q_idx]
        q_camid = q_camids[q_idx]

        # remove gallery samples that have the same pid and camid with query
        order = indices[q_idx]  # select one row
        remove = (g_pids[order] == q_pid) & (g_camids[order] == q_camid)
        keep = np.invert(remove)

        # compute cmc curve
        # binary vector, positions with value 1 are correct matches
        orig_cmc = matches[q_idx][keep]
        if not np.any(orig_cmc):
            # this condition is true when query identity does not appear in gallery
            continue

        cmc = orig_cmc.cumsum()
        cmc[cmc > 1] = 1

        all_cmc.append(cmc[:max_rank])
        num_valid_q += 1.

        # compute average precision
        # reference: https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval)#Average_precision
        num_rel = orig_cmc.sum()
        tmp_cmc = orig_cmc.cumsum()
        #tmp_cmc = [x / (i + 1.) for i, x in enumerate(tmp_cmc)]
        y = np.arange(1, tmp_cmc.shape[0] + 1) * 1.0
        tmp_cmc = tmp_cmc / y
        tmp_cmc = np.asarray(tmp_cmc) * orig_cmc
        AP = tmp_cmc.sum() / num_rel
        all_AP.append(AP)

    assert num_valid_q > 0, "Error: all query identities do not appear in gallery"

    all_cmc = np.asarray(all_cmc).astype(np.float32)
    all_cmc = all_cmc.sum(0) / num_valid_q
    mAP = np.mean(all_AP)

    return all_cmc, mAP

class ReIDEvaluator(DatasetEvaluator):
    """
    Evaluate ReID metric
    """

    def __init__(
        self,
        dataset_name,
        config,
        distributed=True,
        output_dir,
    ):
        """
        Args:
            dataset_name (str): name of the dataset to be evaluated.
            distributed (bool): if True, will collect results from all ranks for evaluation.
                Otherwise, will evaluate the results in the current process.
            output_dir (str): an output directory to dump results.
        """
        self._logger = logging.getLogger(__name__)

        self._dataset_name = dataset_name
        self._distributed = distributed
        self._output_dir = output_dir

        self._cpu_device = torch.device("cpu")

    def reset(self):
        self.feats = []
        self.pids = []
        self.camids = []
        self.image_type = []

    def process(self, inputs, outputs):
        """
        Args:
            inputs: the inputs to a model
                It is a list of dicts. Each dict corresponds to an image and
                contains keys like "height", "width", "file_name".
            outputs: the outputs of a model. It is either reid features (Tensor [N, C])
                or list of dicts with key "features" that contains reid features
                in the same format.
        """
        feat = outputs['feature_nobn']
        pid = inputs['label']
        camid = inputs['camera']
        image_type = inputs['image_type']

        self.feats.append(feat.cpu())
        self.pids.extend(np.asarray(pid))
        self.camids.extend(np.asarray(camid))
        self.image_type.extend(image_type)

    def evaluate(self):
        """
        Evaluates standard Person ReID metrics (https://github.com/damo-cv/TransReID/blob/639bc460b85224942dbee6c53b43bbe72ad712bd/utils/metrics.py#L90):

        * Rank 1
        * mAP: mean average precision
        """
        feats = torch.cat(self.feats, dim=0)
        if self.feat_norm:
            print("The test feature is normalized")
            feats = torch.nn.functional.normalize(feats, dim=1, p=2)  # along channel

        query_index = [i for i, type in enumerate(self.image_type) if type == 'query']
        gallery_index = [i for i, type in enumerate(self.image_type) if type == 'gallery']

        query_index = np.asarray(query_index)
        gallery_index = np.asarray(gallery_index)

        pids_asarray = np.asarray(self.pids)
        camera_ids_asarray = np.asarray(self.camids)

        # query
        qf = feats[query_index]
        q_pids = pids_asarray[query_index]
        q_camids = camera_ids_asarray[query_index]
        # gallery
        gf = feats[gallery_index]
        g_pids = pids_asarray[gallery_index]
        g_camids = camera_ids_asarray[gallery_index]

        print('=> Computing DistMat with euclidean_distance')
        distmat = euclidean_distance(qf, gf)
        cmc, mAP = eval_func(distmat, q_pids, g_pids, q_camids, g_camids)

        return cmc, mAP, distmat, self.pids, self.camids, qf, gf
