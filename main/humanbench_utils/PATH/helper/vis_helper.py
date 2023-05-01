# -*- coding: utf-8 -*-
import cv2
import os
import numpy as np
import torch
import pickle as pk


def inv_normalize_batch(image, mean_arr, stddev_arr):
    # normalize image color channels
    inv_normed_image = image.clone()
    for c in range(3):
        if len(image.size()) == 4:
            inv_normed_image[:, c, :, :] = (image[:, c, :, :] * stddev_arr[c] + mean_arr[c])
        else:
            inv_normed_image[c, :, :] = (image[c, :, :] * stddev_arr[c] + mean_arr[c])
    return inv_normed_image


def get_vis_data(input, range_low=-1, range_high=1, vis_height=-1, vis_width=-1, to_rgb=True):
    if input is None:
        return None

    data = ((input.permute(1, 2, 0) - range_low) / (
                range_high - range_low) * 255.0).data.cpu().numpy()
    if vis_height > 0 and vis_width > 0:
        if data.shape[0] != vis_height or data.shape[1] != vis_width:
            data = cv2.resize(data, (vis_width, vis_height))
    if len(data.shape) == 2:
        data = cv2.cvtColor(data, cv2.COLOR_GRAY2RGB)
    else:
        if to_rgb:
            data = cv2.cvtColor(data, cv2.COLOR_BGR2RGB)
    return data


def vis_one_from_batch(vis_list, range_low=0, range_high=1,
                       vis_height=140, vis_width=140, vis_channel=3, to_rgb=True, return_CHW=True):

    vis_dict = dict()
    for item in vis_list:
        '''
        vis_list = [{
            'name': 'lap_adv',
            'image': laplace_adv
        }]
        '''
        vis_image = get_vis_data(item['image'], range_low, range_high, vis_height, vis_width, to_rgb=to_rgb)
        vis_dict[item['name']] = vis_image

    cnt = 0
    for tag, item in vis_dict.items():
        if item is not None:
            cnt += 1

    # adapt to visualize format
    rst = np.zeros((vis_height, vis_width * cnt, vis_channel))
    pos = 0
    for tag, item in vis_dict.items():
        if item is not None:
            left = vis_width * pos
            right = vis_width * (pos + 1)
            rst[:, left: right] = item
            cv2.putText(rst, tag, (left + 2, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
            pos += 1

    rst = rst.clip(0, 255).astype(np.uint8, copy=False)
    if return_CHW:
        # prepare for tensorboard [RGB, CHW]
        rst = rst.transpose((2, 0, 1))  # HWC -> CHW

    return rst


def get_features_indices(feature_maps, topk=1):
    """

    :param topk:
    :param feature_maps: floatTensor [N, C, H, W]
    :return:
    """
    input_dim = 4
    if len(feature_maps.size()) == 2:
        input_dim = 2
        feature_maps = feature_maps.unsqueeze(0)
    if len(feature_maps.size()) == 3:
        input_dim = 3
        feature_maps = feature_maps.unsqueeze(0)
    N, C, H, W = feature_maps.size()
    feats = feature_maps.view(N, C, -1)
    feats_sum = torch.sum(feats, dim=2)
    y, ind = torch.sort(feats_sum, dim=1, descending=True)
    selected_ind = ind[:, :topk]
    if input_dim < 4:
        return selected_ind.squeeze(0)
    return selected_ind


def show_feature_map(feature_map, reference=None, range_low=-1, range_high=1):
    """
    可视化特征图
    :param feature_map: floatTensor  [C, H, W]
    :param reference: floatTensor  [3, H, W]
    :return:
    """
    if isinstance(feature_map, torch.Tensor):
        feature_map = feature_map.cpu().numpy()
    feature_map_num = feature_map.shape[0]
    row_num = np.ceil(np.sqrt(feature_map_num))

    if reference is not None:
        if isinstance(reference, torch.Tensor):
            reference = ((reference.permute(1, 2, 0) - range_low) / (
                    range_high - range_low)).data.cpu().numpy()
            reference = np.uint8(255 * reference)
            reference = cv2.cvtColor(reference, cv2.COLOR_RGB2BGR)
        all_vis = reference
    else:
        all_vis = None

    for index in range(0, feature_map_num):
        feat = feature_map[index]
        heatmap = feat / np.max(feat)
        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        if reference is not None:
            if not heatmap.shape == reference.shape:
                heatmap = cv2.resize(heatmap, (reference.shape[1], reference.shape[0]), interpolation=cv2.INTER_CUBIC)
            vis = cv2.addWeighted(heatmap, 0.5, reference, 0.5, 0)
        else:
            vis = heatmap
        if index == 0:
            all_vis = vis
        else:
            all_vis = np.hstack([all_vis, vis])

    return all_vis


def dump_to_pickle(file_path, data):
    with open(file_path, "wb") as f:
        pk.dump(data, f)


def load_from_pickle(file_path):
    assert os.path.exists(file_path), "file not exist: {}".format(file_path)
    with open(file_path, "rb") as f:
        meta = pk.load(f)
    return meta
