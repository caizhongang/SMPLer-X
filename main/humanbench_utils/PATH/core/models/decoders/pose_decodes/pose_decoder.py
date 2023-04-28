import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
import math
from torch.nn.parameter import Parameter

try:
    import spring.linklink as link
    from spring.linklink.nn import SyncBatchNorm2d
except:
    import linklink as link
    from linklink.nn import SyncBatchNorm2d

from core.data.transforms.post_transforms import flip_back, pose_pck_accuracy, transform_preds

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

def resize(input,
           size=None,
           scale_factor=None,
           mode='nearest',
           align_corners=None,
           warning=True):
    if warning:
        if size is not None and align_corners:
            input_h, input_w = tuple(int(x) for x in input.shape[2:])
            output_h, output_w = tuple(int(x) for x in size)
            if output_h > input_h or output_w > output_h:
                if ((output_h > 1 and output_w > 1 and input_h > 1
                     and input_w > 1) and (output_h - 1) % (input_h - 1)
                        and (output_w - 1) % (input_w - 1)):
                    warnings.warn(
                        f'When align_corners={align_corners}, '
                        'the output would more aligned if '
                        f'input size {(input_h, input_w)} is `x+1` and '
                        f'out size {(output_h, output_w)} is `nx+1`')
    if isinstance(size, torch.Size):
        size = tuple(int(x) for x in size)
    return F.interpolate(input, size, scale_factor, mode, align_corners)

__all__ = ['TopDownSimpleHead']

criterion = nn.MSELoss()


def myloss(output, target, target_weight):
    """Forward function."""
    use_target_weight = True
    batch_size = output.size(0)
    num_joints = output.size(1)

    heatmaps_pred = output.reshape(
        (batch_size, num_joints, -1)).split(1, 1)
    heatmaps_gt = target.reshape((batch_size, num_joints, -1)).split(1, 1)

    loss = 0.

    for idx in range(num_joints):
        heatmap_pred = heatmaps_pred[idx].squeeze(1)
        heatmap_gt = heatmaps_gt[idx].squeeze(1)
        if use_target_weight:
            loss += criterion(heatmap_pred * target_weight[:, idx],
                              heatmap_gt * target_weight[:, idx])
        else:
            loss += criterion(heatmap_pred, heatmap_gt)

    return loss / num_joints * 1.0


def _get_max_preds(heatmaps):
    """Get keypoint predictions from score maps.

    Note:
        batch_size: N
        num_keypoints: K
        heatmap height: H
        heatmap width: W

    Args:
        heatmaps (np.ndarray[N, K, H, W]): model predicted heatmaps.

    Returns:
        tuple: A tuple containing aggregated results.

        - preds (np.ndarray[N, K, 2]): Predicted keypoint location.
        - maxvals (np.ndarray[N, K, 1]): Scores (confidence) of the keypoints.
    """
    assert isinstance(heatmaps,
                      np.ndarray), ('heatmaps should be numpy.ndarray')
    assert heatmaps.ndim == 4, 'batch_images should be 4-ndim'

    N, K, _, W = heatmaps.shape
    heatmaps_reshaped = heatmaps.reshape((N, K, -1))
    idx = np.argmax(heatmaps_reshaped, 2).reshape((N, K, 1))
    maxvals = np.amax(heatmaps_reshaped, 2).reshape((N, K, 1))

    preds = np.tile(idx, (1, 1, 2)).astype(np.float32)
    preds[:, :, 0] = preds[:, :, 0] % W
    preds[:, :, 1] = preds[:, :, 1] // W

    preds = np.where(np.tile(maxvals, (1, 1, 2)) > 0.0, preds, -1)
    return preds, maxvals


def _taylor(heatmap, coord):
    """Distribution aware coordinate decoding method.

    Note:
        heatmap height: H
        heatmap width: W

    Args:
        heatmap (np.ndarray[H, W]): Heatmap of a particular joint type.
        coord (np.ndarray[2,]): Coordinates of the predicted keypoints.

    Returns:
        np.ndarray[2,]: Updated coordinates.
    """
    H, W = heatmap.shape[:2]
    px, py = int(coord[0]), int(coord[1])
    if 1 < px < W - 2 and 1 < py < H - 2:
        dx = 0.5 * (heatmap[py][px + 1] - heatmap[py][px - 1])
        dy = 0.5 * (heatmap[py + 1][px] - heatmap[py - 1][px])
        dxx = 0.25 * (
            heatmap[py][px + 2] - 2 * heatmap[py][px] + heatmap[py][px - 2])
        dxy = 0.25 * (
            heatmap[py + 1][px + 1] - heatmap[py - 1][px + 1] -
            heatmap[py + 1][px - 1] + heatmap[py - 1][px - 1])
        dyy = 0.25 * (
            heatmap[py + 2 * 1][px] - 2 * heatmap[py][px] +
            heatmap[py - 2 * 1][px])
        derivative = np.array([[dx], [dy]])
        hessian = np.array([[dxx, dxy], [dxy, dyy]])
        if dxx * dyy - dxy**2 != 0:
            hessianinv = np.linalg.inv(hessian)
            offset = -hessianinv @ derivative
            offset = np.squeeze(np.array(offset.T), axis=0)
            coord += offset
    return coord


def post_dark_udp(coords, batch_heatmaps, kernel=3):
    """DARK post-pocessing. Implemented by udp. Paper ref: Huang et al. The
    Devil is in the Details: Delving into Unbiased Data Processing for Human
    Pose Estimation (CVPR 2020). Zhang et al. Distribution-Aware Coordinate
    Representation for Human Pose Estimation (CVPR 2020).

    Note:
        - batch size: B
        - num keypoints: K
        - num persons: N
        - height of heatmaps: H
        - width of heatmaps: W

        B=1 for bottom_up paradigm where all persons share the same heatmap.
        B=N for top_down paradigm where each person has its own heatmaps.

    Args:
        coords (np.ndarray[N, K, 2]): Initial coordinates of human pose.
        batch_heatmaps (np.ndarray[B, K, H, W]): batch_heatmaps
        kernel (int): Gaussian kernel size (K) for modulation.

    Returns:
        np.ndarray([N, K, 2]): Refined coordinates.
    """
    if not isinstance(batch_heatmaps, np.ndarray):
        batch_heatmaps = batch_heatmaps.cpu().numpy()
    B, K, H, W = batch_heatmaps.shape
    N = coords.shape[0]
    assert (B == 1 or B == N)
    for heatmaps in batch_heatmaps:
        for heatmap in heatmaps:
            cv2.GaussianBlur(heatmap, (kernel, kernel), 0, heatmap)
    np.clip(batch_heatmaps, 0.001, 50, batch_heatmaps)
    np.log(batch_heatmaps, batch_heatmaps)

    batch_heatmaps_pad = np.pad(
        batch_heatmaps, ((0, 0), (0, 0), (1, 1), (1, 1)),
        mode='edge').flatten()

    index = coords[..., 0] + 1 + (coords[..., 1] + 1) * (W + 2)
    index += (W + 2) * (H + 2) * np.arange(0, B * K).reshape(-1, K)
    index = index.astype(int).reshape(-1, 1)
    i_ = batch_heatmaps_pad[index]
    ix1 = batch_heatmaps_pad[index + 1]
    iy1 = batch_heatmaps_pad[index + W + 2]
    ix1y1 = batch_heatmaps_pad[index + W + 3]
    ix1_y1_ = batch_heatmaps_pad[index - W - 3]
    ix1_ = batch_heatmaps_pad[index - 1]
    iy1_ = batch_heatmaps_pad[index - 2 - W]

    dx = 0.5 * (ix1 - ix1_)
    dy = 0.5 * (iy1 - iy1_)
    derivative = np.concatenate([dx, dy], axis=1)
    derivative = derivative.reshape(N, K, 2, 1)
    dxx = ix1 - 2 * i_ + ix1_
    dyy = iy1 - 2 * i_ + iy1_
    dxy = 0.5 * (ix1y1 - ix1 - iy1 + i_ + i_ - ix1_ - iy1_ + ix1_y1_)
    hessian = np.concatenate([dxx, dxy, dxy, dyy], axis=1)
    hessian = hessian.reshape(N, K, 2, 2)
    hessian = np.linalg.inv(hessian + np.finfo(np.float32).eps * np.eye(2))
    coords -= np.einsum('ijmn,ijnk->ijmk', hessian, derivative).squeeze()
    return coords


def _gaussian_blur(heatmaps, kernel=11):
    """Modulate heatmap distribution with Gaussian.
     sigma = 0.3*((kernel_size-1)*0.5-1)+0.8
     sigma~=3 if k=17
     sigma=2 if k=11;
     sigma~=1.5 if k=7;
     sigma~=1 if k=3;

    Note:
        batch_size: N
        num_keypoints: K
        heatmap height: H
        heatmap width: W

    Args:
        heatmaps (np.ndarray[N, K, H, W]): model predicted heatmaps.
        kernel (int): Gaussian kernel size (K) for modulation, which should
            match the heatmap gaussian sigma when training.
            K=17 for sigma=3 and k=11 for sigma=2.

    Returns:
        np.ndarray[N, K, H, W]: Modulated heatmap distribution.
    """
    assert kernel % 2 == 1

    border = (kernel - 1) // 2
    batch_size = heatmaps.shape[0]
    num_joints = heatmaps.shape[1]
    height = heatmaps.shape[2]
    width = heatmaps.shape[3]
    for i in range(batch_size):
        for j in range(num_joints):
            origin_max = np.max(heatmaps[i, j])
            dr = np.zeros((height + 2 * border, width + 2 * border),
                          dtype=np.float32)
            dr[border:-border, border:-border] = heatmaps[i, j].copy()
            dr = cv2.GaussianBlur(dr, (kernel, kernel), 0)
            heatmaps[i, j] = dr[border:-border, border:-border].copy()
            heatmaps[i, j] *= origin_max / np.max(heatmaps[i, j])
    return heatmaps


def keypoints_from_heatmaps(heatmaps,
                            center,
                            scale,
                            unbiased=False,
                            post_process='default',
                            kernel=11,
                            valid_radius_factor=0.0546875,
                            use_udp=False,
                            target_type='GaussianHeatMap'):
    """Get final keypoint predictions from heatmaps and transform them back to
    the image.

    Note:
        batch size: N
        num keypoints: K
        heatmap height: H
        heatmap width: W

    Args:
        heatmaps (np.ndarray[N, K, H, W]): model predicted heatmaps.
        center (np.ndarray[N, 2]): Center of the bounding box (x, y).
        scale (np.ndarray[N, 2]): Scale of the bounding box
            wrt height/width.
        post_process (str/None): Choice of methods to post-process
            heatmaps. Currently supported: None, 'default', 'unbiased',
            'megvii'.
        unbiased (bool): Option to use unbiased decoding. Mutually
            exclusive with megvii.
            Note: this arg is deprecated and unbiased=True can be replaced
            by post_process='unbiased'
            Paper ref: Zhang et al. Distribution-Aware Coordinate
            Representation for Human Pose Estimation (CVPR 2020).
        kernel (int): Gaussian kernel size (K) for modulation, which should
            match the heatmap gaussian sigma when training.
            K=17 for sigma=3 and k=11 for sigma=2.
        valid_radius_factor (float): The radius factor of the positive area
            in classification heatmap for UDP.
        use_udp (bool): Use unbiased data processing.
        target_type (str): 'GaussianHeatMap' or 'CombinedTarget'.
            GaussianHeatMap: Classification target with gaussian distribution.
            CombinedTarget: The combination of classification target
            (response map) and regression target (offset map).
            Paper ref: Huang et al. The Devil is in the Details: Delving into
            Unbiased Data Processing for Human Pose Estimation (CVPR 2020).

    Returns:
        tuple: A tuple containing keypoint predictions and scores.

        - preds (np.ndarray[N, K, 2]): Predicted keypoint location in images.
        - maxvals (np.ndarray[N, K, 1]): Scores (confidence) of the keypoints.
    """
    # Avoid being affected
    heatmaps = heatmaps.copy()

    # detect conflicts
    if unbiased:
        assert post_process not in [False, None, 'megvii']
    if post_process in ['megvii', 'unbiased']:
        assert kernel > 0
    if use_udp:
        assert not post_process == 'megvii'

    # normalize configs
    if post_process is False:
        warnings.warn(
            'post_process=False is deprecated, '
            'please use post_process=None instead', DeprecationWarning)
        post_process = None
    elif post_process is True:
        if unbiased is True:
            warnings.warn(
                'post_process=True, unbiased=True is deprecated,'
                " please use post_process='unbiased' instead",
                DeprecationWarning)
            post_process = 'unbiased'
        else:
            warnings.warn(
                'post_process=True, unbiased=False is deprecated, '
                "please use post_process='default' instead",
                DeprecationWarning)
            post_process = 'default'
    elif post_process == 'default':
        if unbiased is True:
            warnings.warn(
                'unbiased=True is deprecated, please use '
                "post_process='unbiased' instead", DeprecationWarning)
            post_process = 'unbiased'

    # start processing
    if post_process == 'megvii':
        heatmaps = _gaussian_blur(heatmaps, kernel=kernel)

    N, K, H, W = heatmaps.shape
    if use_udp:
        assert target_type in ['GaussianHeatMap', 'CombinedTarget']
        if target_type == 'GaussianHeatMap':
            preds, maxvals = _get_max_preds(heatmaps)
            preds = post_dark_udp(preds, heatmaps, kernel=kernel)
        elif target_type == 'CombinedTarget':
            for person_heatmaps in heatmaps:
                for i, heatmap in enumerate(person_heatmaps):
                    kt = 2 * kernel + 1 if i % 3 == 0 else kernel
                    cv2.GaussianBlur(heatmap, (kt, kt), 0, heatmap)
            # valid radius is in direct proportion to the height of heatmap.
            valid_radius = valid_radius_factor * H
            offset_x = heatmaps[:, 1::3, :].flatten() * valid_radius
            offset_y = heatmaps[:, 2::3, :].flatten() * valid_radius
            heatmaps = heatmaps[:, ::3, :]
            preds, maxvals = _get_max_preds(heatmaps)
            index = preds[..., 0] + preds[..., 1] * W
            index += W * H * np.arange(0, N * K / 3)
            index = index.astype(np.int).reshape(N, K // 3, 1)
            preds += np.concatenate((offset_x[index], offset_y[index]), axis=2)
        else:
            raise ValueError('target_type should be either '
                             "'GaussianHeatMap' or 'CombinedTarget'")
    else:
        preds, maxvals = _get_max_preds(heatmaps)
        if post_process == 'unbiased':  # alleviate biased coordinate
            # apply Gaussian distribution modulation.
            heatmaps = np.log(
                np.maximum(_gaussian_blur(heatmaps, kernel), 1e-10))
            for n in range(N):
                for k in range(K):
                    preds[n][k] = _taylor(heatmaps[n][k], preds[n][k])
        elif post_process is not None:
            # add +/-0.25 shift to the predicted locations for higher acc.
            for n in range(N):
                for k in range(K):
                    heatmap = heatmaps[n][k]
                    px = int(preds[n][k][0])
                    py = int(preds[n][k][1])
                    if 1 < px < W - 1 and 1 < py < H - 1:
                        diff = np.array([
                            heatmap[py][px + 1] - heatmap[py][px - 1],
                            heatmap[py + 1][px] - heatmap[py - 1][px]
                        ])
                        preds[n][k] += np.sign(diff) * .25
                        if post_process == 'megvii':
                            preds[n][k] += 0.5

    # Transform back to the image
    for i in range(N):
        preds[i] = transform_preds(preds[i], center[i], scale[i], [W, H], use_udp=use_udp)

    if post_process == 'megvii':
        maxvals = maxvals / 255.0 + 0.5

    return preds, maxvals


class TopDownSimpleHead(nn.Module):
    """Top-down model head of simple baseline paper ref: Bin Xiao. ``Simple
    Baselines for Human Pose Estimation and Tracking``.

    TopDownSimpleHead is consisted of (>=0) number of deconv layers
    and a simple conv2d layer.

    Args:
        in_channels (int): Number of input channels
        out_channels (int): Number of output channels
        num_deconv_layers (int): Number of deconv layers.
            num_deconv_layers should >= 0. Note that 0 means
            no deconv layers.
        num_deconv_filters (list|tuple): Number of filters.
            If num_deconv_layers > 0, the length of
        num_deconv_kernels (list|tuple): Kernel sizes.
        in_index (int|Sequence[int]): Input feature index. Default: -1
        input_transform (str|None): Transformation type of input features.
            Options: 'resize_concat', 'multiple_select', None.
            'resize_concat': Multiple feature maps will be resize to the
                same size as first one and than concat together.
                Usually used in FCN head of HRNet.
            'multiple_select': Multiple feature maps will be bundle into
                a list and passed into decode head.
            None: Only one select feature map is allowed.
            Default: None.
        align_corners (bool): align_corners argument of F.interpolate.
            Default: False.
        loss_keypoint (dict): Config for keypoint loss. Default: None.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 backbone,
                 neck,
                 num_deconv_layers=3,
                 num_deconv_filters=(256, 256, 256),
                 num_deconv_kernels=(4, 4, 4),
                 extra=None,
                 in_index=0,
                 input_transform=None,
                 align_corners=False,
                 loss_keypoint=None,
                 train_cfg=None,
                 test_cfg=None,
                 norm_cfg='BN',
                 upsample=0,
                 use_sync_bn=False,
                 bn_group=None,
                 layer_norm=False,
                 **kwargs
                 ):
        super().__init__()

        self.in_channels = in_channels
        self.backbone = [backbone]
        self.neck = [neck]
        self.upsample=upsample
        self.train_cfg = {} if train_cfg is None else train_cfg
        self.test_cfg = {} if test_cfg is None else test_cfg
        self.target_type = self.test_cfg.get('target_type', 'GaussianHeatMap')

        self._init_inputs(in_channels, in_index, input_transform)
        self.in_index = in_index
        self.align_corners = align_corners
        self.use_sync_bn = use_sync_bn
        self.bn_group = bn_group

        self.layer_norm = layer_norm

        if extra is not None and not isinstance(extra, dict):
            raise TypeError('extra should be dict or None.')

        if num_deconv_layers > 0:
            self.deconv_layers = self._make_deconv_layer(
                num_deconv_layers,
                num_deconv_filters,
                num_deconv_kernels,
            )
        elif num_deconv_layers == 0:
            self.deconv_layers = nn.Identity()
        else:
            raise ValueError(
                f'num_deconv_layers ({num_deconv_layers}) should >= 0.')

        identity_final_layer = False
        if extra is not None and 'final_conv_kernel' in extra:
            assert extra['final_conv_kernel'] in [0, 1, 3]
            if extra['final_conv_kernel'] == 3:
                padding = 1
            elif extra['final_conv_kernel'] == 1:
                padding = 0
            else:
                # 0 for Identity mapping.
                identity_final_layer = True
            kernel_size = extra['final_conv_kernel']
        else:
            kernel_size = 1
            padding = 0

        if identity_final_layer:
            self.final_layer = nn.Identity()
        else:
            conv_channels = num_deconv_filters[
                -1] if num_deconv_layers > 0 else self.in_channels

            layers = []
            if extra is not None:
                num_conv_layers = extra.get('num_conv_layers', 0)
                num_conv_kernels = extra.get('num_conv_kernels',
                                             [1] * num_conv_layers)

                for i in range(num_conv_layers):
                    layers.append(nn.Conv2d(
                        conv_channels,
                        conv_channels,
                        kernel_size=num_conv_kernels[i],
                        stride=1,
                        padding=(num_conv_kernels[i] - 1) // 2))
                    
                    if self.layer_norm:
                        layers.append(
                            LayerNorm(conv_channels, data_format="channels_first")
                        )
                    elif self.use_sync_bn:
                        layers.append(
                            SyncBatchNorm2d(num_features=conv_channels, group=self.bn_group, sync_stats=False)
                        )
                    else:
                        layers.append(
                            nn.BatchNorm(conv_channels))
                    layers.append(nn.ReLU(inplace=True))

            layers.append(
                nn.Conv2d(
                    conv_channels,
                    out_channels,
                    kernel_size=kernel_size,  # 1
                    stride=1,
                    padding=padding))

            if len(layers) > 1:
                self.final_layer = nn.Sequential(*layers)
            else:
                self.final_layer = layers[0]
            self.init_weights()

    def get_accuracy(self, output, target, target_weight):
        """Calculate accuracy for top-down keypoint loss.

        Note:
            batch_size: N
            num_keypoints: K
            heatmaps height: H
            heatmaps weight: W

        Args:
            output (torch.Tensor[NxKxHxW]): Output heatmaps.
            target (torch.Tensor[NxKxHxW]): Target heatmaps.
            target_weight (torch.Tensor[NxKx1]):
                Weights across different joint types.
        """

        if self.target_type == 'GaussianHeatMap':
            _, avg_acc, _ = pose_pck_accuracy(
                output.detach().cpu().numpy(),
                target.detach().cpu().numpy(),
                target_weight.detach().cpu().numpy().squeeze(-1) > 0)
            accuracy = avg_acc

        return torch.Tensor([accuracy]).cuda()

    def forward(self, features):
        """Forward function."""
        x = self.layers(features['neck_output'])
        output = {}
        if self.training:
            target = features['label']
            target_weight = features['target_weight']

            pos_loss = myloss(x, target, target_weight)
            acc = self.get_accuracy(x, target, target_weight)
            output['feature'] = x
            output['loss'] = pos_loss
            output['top1'] = acc
        else:
            assert features['image'].size(0) == len(features['img_metas'])
            batch_size, _, img_height, img_width = features['image'].shape
            if batch_size > 1:
                assert 'bbox_id' in features['img_metas'][0].data
            output_heatmap = self.inference_model(x, flip_pairs=None)

            if self.test_cfg.get('flip_test', True):  # True
                img_flipped = {'image': features['image'].flip(3)}
                features_flipped = self.backbone[0](img_flipped)
                features_flipped = self.neck[0](features_flipped)
                features_flipped = self.layers(features_flipped['neck_output'])

                output_flipped_heatmap = self.inference_model(features_flipped, features['img_metas'][0].data['flip_pairs'])
                output_heatmap = (output_heatmap + output_flipped_heatmap) * 0.5

            # import pdb; pdb.set_trace();
            keypoint_result = self.decode(features['img_metas'], output_heatmap, img_size=[img_width, img_height])
            output.update(keypoint_result)

        return output

    def layers(self, x):
        x = self._transform_inputs(x)  # useless
        x = self.deconv_layers(x)  # useless
        x = self.final_layer(x)  # 18 -> 17
        return x

    def inference_model(self, x, flip_pairs=None):
        """Inference function.

        Returns:
            output_heatmap (np.ndarray): Output heatmaps.

        Args:
            x (torch.Tensor[NxKxHxW]): Input features.
            flip_pairs (None | list[tuple()):
                Pairs of keypoints which are mirrored.
        """
        if flip_pairs is not None:
            output_heatmap = flip_back(
                x.detach().cpu().numpy(),
                flip_pairs,
                target_type=self.target_type)
            # feature is not aligned, shift flipped heatmap for higher accuracy
            if self.test_cfg.get('shift_heatmap', False):  # True
                output_heatmap[:, :, :, 1:] = output_heatmap[:, :, :, :-1]
        else:
            output_heatmap = x.detach().cpu().numpy()
        return output_heatmap

    def _init_inputs(self, in_channels, in_index, input_transform):
        """Check and initialize input transforms.

        The in_channels, in_index and input_transform must match.
        Specifically, when input_transform is None, only single feature map
        will be selected. So in_channels and in_index must be of type int.
        When input_transform is not None, in_channels and in_index must be
        list or tuple, with the same length.

        Args:
            in_channels (int|Sequence[int]): Input channels.
            in_index (int|Sequence[int]): Input feature index.
            input_transform (str|None): Transformation type of input features.
                Options: 'resize_concat', 'multiple_select', None.
                'resize_concat': Multiple feature maps will be resize to the
                    same size as first one and than concat together.
                    Usually used in FCN head of HRNet.
                'multiple_select': Multiple feature maps will be bundle into
                    a list and passed into decode head.
                None: Only one select feature map is allowed.
        """

        if input_transform is not None:
            assert input_transform in ['resize_concat', 'multiple_select']
        self.input_transform = input_transform
        self.in_index = in_index
        if input_transform is not None:
            assert isinstance(in_channels, (list, tuple))
            assert isinstance(in_index, (list, tuple))
            assert len(in_channels) == len(in_index)
            if input_transform == 'resize_concat':
                self.in_channels = sum(in_channels)
            else:
                self.in_channels = in_channels
        else:
            assert isinstance(in_channels, int)
            assert isinstance(in_index, int)
            self.in_channels = in_channels

    def _transform_inputs(self, inputs):  # useless
        """Transform inputs for decoder.

        Args:
            inputs (list[Tensor] | Tensor): multi-level img features.

        Returns:
            Tensor: The transformed inputs
        """
        if not isinstance(inputs, list):
            if not isinstance(inputs, list):
                if self.upsample > 0:
                    inputs = resize(
                        input=F.relu(inputs),
                        scale_factor=self.upsample,
                        mode='bilinear',
                        align_corners=self.align_corners
                        )
            return inputs

        if self.input_transform == 'resize_concat':
            inputs = [inputs[i] for i in self.in_index]
            upsampled_inputs = [
                F.interpolate(
                    input=x,
                    size=inputs[0].shape[2:],
                    mode='bilinear',
                    align_corners=self.align_corners) for x in inputs
            ]
            inputs = torch.cat(upsampled_inputs, dim=1)
        elif self.input_transform == 'multiple_select':
            inputs = [inputs[i] for i in self.in_index]
        else:
            inputs = inputs[self.in_index]

        return inputs

    def _make_deconv_layer(self, num_layers, num_filters, num_kernels, norm_cfg=dict(type='BN')):
        """Make deconv layers."""
        if num_layers != len(num_filters):
            error_msg = f'num_layers({num_layers}) ' \
                        f'!= length of num_filters({len(num_filters)})'
            raise ValueError(error_msg)
        if num_layers != len(num_kernels):
            error_msg = f'num_layers({num_layers}) ' \
                        f'!= length of num_kernels({len(num_kernels)})'
            raise ValueError(error_msg)

        layers = []
        for i in range(num_layers):
            kernel, padding, output_padding = \
                self._get_deconv_cfg(num_kernels[i])

            planes = num_filters[i]
            layers.append(nn.ConvTranspose2d(in_channels=self.in_channels,
                                             out_channels=planes,
                                             kernel_size=kernel,
                                             stride=2,
                                             padding=padding,
                                             output_padding=output_padding,
                                             bias=False)
                          )
            if self.layer_norm:
                layers.append(LayerNorm(planes, data_format="channels_first"))
            elif self.use_sync_bn:
                layers.append(SyncBatchNorm2d(num_features=planes, group=self.bn_group, sync_stats=False))
                # layers.append(self.BN(planes))
            else:
                layers.append(nn.BatchNorm2d(planes))
            layers.append(nn.ReLU(inplace=True))
            self.in_channels = planes

        return nn.Sequential(*layers)

    def _get_deconv_cfg(self, deconv_kernel):
        """Get configurations for deconv layers."""
        if deconv_kernel == 4:
            padding = 1
            output_padding = 0
        elif deconv_kernel == 3:
            padding = 1
            output_padding = 1
        elif deconv_kernel == 2:
            padding = 0
            output_padding = 0
        else:
            raise ValueError(f'Not supported num_kernels ({deconv_kernel}).')

        return deconv_kernel, padding, output_padding

    def constant_init(self, module, val, bias=0):
        if hasattr(module, 'weight') and module.weight is not None:
            nn.init.constant_(module.weight, val)
        if hasattr(module, 'bias') and module.bias is not None:
            nn.init.constant_(module.bias, bias)

    def normal_init(self, module, mean=0, std=1, bias=0):
        if hasattr(module, 'weight') and module.weight is not None:
            nn.init.normal_(module.weight, mean, std)
        if hasattr(module, 'bias') and module.bias is not None:
            nn.init.constant_(module.bias, bias)

    def init_weights(self):
        """Initialize model weights."""
        for _, m in self.deconv_layers.named_modules():
            if isinstance(m, nn.ConvTranspose2d):
                self.normal_init(m, std=0.001)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, SyncBatchNorm2d):
                self.constant_init(m, 1)
        for m in self.final_layer.modules():
            if isinstance(m, nn.Conv2d):
                self.normal_init(m, std=0.001, bias=0)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, SyncBatchNorm2d):
                self.constant_init(m, 1)

    def decode(self, img_metas, output, **kwargs):
        """Decode keypoints from heatmaps.

        Args:
            img_metas (list(dict)): Information about data augmentation
                By default this includes:
                - "image_file: path to the image file
                - "center": center of the bbox
                - "scale": scale of the bbox
                - "rotation": rotation of the bbox
                - "bbox_score": score of bbox
            output (np.ndarray[N, K, H, W]): model predicted heatmaps.
        """
        batch_size = len(img_metas)

        if 'bbox_id' in img_metas[0].data:
            bbox_ids = []
        else:
            bbox_ids = None

        c = np.zeros((batch_size, 2), dtype=np.float32)
        s = np.zeros((batch_size, 2), dtype=np.float32)
        image_paths = []
        score = np.ones(batch_size)
        for i in range(batch_size):
            c[i, :] = img_metas[i].data['center']
            s[i, :] = img_metas[i].data['scale']
            image_paths.append(img_metas[i].data['image_file'])

            if 'bbox_score' in img_metas[i].data:
                score[i] = np.array(img_metas[i].data['bbox_score']).reshape(-1)
            if bbox_ids is not None:
                bbox_ids.append(img_metas[i].data['bbox_id'])
        
        # import pdb; pdb.set_trace()
        preds, maxvals = keypoints_from_heatmaps(
            output,
            c,
            s,
            unbiased=self.test_cfg.get('unbiased_decoding', False),
            post_process=self.test_cfg.get('post_process', 'default'),
            kernel=self.test_cfg.get('modulate_kernel', 11),
            valid_radius_factor=self.test_cfg.get('valid_radius_factor',
                                                  0.0546875),
            use_udp=self.test_cfg.get('use_udp', False),
            target_type=self.test_cfg.get('target_type', 'GaussianHeatMap'))

        all_preds = np.zeros((batch_size, preds.shape[1], 3), dtype=np.float32)
        all_boxes = np.zeros((batch_size, 6), dtype=np.float32)
        all_preds[:, :, 0:2] = preds[:, :, 0:2]
        all_preds[:, :, 2:3] = maxvals
        all_boxes[:, 0:2] = c[:, 0:2]
        all_boxes[:, 2:4] = s[:, 0:2]
        all_boxes[:, 4] = np.prod(s * 200.0, axis=1)
        all_boxes[:, 5] = score

        result = {}

        # true_boxes = []
        # for i in range(batch_size):
        #     true_boxes.append(img_metas[i].data['bbox'])

        # result['true_boxes'] = true_boxes
        result['preds'] = all_preds
        result['boxes'] = all_boxes
        result['image_paths'] = image_paths
        result['bbox_ids'] = bbox_ids

        return result

