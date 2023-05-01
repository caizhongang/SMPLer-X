import math
import numbers
import random
import functools
import numpy as np
import torch
import cv2
import warnings
from torchvision.transforms import functional as F
from core.data.transforms.post_transforms import (affine_transform, fliplr_joints,
                                                  get_affine_transform, get_warp_matrix,
                                                  warp_affine_joints)

from petrelbox.io import PetrelHelper


def assert_tensor_type(func):

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if not isinstance(args[0].data, torch.Tensor):
            raise AttributeError(
                f'{args[0].__class__.__name__} has no attribute '
                f'{func.__name__} for type {args[0].datatype}')
        return func(*args, **kwargs)

    return wrapper

class TopDownGetBboxCenterScale:
    """Convert bbox from [x, y, w, h] to center and scale.
    The center is the coordinates of the bbox center, and the scale is the
    bbox width and height normalized by a scale factor.
    Required key: 'bbox', 'ann_info'
    Modifies key: 'center', 'scale'
    Args:
        padding (float): bbox padding scale that will be multilied to scale.
            Default: 1.25
    """
    # Pixel std is 200.0, which serves as the normalization factor to
    # to calculate bbox scales.
    pixel_std: float = 200.0

    def __init__(self, padding: float = 1.25):
        self.padding = padding

    def __call__(self, results):

        if 'center' in results and 'scale' in results:
            warnings.warn(
                'Use the "center" and "scale" that already exist in the data '
                'sample. The padding will still be applied.')
            results['scale'] *= self.padding
        else:
            bbox = results['bbox']
            image_size = results['ann_info']['image_size']
            aspect_ratio = image_size[0] / image_size[1]

            center, scale = bbox_xywh2cs(
                bbox,
                aspect_ratio=aspect_ratio,
                padding=self.padding,
                pixel_std=self.pixel_std)

            results['center'] = center
            results['scale'] = scale
        return results

class TopDownRandomShiftBboxCenter:
    """Random shift the bbox center.
    Required key: 'center', 'scale'
    Modifies key: 'center'
    Args:
        shift_factor (float): The factor to control the shift range, which is
            scale*pixel_std*scale_factor. Default: 0.16
        prob (float): Probability of applying random shift. Default: 0.3
    """
    # Pixel std is 200.0, which serves as the normalization factor to
    # to calculate bbox scales.
    pixel_std: float = 200.0

    def __init__(self, shift_factor: float = 0.16, prob: float = 0.3):
        self.shift_factor = shift_factor
        self.prob = prob

    def __call__(self, results):

        center = results['center']
        scale = results['scale']
        if np.random.rand() < self.prob:
            center += np.random.uniform(
                -1, 1, 2) * self.shift_factor * scale * self.pixel_std

        results['center'] = center
        return results


class DataContainer:
    """A container for any type of objects.

    Typically tensors will be stacked in the collate function and sliced along
    some dimension in the scatter function. This behavior has some limitations.
    1. All tensors have to be the same size.
    2. Types are limited (numpy array or Tensor).

    We design `DataContainer` and `MMDataParallel` to overcome these
    limitations. The behavior can be either of the following.

    - copy to GPU, pad all tensors to the same size and stack them
    - copy to GPU without stacking
    - leave the objects as is and pass it to the model
    - pad_dims specifies the number of last few dimensions to do padding
    """

    def __init__(self,
                 data,
                 stack=False,
                 padding_value=0,
                 cpu_only=False,
                 pad_dims=2):
        self._data = data
        self._cpu_only = cpu_only
        self._stack = stack
        self._padding_value = padding_value
        assert pad_dims in [None, 1, 2, 3]
        self._pad_dims = pad_dims

    def __repr__(self):
        return f'{self.__class__.__name__}({repr(self.data)})'

    def __len__(self):
        return len(self._data)

    @property
    def data(self):
        return self._data

    @property
    def datatype(self):
        if isinstance(self.data, torch.Tensor):
            return self.data.type()
        else:
            return type(self.data)

    @property
    def cpu_only(self):
        return self._cpu_only

    @property
    def stack(self):
        return self._stack

    @property
    def padding_value(self):
        return self._padding_value

    @property
    def pad_dims(self):
        return self._pad_dims

    @assert_tensor_type
    def size(self, *args, **kwargs):
        return self.data.size(*args, **kwargs)

    @assert_tensor_type
    def dim(self):
        return self.data.dim()


class ComposeX:
    def __init__(self, augmentations):
        self.augmentations = augmentations

    def __call__(self, data):
        # img, mask = Image.fromarray(img, mode='RGB'), Image.fromarray(mask, mode='L')

        for a in self.augmentations:
            data = a(data)
            if data is None:
                return None
        return data


class LoadImageFromFile:
    """Loading image from file.

    Args:
        color_type (str): Flags specifying the color type of a loaded image,
          candidates are 'color', 'grayscale' and 'unchanged'.
        channel_order (str): Order of channel, candidates are 'bgr' and 'rgb'.
    """

    def __init__(self,
                 to_float32=False,
                 color_type='color',
                 channel_order='rgb'):
        self.to_float32 = to_float32
        self.color_type = color_type
        self.channel_order = channel_order

    def __call__(self, results):
        """Loading image from file."""
        image_file = results['image_file']
        img = PetrelHelper.imread(image_file, cv2.IMREAD_COLOR)

        img = img[:, :, ::-1]  # BGR -> RGB
        if img is None:
            raise ValueError('Fail to read {}'.format(image_file))

        results['image'] = img
        return results


class TopDownRandomFlip:
    """Data augmentation with random image flip.

    Required keys: 'img', 'joints_3d', 'joints_3d_visible', 'center' and
    'ann_info'.
    Modifies key: 'img', 'joints_3d', 'joints_3d_visible', 'center' and
    'flipped'.

    Args:
        flip (bool): Option to perform random flip.
        flip_prob (float): Probability of flip.
    """

    def __init__(self, flip_prob=0.5):
        self.flip_prob = flip_prob

    def __call__(self, results):
        """Perform data augmentation with random image flip."""
        img = results['image']
        joints_3d = results['joints_3d']
        joints_3d_visible = results['joints_3d_visible']
        center = results['center']

        # A flag indicating whether the image is flipped,
        # which can be used by child class.
        flipped = False
        if np.random.rand() <= self.flip_prob:
            flipped = True
            img = img[:, ::-1, :]
            joints_3d, joints_3d_visible = fliplr_joints(
                joints_3d, joints_3d_visible, img.shape[1],
                results['ann_info']['flip_pairs'])
            center[0] = img.shape[1] - center[0] - 1

        results['image'] = img
        results['joints_3d'] = joints_3d
        results['joints_3d_visible'] = joints_3d_visible
        results['center'] = center
        results['flipped'] = flipped

        return results


class TopDownHalfBodyTransform:
    """Data augmentation with half-body transform. Keep only the upper body or
    the lower body at random.

    Required keys: 'joints_3d', 'joints_3d_visible', and 'ann_info'.
    Modifies key: 'scale' and 'center'.

    Args:
        num_joints_half_body (int): Threshold of performing
            half-body transform. If the body has fewer number
            of joints (< num_joints_half_body), ignore this step.
        prob_half_body (float): Probability of half-body transform.
    """

    def __init__(self, num_joints_half_body=8, prob_half_body=0.3):
        self.num_joints_half_body = num_joints_half_body
        self.prob_half_body = prob_half_body

    @staticmethod
    def half_body_transform(cfg, joints_3d, joints_3d_visible):
        """Get center&scale for half-body transform."""
        upper_joints = []
        lower_joints = []
        for joint_id in range(cfg['num_joints']):
            if joints_3d_visible[joint_id][0] > 0:
                if joint_id in cfg['upper_body_ids']:
                    upper_joints.append(joints_3d[joint_id])
                else:
                    lower_joints.append(joints_3d[joint_id])

        if np.random.randn() < 0.5 and len(upper_joints) > 2:
            selected_joints = upper_joints
        elif len(lower_joints) > 2:
            selected_joints = lower_joints
        else:
            selected_joints = upper_joints

        if len(selected_joints) < 2:
            return None, None

        selected_joints = np.array(selected_joints, dtype=np.float32)
        center = selected_joints.mean(axis=0)[:2]

        left_top = np.amin(selected_joints, axis=0)

        right_bottom = np.amax(selected_joints, axis=0)

        w = right_bottom[0] - left_top[0]
        h = right_bottom[1] - left_top[1]

        aspect_ratio = cfg['image_size'][0] / cfg['image_size'][1]

        if w > aspect_ratio * h:
            h = w * 1.0 / aspect_ratio
        elif w < aspect_ratio * h:
            w = h * aspect_ratio

        scale = np.array([w / 200.0, h / 200.0], dtype=np.float32)
        scale = scale * 1.5
        return center, scale

    def __call__(self, results):
        """Perform data augmentation with half-body transform."""
        joints_3d = results['joints_3d']
        joints_3d_visible = results['joints_3d_visible']

        if (np.sum(joints_3d_visible[:, 0]) > self.num_joints_half_body
                and np.random.rand() < self.prob_half_body):

            c_half_body, s_half_body = self.half_body_transform(
                results['ann_info'], joints_3d, joints_3d_visible)

            if c_half_body is not None and s_half_body is not None:
                results['center'] = c_half_body
                results['scale'] = s_half_body

        return results


class TopDownGetRandomScaleRotation:
    """Data augmentation with random scaling & rotating.

    Required key: 'scale'. Modifies key: 'scale' and 'rotation'.

    Args:
        rot_factor (int): Rotating to ``[-2*rot_factor, 2*rot_factor]``.
        scale_factor (float): Scaling to ``[1-scale_factor, 1+scale_factor]``.
        rot_prob (float): Probability of random rotation.
    """

    def __init__(self, rot_factor=40, scale_factor=0.5, rot_prob=0.6):
        self.rot_factor = rot_factor
        self.scale_factor = scale_factor
        self.rot_prob = rot_prob

    def __call__(self, results):
        """Perform data augmentation with random scaling & rotating."""
        s = results['scale']

        sf = self.scale_factor
        rf = self.rot_factor

        s_factor = np.clip(np.random.randn() * sf + 1, 1 - sf, 1 + sf)
        s = s * s_factor

        r_factor = np.clip(np.random.randn() * rf, -rf * 2, rf * 2)
        r = r_factor if np.random.rand() <= self.rot_prob else 0

        results['scale'] = s
        results['rotation'] = r

        return results


class TopDownAffine:
    """Affine transform the image to make input.
    Required keys:'img', 'joints_3d', 'joints_3d_visible', 'ann_info','scale',
    'rotation' and 'center'.
    Modified keys:'img', 'joints_3d', and 'joints_3d_visible'.
    Args:
        use_udp (bool): To use unbiased data processing.
            Paper ref: Huang et al. The Devil is in the Details: Delving into
            Unbiased Data Processing for Human Pose Estimation (CVPR 2020).
    """

    def __init__(self, use_udp=False):
        self.use_udp = use_udp

    def __call__(self, results):
        image_size = results['ann_info']['image_size']

        img = results['image']
        joints_3d = results['joints_3d']
        joints_3d_visible = results['joints_3d_visible']
        c = results['center']
        s = results['scale']
        r = results['rotation']

        if self.use_udp:
            trans = get_warp_matrix(r, c * 2.0, image_size - 1.0, s * 200.0)
            if not isinstance(img, list):
                img = cv2.warpAffine(
                    img,
                    trans, (int(image_size[0]), int(image_size[1])),
                    flags=cv2.INTER_LINEAR)
            else:
                img = [
                    cv2.warpAffine(
                        i,
                        trans, (int(image_size[0]), int(image_size[1])),
                        flags=cv2.INTER_LINEAR) for i in img
                ]

            joints_3d[:, 0:2] = \
                warp_affine_joints(joints_3d[:, 0:2].copy(), trans)

        else:
            trans = get_affine_transform(c, s, r, image_size)
            if not isinstance(img, list):
                img = cv2.warpAffine(
                    img,
                    trans, (int(image_size[0]), int(image_size[1])),
                    flags=cv2.INTER_LINEAR)
            else:
                img = [
                    cv2.warpAffine(
                        i,
                        trans, (int(image_size[0]), int(image_size[1])),
                        flags=cv2.INTER_LINEAR) for i in img
                ]
            for i in range(results['ann_info']['num_joints']):
                if joints_3d_visible[i, 0] > 0.0:
                    joints_3d[i, 0:2] = affine_transform(joints_3d[i, 0:2], trans)

        results['image'] = img
        results['joints_3d'] = joints_3d
        results['joints_3d_visible'] = joints_3d_visible

        return results


class ToTensor:
    """Transform image to Tensor.

    Required key: 'img'. Modifies key: 'img'.

    Args:
        results (dict): contain all information about training.
    """

    def __call__(self, results):
        results['image'] = F.to_tensor(results['image'])
        return results


class ToUNTensor:
    """Transform image to Tensor.

    Required key: 'img'. Modifies key: 'img'.

    Args:
        results (dict): contain all information about training.
    """

    def __call__(self, results):
        results['image'] = torch.as_tensor(np.ascontiguousarray(results['image'].transpose(2, 0, 1)))
        # results['image'] = F.to_tensor(results['image'])
        return results


class NormalizeTensor:
    """Normalize the Tensor image (CxHxW), with mean and std.

    Required key: 'img'. Modifies key: 'img'.

    Args:
        mean (list[float]): Mean values of 3 channels.
        std (list[float]): Std values of 3 channels.
    """

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, results):
        results['image'] = F.normalize(
            results['image'], mean=self.mean, std=self.std)
        return results


class TopDownGenerateTarget:
    """Generate the target heatmap.
    Required keys: 'joints_3d', 'joints_3d_visible', 'ann_info'.
    Modified keys: 'target', and 'target_weight'.
    Args:
        sigma: Sigma of heatmap gaussian for 'MSRA' approach.
        kernel: Kernel of heatmap gaussian for 'Megvii' approach.
        encoding (str): Approach to generate target heatmaps.
            Currently supported approaches: 'MSRA', 'Megvii', 'UDP'.
            Default:'MSRA'
        unbiased_encoding (bool): Option to use unbiased
            encoding methods.
            Paper ref: Zhang et al. Distribution-Aware Coordinate
            Representation for Human Pose Estimation (CVPR 2020).
        keypoint_pose_distance: Keypoint pose distance for UDP.
            Paper ref: Huang et al. The Devil is in the Details: Delving into
            Unbiased Data Processing for Human Pose Estimation (CVPR 2020).
        target_type (str): supported targets: 'GaussianHeatmap',
            'CombinedTarget'. Default:'GaussianHeatmap'
            CombinedTarget: The combination of classification target
            (response map) and regression target (offset map).
            Paper ref: Huang et al. The Devil is in the Details: Delving into
            Unbiased Data Processing for Human Pose Estimation (CVPR 2020).
    """

    def __init__(self,
                 sigma=2,
                 kernel=(11, 11),
                 valid_radius_factor=0.0546875,
                 target_type='GaussianHeatmap',
                 encoding='MSRA',
                 unbiased_encoding=False):
        self.sigma = sigma
        self.unbiased_encoding = unbiased_encoding
        self.kernel = kernel
        self.valid_radius_factor = valid_radius_factor
        self.target_type = target_type
        self.encoding = encoding

    def _msra_generate_target(self, cfg, joints_3d, joints_3d_visible, sigma):
        """Generate the target heatmap via "MSRA" approach.
        Args:
            cfg (dict): data config
            joints_3d: np.ndarray ([num_joints, 3])
            joints_3d_visible: np.ndarray ([num_joints, 3])
            sigma: Sigma of heatmap gaussian
        Returns:
            tuple: A tuple containing targets.
            - target: Target heatmaps.
            - target_weight: (1: visible, 0: invisible)
        """
        num_joints = cfg['num_joints']
        image_size = cfg['image_size']
        W, H = cfg['heatmap_size']
        joint_weights = cfg['joint_weights']
        use_different_joint_weights = cfg['use_different_joint_weights']

        target_weight = np.zeros((num_joints, 1), dtype=np.float32)
        target = np.zeros((num_joints, H, W), dtype=np.float32)

        # 3-sigma rule
        tmp_size = sigma * 3

        if self.unbiased_encoding:
            for joint_id in range(num_joints):
                target_weight[joint_id] = joints_3d_visible[joint_id, 0]

                feat_stride = image_size / [W, H]
                mu_x = joints_3d[joint_id][0] / feat_stride[0]
                mu_y = joints_3d[joint_id][1] / feat_stride[1]
                # Check that any part of the gaussian is in-bounds
                ul = [mu_x - tmp_size, mu_y - tmp_size]
                br = [mu_x + tmp_size + 1, mu_y + tmp_size + 1]
                if ul[0] >= W or ul[1] >= H or br[0] < 0 or br[1] < 0:
                    target_weight[joint_id] = 0

                if target_weight[joint_id] == 0:
                    continue

                x = np.arange(0, W, 1, np.float32)
                y = np.arange(0, H, 1, np.float32)
                y = y[:, None]

                if target_weight[joint_id] > 0.5:
                    target[joint_id] = np.exp(-((x - mu_x)**2 +
                                                (y - mu_y)**2) /
                                              (2 * sigma**2))
        else:
            for joint_id in range(num_joints):
                target_weight[joint_id] = joints_3d_visible[joint_id, 0]

                feat_stride = image_size / [W, H]
                mu_x = int(joints_3d[joint_id][0] / feat_stride[0] + 0.5)
                mu_y = int(joints_3d[joint_id][1] / feat_stride[1] + 0.5)
                # Check that any part of the gaussian is in-bounds
                ul = [int(mu_x - tmp_size), int(mu_y - tmp_size)]
                br = [int(mu_x + tmp_size + 1), int(mu_y + tmp_size + 1)]
                if ul[0] >= W or ul[1] >= H or br[0] < 0 or br[1] < 0:
                    target_weight[joint_id] = 0

                if target_weight[joint_id] > 0.5:
                    size = 2 * tmp_size + 1
                    x = np.arange(0, size, 1, np.float32)
                    y = x[:, None]
                    x0 = y0 = size // 2
                    # The gaussian is not normalized,
                    # we want the center value to equal 1
                    g = np.exp(-((x - x0)**2 + (y - y0)**2) / (2 * sigma**2))

                    # Usable gaussian range
                    g_x = max(0, -ul[0]), min(br[0], W) - ul[0]
                    g_y = max(0, -ul[1]), min(br[1], H) - ul[1]
                    # Image range
                    img_x = max(0, ul[0]), min(br[0], W)
                    img_y = max(0, ul[1]), min(br[1], H)

                    target[joint_id][img_y[0]:img_y[1], img_x[0]:img_x[1]] = \
                        g[g_y[0]:g_y[1], g_x[0]:g_x[1]]

        if use_different_joint_weights:
            target_weight = np.multiply(target_weight, joint_weights)

        return target, target_weight

    def _megvii_generate_target(self, cfg, joints_3d, joints_3d_visible,
                                kernel):
        """Generate the target heatmap via "Megvii" approach.
        Args:
            cfg (dict): data config
            joints_3d: np.ndarray ([num_joints, 3])
            joints_3d_visible: np.ndarray ([num_joints, 3])
            kernel: Kernel of heatmap gaussian
        Returns:
            tuple: A tuple containing targets.
            - target: Target heatmaps.
            - target_weight: (1: visible, 0: invisible)
        """

        num_joints = cfg['num_joints']
        image_size = cfg['image_size']
        W, H = cfg['heatmap_size']
        heatmaps = np.zeros((num_joints, H, W), dtype='float32')
        target_weight = np.zeros((num_joints, 1), dtype=np.float32)

        for i in range(num_joints):
            target_weight[i] = joints_3d_visible[i, 0]

            if target_weight[i] < 1:
                continue

            target_y = int(joints_3d[i, 1] * H / image_size[1])
            target_x = int(joints_3d[i, 0] * W / image_size[0])

            if (target_x >= W or target_x < 0) \
                    or (target_y >= H or target_y < 0):
                target_weight[i] = 0
                continue

            heatmaps[i, target_y, target_x] = 1
            heatmaps[i] = cv2.GaussianBlur(heatmaps[i], kernel, 0)
            maxi = heatmaps[i, target_y, target_x]

            heatmaps[i] /= maxi / 255

        return heatmaps, target_weight

    def _udp_generate_target(self, cfg, joints_3d, joints_3d_visible, factor,
                             target_type):
        """Generate the target heatmap via 'UDP' approach. Paper ref: Huang et
        al. The Devil is in the Details: Delving into Unbiased Data Processing
        for Human Pose Estimation (CVPR 2020).
        Note:
            - num keypoints: K
            - heatmap height: H
            - heatmap width: W
            - num target channels: C
            - C = K if target_type=='GaussianHeatmap'
            - C = 3*K if target_type=='CombinedTarget'
        Args:
            cfg (dict): data config
            joints_3d (np.ndarray[K, 3]): Annotated keypoints.
            joints_3d_visible (np.ndarray[K, 3]): Visibility of keypoints.
            factor (float): kernel factor for GaussianHeatmap target or
                valid radius factor for CombinedTarget.
            target_type (str): 'GaussianHeatmap' or 'CombinedTarget'.
                GaussianHeatmap: Heatmap target with gaussian distribution.
                CombinedTarget: The combination of classification target
                (response map) and regression target (offset map).
        Returns:
            tuple: A tuple containing targets.
            - target (np.ndarray[C, H, W]): Target heatmaps.
            - target_weight (np.ndarray[K, 1]): (1: visible, 0: invisible)
        """
        num_joints = cfg['num_joints']
        image_size = cfg['image_size']
        heatmap_size = cfg['heatmap_size']
        joint_weights = cfg['joint_weights']
        use_different_joint_weights = cfg['use_different_joint_weights']

        target_weight = np.ones((num_joints, 1), dtype=np.float32)
        target_weight[:, 0] = joints_3d_visible[:, 0]

        if target_type.lower() == 'GaussianHeatmap'.lower():
            target = np.zeros((num_joints, heatmap_size[1], heatmap_size[0]),
                              dtype=np.float32)

            tmp_size = factor * 3

            # prepare for gaussian
            size = 2 * tmp_size + 1
            x = np.arange(0, size, 1, np.float32)
            y = x[:, None]

            for joint_id in range(num_joints):
                feat_stride = (image_size - 1.0) / (heatmap_size - 1.0)
                mu_x = int(joints_3d[joint_id][0] / feat_stride[0] + 0.5)
                mu_y = int(joints_3d[joint_id][1] / feat_stride[1] + 0.5)
                # Check that any part of the gaussian is in-bounds
                ul = [int(mu_x - tmp_size), int(mu_y - tmp_size)]
                br = [int(mu_x + tmp_size + 1), int(mu_y + tmp_size + 1)]
                if ul[0] >= heatmap_size[0] or ul[1] >= heatmap_size[1] \
                        or br[0] < 0 or br[1] < 0:
                    # If not, just return the image as is
                    target_weight[joint_id] = 0
                    continue

                # # Generate gaussian
                mu_x_ac = joints_3d[joint_id][0] / feat_stride[0]
                mu_y_ac = joints_3d[joint_id][1] / feat_stride[1]
                x0 = y0 = size // 2
                x0 += mu_x_ac - mu_x
                y0 += mu_y_ac - mu_y
                g = np.exp(-((x - x0)**2 + (y - y0)**2) / (2 * factor**2))

                # Usable gaussian range
                g_x = max(0, -ul[0]), min(br[0], heatmap_size[0]) - ul[0]
                g_y = max(0, -ul[1]), min(br[1], heatmap_size[1]) - ul[1]
                # Image range
                img_x = max(0, ul[0]), min(br[0], heatmap_size[0])
                img_y = max(0, ul[1]), min(br[1], heatmap_size[1])

                v = target_weight[joint_id]
                if v > 0.5:
                    target[joint_id][img_y[0]:img_y[1], img_x[0]:img_x[1]] = \
                        g[g_y[0]:g_y[1], g_x[0]:g_x[1]]

        elif target_type.lower() == 'CombinedTarget'.lower():
            target = np.zeros(
                (num_joints, 3, heatmap_size[1] * heatmap_size[0]),
                dtype=np.float32)
            feat_width = heatmap_size[0]
            feat_height = heatmap_size[1]
            feat_x_int = np.arange(0, feat_width)
            feat_y_int = np.arange(0, feat_height)
            feat_x_int, feat_y_int = np.meshgrid(feat_x_int, feat_y_int)
            feat_x_int = feat_x_int.flatten()
            feat_y_int = feat_y_int.flatten()
            # Calculate the radius of the positive area in classification
            #   heatmap.
            valid_radius = factor * heatmap_size[1]
            feat_stride = (image_size - 1.0) / (heatmap_size - 1.0)
            for joint_id in range(num_joints):
                mu_x = joints_3d[joint_id][0] / feat_stride[0]
                mu_y = joints_3d[joint_id][1] / feat_stride[1]
                x_offset = (mu_x - feat_x_int) / valid_radius
                y_offset = (mu_y - feat_y_int) / valid_radius
                dis = x_offset**2 + y_offset**2
                keep_pos = np.where(dis <= 1)[0]
                v = target_weight[joint_id]
                if v > 0.5:
                    target[joint_id, 0, keep_pos] = 1
                    target[joint_id, 1, keep_pos] = x_offset[keep_pos]
                    target[joint_id, 2, keep_pos] = y_offset[keep_pos]
            target = target.reshape(num_joints * 3, heatmap_size[1],
                                    heatmap_size[0])
        else:
            raise ValueError('target_type should be either '
                             "'GaussianHeatmap' or 'CombinedTarget'")

        if use_different_joint_weights:
            target_weight = np.multiply(target_weight, joint_weights)

        return target, target_weight

    def __call__(self, results):
        """Generate the target heatmap."""
        joints_3d = results['joints_3d']
        joints_3d_visible = results['joints_3d_visible']

        assert self.encoding in ['MSRA', 'Megvii', 'UDP']

        if self.encoding == 'MSRA':
            if isinstance(self.sigma, list):
                num_sigmas = len(self.sigma)
                cfg = results['ann_info']
                num_joints = cfg['num_joints']
                heatmap_size = cfg['heatmap_size']

                target = np.empty(
                    (0, num_joints, heatmap_size[1], heatmap_size[0]),
                    dtype=np.float32)
                target_weight = np.empty((0, num_joints, 1), dtype=np.float32)
                for i in range(num_sigmas):
                    target_i, target_weight_i = self._msra_generate_target(
                        cfg, joints_3d, joints_3d_visible, self.sigma[i])
                    target = np.concatenate([target, target_i[None]], axis=0)
                    target_weight = np.concatenate(
                        [target_weight, target_weight_i[None]], axis=0)
            else:
                target, target_weight = self._msra_generate_target(
                    results['ann_info'], joints_3d, joints_3d_visible,
                    self.sigma)

        elif self.encoding == 'Megvii':
            if isinstance(self.kernel, list):
                num_kernels = len(self.kernel)
                cfg = results['ann_info']
                num_joints = cfg['num_joints']
                W, H = cfg['heatmap_size']

                target = np.empty((0, num_joints, H, W), dtype=np.float32)
                target_weight = np.empty((0, num_joints, 1), dtype=np.float32)
                for i in range(num_kernels):
                    target_i, target_weight_i = self._megvii_generate_target(
                        cfg, joints_3d, joints_3d_visible, self.kernel[i])
                    target = np.concatenate([target, target_i[None]], axis=0)
                    target_weight = np.concatenate(
                        [target_weight, target_weight_i[None]], axis=0)
            else:
                target, target_weight = self._megvii_generate_target(
                    results['ann_info'], joints_3d, joints_3d_visible,
                    self.kernel)

        elif self.encoding == 'UDP':
            if self.target_type.lower() == 'CombinedTarget'.lower():
                factors = self.valid_radius_factor
                channel_factor = 3
            elif self.target_type.lower() == 'GaussianHeatmap'.lower():
                factors = self.sigma
                channel_factor = 1
            else:
                raise ValueError('target_type should be either '
                                 "'GaussianHeatmap' or 'CombinedTarget'")
            if isinstance(factors, list):
                num_factors = len(factors)
                cfg = results['ann_info']
                num_joints = cfg['num_joints']
                W, H = cfg['heatmap_size']

                target = np.empty((0, channel_factor * num_joints, H, W),
                                  dtype=np.float32)
                target_weight = np.empty((0, num_joints, 1), dtype=np.float32)
                for i in range(num_factors):
                    target_i, target_weight_i = self._udp_generate_target(
                        cfg, joints_3d, joints_3d_visible, factors[i],
                        self.target_type)
                    target = np.concatenate([target, target_i[None]], axis=0)
                    target_weight = np.concatenate(
                        [target_weight, target_weight_i[None]], axis=0)
            else:
                target, target_weight = self._udp_generate_target(
                    results['ann_info'], joints_3d, joints_3d_visible, factors,
                    self.target_type)
        else:
            raise ValueError(
                f'Encoding approach {self.encoding} is not supported!')

        results['label'] = target
        results['target_weight'] = target_weight

        return results


class Collect:
    """Collect data from the loader relevant to the specific task.

    This keeps the items in `keys` as it is, and collect items in `meta_keys`
    into a meta item called `meta_name`.This is usually the last stage of the
    data loader pipeline.
    For example, when keys='imgs', meta_keys=('filename', 'label',
    'original_shape'), meta_name='img_metas', the results will be a dict with
    keys 'imgs' and 'img_metas', where 'img_metas' is a DataContainer of
    another dict with keys 'filename', 'label', 'original_shape'.

    Args:
        keys (Sequence[str|tuple]): Required keys to be collected. If a tuple
          (key, key_new) is given as an element, the item retrived by key will
          be renamed as key_new in collected data.
        meta_name (str): The name of the key that contains meta infomation.
          This key is always populated. Default: "img_metas".
        meta_keys (Sequence[str|tuple]): Keys that are collected under
          meta_name. The contents of the `meta_name` dictionary depends
          on `meta_keys`.
    """

    def __init__(self, keys, meta_keys, meta_name='img_metas'):
        self.keys = keys
        self.meta_keys = meta_keys
        self.meta_name = meta_name

    def __call__(self, results):
        """Performs the Collect formating.

        Args:
            results (dict): The resulting dict to be modified and passed
              to the next transform in pipeline.
        """
        if 'ann_info' in results:
            results.update(results['ann_info'])

        data = {}
        for key in self.keys:
            if isinstance(key, tuple):
                assert len(key) == 2
                key_src, key_tgt = key[:2]
            else:
                key_src = key_tgt = key
            data[key_tgt] = results[key_src]

        meta = {}
        if len(self.meta_keys) != 0:
            for key in self.meta_keys:
                if isinstance(key, tuple):
                    assert len(key) == 2
                    key_src, key_tgt = key[:2]
                else:
                    key_src = key_tgt = key
                meta[key_tgt] = results[key_src]
        if 'bbox_id' in results:
            meta['bbox_id'] = results['bbox_id']
        data[self.meta_name] = DataContainer(meta, cpu_only=True)

        return data

    def __repr__(self):
        """Compute the string representation."""
        return (f'{self.__class__.__name__}('
                f'keys={self.keys}, meta_keys={self.meta_keys})')
