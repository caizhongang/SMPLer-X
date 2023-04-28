import os
import cv2
import torch
from torch.nn import functional as F
from torch.utils import data
import random
import inspect
import numpy as np

class compose:
    def __init__(self, augs):
        self.augs = augs

    def __call__(self, data, label=None):
        for a in self.augs:
            # import pdb;pdb.set_trace()
            data, label = a(data, label)

        return data, label

    def __repr__(self):
        return 'Compose'


class hflip:
    def __init__(self, p=False, mapped_left_right_pairs=None):
        self.p = p
        self.mapped_left_right_pairs = mapped_left_right_pairs

    def __call__(self, image, label=None,):
        if not self.p:
            return image, label

        assert len(label), "hflip parsing needs label to map left and right pairs"
        flip = random.randrange(2) * 2 -1

        image = image[:, ::flip, :]
        label = label[:, ::flip]

        if flip == -1:
            left_idx = self.mapped_left_right_pairs[:, 0].reshape(-1)
            right_idx = self.mapped_left_right_pairs[:, 1].reshape(-1)
            for i in range(0, self.mapped_left_right_pairs.shape[0]):
                right_pos = np.where(label == right_idx[i])
                left_pos = np.where(label == left_idx[i])
                label[right_pos[0], right_pos[1]] = left_idx[i]
                label[left_pos[0], left_pos[1]] = right_idx[i]

        return image, label

    def __repr__(self):
        return f'Hflip with {self.mapped_left_right_pairs}'

class resize_image:
    def __init__(self, crop_size):
        self.size = crop_size

    def __call__(self, image, label=None):
        image = cv2.resize(image, tuple(self.size), interpolation=cv2.INTER_LINEAR)
        label = cv2.resize(label, tuple(self.size), interpolation=cv2.INTER_NEAREST)
        return image, label

    def __repr__(self):
        return f"Resize with {self.size}"

class resize_image_eval:
    def __init__(self, crop_size):
        self.size = crop_size

    def __call__(self, image, label=None):
        image = cv2.resize(image, tuple(self.size), interpolation=cv2.INTER_LINEAR)
        label = cv2.resize(label, (1000, 1000), interpolation = cv2.INTER_LINEAR_EXACT)
        return image, label

    def __repr__(self):
        return f"Resize_eval with {self.size}"

class multi_scale:
    def __init__(self, is_multi_scale, scale_factor=11,
                 center_crop_test=False, base_size=480,
                 crop_size=(480, 480),
                 ignore_label=-1):
        self.is_multi_scale = is_multi_scale
        self.scale_factor = scale_factor
        self.center_crop_test = center_crop_test
        self.base_size = base_size
        self.crop_size = crop_size
        self.ignore_label = ignore_label

    def multi_scale_aug(self, image, label=None,
            rand_scale=1, rand_crop=True):
        long_size = np.int(self.base_size * rand_scale + 0.5)
        if label is not None:
            image, label = self.image_resize(image, long_size, label)
            if rand_crop:
                image, label = self.rand_crop(image, label)
            return image, label
        else:
            image = self.image_resize(image, long_size)
            return image

    def pad_image(self, image, h, w, size, padvalue):
        pad_image = image.copy()
        pad_h = max(size[0] - h, 0)
        pad_w = max(size[1] - w, 0)
        if pad_h > 0 or pad_w > 0:
            pad_image = cv2.copyMakeBorder(image, 0, pad_h, 0,
                                           pad_w, cv2.BORDER_CONSTANT,
                                           value=padvalue)

        return pad_image

    def rand_crop(self, image, label):
        h, w = image.shape[:-1]
        image = self.pad_image(image, h, w, self.crop_size,
                               (0.0, 0.0, 0.0))
        label = self.pad_image(label, h, w, self.crop_size,
                               (self.ignore_label,))

        new_h, new_w = label.shape
        x = random.randint(0, new_w - self.crop_size[1])
        y = random.randint(0, new_h - self.crop_size[0])
        image = image[y:y + self.crop_size[0], x:x + self.crop_size[1]]
        label = label[y:y + self.crop_size[0], x:x + self.crop_size[1]]

        return image, label

    def image_resize(self, image, long_size, label=None):
        h, w = image.shape[:2]
        if h > w:
            new_h = long_size
            new_w = np.int(w * long_size / h + 0.5)
        else:
            new_w = long_size
            new_h = np.int(h * long_size / w + 0.5)

        image = cv2.resize(image, (new_w, new_h),
                           interpolation=cv2.INTER_LINEAR)
        if label is not None:
            label = cv2.resize(label, (new_w, new_h),
                               interpolation=cv2.INTER_NEAREST)
        else:
            return image

        return image, label

    def center_crop(self, image, label):
        h, w = image.shape[:2]
        x = int(round((w - self.crop_size[1]) / 2.))
        y = int(round((h - self.crop_size[0]) / 2.))
        image = image[y:y+self.crop_size[0], x:x+self.crop_size[1]]
        label = label[y:y+self.crop_size[0], x:x+self.crop_size[1]]

        return image, label

    def __call__(self, image, label=None):
        if self.is_multi_scale:
            rand_scale = 0.5 + random.randint(0, self.scale_factor) / 10.0
            image, label = self.multi_scale_aug(image, label,
                                                rand_scale=rand_scale)

        # if center_crop_test:
        #     image, label = self.image_resize(image,
        #                                      self.base_size,
        #                                      label)
        #     image, label = self.center_crop(image, label)

        return image, label





class normalize:
    def __init__(self, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        self.mean = mean
        self.std = std

    def __call__(self, image, label = None):
        image = image.astype(np.float32)
        image = image / 255.0
        image -= self.mean
        image /= self.std
        return image, label

    def __repr__(self):
        return f"Normalize with {self.mean} and {self.std}"

class transpose:
    def __repr__(self):
        return 'Transpose'

    def __call__(self, image, label=None):
        return image.transpose((2,0,1)), label

class rotate:

    cv2_interp_codes = {
        'nearest': cv2.INTER_NEAREST,
        'bilinear': cv2.INTER_LINEAR,
        'bicubic': cv2.INTER_CUBIC,
        'area': cv2.INTER_AREA,
        'lanczos': cv2.INTER_LANCZOS4
    }

    def __init__(self, is_rotate=False, degree=0, p=0.5, pad_val=0, seg_pad_val=255,
                 center=None, auto_bound=False):
        self.is_rotate = is_rotate
        if isinstance(degree, (float, int)):
            assert degree > 0, f'degree {degree} should be positive'
            self.degree = (-degree, degree)
        else:
            self.degree = degree
        self.p = p
        self.pad_val = pad_val
        self.seg_pad_val = seg_pad_val
        self.center = center
        self.auto_bound=auto_bound

    def __call__(self, image, label=None):
        if not self.is_rotate or random.random() < self.p:
            return image, label
        degree = random.uniform(min(*self.degree), max(*self.degree))
        image = self._rotate(
            image,
            angle=degree,
            border_value=self.pad_val,
            center=self.center,
            auto_bound=self.auto_bound)
        label = self._rotate(
            label,
            angle=degree,
            border_value=self.seg_pad_val,
            center=self.center,
            auto_bound=self.auto_bound,
            interpolation='nearest')
        return image, label

    def _rotate(self, img, angle, center=None, scale=1.0, border_value=0, interpolation='bilinear', auto_bound=False):
        if center is not None and auto_bound:
            raise ValueError('`auto_bound` conflicts with `center`')
        h, w = img.shape[:2]
        if center is None:
            center = ((w - 1) * 0.5, (h - 1) * 0.5)
        assert isinstance(center, tuple)

        matrix = cv2.getRotationMatrix2D(center, -angle, scale)
        if auto_bound:
            cos = np.abs(matrix[0, 0])
            sin = np.abs(matrix[0, 1])
            new_w = h * sin + w * cos
            new_h = h * cos + w * sin
            matrix[0, 2] += (new_w - w) * 0.5
            matrix[1, 2] += (new_h - h) * 0.5
            w = int(np.round(new_w))
            h = int(np.round(new_h))
        rotated = cv2.warpAffine(
            img,
            matrix, (w, h),
            flags=self.cv2_interp_codes[interpolation],
            borderValue=border_value)
        return rotated

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(prob={self.prob}, ' \
                    f'degree={self.degree}, ' \
                    f'pad_val={self.pal_val}, ' \
                    f'seg_pad_val={self.seg_pad_val}, ' \
                    f'center={self.center}, ' \
                    f'auto_bound={self.auto_bound})'
        return repr_str

# copied from https://github.com/open-mmlab/mmsegmentation/blob/2d66179630035097dcae08ee958f60d4b5a7fcae/mmseg/datasets/pipelines/transforms.py
class PhotoMetricDistortion:
    def __init__(self,
                 is_PhotoMetricDistortio=False,
                 brightness_delta=32,
                 contrast_range=(0.5, 1.5),
                 saturation_range=(0.5, 1.5),
                 hue_delta=18):
        self.brightness_delta = brightness_delta
        self.contrast_lower, self.contrast_upper = contrast_range
        self.saturation_lower, self.saturation_upper = saturation_range
        self.hue_delta = hue_delta
        self.is_PhotoMetricDistortio = is_PhotoMetricDistortio

    def convert(self, img, alpha=1, beta=0):
        """Multiple with alpha and add beat with clip."""
        img = img.astype(np.float32) * alpha + beta
        img = np.clip(img, 0, 255)
        return img.astype(np.uint8)

    def brightness(self, img):
        """Brightness distortion."""
        if random.randint(0,1):
            return self.convert(
                img,
                beta=random.uniform(-self.brightness_delta,
                                    self.brightness_delta))
        return img

    def contrast(self, img):
        """Contrast distortion."""
        if random.randint(0,1):
            return self.convert(
                img,
                alpha=random.uniform(self.contrast_lower, self.contrast_upper))
        return img

    def saturation(self, img):
        """Saturation distortion."""
        if random.randint(0,1):
            img = bgr2hsv(img)
            img[:, :, 1] = self.convert(
                img[:, :, 1],
                alpha=random.uniform(self.saturation_lower,
                                     self.saturation_upper))
            img = hsv2bgr(img)
        return img

    def hue(self, img):
        """Hue distortion."""
        if random.randint(0,1):
            img = bgr2hsv(img)
            img[:, :,
            0] = (img[:, :, 0].astype(int) +
                  random.randint(-self.hue_delta, self.hue_delta)) % 180
            img = hsv2bgr(img)
        return img

    def __call__(self, img, label=None):
        """Call function to perform photometric distortion on images.
        Args:
            results (dict): Result dict from loading pipeline.
        Returns:
            dict: Result dict with images distorted.
        """
        if not self.is_PhotoMetricDistortio:
            return img, label
        img = img
        # random brightness
        img = self.brightness(img)

        # mode == 0 --> do random contrast first
        # mode == 1 --> do random contrast last
        mode = random.randint(0,1)
        if mode == 1:
            img = self.contrast(img)

        # random saturation
        img = self.saturation(img)

        # random hue
        img = self.hue(img)

        # random contrast
        if mode == 0:
            img = self.contrast(img)

        # results['img'] = img
        return img, label

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += (f'(brightness_delta={self.brightness_delta}, '
                     f'contrast_range=({self.contrast_lower}, '
                     f'{self.contrast_upper}), '
                     f'saturation_range=({self.saturation_lower}, '
                     f'{self.saturation_upper}), '
                     f'hue_delta={self.hue_delta})')
        return

def convert_color_factory(src: str, dst: str):

    code = getattr(cv2, f'COLOR_{src.upper()}2{dst.upper()}')

    def convert_color(img: np.ndarray) -> np.ndarray:
        out_img = cv2.cvtColor(img, code)
        return out_img

    convert_color.__doc__ = f"""Convert a {src.upper()} image to {dst.upper()}
        image.
    Args:
        img (ndarray or str): The input image.
    Returns:
        ndarray: The converted {dst.upper()} image.
    """

    return convert_color


bgr2rgb = convert_color_factory('bgr', 'rgb')

rgb2bgr = convert_color_factory('rgb', 'bgr')

bgr2hsv = convert_color_factory('bgr', 'hsv')

hsv2bgr = convert_color_factory('hsv', 'bgr')

bgr2hls = convert_color_factory('bgr', 'hls')

hls2bgr = convert_color_factory('hls', 'bgr')



def main():
    img_path = ''