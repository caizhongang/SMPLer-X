import torch.utils.data as data
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES=True

import os
import os.path

import random
import torch
import numpy as np
import copy

import time
from core.data.transforms.peddet_transforms import PedestrainDetectionAugmentation, SparseRCNNPedestrainDetectionAugmentation

from core.data.datasets.images.seg_dataset_dev import Instances
from typing import *

from core import distributed_utils as dist

from petrelbox.io import PetrelHelper
from pycocotools.coco import COCO

from collections import defaultdict

__all__ = ['PedestrainDetectionDataset']

class PetrelCOCO(COCO):
    def __init__(self, annotation_file=None, annotation=None):
        """
        Constructor of Microsoft COCO helper class for reading and visualizing annotations.
        :param annotation_file (str): location of annotation file
        :param image_folder (str): location to the folder that hosts images.
        :return:
        """
        # load dataset
        self.dataset, self.anns, self.cats, self.imgs = dict(), dict(), dict(), dict()
        self.imgToAnns, self.catToImgs = defaultdict(list), defaultdict(list)
        assert not (annotation_file is not None and annotation_file is not None)
        if annotation_file is not None:
            print('loading annotations into memory...')
            tic = time.time()
            dataset = PetrelHelper.load_json(annotation_file)
            assert type(dataset) == dict, 'annotation file format {} not supported'.format(type(dataset))
            print('Done (t={:0.2f}s)'.format(time.time() - tic))
            self.dataset = dataset
            self.createIndex()

        if annotation is not None:
            print('adding annotations into memory...')
            tic = time.time()
            dataset = annotation
            self.dataset = dataset
            self.createIndex()

def convert_coco_poly_to_mask(segmentations, height, width):
    masks = []
    for polygons in segmentations:
        rles = coco_mask.frPyObjects(polygons, height, width)
        mask = coco_mask.decode(rles)
        if len(mask.shape) < 3:
            mask = mask[..., None]
        mask = torch.as_tensor(mask, dtype=torch.uint8)
        mask = mask.any(dim=2)
        masks.append(mask)
    if masks:
        masks = torch.stack(masks, dim=0)
    else:
        masks = torch.zeros((0, height, width), dtype=torch.uint8)
    return masks

class ConvertCocoPolysToMask(object):
    def __init__(self, return_masks=False):
        self.return_masks = return_masks

    def __call__(self, image, target):
        w, h = image.size

        image_id = target["image_id"]
        image_id = torch.tensor([image_id])

        anno = target["annotations"]

        boxes = [obj["bbox"] for obj in anno]
        # guard against no boxes via resizing
        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        boxes[:, 2:] += boxes[:, :2]
        boxes[:, 0::2].clamp_(min=0, max=w)
        boxes[:, 1::2].clamp_(min=0, max=h)

        classes = [obj["category_id"] for obj in anno]
        classes = torch.tensor(classes, dtype=torch.int64)

        if self.return_masks:
            segmentations = [obj["segmentation"] for obj in anno]
            masks = convert_coco_poly_to_mask(segmentations, h, w)

        keypoints = None
        if anno and "keypoints" in anno[0]:
            keypoints = [obj["keypoints"] for obj in anno]
            keypoints = torch.as_tensor(keypoints, dtype=torch.float32)
            num_keypoints = keypoints.shape[0]
            if num_keypoints:
                keypoints = keypoints.view(num_keypoints, -1, 3)

        keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])

        # for conversion to coco api
        area = torch.tensor([obj["area"] for obj in anno])
        iscrowd = torch.BoolTensor([obj["iscrowd"] if "iscrowd" in obj else 0 for obj in anno])
        iscrowd |= classes != 0

        target = {}
        target["boxes"] = boxes[keep]
        target["labels"] = classes[keep]
        if self.return_masks:
            target["masks"] = masks[keep]
        target["image_id"] = image_id
        if keypoints is not None:
            target["keypoints"] = keypoints[keep]

        target["area"] = area[keep]
        target["iscrowd"] = iscrowd[keep]

        target["orig_size"] = torch.as_tensor([int(h), int(w)])
        target["size"] = torch.as_tensor([int(h), int(w)])

        return image, target


class CocoDetection(data.Dataset):
    """`MS Coco Detection <http://mscoco.org/dataset/#detections-challenge2016>`_ Dataset.

    Args:
        root (string): Root directory where images are downloaded to.
        annFile (string): Path to json annotation file.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.ToTensor``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
    """

    def __init__(self, ann, phase, transform=None, target_transform=None, data_use_ratio=1, dataset_name=None):
        self.dataset_name = dataset_name
        self.coco = PetrelCOCO(annotation=ann)

        self.ids = list(self.coco.imgs.keys())

        if data_use_ratio != 1:
            self.ids = random.sample(self.ids, int(len(self.ids) * data_use_ratio))

        assert phase in ['train', 'val']
        self.transform = transform
        self.phase = phase
        self.target_transform = target_transform

        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()

        self.initialized = True

    def _init_memcached(self):
        if not self.initialized:
            ## only use mc default
            print("==> will load files from local machine")
            server_list_config_file = "/mnt/lustre/share/memcached_client/server_list.conf"
            client_config_file = "/mnt/lustre/share/memcached_client/client.conf"
            self.memcached_mclient = mc.MemcachedClient.GetInstance(server_list_config_file, client_config_file)
            ## mc-support-ceph
            print('mc-support-ceph')
            self.ceph_mclient = s3client

            self.initialized = True

    def _read_one(self, index=None):
        """
        Args:
            index (int): Index

        Returns:
            tuple: Tuple (image, target). target is the object returned by ``coco.loadAnns``.
        """
        if index is None:
            index = np.random.randint(len(self.ids))

        coco = self.coco
        img_id = self.ids[index]
        flag = self.flag[index]

        ann_ids = coco.getAnnIds(imgIds=img_id)
        target = copy.deepcopy(coco.loadAnns(ann_ids))

        for one_target in target:
            if 'segmentation' in one_target: del one_target['segmentation']
            if 'keypoints' in one_target: del one_target['keypoints']

        path = coco.loadImgs(img_id)[0]['file_name']
        img_root = coco.loadImgs(img_id)[0]['img_root']
        imgname = os.path.splitext(path)[0]

        if self.dataset_name == 'CrowdHuman' and path.endswith('.png'):
            path = path.replace('.png', '.jpg')

        filename = os.path.join(img_root, path)
        try:
            img = PetrelHelper.pil_open(filename, "RGB")
            if img is None:
                raise Exception("None Image")
        except:
            outputName = "failed_to_read_in_train.txt"
            with open(outputName,"a") as g:
                g.write("%s\n"%(filename))
            print('Read image[{}] failed ({})'.format(index, filename))
            ## if fail then recursive call _read_one without idx
            return self._read_one()
        else:
            output = dict()
            ##set random_seed with img idx
            random.seed(index+self.rank)
            np.random.seed(index+self.rank)

            if self.transform is not None:
                img = self.transform(img)

            if self.target_transform is not None:
                target = self.target_transform(target)

            return img, target, imgname, flag

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: Tuple (image, target). target is the object returned by ``coco.loadAnns``.
        """
        self._init_memcached()
        img, target, imgname, flag = self._read_one(index)

        return img, target, imgname, flag

    def __len__(self):
        return len(self.ids)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str


def coco_merge(
    img_root_list: List[str], input_list: List[str],
    indent: Optional[int] = None,
) -> str:
    """Merge COCO annotation files.

    Args:
        input_extend: Path to input file to be extended.
        input_add: Path to input file to be added.
        output_file : Path to output file with merged annotations.
        indent: Argument passed to `json.dump`. See https://docs.python.org/3/library/json.html#json.dump.
    """
    data_list = []

    for input in input_list:
        data_extend = PetrelHelper.load_json(input)
        data_list.append(data_extend)

    output= {'categories': data_list[0]['categories']}

    output["images"], output["annotations"] = [], []

    for i, (data, img_root) in enumerate(zip(data_list, img_root_list)):

        print(
            "Input {}: {} images, {} annotations".format(
                i + 1, len(data["images"]), len(data["annotations"])
            )
        )

        cat_id_map = {}
        for new_cat in data["categories"]:
            new_id = None
            for output_cat in output["categories"]:
                if new_cat["name"] == output_cat["name"]:
                    new_id = output_cat["id"]
                    break

            if new_id is not None:
                cat_id_map[new_cat["id"]] = new_id
            else:
                new_cat_id = max(c["id"] for c in output["categories"]) + 1
                cat_id_map[new_cat["id"]] = new_cat_id
                new_cat["id"] = new_cat_id
                output["categories"].append(new_cat)

        img_id_map = {}
        for image in data["images"]:
            n_imgs = len(output["images"])
            img_id_map[image["id"]] = n_imgs
            image["id"] = n_imgs
            image["img_root"] = img_root

            output["images"].append(image)

        for annotation in data["annotations"]:
            n_anns = len(output["annotations"])
            annotation["id"] = n_anns
            annotation["image_id"] = img_id_map[annotation["image_id"]]
            annotation["category_id"] = cat_id_map[annotation["category_id"]]

            output["annotations"].append(annotation)

    print(
        "Result: {} images, {} annotations".format(
            len(output["images"]), len(output["annotations"])
        )
    )

    return output


class PedestrainDetectionDataset(CocoDetection):
    def __init__(self, ginfo, augmentation, task_spec, train=True, vit=False, sparsercnn=False, data_use_ratio=1, dataset_name=None, **kwargs):
        img_folder = task_spec['img_folder']
        ann_file = task_spec['ann_file']

        ann = coco_merge(img_folder, ann_file)

        return_masks = task_spec['return_masks']
        phase = 'train' if train else 'val'
        super(PedestrainDetectionDataset, self).__init__(ann=ann, phase=phase, data_use_ratio=data_use_ratio, dataset_name=dataset_name)
        if sparsercnn:
            max_size = augmentation.get('max_size', 1024)
            transforms = SparseRCNNPedestrainDetectionAugmentation(phase=phase, vit=vit, max_size=max_size)
        else:
            max_size = augmentation.get('max_size', 1024)
            transforms = PedestrainDetectionAugmentation(phase=phase, vit=vit, max_size=max_size)

        name2wh = {}
        for img_id in self.ids:
            img_name = self.coco.loadImgs(img_id)[0]['file_name'].split('.')[0]
            height = self.coco.loadImgs(img_id)[0]['height']
            width = self.coco.loadImgs(img_id)[0]['width']
            name2wh[img_name]={'width':width, 'height': height}

        self.flag = np.zeros(len(self.ids), dtype=np.uint8)
        self.img_id2index = {}
        for i, img_id in enumerate(self.ids):
            self.img_id2index[img_id] = i
            img_info = self.coco.loadImgs(img_id)[0]['file_name'].split('.')[0]
            if name2wh[img_info]['width'] / name2wh[img_info]['height'] > 1:
                self.flag[i] = 1

        self._transforms = transforms
        self.phase = phase
        self.prepare = ConvertCocoPolysToMask(return_masks)
        self.task_name = ginfo.task_name

    def _filter_ignores(self, target):

        # annotations = target['annotations']
        # cates = np.array([rb['category_id'] for rb in annotations])
        target = list(filter(lambda rb: rb['category_id'] > -1, target))
        # target['annotations'] = annotations
        return target

    def _minus_target_label(self, target, value):

        results = []
        for t in target:
            t['category_id'] -= value
            results.append(t)
        return results

    def __getitem__(self, idx):
        dataset_dict = {}
        img, target, imgname, flag = super(PedestrainDetectionDataset, self).__getitem__(idx)

        target = self._minus_target_label(target, 1)
        total = len(target)
        image_id = self.ids[idx]

        target = {'image_id': image_id, 'annotations': target}
        img, target = self.prepare(img, target)
        image_shape = (img.size[-1], img.size[-2])  # h, w
        self._record_image_size(dataset_dict, img)

        if self._transforms is not None:
            img, target = self._transforms(img, target)

        dataset_dict['orig_size'] = target['orig_size']
        dataset_dict['size'] = target['size']
        del target['image_id']
        del target['orig_size']
        del target['size']

        instances = Instances(image_shape, **target)

        dataset_dict["image"] = img
        dataset_dict["image_id"] = image_id
        dataset_dict["label"] = -1
        dataset_dict["instances"] = instances
        dataset_dict["filename"] = imgname
        dataset_dict["flag"] = flag

        return dataset_dict

    @staticmethod
    def _record_image_size(dataset_dict, image):
        """
        Raise an error if the image does not match the size specified in the dict.
        """
        # To ensure bbox always remap to original image size
        if "width" not in dataset_dict:
            dataset_dict["width"] = image.size[1]
        if "height" not in dataset_dict:
            dataset_dict["height"] = image.size[0]
