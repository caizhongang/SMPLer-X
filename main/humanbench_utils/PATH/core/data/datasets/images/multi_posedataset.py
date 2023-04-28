import copy
from abc import ABCMeta, abstractmethod
import numpy as np
from torch.utils.data import Dataset
from pathlib import Path

import os
import cv2
import random
import time
import os.path as osp
import os
import warnings
from collections import OrderedDict, defaultdict
from core.data.transforms.pose_transforms import *
import json_tricks as json
import numpy as np
from xtcocotools.coco import COCO
from xtcocotools.cocoeval import COCOeval

from core.utils import sync_print

try:
    from petrelbox.io import PetrelHelper
except:
    print("ceph can not be used")


class PetrelCOCO(COCO):
    def __init__(self, annotation_file=None, test_index=None, ann_data=None):
        """
        Constructor of Microsoft COCO helper class for reading and visualizing annotations.
        :param annotation_file (str): location of annotation file
        :param image_folder (str): location to the folder that hosts images.
        :return:
        """
        self.dataset,self.anns,self.cats,self.imgs = dict(),dict(),dict(),dict()
        self.imgToAnns, self.catToImgs = defaultdict(list), defaultdict(list)
        self.anno_file = [annotation_file]
        self.test_index = test_index
        if annotation_file is not None:
            print('loading annotations into memory...')
            tic = time.time()
            # https://github.com/cocodataset/cocoapi/pull/453/
            if ann_data == None:
                dataset = PetrelHelper.load_json(annotation_file)
            else:
                dataset = ann_data
            assert type(dataset)==dict, 'annotation file format {} not supported'.format(type(dataset))
            print('Done (t={:0.2f}s)'.format(time.time()- tic))
            self.dataset = dataset
            self.createIndex()
        if 'annotations' in self.dataset:
            for i in range(len(self.dataset['annotations'])):
                if self.test_index is not None:
                    keypoints = np.array(self.dataset['annotations'][i]['keypoints']).reshape([-1, 3])
                    keypoints = keypoints[self.test_index, :]
                    self.dataset['annotations'][i]['keypoints'] = keypoints.reshape([-1]).tolist()
                if 'iscrowd' not in self.dataset['annotations'][i]:
                    self.dataset['annotations'][i]['iscrowd'] = False


class MultiPoseDatasetDev(Dataset):
    def __init__(self,
                 ginfo,
                 ann_file,
                 img_prefix,
                 data_cfg,
                 test_mode=False,
                 use_udp=False,
                 dataset_name='coco',
                 data_use_ratio=1,
                 **kwargs):

        assert dataset_name in ['coco', 'aic', 'posetrack', 'halpe', 'JRDB2022', 'h36m', 'mhp', 'penn_action', '3DPW', '3DHP', 'AIST'], "invalid dataset name input"
        self.dataset_name = dataset_name
        self.image_info = {}
        self.ann_info = {}
        self.initialized = False

        self.use_ceph = True
        self.annotations_path = ann_file
        self.img_prefix = img_prefix
        self.test_mode = test_mode
        print('data_cfg0',data_cfg)
        # data_cfg=demjson.decode(data_cfg)
        # print('data_cfg',data_cfg)
        self.ann_info['image_size'] = np.array(data_cfg['image_size'])
        self.ann_info['heatmap_size'] = np.array(data_cfg['heatmap_size'])
        self.ann_info['num_joints'] = data_cfg['num_joints']

        self.ann_info['inference_channel'] = data_cfg['inference_channel']
        self.ann_info['num_output_channels'] = data_cfg['num_output_channels']
        self.ann_info['dataset_channel'] = data_cfg['dataset_channel']

        self.db = []
        self.task_name = ginfo.task_name

        if test_mode:
            pipeline = [
                LoadImageFromFile(),
                TopDownAffine(use_udp=use_udp),
                ToUNTensor(),
                Collect(keys=['image'],
                        meta_keys=['image_file', 'center', 'scale', 'rotation', 'bbox_score', 'flip_pairs'])
            ]
        else:
            if self.dataset_name == 'coco' or self.dataset_name == 'aic' or self.dataset_name == 'penn_action' or self.dataset_name == 'mhp':
                pipeline = [
                    LoadImageFromFile(),
                    TopDownRandomFlip(flip_prob=0.5),
                    TopDownHalfBodyTransform(num_joints_half_body=8, prob_half_body=0.3),
                    TopDownGetRandomScaleRotation(rot_factor=40, scale_factor=0.5),
                    TopDownAffine(use_udp=use_udp),
                    ToUNTensor(),
                    TopDownGenerateTarget(sigma=2, encoding='UDP' if use_udp else 'MSRA'),
                    Collect(keys=['image', 'label', 'target_weight'],
                            meta_keys=['image_file', 'joints_3d', 'joints_3d_visible', 'center', 'scale','rotation',
                                    'bbox_score', 'flip_pairs'])
                ]

            else:
                pipeline = [
                    LoadImageFromFile(),
                    # TopDownGetBboxCenterScale(padding=1.25),
                    TopDownRandomShiftBboxCenter(shift_factor=0.16, prob=0.3),
                    TopDownRandomFlip(flip_prob=0.5),
                    TopDownHalfBodyTransform(num_joints_half_body=8, prob_half_body=0.3),
                    TopDownGetRandomScaleRotation(rot_factor=40, scale_factor=0.5),
                    TopDownAffine(use_udp=use_udp),
                    ToUNTensor(),
                    TopDownGenerateTarget(sigma=2, encoding='UDP' if use_udp else 'MSRA'),
                    Collect(keys=['image', 'label', 'target_weight'],
                            meta_keys=['image_file', 'joints_3d', 'joints_3d_visible', 'center', 'scale','rotation',
                                    'bbox_score', 'flip_pairs'])
                ]

        self.pipeline = ComposeX(pipeline)

        self.use_gt_bbox = data_cfg['use_gt_bbox']
        self.bbox_file = data_cfg['bbox_file'] if data_cfg['bbox_file'].startswith('/mnt') else (Path(__file__).parent / 'resources' / data_cfg['bbox_file']).resolve()
        self.det_bbox_thr = data_cfg.get('det_bbox_thr', 0.0)
        if 'image_thr' in data_cfg:
            warnings.warn(
                'image_thr is deprecated, '
                'please use det_bbox_thr instead', DeprecationWarning)
            self.det_bbox_thr = data_cfg['image_thr']


        self.ann_info['flip_pairs'] =  data_cfg['flip_pairs']
        self.ann_info['upper_body_ids'] = data_cfg['upper_body_ids']
        self.ann_info['lower_body_ids'] = data_cfg['lower_body_ids']

        self.ann_info['use_different_joint_weights'] = False
        self.ann_info['joint_weights'] = np.array(
            data_cfg['joint_weights'],
            dtype=np.float32).reshape((self.ann_info['num_joints'], 1))

        # 'https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/'
        # 'pycocotools/cocoeval.py#L523'

        self.coco = PetrelCOCO(ann_file)

        cats = [
            cat['name'] for cat in self.coco.loadCats(self.coco.getCatIds())
        ]
        self.classes = ['__background__'] + cats
        self.num_classes = len(self.classes)
        self._class_to_ind = dict(zip(self.classes, range(self.num_classes)))
        self._class_to_coco_ind = dict(zip(cats, self.coco.getCatIds()))
        self._coco_ind_to_class_ind = dict(
            (self._class_to_coco_ind[cls], self._class_to_ind[cls])
            for cls in self.classes[1:])
        self.img_ids = self.coco.getImgIds()
        self.num_images = len(self.img_ids)
        self.id2name, self.name2id = self._get_mapping_id_name(self.coco.imgs)


        self.db = self._get_db()
        if data_use_ratio != 1:
            self.db = random.sample(self.db, int(len(self.db) * data_use_ratio))

        print(f'=> num_images: {self.num_images}')
        print(f'=> load {len(self.db)} samples')

    @staticmethod
    def _get_mapping_id_name(imgs):
        """
        Args:
            imgs (dict): dict of image info.

        Returns:
            tuple: Image name & id mapping dicts.

            - id2name (dict): Mapping image id to name.
            - name2id (dict): Mapping image name to id.
        """
        id2name = {}
        name2id = {}
        for image_id, image in imgs.items():
            file_name = image['file_name']
            id2name[image_id] = file_name
            name2id[file_name] = image_id

        return id2name, name2id

    def _get_db(self):
        """Load dataset."""
        if (not self.test_mode) or self.use_gt_bbox:
            # use ground truth bbox
            gt_db = self._load_coco_keypoint_annotations()
        else:
            # use bbox from detection
            gt_db = self._load_coco_person_detection_results()
        return gt_db

    def _load_coco_keypoint_annotations(self):
        """Ground truth bbox and keypoints."""
        gt_db = []
        for img_id in self.img_ids:
            gt_db.extend(self._load_coco_keypoint_annotation_kernel(img_id))
        return gt_db

    def _load_coco_keypoint_annotation_kernel(self, img_id):
        """load annotation from COCOAPI.

        Note:
            bbox:[x1, y1, w, h]
        Args:
            img_id: coco image id
        Returns:
            dict: db entry
        """
        img_ann = self.coco.loadImgs(img_id)[0]
        width = img_ann['width']
        height = img_ann['height']
        num_joints = self.ann_info['num_joints']

        ann_ids = self.coco.getAnnIds(imgIds=img_id, iscrowd=False)
        objs = self.coco.loadAnns(ann_ids)

        # sanitize bboxes
        valid_objs = []
        for obj in objs:
            if 'bbox' not in obj:
                continue
            x, y, w, h = obj['bbox']
            x1 = max(0, x)
            y1 = max(0, y)
            x2 = min(width - 1, x1 + max(0, w - 1))
            y2 = min(height - 1, y1 + max(0, h - 1))
            if ('area' not in obj or obj['area'] > 0) and x2 > x1 and y2 > y1:
                obj['clean_bbox'] = [x1, y1, x2 - x1, y2 - y1]
                valid_objs.append(obj)
        objs = valid_objs

        bbox_id = 0
        rec = []
        for obj in objs:
            if 'keypoints' not in obj:
                continue
            if max(obj['keypoints']) == 0:
                continue
            if 'num_keypoints' in obj and obj['num_keypoints'] == 0:
                continue
            joints_3d = np.zeros((num_joints, 3), dtype=np.float32)
            joints_3d_visible = np.zeros((num_joints, 3), dtype=np.float32)

            keypoints = np.array(obj['keypoints']).reshape(-1, 3)

            if self.dataset_name == 'posetrack':
                keypoints = np.delete(keypoints, [3, 4], axis=0)  # keypoint idx == 3 and 4 not annot

            joints_3d[:, :2] = keypoints[:, :2]
            joints_3d_visible[:, :2] = np.minimum(1, keypoints[:, 2:3])

            center, scale = self._xywh2cs(*obj['clean_bbox'][:4])

            image_file = os.path.join(self.img_prefix, self.id2name[img_id])
            rec.append({
                'image_file': image_file,
                'center': center,
                'scale': scale,
                'bbox': obj['clean_bbox'][:4],
                'rotation': 0,
                'joints_3d': joints_3d,
                'joints_3d_visible': joints_3d_visible,
                'dataset': self.dataset_name,
                'bbox_score': 1,
                'bbox_id': bbox_id
            })
            bbox_id = bbox_id + 1

        return rec

    def _xywh2cs(self, x, y, w, h):
        """This encodes bbox(x,y,w,w) into (center, scale)

        Args:
            x, y, w, h

        Returns:
            tuple: A tuple containing center and scale.

            - center (np.ndarray[float32](2,)): center of the bbox (x, y).
            - scale (np.ndarray[float32](2,)): scale of the bbox w & h.
        """
        aspect_ratio = self.ann_info['image_size'][0] / self.ann_info[
            'image_size'][1]
        center = np.array([x + w * 0.5, y + h * 0.5], dtype=np.float32)

        if (not self.test_mode) and np.random.rand() < 0.3:
            center += 0.4 * (np.random.rand(2) - 0.5) * [w, h]

        if w > aspect_ratio * h:
            h = w * 1.0 / aspect_ratio
        elif w < aspect_ratio * h:
            w = h * aspect_ratio

        # pixel std is 200.0
        scale = np.array([w / 200.0, h / 200.0], dtype=np.float32)
        # padding to include proper amount of context
        scale = scale * 1.25

        return center, scale

    def _load_coco_person_detection_results(self):
        """Load coco person detection results."""
        num_joints = self.ann_info['num_joints']
        all_boxes = None
        with open(self.bbox_file, 'r') as f:
            all_boxes = json.load(f)

        if not all_boxes:
            raise ValueError('=> Load %s fail!' % self.bbox_file)

        print(f'=> Total boxes: {len(all_boxes)}')

        kpt_db = []
        bbox_id = 0
        for det_res in all_boxes:
            if det_res['category_id'] != 1:
                continue

            image_file = os.path.join(self.img_prefix,
                                      self.id2name[det_res['image_id']])
            box = det_res['bbox']
            score = det_res['score']

            if score < self.det_bbox_thr:
                continue

            center, scale = self._xywh2cs(*box[:4])
            joints_3d = np.zeros((num_joints, 3), dtype=np.float32)
            joints_3d_visible = np.ones((num_joints, 3), dtype=np.float32)
            kpt_db.append({
                'image_file': image_file,
                'center': center,
                'scale': scale,
                'rotation': 0,
                'bbox': box[:4],
                'bbox_score': score,
                'dataset': self.dataset_name,
                'joints_3d': joints_3d,
                'joints_3d_visible': joints_3d_visible,
                'bbox_id': bbox_id
            })
            bbox_id = bbox_id + 1
        print(f'=> Total boxes after filter '
              f'low score@{self.det_bbox_thr}: {bbox_id}')
        return kpt_db

    def __len__(self):
        """Get the size of the dataset."""
        return len(self.db)

    def __getitem__(self, idx):
        """Get the sample given index."""
        results = copy.deepcopy(self.db[idx])
        results['ann_info'] = self.ann_info
        out = self.pipeline(results)
        # del out['ann_info']
        return out  # dict_keys(['image_file', 'center', 'scale', 'bbox', 'rotation', 'joints_3d', 'joints_3d_visible',
                               # 'dataset', 'bbox_score', 'bbox_id', 'ann_info', 'image', 'flipped', 'label',
                               # 'target_weight'])
