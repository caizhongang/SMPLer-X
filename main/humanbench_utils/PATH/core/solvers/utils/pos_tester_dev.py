import logging
import os
import torch
from scipy.io import loadmat, savemat
import json_tricks as json
import numpy as np
import time

from collections import OrderedDict, defaultdict
from xtcocotools.cocoeval import COCOeval

from .seg_tester_dev import DatasetEvaluator
from .nms import oks_nms, soft_oks_nms


class PoseEvaluator(DatasetEvaluator):
    """
    Evaluate semantic segmentation metrics.
    """

    def __init__(
        self,
        dataset_name,
        config,
        distributed=True,
        output_dir=None,
    ):
        """
        Args:
            dataset_name (str): name of the dataset to be evaluated.
            distributed (bool): if True, will collect results from all ranks for evaluation.
                Otherwise, will evaluate the results in the current process.
            output_dir (str): an output directory to dump results.
            num_classes, ignore_label: deprecated argument
        """
        self._logger = logging.getLogger(__name__)

        self._dataset_name = dataset_name
        self._distributed = distributed
        self._output_dir = output_dir

        self._cpu_device = torch.device("cpu")

        self.use_nms = config.evaluation.cfg.get('use_nms', True)
        self.soft_nms = config.evaluation.cfg.soft_nms
        self.nms_thr = config.evaluation.cfg.nms_thr
        self.oks_thr = config.evaluation.cfg.oks_thr
        self.vis_thr = config.evaluation.cfg.vis_thr
        self.cls_logits_vis_thr = config.evaluation.cfg.get('cls_logits_vis_thr', -1)

        self.dataset = config.evaluation.cfg.dataset

        if 'coco' in dataset_name.lower():
            self.sigmas = np.array([
                .26, .25, .25, .35, .35, .79, .79, .72, .72, .62, .62, 1.07, 1.07,
                .87, .87, .89, .89
            ]) / 10.0
        elif 'aic' in dataset_name.lower():
            self.sigmas = np.array([
                0.01388152, 0.01515228, 0.01057665, 0.01417709, 0.01497891, 0.01402144,
                0.03909642, 0.03686941, 0.01981803, 0.03843971, 0.03412318, 0.02415081,
                0.01291456, 0.01236173
            ])

        self.interval = config.evaluation.cfg.interval
        self.metric = config.evaluation.cfg.metric
        self.key_indicator = config.evaluation.cfg.key_indicator

        self.ann_info = {}
        self.ann_info['num_joints'] = config.dataset.kwargs.data_cfg['num_joints']
        
        self.annot_json_path = config.dataset.kwargs.ann_file

        self.use_area = config.evaluation.cfg.get('use_area', True)
        
        # for pseudo_label
        self.pseudo_labels_results = []
        self.annot_id = 1

    def reset(self):
        self.results = []
    
    def generate_pseudo_labels(self, inputs, outputs, save_path='./', dataset=None):
        """
        outputs: dict_keys(['preds', 'boxes', 'image_paths', 'bbox_ids'])
        """
        assert dataset is not None
        
        for batch_idx, per_img_path in enumerate(outputs['image_paths']):
            # img_name = per_img_path.split('/')[-1]
            img_id = dataset.name2id[per_img_path[len(self.dataset.img_prefix):]]
            
            # boxes
            bbox = outputs['true_boxes'][batch_idx]
            # w, h = (bbox[2:4] / 1.25 * 200)
            # x1, y1 = bbox[0] - w / 2, bbox[1] - h / 2
            
            # kpts
            kpt = outputs['preds'][batch_idx]
            kpt[:, 2] = kpt[:, 2] > self.vis_thr
            num_keypoints = int(sum(kpt[:, 2]))

            kpt = kpt.astype(np.int)

            for per_kpt in kpt:
                if per_kpt[2] == False:
                    per_kpt[0] = 0
                    per_kpt[1] = 0

            kpt = kpt.flatten().tolist()

            curr_instance_annot = {}
            curr_instance_annot['image_id'] = img_id
            # curr_instance_annot['bbox'] = [x1, y1, w, h]
            curr_instance_annot['bbox'] = bbox
            curr_instance_annot['keypoints'] = kpt
            curr_instance_annot['num_keypoints'] = num_keypoints
            curr_instance_annot['segmentation'] = []
            curr_instance_annot['area'] = int(bbox[2]) * int(bbox[3])
            curr_instance_annot['iscrowd'] = 0
            curr_instance_annot['category_id'] = 1
            curr_instance_annot['id'] = self.annot_id
            self.annot_id += 1

            self.pseudo_labels_results.append(curr_instance_annot)

        return
    
    def combine_pseudo_labels(self, root=None, dataset_name=None):
        with open(self.annot_json_path, 'r') as f:
            origin_json = json.load(f)
        
        origin_json['annotations'] = self.pseudo_labels_results
        origin_json_name = self.annot_json_path.split('/')[-1]

        if dataset_name != None:
            save_path = os.path.join(root, dataset_name + '_with_PL.json')
        else:
            save_path = os.path.join(root, 'with_PL.json')
        with open(save_path, 'w') as f:
            json.dump(origin_json, f, indent=4)
        
        return

    def process(self, inputs, outputs):
        """
        Args:
            inputs: the inputs to a model.
                It is a list of dicts. Each dict corresponds to an image and
                contains keys like "height", "width", "file_name".
            outputs: the outputs of a model. It is either list of semantic segmentation predictions
                (Tensor [H, W]) or list of dicts with key "sem_seg" that contains semantic
                segmentation prediction in the same format.
        """
        # for input, output in zip(inputs, outputs):
        self.results.append(outputs)

        #  note: sync if multi-gpu

    def evaluate(self):
        """Evaluate coco keypoint results. The pose prediction results will be
        saved in `${res_folder}/result_keypoints.json`.

        Note:
            batch_size: N
            num_keypoints: K
            heatmap height: H
            heatmap width: W

        Args:
            outputs (list(dict))
                :preds (np.ndarray[N,K,3]): The first two dimensions are
                    coordinates, score is the third dimension of the array.
                :boxes (np.ndarray[N,6]): [center[0], center[1], scale[0]
                    , scale[1],area, score]
                :image_paths (list[str]): For example, ['data/coco/val2017
                    /000000393226.jpg']
                :heatmap (np.ndarray[N, K, H, W]): model output heatmap
                :bbox_id (list(int)).
            res_folder (str): Path of directory to save the results.
            metric (str | list[str]): Metric to be performed. Defaults: 'mAP'.

        Returns:
            dict: Evaluation results for evaluation metric.
        """
        metrics = self.metric if isinstance(self.metric, list) else [self.metric]
        allowed_metrics = ['mAP']
        for metric in metrics:
            if metric not in allowed_metrics:
                raise KeyError(f'metric {metric} is not supported')

        assert self._output_dir

        os.makedirs(self._output_dir, exist_ok=True)
        res_file = os.path.join(self._output_dir, f'result_keypoints-{time.time()}.json')

        kpts = defaultdict(list)

        for output in self.results:
            preds = output['preds']
            boxes = output['boxes']
            image_paths = output['image_paths']
            bbox_ids = output['bbox_ids']

            batch_size = len(image_paths)
            for i in range(batch_size):
                image_id = self.dataset.name2id[image_paths[i][len(self.dataset.img_prefix):]]
                img_dict = {
                    'keypoints': preds[i],
                    'center': boxes[i][0:2],
                    'scale': boxes[i][2:4],
                    'area': boxes[i][4],
                    'score': boxes[i][5],
                    'image_id': image_id,
                    'bbox_id': bbox_ids[i], }
                if 'pred_logits' in output:
                    img_dict['pred_logits']=output['pred_logits'][i]
                kpts[image_id].append(img_dict)
        kpts = self._sort_and_unique_bboxes(kpts)

        # rescoring and oks nms
        num_joints = self.ann_info['num_joints']
        vis_thr = self.vis_thr
        oks_thr = self.oks_thr
        valid_kpts = []
        for image_id in kpts.keys():
            img_kpts = kpts[image_id]
            for n_p in img_kpts:
                box_score = n_p['score']
                kpt_score = 0
                valid_num = 0
                for n_jt in range(0, num_joints):
                    t_s = n_p['keypoints'][n_jt][2]
                    if t_s > vis_thr:
                        kpt_score = kpt_score + t_s
                        valid_num = valid_num + 1
                if valid_num != 0:
                    kpt_score = kpt_score / valid_num
                # rescoring
                n_p['score'] = kpt_score * box_score

            if self.use_nms:
                nms = soft_oks_nms if self.soft_nms else oks_nms
                keep = nms(list(img_kpts), oks_thr, sigmas=self.sigmas)
                valid_kpts.append([img_kpts[_keep] for _keep in keep])
            else:
                valid_kpts.append(img_kpts)

        self._write_coco_keypoint_results(valid_kpts, res_file)

        info_str = self._do_python_keypoint_eval(res_file)
        results = OrderedDict({"key_point": info_str})
        self._logger.info(results)

        if self.cls_logits_vis_thr>=0 and 'pred_logits' in self.results[0]:
            print(f"Test with CLS logits and new vis_threshold {self.cls_logits_vis_thr}")

            valid_kpts = []
            vis_thr = self.cls_logits_vis_thr
            # import pdb;pdb.set_trace()
            for image_id in kpts.keys():
                img_kpts = kpts[image_id]
                for n_p in img_kpts:
                    box_score = n_p['score']
                    kpt_score = 0
                    valid_num = 0
                    for n_jt in range(0, num_joints):
                        t_s = n_p['keypoints'][n_jt][2] * n_p['pred_logits'][n_jt][0]
                        if t_s > vis_thr:
                            kpt_score = kpt_score + t_s
                            valid_num = valid_num + 1
                    if valid_num != 0:
                        kpt_score = kpt_score / valid_num
                    # rescoring
                    n_p['score'] = kpt_score * box_score

                if self.use_nms:
                    nms = soft_oks_nms if self.soft_nms else oks_nms
                    keep = nms(list(img_kpts), oks_thr, sigmas=self.sigmas)
                    valid_kpts.append([img_kpts[_keep] for _keep in keep])
                else:
                    valid_kpts.append(img_kpts)

            self._write_coco_keypoint_results(valid_kpts, res_file)

            info_str = self._do_python_keypoint_eval(res_file)
            results = OrderedDict({"key_point": info_str})
            self._logger.info(results)

        os.remove(res_file)
        print(f"{res_file} deleted")
        return results

    def _write_coco_keypoint_results(self, keypoints, res_file):
        """Write results into a json file."""
        data_pack = [{
            'cat_id': self.dataset._class_to_coco_ind[cls],
            'cls_ind': cls_ind,
            'cls': cls,
            'ann_type': 'keypoints',
            'keypoints': keypoints
        } for cls_ind, cls in enumerate(self.dataset.classes)
                     if not cls == '__background__']

        results = self._coco_keypoint_results_one_category_kernel(data_pack[0])

        print(f"save results to {res_file}")
        with open(res_file, 'w') as f:
            json.dump(results, f, sort_keys=True, indent=4)

    def _coco_keypoint_results_one_category_kernel(self, data_pack):
        """Get coco keypoint results."""
        cat_id = data_pack['cat_id']
        keypoints = data_pack['keypoints']
        cat_results = []

        for img_kpts in keypoints:
            if len(img_kpts) == 0:
                continue

            _key_points = np.array(
                [img_kpt['keypoints'] for img_kpt in img_kpts])
            key_points = _key_points.reshape(-1,
                                             self.ann_info['num_joints'] * 3)

            result = [{
                'image_id': img_kpt['image_id'],
                'category_id': cat_id,
                'keypoints': key_point.tolist(),
                'score': float(img_kpt['score']),
                'center': img_kpt['center'].tolist(),
                'scale': img_kpt['scale'].tolist()
            } for img_kpt, key_point in zip(img_kpts, key_points)]

            cat_results.extend(result)

        return cat_results

    def _do_python_keypoint_eval(self, res_file):
        """Keypoint evaluation using COCOAPI."""
        coco_det = self.dataset.coco.loadRes(res_file)
        coco_eval = COCOeval(self.dataset.coco, coco_det, 'keypoints', self.sigmas, use_area=self.use_area)
        coco_eval.params.useSegm = None
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

        stats_names = [
            'AP', 'AP .5', 'AP .75', 'AP (M)', 'AP (L)', 'AR', 'AR .5',
            'AR .75', 'AR (M)', 'AR (L)'
        ]

        info_str = list(zip(stats_names, coco_eval.stats))

        return info_str

    def _sort_and_unique_bboxes(self, kpts, key='bbox_id'):
        """sort kpts and remove the repeated ones."""
        for img_id, persons in kpts.items():
            num = len(persons)
            kpts[img_id] = sorted(kpts[img_id], key=lambda x: x[key])
            for i in range(num - 1, 0, -1):
                if kpts[img_id][i][key] == kpts[img_id][i - 1][key]:
                    del kpts[img_id][i]

        return kpts


class MPIIPoseEvaluator(DatasetEvaluator):
    """
    Evaluate semantic segmentation metrics.
    """

    def __init__(
        self,
        dataset_name,
        config,
        distributed=True,
        output_dir=None,
    ):
        """
        Args:
            dataset_name (str): name of the dataset to be evaluated.
            distributed (bool): if True, will collect results from all ranks for evaluation.
                Otherwise, will evaluate the results in the current process.
            output_dir (str): an output directory to dump results.
            num_classes, ignore_label: deprecated argument
        """
        self._logger = logging.getLogger(__name__)
        self._cpu_device = torch.device("cpu")        
        self.annot_root = config.dataset.kwargs.ann_file

        # for pseudo_label
        self.pseudo_labels_results = []

    def reset(self):
        self.results = []
    
    def process(self, inputs, outputs):
        """
        Args:
            inputs: the inputs to a model.
                It is a list of dicts. Each dict corresponds to an image and
                contains keys like "height", "width", "file_name".
            outputs: the outputs of a model. It is either list of semantic segmentation predictions
                (Tensor [H, W]) or list of dicts with key "sem_seg" that contains semantic
                segmentation prediction in the same format.
        """
        # for input, output in zip(inputs, outputs):
        self.results.append(outputs)

        #  note: sync if multi-gpu

    def evaluate(self, res_folder=None, metric='PCKh', **kwargs):
        """Evaluate PCKh for MPII dataset. Adapted from
        https://github.com/leoxiaobin/deep-high-resolution-net.pytorch
        Copyright (c) Microsoft, under the MIT License.
        Note:
            - batch_size: N
            - num_keypoints: K
            - heatmap height: H
            - heatmap width: W
        Args:
            results (list[dict]): Testing results containing the following
                items:
                - preds (np.ndarray[N,K,3]): The first two dimensions are \
                    coordinates, score is the third dimension of the array.
                - boxes (np.ndarray[N,6]): [center[0], center[1], scale[0], \
                    scale[1],area, score]
                - image_paths (list[str]): For example, ['/val2017/000000\
                    397133.jpg']
                - heatmap (np.ndarray[N, K, H, W]): model output heatmap.
            res_folder (str, optional): The folder to save the testing
                results. Default: None.
            metric (str | list[str]): Metrics to be performed.
                Defaults: 'PCKh'.
        Returns:
            dict: PCKh for each joint
        """

        metrics = metric if isinstance(metric, list) else [metric]
        allowed_metrics = ['PCKh']
        for metric in metrics:
            if metric not in allowed_metrics:
                raise KeyError(f'metric {metric} is not supported')

        kpts = []
        for result in self.results:
            preds = result['preds']
            bbox_ids = result['bbox_ids']
            batch_size = len(bbox_ids)
            for i in range(batch_size):
                kpts.append({'keypoints': preds[i], 'bbox_id': bbox_ids[i]})
        kpts = self._sort_and_unique_bboxes(kpts)

        preds = np.stack([kpt['keypoints'] for kpt in kpts])

        # convert 0-based index to 1-based index,
        # and get the first two dimensions.
        preds = preds[..., :2] + 1.0

        if res_folder:
            pred_file = os.path.join(res_folder, 'pred.mat')
            savemat(pred_file, mdict={'preds': preds})

        SC_BIAS = 0.6
        threshold = 0.5

        gt_file = os.path.join(os.path.dirname(self.annot_root), 'mpii_gt_val.mat')
        gt_dict = loadmat(gt_file)
        dataset_joints = gt_dict['dataset_joints']
        jnt_missing = gt_dict['jnt_missing']
        pos_gt_src = gt_dict['pos_gt_src']
        headboxes_src = gt_dict['headboxes_src']

        pos_pred_src = np.transpose(preds, [1, 2, 0])

        head = np.where(dataset_joints == 'head')[1][0]
        lsho = np.where(dataset_joints == 'lsho')[1][0]
        lelb = np.where(dataset_joints == 'lelb')[1][0]
        lwri = np.where(dataset_joints == 'lwri')[1][0]
        lhip = np.where(dataset_joints == 'lhip')[1][0]
        lkne = np.where(dataset_joints == 'lkne')[1][0]
        lank = np.where(dataset_joints == 'lank')[1][0]

        rsho = np.where(dataset_joints == 'rsho')[1][0]
        relb = np.where(dataset_joints == 'relb')[1][0]
        rwri = np.where(dataset_joints == 'rwri')[1][0]
        rkne = np.where(dataset_joints == 'rkne')[1][0]
        rank = np.where(dataset_joints == 'rank')[1][0]
        rhip = np.where(dataset_joints == 'rhip')[1][0]

        jnt_visible = 1 - jnt_missing
        uv_error = pos_pred_src - pos_gt_src
        uv_err = np.linalg.norm(uv_error, axis=1)
        headsizes = headboxes_src[1, :, :] - headboxes_src[0, :, :]
        headsizes = np.linalg.norm(headsizes, axis=0)
        headsizes *= SC_BIAS
        scale = headsizes * np.ones((len(uv_err), 1), dtype=np.float32)
        scaled_uv_err = uv_err / scale
        scaled_uv_err = scaled_uv_err * jnt_visible
        jnt_count = np.sum(jnt_visible, axis=1)
        less_than_threshold = (scaled_uv_err <= threshold) * jnt_visible
        PCKh = 100. * np.sum(less_than_threshold, axis=1) / jnt_count

        # save
        rng = np.arange(0, 0.5 + 0.01, 0.01)
        pckAll = np.zeros((len(rng), 16), dtype=np.float32)

        for r, threshold in enumerate(rng):
            less_than_threshold = (scaled_uv_err <= threshold) * jnt_visible
            pckAll[r, :] = 100. * np.sum(
                less_than_threshold, axis=1) / jnt_count

        PCKh = np.ma.array(PCKh, mask=False)
        PCKh.mask[6:8] = True

        jnt_count = np.ma.array(jnt_count, mask=False)
        jnt_count.mask[6:8] = True
        jnt_ratio = jnt_count / np.sum(jnt_count).astype(np.float64)

        name_value = [('Head', PCKh[head]),
                      ('Shoulder', 0.5 * (PCKh[lsho] + PCKh[rsho])),
                      ('Elbow', 0.5 * (PCKh[lelb] + PCKh[relb])),
                      ('Wrist', 0.5 * (PCKh[lwri] + PCKh[rwri])),
                      ('Hip', 0.5 * (PCKh[lhip] + PCKh[rhip])),
                      ('Knee', 0.5 * (PCKh[lkne] + PCKh[rkne])),
                      ('Ankle', 0.5 * (PCKh[lank] + PCKh[rank])),
                      ('PCKh', np.sum(PCKh * jnt_ratio)),
                      ('PCKh@0.1', np.sum(pckAll[10, :] * jnt_ratio))]
        name_value = OrderedDict(name_value)

        return name_value

    def _sort_and_unique_bboxes(self, kpts, key='bbox_id'):
        """sort kpts and remove the repeated ones."""
        kpts = sorted(kpts, key=lambda x: x[key])
        num = len(kpts)
        for i in range(num - 1, 0, -1):
            if kpts[i][key] == kpts[i - 1][key]:
                del kpts[i]

        return kpts
