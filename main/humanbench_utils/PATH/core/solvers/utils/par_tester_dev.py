import itertools
import json
import logging
import os
from collections import OrderedDict

import numpy as np
import pycocotools.mask as mask_util
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn import functional as F
import cv2
from ast import literal_eval
try:
    import spring.linklink as link
except:
    import linklink as link

from .seg_tester_dev import DatasetEvaluator
import sklearn.metrics as metrics
from PIL import Image

class HumParEvaluator(DatasetEvaluator):
    """
    Evaluate human parsing metrics, specifically, for Human3.6M
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

        self._class_names = config.dataset.kwargs.cfg.label_list #[1:] # 0 as background
        self._num_classes = len(self._class_names)
        assert self._num_classes == config.dataset.kwargs.cfg.num_classes, f"{self._num_classes} != {config.dataset.kwargs.cfg.num_classes}"
        self._contiguous_id_to_dataset_id = {i: k for i, k in enumerate(
            self._class_names)}  # Dict that maps contiguous training ids to COCO category ids
        self._ignore_label = config.dataset.kwargs.cfg.ignore_value

    def reset(self):
        self._conf_matrix = np.zeros((self._num_classes, self._num_classes), dtype=np.int64)
        self._predictions = []

    def generate_pseudo_labels(self, inputs, outputs, dataset=None, save_dir='./'):
        assert dataset is not None
        assert dataset.pseudo_labels_palette is not None, "palette follows the default property of the Human3.6M dataset."
        # import pdb;
        # pdb.set_trace()
        palette = np.array(dataset.pseudo_labels_palette)
        assert palette.shape[1] == 3
        assert len(palette.shape) == 2
        palette = palette.flatten().tolist()
        palette = palette + [255] * (256 * 3 - len(palette))

        for _idx, output in enumerate(outputs):
            par_pred = output["sem_seg"]

            try:
                gt = np.array([inputs["height"][_idx].to(self._cpu_device), inputs["width"][_idx].to(self._cpu_device)]).astype(np.int)
            except:
                raise OSError("Height and width are not recorded during dataloading!")
            # import pdb;
            # pdb.set_trace()
            par_pred = output["sem_seg"]
            par_pred_size = par_pred.size()
            gt_h, gt_w = gt[-2], gt[-1]

            if par_pred_size[-2]!=gt_h or par_pred_size[-1]!=gt_w:
                par_pred = F.upsample(par_pred.unsqueeze(0), (gt_h, gt_w),mode='bilinear')
                output = par_pred[0].argmax(dim=0).to(self._cpu_device)
            else:
                output = par_pred.argmax(dim=0).to(self._cpu_device)

            pred = np.array(output, dtype=np.int)

            png_img = Image.fromarray(np.array(pred).astype(np.uint8))
            png_img.putpalette(palette)
            img_name = inputs['filename'][_idx]
            # TODO auto postfix exchanging
            if img_name[-4:] in ['.jpg', '.png', '.JPG']:
                save_png_name = img_name.replace(img_name[-4:],'.png')
            elif '.' not in img_name[-4:]:
                save_png_name = img_name+'.png'
            dir_path = '/'.join(img_name.split('/')[:-1])
            dir_path = os.path.join(save_dir, dir_path)
            os.makedirs(dir_path,exist_ok=True)
            png_img.save(os.path.join(save_dir,save_png_name))

    def combine_pseudo_labels(self, save_dir=None):
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

        for _idx, output in enumerate(outputs):
            par_pred = output["sem_seg"]

            try:
                gt = np.array(inputs["gt"][_idx].to(self._cpu_device)).astype(np.int)
            except:
                gt = inputs["gt"][_idx].data.astype(np.int)
            # import pdb;
            # pdb.set_trace()
            par_pred = output["sem_seg"]
            par_pred_size = par_pred.size()
            gt_h, gt_w = gt.shape[-2], gt.shape[-1]

            if par_pred_size[-2]!=gt_h or par_pred_size[-1]!=gt_w:
                par_pred = F.upsample(par_pred.unsqueeze(0), (gt_h, gt_w),mode='bilinear')
                output = par_pred[0].argmax(dim=0).to(self._cpu_device)
            else:
                output = par_pred.argmax(dim=0).to(self._cpu_device)

            pred = np.array(output, dtype=np.int)

            if len(pred.shape)!=2:
                import pdb;
                pdb.set_trace()

            self._conf_matrix += self.get_confusion_matrix(gt, pred, self._num_classes, self._ignore_label).astype(np.int64)


    def get_confusion_matrix(self, seg_gt, seg_pred, num_class, ignore=-1):
        import time
        start = time.time()
        ignore_index = seg_gt != ignore
        seg_gt = seg_gt[ignore_index]
        try:
            seg_pred = seg_pred[ignore_index]
        except:
            import pdb;pdb.set_trace()

        index = (seg_gt * num_class + seg_pred).astype('int32')
        label_count = np.bincount(index)
        confusion_matrix = np.zeros((num_class, num_class))

        for i_label in range(num_class):
            for i_pred in range(num_class):
                cur_index = i_label * num_class + i_pred
                if cur_index < len(label_count):
                    confusion_matrix[i_label,
                                     i_pred] = label_count[cur_index]
        return confusion_matrix

    @staticmethod
    def all_gather(data, group=0):
        assert link.get_world_size() == 1, f"distributed eval unsupported yet, uncertain if we can use torch.dist with link jointly"
        if link.get_world_size() == 1:
            return [data]

        # output = [None for _ in range(link.get_world_size())]
        # dist.all_gather_object(output, data, group=group)
        # return output
        # import pdb;pdb.set_trace()
        world_size = link.get_world_size()
        tensors_gather = [torch.ones_like(data) for _ in range(world_size)]
        link.allgather(tensors_gather, data, group=group)
        return tensors_gather

    
    def evaluate(self):
        """
        
        :return: mean_IoU, IoU_array, pixel_acc, mean_acc 
        """
        
        if self._distributed:
            link.synchronize()

            conf_matrix_list = self.all_gather(self._conf_matrix)
            # self._predictions = self.all_gather(self._predictions)
            # self._predictions = list(itertools.chain(*self._predictions))
            if link.get_rank() != 0:
                return

            self._conf_matrix = np.zeros_like(self._conf_matrix)
            for conf_matrix in conf_matrix_list:
                self._conf_matrix += conf_matrix

        # if self._output_dir:
        #     os.makedirs(self._output_dir, exist_ok=True)
        #     file_path = os.path.join(self._output_dir, "humam_parsing_predictions.json")
        #     with open(file_path, "w") as f:
        #         f.write(json.dumps(self._predictions))

        acc = np.full(self._num_classes, np.nan, dtype=np.float)
        iou = np.full(self._num_classes, np.nan, dtype=np.float)
        tp = self._conf_matrix.diagonal().astype(np.float)
        pos_gt = np.sum(self._conf_matrix, axis=0).astype(np.float)
        # class_weights = pos_gt / np.sum(pos_gt)
        pos_pred = np.sum(self._conf_matrix, axis=1).astype(np.float)
        acc_valid = pos_gt > 0
        acc[acc_valid] = tp[acc_valid] / pos_gt[acc_valid]
        iou_valid = (pos_gt + pos_pred) > 0
        union = pos_gt + pos_pred - tp
        iou[acc_valid] = tp[acc_valid] / union[acc_valid]
        macc = np.sum(acc[acc_valid]) / np.sum(acc_valid)
        miou = np.sum(iou[acc_valid]) / np.sum(iou_valid)
        # fiou = np.sum(iou[acc_valid] * class_weights[acc_valid])
        pacc = np.sum(tp) / np.sum(pos_gt)

        res = {}
        res["mIoU"] = 100 * miou
        # res["fwIoU"] = 100 * fiou
        for i, name in enumerate(self._class_names):
            res["IoU-{}".format(name)] = 100 * iou[i]
        res["mACC"] = 100 * macc
        res["pACC"] = 100 * pacc
        for i, name in enumerate(self._class_names):
            res["ACC-{}".format(name)] = 100 * acc[i]

        if self._output_dir:
            file_path = os.path.join(self._output_dir, "human_parsing_evaluation.pth")
            with open(file_path, "wb") as f:
                torch.save(res, f)
        results = OrderedDict({"human_parsing": res})
        self._logger.info(results)
        return results

    def encode_json_sem_seg(self, sem_seg, input_file_name):
        """
        Convert semantic segmentation to COCO stuff format with segments encoded as RLEs.
        See http://cocodataset.org/#format-results
        """
        json_list = []
        for label in np.unique(sem_seg):
            if self._contiguous_id_to_dataset_id is not None:
                assert (
                    label in self._contiguous_id_to_dataset_id
                ), "Label {} is not in the metadata info for {}".format(label, self._dataset_name)
                dataset_id = self._contiguous_id_to_dataset_id[label]
            else:
                dataset_id = int(label)
            mask = (sem_seg == label).astype(np.uint8)
            mask_rle = mask_util.encode(np.array(mask[:, :, None], order="F"))[0]
            mask_rle["counts"] = mask_rle["counts"].decode("utf-8")
            json_list.append(
                {"file_name": input_file_name, "category_id": dataset_id, "segmentation": mask_rle}
            )
        return json_list

class HumParEvaluator_bce_cls(DatasetEvaluator):
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

        self._class_names = config.dataset.kwargs.cfg.label_list #[1:] # 0 as background
        self._num_classes = len(self._class_names)
        assert self._num_classes == config.dataset.kwargs.cfg.num_classes, f"{self._num_classes} != {config.dataset.kwargs.cfg.num_classes}"
        self._contiguous_id_to_dataset_id = {i: k for i, k in enumerate(
            self._class_names)}  # Dict that maps contiguous training ids to COCO category ids
        self._ignore_label = config.dataset.kwargs.cfg.ignore_value
    
    def reset(self):
        self._conf_matrix = [np.zeros((2, 2), dtype=np.int64) for _ in range(self._num_classes)]
        self._predictions = []
        self._labels = []
    
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
        # import pdb;pdb.set_trace()
        for _idx in range(len(outputs)):
            # par_pred = output["sem_cls"]

            try:
                gt = np.array(inputs["gt"][_idx].to(self._cpu_device)).astype(np.int)
            except:
                gt = inputs["gt"][_idx].data.astype(np.int)
            classes = np.unique(gt)
            label = np.zeros(self._num_classes)
            label[classes] = 1
            self._labels.append(label)
        # labels = np.vstack(labels)
        pred = ((outputs>0.5)*1).cpu().numpy()
        self._predictions.append(pred)

    def evaluate(self):
        if self._distributed:
            link.synchronize()
        # add a false prediction to force a 2x2 confusion matrix
        self._predictions.append(np.ones(self._num_classes))
        self._labels.append(np.zeros(self._num_classes))
        preds = np.vstack(self._predictions)
        labels = np.vstack(self._labels)
        acc = np.full(self._num_classes, np.nan, dtype=np.float)
        for i in range(self._num_classes):
            pred = preds[:,i]
            label = labels[:,i]
            confusion_matrix = metrics.confusion_matrix(label, pred)
            # sub 1 to get the right 2x2 confusion matrix
            confusion_matrix[0,1] -= 1
            self._conf_matrix[i] = confusion_matrix

            tp_i = self._conf_matrix[i].diagonal().astype(np.float).sum()
            acc[i] = tp_i / self._conf_matrix[i].sum()

        res = {}
        macc = acc.mean()
        res['mACC'] = macc
        for i, name in enumerate(self._class_names):
            res["ACC-{}".format(name)] = 100 * acc[i]
        for i, name in enumerate(self._class_names):
            res['conf-{}'.format(name)] = self._conf_matrix[i]
        results = OrderedDict({"human_parsing": res})
        self._logger.info(results)
        return results


