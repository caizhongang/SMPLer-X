import itertools
import json
import logging
import os
import time
from collections import OrderedDict

import numpy as np
from .detools.box import DetBox
from pycocotools.cocoeval import COCOeval
from pycocotools.coco import COCO
import pycocotools.mask as mask_util
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn import functional as F
from pathlib import Path

import cv2, math
from tqdm import tqdm
from multiprocessing import Queue, Process
from scipy.optimize import linear_sum_assignment
from core.data.datasets.images import peddet_dataset

try:
    import spring.linklink as link
except:
    import linklink as link

from .seg_tester_dev import DatasetEvaluator

from petrelbox.io import PetrelHelper

PERSON_CLASSES = ['background', 'person']

class PedDetEvaluator(DatasetEvaluator):
    """
    Evaluate Pedestrain Detection metrics
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
        self._thr = config.tester.kwargs.pos_thr
        # self._gt_path = config.tester.kwargs.gt_path if config.tester.kwargs.gt_path.startswith('/mnt') or 's3://' in config.tester.kwargs.gt_path else str((Path(peddet_dataset.__file__).parent / 'resources' / config.tester.kwargs.gt_path).resolve())
        self._gt_path = config.tester.kwargs.gt_path

    def reset(self):
        self._predictions = []

    def process(self, inputs, outputs):
        """
        Args:
            inputs: the inputs to a model
                It is a list of dicts. Each dict corresponds to an image and
                contains keys like "height", "width", "file_name".
            outputs: the outputs of a model. It is list of dicts with key "logits" and "bbox"
        """
        orig_target_sizes = inputs["orig_size"]
        filtered_outputs = [self.deplicate(r, self._thr) for r in outputs]
        results = [{k:v.cpu().numpy() for k, v in r.items()} for r in filtered_outputs]
        dtboxes = [np.hstack([r['boxes'], r['scores'][:, np.newaxis]]) for r in results]
        dtboxes = [self.boxes_dump(db) for db in dtboxes]
        filenames = inputs['filename']
        res = [{'ID':name, 'dtboxes':db} for name, db in zip(filenames, dtboxes)]
        assert len(res) == len(outputs)
        self._predictions.extend(res)

    def deplicate(self, record, thr):

        assert 'scores' in record
        names = [k for (k, v) in record.items()]
        flag = record['scores'] >= thr
        for name in names:
            record[name] = record[name][flag]
        return record

    @staticmethod
    def all_gather(data):
        """
        Run all_gather on arbitrary picklable data (not necessarily tensors)
        Args:
            data: any picklable object
        Returns:
            list[data]: list of data gathered from each rank
        """
        world_size = link.get_world_size()
        if world_size == 1:
            return [data]

        # serialized to a Tensor
        buffer = pickle.dumps(data)
        storage = torch.ByteStorage.from_buffer(buffer)
        tensor = torch.ByteTensor(storage).to("cuda")

        # obtain Tensor size of each rank
        local_size = torch.tensor([tensor.numel()], device="cuda")
        size_list = [torch.tensor([0], device="cuda") for _ in range(world_size)]
        link.allgather(size_list, local_size)
        size_list = [int(size.item()) for size in size_list]
        max_size = max(size_list)

        # receiving Tensor from all ranks
        # we pad the tensor because torch all_gather does not support
        # gathering tensors of different shapes
        tensor_list = []
        for _ in size_list:
            tensor_list.append(torch.empty((max_size,), dtype=torch.uint8, device="cuda"))
        if local_size != max_size:
            padding = torch.empty(size=(max_size - local_size,), dtype=torch.uint8, device="cuda")
            tensor = torch.cat((tensor, padding), dim=0)
        link.allgather(tensor_list, tensor)

        data_list = []
        for size, tensor in zip(size_list, tensor_list):
            buffer = tensor.cpu().numpy().tobytes()[:size]
            data_list.append(pickle.loads(buffer))

        return data_list

    def evaluate(self):
        """
        : return: "AP", "MR", "JI", "Recall"
        """
        if self._distributed:
            link.synchronize()
            predictions_list = self.all_gather(self._predictions)
            if link.get_rank() != 0:
                return

        proxy_number = time.time()
        file_path = os.path.join('tmp', 'CrowdHuman', f'Iter-{proxy_number}.human')
        self.save_results(self._predictions, file_path)
        eval_results = self._evaluate_predictions_on_crowdhuman(self._gt_path, file_path)

        res = {}
        metric_names = ["AP", "MR", "JI", "Recall"]
        for k, v in zip(metric_names, eval_results):
            print(f"{k}: {v}")
            res[k] = v
        results = OrderedDict({"pedestrain_detection": res})
        self._logger.info(results)
        os.remove(file_path)
        print(f"{file_path} deleted")

        return results

    def save_results(self, content, fpath):
        os.makedirs(os.path.dirname(fpath), exist_ok=True)
        print("save results to {}".format(fpath))
        with open(fpath,'w') as fid:
            for db in content:
                line = json.dumps(db)+'\n'
                fid.write(line)

    def boxes_dump(self, dtboxes):

        n, boxes = dtboxes.shape[0], []
        for i in range(n):
            db = np.float64(dtboxes[i,:])
            dbox = DetBox(db[0], db[1], db[2]-db[0],
                db[3]-db[1], tag = 1, score = db[4])
            boxes.append(dbox.dumpOdf())
        return boxes

    def _evaluate_predictions_on_crowdhuman(self, gt_path, dt_path, target_key="box", mode=0):
        """
        Evaluate the coco results using COCOEval API.
        """
        database = Database(gt_path, dt_path, target_key, None, mode)
        database.compare()
        AP, recall, data = database.eval_AP()
        mMR, _ = database.eval_MR(fppiX=data[-2], fppiY=data[-1])
        # return AP, mMR, computeJaccard(gt_path, dt_path), recall
        return AP, mMR, 0, recall
    

class Image(object):
    def __init__(self, mode):
        self.ID = None
        self._width = None
        self._height = None
        self.dtboxes = None
        self.gtboxes = None
        self.eval_mode = mode

        self._ignNum = None
        self._gtNum = None
        self._dtNum = None

    def load(self, record, body_key, head_key, class_names, gtflag):
        """
        :meth: read the object from a dict
        """
        if "ID" in record and self.ID is None:
            self.ID = record['ID']
        if "width" in record and self._width is None:
            self._width = record["width"]
        if "height" in record and self._height is None:
            self._height = record["height"]
        if gtflag:
            self._gtNum = len(record["gtboxes"])
            body_bbox, head_bbox = self.load_gt_boxes(record, 'gtboxes', class_names)
            if self.eval_mode == 0:
                self.gtboxes = body_bbox
                self._ignNum = (body_bbox[:, -1] == -1).sum()
            elif self.eval_mode == 1:
                self.gtboxes = head_bbox
                self._ignNum = (head_bbox[:, -1] == -1).sum()
            elif self.eval_mode == 2:
                gt_tag = np.array(
                    [body_bbox[i, -1] != -1 and head_bbox[i, -1] != -1
                     for i in range(len(body_bbox))]
                )
                self._ignNum = (gt_tag == 0).sum()
                self.gtboxes = np.hstack(
                    (body_bbox[:, :-1], head_bbox[:, :-1], gt_tag.reshape(-1, 1))
                )
            else:
                raise Exception('Unknown evaluation mode!')
        if not gtflag:
            self._dtNum = len(record["dtboxes"])
            if self.eval_mode == 0:
                self.dtboxes = self.load_det_boxes(record, 'dtboxes', body_key, 'score')
            elif self.eval_mode == 1:
                self.dtboxes = self.load_det_boxes(record, 'dtboxes', head_key, 'score')
            elif self.eval_mode == 2:
                body_dtboxes = self.load_det_boxes(record, 'dtboxes', body_key)
                head_dtboxes = self.load_det_boxes(record, 'dtboxes', head_key, 'score')
                self.dtboxes = np.hstack((body_dtboxes, head_dtboxes))
            else:
                raise Exception('Unknown evaluation mode!')

    def compare_caltech(self, thres):
        """
        :meth: match the detection results with the groundtruth by Caltech matching strategy
        :param thres: iou threshold
        :type thres: float
        :return: a list of tuples (dtbox, imageID), in the descending sort of dtbox.score
        """
        if self.dtboxes is None or self.gtboxes is None:
            return list()

        dtboxes = self.dtboxes if self.dtboxes is not None else list()
        gtboxes = self.gtboxes if self.gtboxes is not None else list()

        dtboxes = np.array(sorted(dtboxes, key=lambda x: x[-1], reverse=True))
        gtboxes = np.array(sorted(gtboxes, key=lambda x: x[-1], reverse=True))
        if len(dtboxes) and len(gtboxes):
            overlap_iou = self.box_overlap_opr(dtboxes, gtboxes[gtboxes[:, -1] > 0], True)
            overlap_ioa = self.box_overlap_opr(dtboxes, gtboxes[gtboxes[:, -1] <= 0], False)
            ign = np.any(overlap_ioa > thres, 1)
            pos = np.any(overlap_iou > thres, 1)
        else:
            return list()

        scorelist = list()
        for i, dt in enumerate(dtboxes):
            maxpos = np.argmax(overlap_iou[i])
            if overlap_iou[i, maxpos] > thres:
                overlap_iou[:, maxpos] = 0
                scorelist.append((dt, 1, self.ID, pos[i]))
            elif not ign[i]:
                scorelist.append((dt, 0, self.ID, pos[i]))
        return scorelist

    def compare_caltech_union(self, thres):
        """
        :meth: match the detection results with the groundtruth by Caltech matching strategy
        :param thres: iou threshold
        :type thres: float
        :return: a list of tuples (dtbox, imageID), in the descending sort of dtbox.score
        """
        dtboxes = self.dtboxes if self.dtboxes is not None else list()
        gtboxes = self.gtboxes if self.gtboxes is not None else list()
        if len(dtboxes) == 0:
            return list()
        dt_matched = np.zeros(dtboxes.shape[0])
        gt_matched = np.zeros(gtboxes.shape[0])

        dtboxes = np.array(sorted(dtboxes, key=lambda x: x[-1], reverse=True))
        gtboxes = np.array(sorted(gtboxes, key=lambda x: x[-1], reverse=True))
        dt_body_boxes = np.hstack((dtboxes[:, :4], dtboxes[:, -1][:, None]))
        dt_head_boxes = dtboxes[:, 4:8]
        gt_body_boxes = np.hstack((gtboxes[:, :4], gtboxes[:, -1][:, None]))
        gt_head_boxes = gtboxes[:, 4:8]
        overlap_iou = self.box_overlap_opr(dt_body_boxes, gt_body_boxes, True)
        overlap_head = self.box_overlap_opr(dt_head_boxes, gt_head_boxes, True)
        overlap_ioa = self.box_overlap_opr(dt_body_boxes, gt_body_boxes, False)

        scorelist = list()
        for i, dt in enumerate(dtboxes):
            maxpos = -1
            maxiou = thres
            for j, gt in enumerate(gtboxes):
                if gt_matched[j] == 1:
                    continue
                if gt[-1] > 0:
                    o_body = overlap_iou[i][j]
                    o_head = overlap_head[i][j]
                    if o_body > maxiou and o_head > maxiou:
                        maxiou = o_body
                        maxpos = j
                else:
                    if maxpos >= 0:
                        break
                    else:
                        o_body = overlap_ioa[i][j]
                        if o_body > thres:
                            maxiou = o_body
                            maxpos = j
            if maxpos >= 0:
                if gtboxes[maxpos, -1] > 0:
                    gt_matched[maxpos] = 1
                    dt_matched[i] = 1
                    scorelist.append((dt, 1, self.ID))
                else:
                    dt_matched[i] = -1
            else:
                dt_matched[i] = 0
                scorelist.append((dt, 0, self.ID))
        return scorelist

    def box_overlap_opr(self, dboxes: np.ndarray, gboxes: np.ndarray, if_iou) -> np.ndarray:
        eps = 1e-6
        assert dboxes.shape[-1] >= 4 and gboxes.shape[-1] >= 4
        N, K = dboxes.shape[0], gboxes.shape[0]
        dtboxes = np.tile(np.expand_dims(dboxes, axis=1), (1, K, 1))
        gtboxes = np.tile(np.expand_dims(gboxes, axis=0), (N, 1, 1))

        iw = (np.minimum(dtboxes[:, :, 2], gtboxes[:, :, 2])
              - np.maximum(dtboxes[:, :, 0], gtboxes[:, :, 0]))
        ih = (np.minimum(dtboxes[:, :, 3], gtboxes[:, :, 3])
              - np.maximum(dtboxes[:, :, 1], gtboxes[:, :, 1]))
        inter = np.maximum(0, iw) * np.maximum(0, ih)

        dtarea = (dtboxes[:, :, 2] - dtboxes[:, :, 0]) * (dtboxes[:, :, 3] - dtboxes[:, :, 1])
        if if_iou:
            gtarea = (gtboxes[:, :, 2] - gtboxes[:, :, 0]) * (gtboxes[:, :, 3] - gtboxes[:, :, 1])
            ious = inter / (dtarea + gtarea - inter + eps)
        else:
            ious = inter / (dtarea + eps)
        return ious

    def clip_all_boader(self):

        def _clip_boundary(boxes, height, width):
            assert boxes.shape[-1] >= 4
            boxes[:, 0] = np.minimum(np.maximum(boxes[:, 0], 0), width - 1)
            boxes[:, 1] = np.minimum(np.maximum(boxes[:, 1], 0), height - 1)
            boxes[:, 2] = np.maximum(np.minimum(boxes[:, 2], width), 0)
            boxes[:, 3] = np.maximum(np.minimum(boxes[:, 3], height), 0)
            return boxes

        assert self.dtboxes.shape[-1] >= 4
        assert self.gtboxes.shape[-1] >= 4
        assert self._width is not None and self._height is not None
        if self.eval_mode == 2:
            self.dtboxes[:, :4] = _clip_boundary(self.dtboxes[:, :4], self._height, self._width)
            self.gtboxes[:, :4] = _clip_boundary(self.gtboxes[:, :4], self._height, self._width)
            self.dtboxes[:, 4:8] = _clip_boundary(self.dtboxes[:, 4:8], self._height, self._width)
            self.gtboxes[:, 4:8] = _clip_boundary(self.gtboxes[:, 4:8], self._height, self._width)
        else:
            self.dtboxes = _clip_boundary(self.dtboxes, self._height, self._width)
            self.gtboxes = _clip_boundary(self.gtboxes, self._height, self._width)

    def load_gt_boxes(self, dict_input, key_name, class_names):
        assert key_name in dict_input
        if len(dict_input[key_name]) < 1:
            return np.empty([0, 5]), np.empty([0, 5])
        head_bbox = []
        body_bbox = []
        for rb in dict_input[key_name]:
            if rb['tag'] in class_names:
                body_tag = class_names.index(rb['tag'])
                head_tag = 1
            else:
                body_tag = -1
                head_tag = -1
            if 'extra' in rb:
                if 'ignore' in rb['extra']:
                    if rb['extra']['ignore'] != 0:
                        body_tag = -1
                        head_tag = -1
            if 'head_attr' in rb:
                if 'ignore' in rb['head_attr']:
                    if rb['head_attr']['ignore'] != 0:
                        head_tag = -1
            # head_bbox.append(np.hstack((rb['hbox'], head_tag)))
            body_bbox.append((*rb['fbox'], body_tag))
        # head_bbox = np.array(head_bbox)
        # head_bbox[:, 2:4] += head_bbox[:, :2]
        body_bbox = np.array(body_bbox)
        body_bbox[:, 2:4] += body_bbox[:, :2]
        return body_bbox, head_bbox

    def load_det_boxes(self, dict_input, key_name, key_box, key_score=None, key_tag=None):
        assert key_name in dict_input
        if len(dict_input[key_name]) < 1:
            return np.empty([0, 5])
        else:
            assert key_box in dict_input[key_name][0]
            if key_score:
                assert key_score in dict_input[key_name][0]
            if key_tag:
                assert key_tag in dict_input[key_name][0]
        if key_score:
            if key_tag:
                bboxes = np.vstack(
                    [
                        np.hstack(
                            (rb[key_box], rb[key_score], rb[key_tag])
                        ) for rb in dict_input[key_name]
                    ]
                )
            else:
                bboxes = np.array([(*rb[key_box], rb[key_score]) for rb in dict_input[key_name]])
        else:
            if key_tag:
                bboxes = np.vstack(
                    [np.hstack((rb[key_box], rb[key_tag])) for rb in dict_input[key_name]]
                )
            else:
                bboxes = np.vstack([rb[key_box] for rb in dict_input[key_name]])
        bboxes[:, 2:4] += bboxes[:, :2]
        return bboxes

    def compare_voc(self, thres):
        """
        :meth: match the detection results with the groundtruth by VOC matching strategy
        :param thres: iou threshold
        :type thres: float
        :return: a list of tuples (dtbox, imageID), in the descending sort of dtbox.score
        """
        if self.dtboxes is None:
            return list()
        dtboxes = self.dtboxes
        gtboxes = self.gtboxes if self.gtboxes is not None else list()
        dtboxes.sort(key=lambda x: x.score, reverse=True)
        gtboxes.sort(key=lambda x: x.ign)

        scorelist = list()
        for i, dt in enumerate(dtboxes):
            maxpos = -1
            maxiou = thres

            for j, gt in enumerate(gtboxes):
                overlap = dt.iou(gt)
                if overlap > maxiou:
                    maxiou = overlap
                    maxpos = j

            if maxpos >= 0:
                if gtboxes[maxpos].ign == 0:
                    gtboxes[maxpos].matched = 1
                    dtboxes[i].matched = 1
                    scorelist.append((dt, self.ID))
                else:
                    dtboxes[i].matched = -1
            else:
                dtboxes[i].matched = 0
                scorelist.append((dt, self.ID))
        return scorelist

class Database(object):
    def __init__(self, gtpath=None, dtpath=None, body_key=None, head_key=None, mode=0):
        """
        mode=0: only body; mode=1: only head
        """
        self.images = dict()
        self.eval_mode = mode
        self.loadData(gtpath, body_key, head_key, if_gt=True)
        self.loadData(dtpath, body_key, head_key, if_gt=False)

        self._ignNum = sum([self.images[i]._ignNum for i in self.images])
        self._gtNum = sum([self.images[i]._gtNum for i in self.images])
        self._imageNum = len(self.images)
        self.scorelist = None

    def loadData(self, fpath, body_key=None, head_key=None, if_gt=True):
        # assert os.path.isfile(fpath), fpath + " does not exist!"
        with PetrelHelper.open(fpath) as f:
        # with open(fpath, "r") as f:
            lines = []
            for line in f:
                lines.append(line)
            records = [json.loads(line.strip('\n')) for line in lines]
        if if_gt:
            records = records[0]
            for record in records:
                self.images[record["ID"]] = Image(self.eval_mode)
                self.images[record["ID"]].load(record, body_key, head_key, PERSON_CLASSES, True)
        else:
            for record in records:
                self.images[record["ID"]].load(record, body_key, head_key, PERSON_CLASSES, False)
                self.images[record["ID"]].clip_all_boader()

    def compare(self, thres=0.5, matching=None):
        """
        match the detection results with the groundtruth in the whole database
        """
        assert matching is None or matching == "VOC", matching
        scorelist = list()
        for ID in self.images:
            if matching == "VOC":
                result = self.images[ID].compare_voc(thres)
            else:
                result = self.images[ID].compare_caltech(thres)
            scorelist.extend(result)
        # In the descending sort of dtbox score.
        scorelist.sort(key=lambda x: x[0][-1], reverse=True)
        self.scorelist = scorelist

    def eval_MR(self, ref="CALTECH_-2", fppiX=None, fppiY=None):
        """
        evaluate by Caltech-style log-average miss rate
        ref: str - "CALTECH_-2"/"CALTECH_-4"
        """
        # find greater_than
        def _find_gt(lst, target):
            for idx, item in enumerate(lst):
                if item >= target:
                    return idx
            return len(lst) - 1

        assert ref == "CALTECH_-2" or ref == "CALTECH_-4", ref
        if ref == "CALTECH_-2":
            # CALTECH_MRREF_2: anchor points (from 10^-2 to 1) as in P.Dollar's paper
            ref = [0.0100, 0.0178, 0.03160, 0.0562, 0.1000, 0.1778, 0.3162, 0.5623, 1.000]
        else:
            # CALTECH_MRREF_4: anchor points (from 10^-4 to 1) as in S.Zhang's paper
            ref = [0.0001, 0.0003, 0.00100, 0.0032, 0.0100, 0.0316, 0.1000, 0.3162, 1.000]

        if self.scorelist is None:
            self.compare()

        tp, fp = 0.0, 0.0
        if fppiX is None or fppiY is None:
            fppiX, fppiY = list(), list()
            for i, item in enumerate(self.scorelist):
                if item[1] == 1:
                    tp += 1.0
                elif item[1] == 0:
                    fp += 1.0

                fn = (self._gtNum - self._ignNum) - tp
                recall = tp / (tp + fn)
                missrate = 1.0 - recall
                fppi = fp / self._imageNum
                fppiX.append(fppi)
                fppiY.append(missrate)

        score = list()
        for pos in ref:
            argmin = _find_gt(fppiX, pos)
            if argmin >= 0:
                score.append(fppiY[argmin])
        score = np.array(score)
        MR = np.exp(np.log(score).mean())
        return MR, (fppiX, fppiY)

    def eval_AP(self):
        """
        :meth: evaluate by average precision
        """
        # calculate general ap score
        def _calculate_map(recall, precision):
            assert len(recall) == len(precision)
            area = 0
            for i in range(1, len(recall)):
                delta_h = (precision[i - 1] + precision[i]) / 2
                delta_w = recall[i] - recall[i - 1]
                area += delta_w * delta_h
            return area

        tp, fp, dp = 0.0, 0.0, 0.0
        rpX, rpY = list(), list()
        total_gt = self._gtNum - self._ignNum
        total_images = self._imageNum
        fpn = []
        dpn = []
        recalln = []
        thr = []
        fppi = []
        mr = []
        for i, item in enumerate(self.scorelist):
            if item[1] == 1:
                tp += 1.0
            elif item[1] == 0:
                fp += 1.0
                dp += item[-1]
            fn = total_gt - tp
            recall = tp / (tp + fn)
            precision = tp / (tp + fp)
            rpX.append(recall)
            rpY.append(precision)
            fpn.append(fp)
            dpn.append(dp)
            recalln.append(tp)
            thr.append(item[0][-1])
            fppi.append(fp / total_images)
            mr.append(1 - recall)

        AP = _calculate_map(rpX, rpY)
        return AP, recall, (rpX, rpY, thr, fpn, dpn, recalln, fppi, mr)

def computeJaccard(gt_path, dt_path):
    dt = load_func(dt_path)
    gt = load_func(gt_path)
    ji = 0.
    for i in range(1, 10):
        results = common_process(worker, dt, 1, gt, i * 0.1, 0.5)
        ji = max(ji, np.sum([rb['ratio'] for rb in results]) / 4370)
    return ji

def load_func(fpath):
    assert os.path.exists(fpath)
    with PetrelHelper.open(fpath) as fid:
    # with open(fpath,'r') as fid:
        lines = fid.readlines()
    records = [json.loads(line.strip('\n')) for line in lines]
    if len(records) == 1: records = records[0]
    return records

def worker(result_queue, records, gt, score_thr, bm_thr):

    total, eps = len(records), 1e-6
    for i in range(total):
        record = records[i]
        ID = record['ID']


        if len(record['dtboxes']) < 1:
            result_queue.put_nowait(None)
            continue

        GT = list(filter(lambda rb:rb['ID'] == ID, gt))
        if len(GT) < 1:
            result_queue.put_nowait(None)
            continue

        GT = GT[0]
        if 'height' in record and 'width' in record:
            height, width = record['height'], record['width']
        else:
            height, width = GT['height'], GT['width']
        flags = np.array([is_ignore(rb) for rb in GT['gtboxes']])
        rows = np.where(~flags)[0]
        ignores = np.where(flags)[0]

        gtboxes = np.vstack([GT['gtboxes'][j]['fbox'] for j in rows])
        gtboxes = recover_func(gtboxes)
        gtboxes = clip_boundary(gtboxes, height, width)

        if ignores.size:
            ignores = np.vstack([GT['gtboxes'][j]['fbox'] for j in ignores])
            ignores = recover_func(ignores)
            ignores = clip_boundary(ignores, height, width)

        dtboxes = np.vstack([np.hstack([rb['box'], rb['score']]) for rb in record['dtboxes']])
        dtboxes = recover_func(dtboxes)
        dtboxes = clip_boundary(dtboxes, height, width)
        rows = np.where(dtboxes[:,-1]> score_thr)[0]
        dtboxes = dtboxes[rows,...]

        matches = compute_JC(dtboxes, gtboxes, bm_thr)
        dt_ign, gt_ign = 0, 0

        if ignores.size:
            indices = np.array([j for (j,_) in matches])
            dt_ign = get_ignores(indices, dtboxes, ignores, bm_thr)
            indices = np.array([j for (_,j) in matches])
            gt_ign = get_ignores(indices, gtboxes, ignores, bm_thr)

        k = len(matches)
        m = gtboxes.shape[0] - gt_ign
        n = dtboxes.shape[0] - dt_ign

        ratio = k / (m + n -k + eps)
        recall = k / (m + eps)
        cover = k / (n + eps)
        noise = 1 - cover

        result_dict = dict(ID = ID, ratio = ratio, recall = recall , noise = noise ,
            cover = cover, k= k ,n = n, m = m)
        result_queue.put_nowait(result_dict)

def common_process(func, data, nr_procs, *args):

    total = len(data)
    stride = math.ceil(total / nr_procs)
    result_queue = Queue(10000)
    results, procs = [], []
    tqdm.monitor_interval = 0
    pbar = tqdm(total = total, leave = False, ascii = True)
    for i in range(nr_procs):
        start = i*stride
        end = np.min([start+stride,total])
        sample_data = data[start:end]
        # import pdb; pdb.set_trace()
        # func(result_queue, sample_data, *args)
        p = Process(target= func,args=(result_queue, sample_data, *args))
        p.start()
        procs.append(p)

    for i in range(total):

        t = result_queue.get()
        if t is None:
            pbar.update(1)
            continue
        results.append(t)
        pbar.update()
    for p in procs:
        p.join()
    return results

def recover_func(bboxes):

    assert bboxes.shape[1]>=4
    bboxes[:, 2:4] += bboxes[:,:2]
    return bboxes

def clip_boundary(dtboxes,height,width):

    assert dtboxes.shape[-1]>=4
    dtboxes[:,0] = np.minimum(np.maximum(dtboxes[:,0],0), width - 1)
    dtboxes[:,1] = np.minimum(np.maximum(dtboxes[:,1],0), height - 1)
    dtboxes[:,2] = np.maximum(np.minimum(dtboxes[:,2],width), 0)
    dtboxes[:,3] = np.maximum(np.minimum(dtboxes[:,3],height), 0)
    return dtboxes

def get_ignores(indices, boxes, ignores, ioa_thr):

    indices = list(set(np.arange(boxes.shape[0])) - set(indices))
    rboxes = boxes[indices, :]
    ioas = compute_ioa_matrix(rboxes, ignores)
    ioas = np.max(ioas, axis = 1)
    rows = np.where(ioas > ioa_thr)[0]
    return rows.size

def compute_ioa_matrix(dboxes: np.ndarray, gboxes: np.ndarray):

    assert dboxes.shape[-1] >= 4 and gboxes.shape[-1] >= 4
    N, K = dboxes.shape[0], gboxes.shape[0]
    eps = 1e-6
    dtboxes = np.tile(np.expand_dims(dboxes, axis = 1), (1, K, 1))
    gtboxes = np.tile(np.expand_dims(gboxes, axis = 0), (N, 1, 1))

    iw = np.minimum(dtboxes[:,:,2], gtboxes[:,:,2]) - np.maximum(dtboxes[:,:,0], gtboxes[:,:,0])
    ih = np.minimum(dtboxes[:,:,3], gtboxes[:,:,3]) - np.maximum(dtboxes[:,:,1], gtboxes[:,:,1])
    inter = np.maximum(0, iw) * np.maximum(0, ih)

    dtarea = np.maximum(dtboxes[:,:,2] - dtboxes[:,:,0], 0) * np.maximum(dtboxes[:,:,3] - dtboxes[:,:,1], 0)
    ioas = inter / (dtarea + eps)
    return ioas

def is_ignore(record):

    flag = False
    if 'extra' in record:
        if 'ignore' in record['extra']:
            flag = True if record['extra']['ignore'] else False
    return flag

def compute_iou_matrix(dboxes:np.ndarray, gboxes:np.ndarray):

    assert dboxes.shape[-1] >= 4 and gboxes.shape[-1] >= 4
    eps = 1e-6
    N, K = dboxes.shape[0], gboxes.shape[0]
    dtboxes = np.tile(np.expand_dims(dboxes, axis = 1), (1, K, 1))
    gtboxes = np.tile(np.expand_dims(gboxes, axis = 0), (N, 1, 1))

    iw = np.minimum(dtboxes[:,:,2], gtboxes[:,:,2]) - np.maximum(dtboxes[:,:,0], gtboxes[:,:,0])
    ih = np.minimum(dtboxes[:,:,3], gtboxes[:,:,3]) - np.maximum(dtboxes[:,:,1], gtboxes[:,:,1])
    inter = np.maximum(0, iw) * np.maximum(0, ih)

    dtarea = (dtboxes[:,:,2] - dtboxes[:,:,0]) * (dtboxes[:,:,3] - dtboxes[:,:,1])
    gtarea = (gtboxes[:,:,2] - gtboxes[:,:,0]) * (gtboxes[:,:,3] - gtboxes[:,:,1])
    ious = inter / (dtarea + gtarea - inter + eps)
    return ious

def compute_lap(dtboxes, gtboxes, thr):

    eps = 1e-7
    n, k = dtboxes.shape[0], gtboxes.shape[0]
    if k + n < 2:
        m, n = np.array([]), np.array([])
        return m, n

    overlaps = compute_iou_matrix(dtboxes, gtboxes)

    if n < 2:
        cols = np.argmax(overlaps, axis = 1)
        rows = np.array([0])
        m, n = (rows, cols) if thr - overlaps[rows, cols] < eps else (np.array([]), np.array([]))
        return m, n

    if k < 2:

        rows = np.argmax(overlaps, axis = 0)
        cols = np.array([0])
        m,n = (rows, cols) if thr - overlaps[rows, cols] < eps else (np.array([]), np.array([]))
        return m, n

    ious = overlaps * (overlaps >= thr)

    matches = minimumWeightMatching(-ious)
    m, n = np.array([i for i, _ in matches]).astype(np.int32), np.array([i for _, i in matches]).astype(np.int32)
    indice = np.where(overlaps[m, n] < thr)[0]

    if indice.size >= m.size:
        m, n = np.array([]), np.array([])
    else:
        index = np.array(list(set(np.arange(m.size)) - set(indice))).astype(np.int)
        m, n = m[index], n[index]

    return m, n

def minimumWeightMatching(costSet : np.ndarray) -> list:
    '''
    Computes a minimum-weight matching in a bipartite graph
    (A union B, E).
    costSet:
    An (m x n)-matrix of real values, where costSet[i, j]
    is the cost of matching the i:th vertex in A to the j:th
    vertex of B. A value of numpy.inf is allowed, and is
    interpreted as missing the (i, j)-edge.
    returns:
    A minimum-weight matching given as a list of pairs (i, j),
    denoting that the i:th vertex of A be paired with the j:th
    vertex of B.
    '''

    m, n = costSet.shape
    nMax = max(m, n)

    # Since the choice of infinity blocks later choices for that index,
    # it is important that the cost matrix is square, so there
    # is enough space to shift the choices for infinity to the unused
    # part of the cost-matrix.
    costSet_ = np.full((nMax, nMax), np.inf)

    mask = costSet < 0
    costSet_[:m, :n][mask] = costSet[mask]
    assert costSet_.shape[0] == costSet_.shape[1]

    # We allow a cost to be infinity. Since scipy does not
    # support this, we use a workaround. We represent infinity
    # by M = 2 * maximum cost + 1. The point is to choose a distinct
    # value, greater than any other cost, so that choosing an
    # infinity-pair is the last resort. The 2 times is for large
    # values for which x + 1 == x in floating point. The plus 1
    # is for zero, for which 2 x == x.
    try:
        practicalInfinity = 2 * costSet[costSet < np.inf].max() + 10
    except ValueError:
        # This is thrown when the indexing set is empty;
        # then all elements are infinities.
        practicalInfinity = 1

    # Replace infinitites with our representation.
    costSet_[costSet_ == np.inf] = practicalInfinity

    # Find a pairing of minimum total cost between matching second-level contours.
    iSet, jSet = linear_sum_assignment(costSet_)
    assert len(iSet) == len(jSet)

    # Return only pairs with finite cost.
    indices = [(iSet[k], jSet[k])
        for k in range(len(iSet))
        if costSet_[iSet[k], jSet[k]] != practicalInfinity]

    return indices


def compute_JC(detection:np.ndarray, gt:np.ndarray, iou_thresh:np.ndarray):

    rows, cols = compute_lap(detection, gt, iou_thresh)
    return [(i, j) for i, j in zip(rows, cols)]
