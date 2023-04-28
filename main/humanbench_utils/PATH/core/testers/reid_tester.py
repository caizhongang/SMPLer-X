import copy
import os
import random
import time
from torch.nn import functional as F

import core
import core.models.decoders as decoders
import core.models.backbones as backbones
import core.models.necks as necks
import core.data.test_datasets as datasets
from core.models.model_entry import model_entry, aio_entry, backbone_aio_entry
import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from core import distributed_utils as dist

import torchvision.utils as vutils
from core.distributed_utils import (DistModule,
                                    DistributedSequentialSampler,
                                    simple_group_split,
                                    vgather, vreduce)
from core.utils import (AverageMeter, AverageMinMaxMeter, IterLRScheduler, accuracy,
                        count_parameters_num, create_logger, load_state,
                        save_state, change_tensor_half, sync_print)
import core.fp16 as fp16

from easydict import EasyDict as edict
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from collections import OrderedDict, defaultdict

from .tester_deter import TesterDeter
from helper.vis_helper import inv_normalize_batch, vis_one_from_batch
from sklearn.metrics import average_precision_score

from .utils.metrics import R1_mAP_eval

class ReIDTester(TesterDeter):

    def __init__(self, C_train, C_test):
        super().__init__(C_train, C_test)

    def forward(self):
        self._set_randomseed(self.randomseed_pool[self.tmp.current_step])

        tmp = self.tmp
        ginfo = self.ginfo
        tmp.drop_this_iter = False
        output = self.model(tmp.input_var, tmp.current_step)
        tmp.features = output[self.C_test.config['common']['tester']['test_feature_name']]
        tmp.labels = output['label']
        tmp.camera_ids = tmp.input_var['camera']

    def gather_result(self):
        tmp = self.tmp
        ginfo = self.ginfo
        tmp.features_list.append(tmp.features.cpu())
        tmp.labels_list.append(tmp.labels.cpu())
        tmp.camera_ids_list.append(tmp.camera_ids.cpu())

    def pre_run(self):
        tmp = self.tmp
        tmp.vbatch_time = AverageMeter(10)
        tmp.vdata_time = AverageMeter(10)
        tmp.features_list = list()
        tmp.labels_list = list()
        tmp.camera_ids_list = list()

        self.model.eval()

    def create_dataset(self):
        ginfo = self.ginfo
        config = self.config
        dataset_args = config.dataset['kwargs']
        dataset_args['image_list_paths'] = dataset_args['query_file_path']
        dataset_args['ginfo'] = ginfo
        self.query_dataset = datasets.dataset_entry(config.dataset)
        dataset_args['image_list_paths'] = dataset_args['gallery_file_path']
        self.gallery_dataset = datasets.dataset_entry(config.dataset)
        dist.barrier()

    def create_dataloader(self):
        config = self.config
        ginfo = self.ginfo

        self.query_sampler = DistributedSequentialSampler(self.query_dataset)
        self.query_loader = DataLoader(self.query_dataset, batch_size=config.sampler.batch_size,
                            shuffle=False, num_workers=config.workers,
                            pin_memory=False, sampler=self.query_sampler)

        # self.query_loader = DataLoader(self.query_dataset, batch_size=config.sampler.batch_size,
        #                     shuffle=False, num_workers=config.workers,
        #                     pin_memory=False)

        self.gallery_sampler = DistributedSequentialSampler(self.gallery_dataset)
        self.gallery_loader = DataLoader(self.gallery_dataset, batch_size=config.sampler.batch_size,
                            shuffle=False, num_workers=config.workers,
                            pin_memory=False, sampler=self.gallery_sampler)
        # self.gallery_loader = DataLoader(self.gallery_dataset, batch_size=config.sampler.batch_size,
        #                     shuffle=False, num_workers=config.workers,
        #                     pin_memory=False)

    def create_model(self):
        config = self.config
        ginfo = self.ginfo

        backbone_bn_group_size = config.backbone['kwargs'].get('bn_group_size', 1)
        assert backbone_bn_group_size == 1, 'other bn group size not support!'
        backbone_bn_group_comm = self.ginfo.backbone_share_group

        ## build backbone
        config.backbone['kwargs']['bn_group'] = backbone_bn_group_comm
        backbone_module = backbones.backbone_entry(config.backbone)
        count_parameters_num(backbone_module)

        ## build necks
        neck_bn_group_size = config.backbone['kwargs'].get('bn_group_size', 1)
        assert neck_bn_group_size == 1, 'other bn group size not support!'
        neck_bn_group_comm = self.ginfo.neck_share_group

        neck_args = config.neck['kwargs']
        neck_args['backbone'] = backbone_module
        neck_args['bn_group'] = neck_bn_group_comm
        neck_module = necks.neck_entry(config.neck)

        ## add decoder
        decoder_bn_group_size = config.backbone['kwargs'].get('bn_group_size', 1)
        assert decoder_bn_group_size == 1, 'other bn group size not support!'
        decoder_bn_group_comm = self.ginfo.decoder_share_group

        decoder_args = config.decoder['kwargs']
        decoder_args['backbone'] = backbone_module
        decoder_args['neck'] = neck_module
        decoder_args['bn_group'] = decoder_bn_group_comm
        decoder_args['feature_only'] = True
        self.config.decoder.kwargs.ginfo = self.ginfo
        self.config.decoder.kwargs.ignore_value = None
        self.config.decoder.kwargs.num_classes = 666
        decoder_module = decoders.decoder_entry(config.decoder)

        # build
        model = globals()[self.config.get('model_entry_type', 'model_entry')](backbone_module,
                                                                              neck_module,
                                                                              decoder_module)

        ## distributed
        model.cuda()

        if self.C_train.rank == 0:
            print(model)

        model = DistModule(model, sync=self.sync, task_grp=self.ginfo.group, \
            share_backbone_group=self.ginfo.backbone_share_group, \
            share_neck_group=self.ginfo.neck_share_group, \
            share_decoder_group=self.ginfo.decoder_share_group)

        self.model = model

    def save(self):
        pass

    def post_run(self):
        pass

    def extract(self, data_loader):
        config = self.config
        ginfo = self.ginfo
        tmp = self.tmp

        self.pre_run()
        end = time.time()
        self.model.eval()

        for i, tmp.input in enumerate(data_loader):
            tmp.vdata_time.update(time.time() - end)
            self.prepare_data()

            tmp.current_step = i + 1
            with torch.no_grad():
                self.forward()
            self.gather_result()

            tmp.vbatch_time.update(time.time() - end)
            end = time.time()

            if tmp.current_step % config.print_freq == 0 and ginfo.task_rank == 0:
                print(
                'Extract Features: [{0}/{1}]\t'
                'task{task_id:<2}: {task_name}\t'
                'Time {batch_time.avg:.3f} (ETA:{eta:.2f}h) ({data_time.avg:.3f})\t'
                .format(i + 1, len(data_loader),
                task_id=ginfo.task_id, task_name=ginfo.task_name,
                batch_time=tmp.vbatch_time,
                eta=(config.max_iter-tmp.current_step)*tmp.vbatch_time.avg/3600,
                data_time=tmp.vdata_time))

        all_features = torch.cat(tmp.features_list)
        all_labels = torch.cat(tmp.labels_list)
        all_camera_ids = torch.cat(tmp.camera_ids_list)
        # all_features = F.normalize(all_features)
        return all_features, all_labels, all_camera_ids

    def run(self):
        config = self.config
        ginfo = self.ginfo
        tmp = self.tmp

        query_all_features, query_all_labels, query_all_camera_ids = self.extract(self.query_loader)
        gallery_all_features, gallery_all_labels, gallery_all_camera_ids = self.extract(self.gallery_loader)

        dist.barrier()
        num_query = len(query_all_features)

        evaluator = R1_mAP_eval(num_query, max_rank=50, feat_norm=True)
        evaluator.reset()

        evaluator.update((query_all_features, query_all_labels, query_all_camera_ids))
        evaluator.update((gallery_all_features, gallery_all_labels, gallery_all_camera_ids))

        cmc, mAP, _, _, _, _, _ = evaluator.compute()
        print("Validation Results ")
        print("mAP: {:.1%}".format(mAP))
        for r in [1, 5, 10]:
            print("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
        dist.barrier()
