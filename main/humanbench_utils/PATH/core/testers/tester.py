import os
import time
import torch
import torch.backends.cudnn as cudnn
from core import distributed_utils as dist
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from easydict import EasyDict as edict

import numpy as np
import random
import copy

import core
import core.models.decoders as decoders
import core.models.backbones as backbones
import core.models.necks as necks
import core.data.test_datasets as datasets
from core.models.model_entry import model_entry, backbone_aio_entry
from core.utils import (AverageMeter, accuracy, load_state, load_last_iter,
                       save_state, create_logger, IterLRScheduler,
                       count_parameters_num, freeze_bn,
                       change_tensor_cuda, sync_print)
from core.distributed_utils import DistModule, DistributedSequentialSampler, simple_group_split, vreduce, vgather
from dict_recursive_update import recursive_update

class Tester(object):
    def __init__(self, C_train, C_test):
        train_config = edict(C_train.config['common'])
        ginfo = C_train.ginfo
        config = train_config

        if C_test.config.get('common') is not None:
            recursive_update(config, C_test.config.get('common'))
        config = edict(config)
        if 'out_dir' in config:
            self.out_dir = config['out_dir']+'test_results/'
        else:
            self.out_dir = "./test_results/"

        if 'expname' in config:
            self.tb_path = '{}events/{}'.format(self.out_dir, config['expname'])
            self.ckpt_path = '{}checkpoints/{}'.format(self.out_dir, config['expname'])
            self.logs_path = '{}logs/{}'.format(self.out_dir, config['expname'])
        else:
            save_path = config.get('save_path', os.path.dirname(os.path.abspath(C_train.config_file)))
            self.save_path = save_path
            self.tb_path = '{}/test_results/events'.format(save_path)
            self.ckpt_path = '{}/test_results/checkpoints'.format(save_path)
            self.logs_path = '{}/test_results/logs'.format(save_path)
        if C_train.rank == 0:
            os.makedirs(self.tb_path, exist_ok=True)
            os.makedirs(self.ckpt_path, exist_ok=True)
            os.makedirs(self.logs_path, exist_ok=True)
            self.tb_logger = SummaryWriter(self.tb_path)
        else:
            while not os.path.exists(self.logs_path):
                time.sleep(1)

        if ginfo.task_rank == 0:
            self.logger = create_logger('global_logger', '{}/log_task_{}.txt'.format(self.logs_path, ginfo.task_id))

        self.sync = config.get('sync', True)
        self.C_train = C_train
        self.C_test = C_test
        self.config = config
        self.ginfo = ginfo

        # change tensor .cuda
        change_tensor_cuda()

        self.tmp = edict()

        ## random seed setting
        rng = np.random.RandomState(self.config.get('random_seed', 0))
        self.randomseed_pool = rng.randint(999999, size=config.max_iter)

    def create_dataset(self):
        ginfo = self.ginfo
        config = self.config
        dataset_args = config.dataset['kwargs']
        dataset_args['ginfo'] = ginfo
        self.dataset = datasets.dataset_entry(config.dataset)
        dist.barrier()

    def create_dataloader(self):
        raise NotImplementedError

    def create_model(self):
        config = self.config

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
        decoder_module = decoders.decoder_entry(config.decoder)

        # build
        model = model_entry(backbone_module, neck_module, decoder_module)

        ## distributed
        model.cuda()

        if self.C_train.rank == 0:
            print(model)

        model = DistModule(model, sync=self.sync, task_grp=self.ginfo.group, \
            share_backbone_group=self.ginfo.backbone_share_group, \
            share_neck_group=self.ginfo.neck_share_group, \
            share_decoder_group=self.ginfo.decoder_share_group)

        self.model = model

    def load(self, args):
        if args.load_path == '':
            return
        if args.recover:
            self.last_iter = load_state(args.load_path.replace('ckpt_task_', 'ckpt_task{}_'.format(\
                self.ginfo.task_id)), self.model, recover=args.recover)
            self.last_iter -= 1
        else:
            if args.load_single:
                load_state(args.load_path, self.model, ignore=args.ignore)
            else:
                load_state(args.load_path.replace('ckpt_task_', 'ckpt_task{}_'.format(\
                    self.ginfo.task_id)), self.model, ignore=args.ignore)

    def initialize(self, args):

        # create dataset to get num_classes
        self.create_dataset()
        self.create_model()

        self.load_args = args

        self.load(args)
        self.create_dataloader()

    def pre_run(self):
        tmp = self.tmp
        tmp.vbatch_time = AverageMeter(10)
        tmp.vdata_time = AverageMeter(10)
        tmp.vtop1 = AverageMeter(10)
        tmp.top1_list = [torch.Tensor(1).cuda() for _ in range(self.C_train.world_size)]

        self.model.eval()

    def prepare_data(self):
        tmp = self.tmp
        tmp.input_var = dict()

        for k,v in tmp.input.items():
            if not isinstance(v, list):
                tmp.input_var[k] = torch.autograd.Variable(v.cuda())

    def _set_randomseed(self, seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
