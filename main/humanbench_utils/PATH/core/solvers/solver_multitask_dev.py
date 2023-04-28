import copy
import os
import re
import collections
import time
import random
import datetime
import traceback
import numpy as np
import functools

import core.models.decoders as decoders
import core.models.backbones as backbones
import core.models.necks as necks
import core.data.datasets as datasets
import core.optimizers as optimizers
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn

from torch._six import string_classes
from torch.utils.data import DataLoader
from core.data.datasets.images.seg_dataset_dev import Instances
from core.data.transforms.pose_transforms import DataContainer
from core.models.model_entry import model_entry, aio_entry, backbone_aio_entry
from core.models.tta import SemanticSegmentorWithTTA
from core.distributed_utils import (DistModule, vgather, vreduce, DistributedGivenIterationSampler,
                                    DistributedSequentialSampler, RandomIdentityBatchSampler,
                                    reduce_dict)
from core.utils import (AverageMeter, count_parameters_num, printlog, change_tensor_cuda,
                        create_logger, load_state_model, load_state_optimizer, save_state, get_num_layer_for_vit)
from core.solvers.utils.seg_tester_dev import SemSegEvaluator
from core.solvers.utils.pos_tester_dev import PoseEvaluator, MPIIPoseEvaluator
from core.solvers.utils.par_tester_dev import HumParEvaluator
from core.solvers.utils.attr_tester_dev import PedAttrEvaluator
from core.solvers.utils.peddet_tester_dev import PedDetEvaluator
from helper.vis_helper import inv_normalize_batch, vis_one_from_batch

from easydict import EasyDict as edict
from tensorboardX import SummaryWriter

from dict_recursive_update import recursive_update
from collections import OrderedDict
from collections.abc import Mapping
from contextlib import ExitStack, contextmanager
from .solver_deter import SolverDeter, WorkerInit
from core.utils import nested_tensor_from_tensor_list

from core import distributed_utils as dist


DEBUG_MODE = False


class SolverMultiTaskDev(SolverDeter):

    def __init__(self, C):
        super().__init__(C)
        if 'SLURM_NODELIST' in os.environ:
            printlog(f"hostnames: {os.environ['SLURM_NODELIST']}")
            printlog(f"NODEID: {os.environ['SLURM_NODEID']} - {os.environ['SLURMD_NODENAME']}")

    def initialize(self, args):
        self.create_dataset()

        self.create_model()
        self.create_optimizer()

        self.load_args = args
        self.load(args)

        self.create_dataloader()
        self.create_lr_scheduler()

    def create_model(self):
        ## build backbone
        self.config.backbone.kwargs.bn_group = self.ginfo.backbone_share_group
        backbone_module = backbones.backbone_entry(self.config.backbone)
        count_parameters_num(backbone_module)

        ## build neck
        self.config.neck.kwargs.backbone = backbone_module
        self.config.neck.kwargs.bn_group = self.ginfo.neck_share_group
        neck_module = necks.neck_entry(self.config.neck)

        ## build decoder(s)
        self.config.decoder.kwargs.backbone = backbone_module
        self.config.decoder.kwargs.neck = neck_module
        self.config.decoder.kwargs.bn_group = self.ginfo.decoder_share_group
        self.config.decoder.kwargs.ginfo = self.ginfo

        if self.config.dataset.type == "COCOStuffSegDatasetDev":
            self.config.decoder.kwargs.ignore_value = self.config.dataset.kwargs.cfg.ignore_value
            self.config.decoder.kwargs.num_classes = self.config.dataset.kwargs.cfg.num_classes
        elif self.config.dataset.type in ["COCOPosDatasetDev", "MultiPoseDatasetDev"]:
            self.config.decoder.kwargs.num_classes = self.dataset.num_classes
            self.config.decoder.kwargs.ignore_value = None
        elif "ParsingDataset" in self.config.dataset.type:
            self.config.decoder.kwargs.ignore_value = self.config.dataset.kwargs.cfg.ignore_value
            self.config.decoder.kwargs.num_classes = self.config.dataset.kwargs.cfg.num_classes
        elif self.config.dataset.type == "ReIDDataset":
            # self.config.decoder.kwargs.feature_only = True
            self.config.decoder.kwargs.ignore_value = None
        elif self.config.dataset.type == 'AttrDataset':
            self.config.decoder.kwargs.ignore_value = None
            self.config.decoder.kwargs.num_classes = 0
        elif self.config.dataset.type == "PedestrainDetectionDataset":
            self.config.decoder.kwargs.ignore_value = None
            self.config.decoder.kwargs.num_classes = 1 # treat pedestrain classificatin as a binary classification

        decoder_module = decoders.decoder_entry(self.config.decoder)

        ## build model
        model = globals()[self.config.get('model_entry_type', 'model_entry')](backbone_module,
                                                                              neck_module,
                                                                              decoder_module)

        # fix backbone finetune
        if self.config.get('fix_backbone', False):
            for name, params in model.named_parameters():
                if 'backbone' in name:
                    params.requires_grad = False

        # partial finetune
        if len(self.config.get('fix_backbone_partial_layers', [])):
            # import pdb; pdb.set_trace();
            backbone_partial_layers = self.config.get('fix_backbone_partial_layers', [])
            for name, params in model.named_parameters():
                if name in backbone_partial_layers:
                    params.requires_grad = False
                    print('fix param: {}'.format(name))
                else:
                    curr_block_num = name.split('.')[2]
                    if curr_block_num in backbone_partial_layers:
                        print('fix param: {}'.format(name))
                        params.requires_grad = False


        ## distributed
        model.cuda()

        if self.C.rank == 0:
            print(model)

        model = DistModule(model, sync=self.sync, task_grp=self.ginfo.group,
                           share_backbone_group=self.ginfo.backbone_share_group,
                           share_neck_group=self.ginfo.neck_share_group,
                           share_decoder_group=self.ginfo.decoder_share_group)

        self.model = model

    def create_optimizer(self):
        ## param_group
        defaults = {}
        defaults["lr"] = self.config.base_lr
        defaults["weight_decay"] = self.config.optimizer.kwargs.weight_decay

        norm_module_types = (
            torch.nn.BatchNorm1d,
            torch.nn.BatchNorm2d,
            torch.nn.BatchNorm3d,
            torch.nn.SyncBatchNorm,
            # NaiveSyncBatchNorm inherits from BatchNorm2d
            torch.nn.GroupNorm,
            torch.nn.InstanceNorm1d,
            torch.nn.InstanceNorm2d,
            torch.nn.InstanceNorm3d,
            torch.nn.LayerNorm,
            torch.nn.LocalResponseNorm,
        )

        memo = set()
        param_groups = []
        for module_name, module in self.model.named_modules():
            for module_param_name, value in module.named_parameters(recurse=False):
                if not value.requires_grad:
                    continue
                # Avoid duplicating parameters
                if value in memo:
                    continue
                memo.add(value)

                hyperparams = copy.copy(defaults)
                if "backbone_module" in module_name:
                    if self.config.get('layer_decay', False):
                        layer_id = get_num_layer_for_vit(module_name, self.config.layer_decay)
                        scale = self.config.layer_decay.layer_decay_rate ** (self.config.layer_decay.num_layers - layer_id - 1)
                        hyperparams["lr"] = hyperparams["lr"] * scale * self.config.get('backbone_multiplier', 1.0)
                    else:
                        hyperparams["lr"] = hyperparams["lr"] * self.config.get('backbone_multiplier', 1.0)

                    if module_name in ("module.backbone_module.pos_embed"):
                        hyperparams["lr"] = hyperparams["lr"] * self.config.get('pos_embed_multiplier', 1.0)
                    if self.config.get('vdp_wd_rule', False) and (len(value.shape) == 1 or module_param_name.endswith(".bias")):
                        hyperparams["weight_decay"] = 0.0
                if "neck_module" in module_name:
                    hyperparams["lr"] = hyperparams["lr"] * self.config.get('neck_multiplier', 1.0)
                if "bias" in module_param_name:
                    hyperparams["lr"] = hyperparams["lr"] * self.config.get('bias_multiplier', 1.0)
                if (
                    "relative_position_bias_table" in module_param_name
                    or "absolute_pos_embed" in module_param_name
                    or "pos_embed" in module_param_name
                    or "cls_token" in module_param_name
                    or 'rel_pos_' in module_param_name
                    or 'bias' in module_param_name
                    or isinstance(module, norm_module_types)
                    or isinstance(module, torch.nn.Embedding)
                ):
                    hyperparams["weight_decay"] = 0.0

                if self.config.get('decoder_wd', None) != None and 'decoder_module' in module_name:
                    hyperparams["weight_decay"] = self.config.get('decoder_wd')
                    # import pdb; pdb.set_trace()
                    if self.ginfo.task_rank == 0:
                        self.logger.info("set {} weight_decay={}".format(module_name, self.config.get('decoder_wd')))

                if value.task_specific and self.config.get('task_specific_lr_scale', False):
                    hyperparams["lr"] = hyperparams["lr"] / self.ginfo.task_weight

                param_groups.append({"params": [value], **hyperparams})

                if self.ginfo.task_rank == 0:
                    self.logger.info(f"task_id: {self.ginfo.task_id} \t"
                                     f"module_name: {module_name} \t\t "
                                     f"module_param_name: {module_param_name} \t\t "
                                     f"specification: {hyperparams}")

        self.config.optimizer.kwargs.params = param_groups
        self.config.optimizer.kwargs.lr = self.config.base_lr
        self.optimizer = optimizers.optim_entry(self.config.optimizer)

    def create_dataset(self):
        self.config.dataset.kwargs.ginfo = self.ginfo
        self.dataset = datasets.dataset_entry(self.config.dataset)
        printlog(self.dataset.__repr__())
        dist.barrier()

    def create_dataloader(self):
        if self.config.sampler.get('type', False) == 'RandomIdentity':
            self.sampler = RandomIdentityBatchSampler(
                self.dataset, self.config.max_iter, self.config.sampler.batch_size,
                world_size=self.ginfo.task_size, rank=self.ginfo.task_rank,
                last_iter=self.last_iter,
                shuffle_strategy=self.config.sampler.shuffle_strategy,
                random_seed=self.ginfo.task_random_seed,
                ret_save_path=self.config.sampler.get('ret_save_path', None))
        else:
            self.sampler = DistributedGivenIterationSampler(
                            self.dataset, self.config.max_iter, self.config.sampler.batch_size,
                            world_size=self.ginfo.task_size, rank=self.ginfo.task_rank,
                            last_iter=self.last_iter,
                            shuffle_strategy=self.config.sampler.shuffle_strategy,
                            random_seed=self.ginfo.task_random_seed,
                            ret_save_path=self.config.sampler.get('ret_save_path', None))

        if self.config.get('collate', 'dev') == 'det':
            collate = det_collate
        else:
            collate = dev_collate

        self.loader = DataLoader(self.dataset, batch_size=self.config.sampler.batch_size,
                            shuffle=False, num_workers=self.config.workers, collate_fn=functools.partial(dev_collate, nested=self.config.dataset.kwargs.get('nested', False)),
                            pin_memory=False, sampler=self.sampler, worker_init_fn=self.worker_init_fn)

    def load(self, args):
        if args.load_path == '':
            return
        load_path = args.load_path if args.load_single else args.load_path.replace('ckpt_task_', f'ckpt_task{self.ginfo.task_id}_')

        try:
            checkpoint = torch.load(load_path, 'cpu')
        except:
            raise FileNotFoundError(f'=> no checkpoint found at {load_path}')

        if self.ginfo.task_rank == 0:
            printlog(f"Recovering from {load_path}, keys={list(checkpoint.keys())}")

        if 'state_dict' in checkpoint:
            pretrained_state_dict = checkpoint['state_dict']
        else:
            pretrained_state_dict = checkpoint

        ignores = args.ignore + self.config.get('load_ignore', [])
        if len(ignores) > 0:
            for k in list(pretrained_state_dict.keys()):
                flag = False
                for prefix in ignores:
                    if k.startswith(prefix):
                        flag = True
                        the_prefix = prefix
                        break
                if flag:
                    print('ignoring {} (prefix: {})'.format(k, the_prefix))
                    del pretrained_state_dict[k]
        load_state_model(self.model, pretrained_state_dict, self.ginfo)
        if args.finetune and not args.recover:
            return
        if 'optimizer' in checkpoint:
            load_state_optimizer(self.optimizer, checkpoint['optimizer'], self.ginfo)
            self.last_iter = checkpoint['step'] - 1
        elif args.recover:
            self.last_iter = checkpoint['step'] - 1

    def pre_run(self):
        tmp = self.tmp
        tmp.vtask_time = AverageMeter(10)
        tmp.vbatch_time = AverageMeter(10)
        tmp.vdata_time = AverageMeter(10)
        tmp.vloss = AverageMeter(10)
        tmp.vtop1 = AverageMeter(10)

        printlog(f">>> sanity check: attempting torch.Tensor(1).cuda(), check task_sp_list if stuck")
        torch.Tensor(1).cuda()
        printlog(f">>> sanity check: torch.Tensor(1).cuda() passed")

        tmp.loss_list = [torch.Tensor(1).cuda() for _ in range(self.C.world_size)]
        tmp.top1_list = [torch.Tensor(1).cuda() for _ in range(self.C.world_size)]

        tmp.vbackbone_grad_norm = AverageMeter(10)
        tmp.backbone_grad_norm_list = [torch.Tensor(1).cuda() for _ in range(self.C.world_size)]
        tmp.vneck_grad_norm = AverageMeter(10)
        tmp.neck_grad_norm_list = [torch.Tensor(1).cuda() for _ in range(self.C.world_size)]
        tmp.vdecoder_grad_norm = AverageMeter(10)
        tmp.decoder_grad_norm_list = [torch.Tensor(1).cuda() for _ in range(self.C.world_size)]

        tmp.vbackbone_grad_thresh = AverageMeter(10)
        tmp.backbone_grad_thresh_list = [torch.Tensor(1).cuda() for _ in range(self.C.world_size)]
        tmp.vneck_grad_thresh = AverageMeter(10)
        tmp.neck_grad_thresh_list = [torch.Tensor(1).cuda() for _ in range(self.C.world_size)]
        tmp.vdecoder_grad_thresh = AverageMeter(10)
        tmp.decoder_grad_thresh_list = [torch.Tensor(1).cuda() for _ in range(self.C.world_size)]

        self.model.train()

    def gather_result(self):
        tmp = self.tmp
        ginfo = self.ginfo

        vreduce(tmp.vloss, tmp.raw_loss.data, group=ginfo.group)
        vreduce(tmp.vtop1, tmp.top1, group=ginfo.group)

        vgather(tmp.loss_list, tmp.vloss.avg)
        vgather(tmp.top1_list, tmp.vtop1.avg)

        if self.config.get('verbose_loss', True):
            tmp.vlosses = reduce_dict(tmp.raw_losses, task_size=self.ginfo.task_size,
                                    task_rank=self.ginfo.task_rank, group=self.ginfo.group)
        else:
            tmp.vlosses = {}

    def tb_logging(self):
        tmp = self.tmp
        ginfo = self.ginfo

        for tid,ii in enumerate(ginfo.task_root_ranks):
            self.tb_logger.add_scalar('loss_{}'.format(ginfo.task_names[tid]), tmp.loss_list[ii], tmp.current_step)
            self.tb_logger.add_scalar('top1_{}'.format(ginfo.task_names[tid]), tmp.top1_list[ii], tmp.current_step)
            for k, v in tmp.vlosses.items():
                self.tb_logger.add_scalar('{}_{}'.format(k, ginfo.task_names[tid]), v, tmp.current_step)

        self.tb_logger.add_scalar('lr', tmp.current_lr, tmp.current_step)

    def logging(self):
        tmp = self.tmp
        config = self.config
        ginfo = self.ginfo

        vlosses = tmp.vlosses

        log_msg = '\t'.join([
            'Iter: [{0}/{1}] ',
            'task{task_id:<2}: {task_name}',
            'TaskFBTime: {task_time.avg:.3f}',
            'Time: {batch_time.avg:.3f} (ETA:{eta:.2f}h) ({data_time.avg:.3f}) ',
            'Loss: {loss.avg:.4f} ',
            'Prec@1: {top1.avg:.3f} ',
            'LR: {current_lr} ',
            '{meters} ',
            'max mem: {memory:.0f}'
        ])

        MB = 1024.0 * 1024.0

        loss_str = []
        for name, meter in vlosses.items():
            loss_str.append(
                "{}: {} ".format(name, str(meter.item()))
            )

        loss_str = '\t'.join(loss_str)
        log_msg = log_msg.format(tmp.current_step, config.max_iter, \
                        task_id=ginfo.task_id, task_name=ginfo.task_name, \
                        task_time=tmp.vtask_time, \
                        batch_time=tmp.vbatch_time, \
                        eta=(config.max_iter-tmp.current_step)*tmp.vbatch_time.avg/3600, \
                        data_time=tmp.vdata_time, \
                        loss=tmp.vloss, \
                        top1=tmp.vtop1, \
                        current_lr=tmp.current_lr, \
                        meters=loss_str, \
                        memory=torch.cuda.max_memory_allocated() / MB)

        self.logger.info(log_msg)

    def save(self):
        if ((self.tmp.current_step + 1) % self.config.get('ckpt_interval', 1000) == 0 or
            self.tmp.current_step + 1 == self.config.max_iter
        ) and self.ginfo.task_rank == 0:
            save_state({
                'step': self.tmp.current_step+1,
                'state_dict': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
            }, '{}/ckpt_task{}'.format(self.ckpt_path, self.ginfo.task_id), 'newest')
        if self.config.get('save_interval', -1) > 0 and (self.tmp.current_step+1) % self.config.save_interval == 0 and self.ginfo.task_rank == 0:
            save_state({
                'step': self.tmp.current_step+1,
                'state_dict': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
            }, '{}/ckpt_task{}'.format(self.ckpt_path, self.ginfo.task_id), self.tmp.current_step+1)

        if (self.tmp.current_step+1) in self.config.get('save_iters', []) and self.ginfo.task_rank == 0:
            save_state({
                'step': self.tmp.current_step+1,
                'state_dict': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
            }, '{}/ckpt_task{}'.format(self.ckpt_path, self.ginfo.task_id), self.tmp.current_step+1)

        if (self.tmp.current_step + 1) % self.config.get('ckpt_interval', 1000) == 0 and self.config.get('cuda_empty_cache', False):
            torch.cuda.empty_cache()

    def prepare_data(self):
        self.tmp.input_var = dict()

        for k, v in self.tmp.input.items():
            if not isinstance(v, list) and not isinstance(v, str) and not isinstance(v, DataContainer):
                self.tmp.input_var[k] = v.cuda()
            elif k == "instances":
                self.tmp.input_var[k] = [_v.cuda() for _v in v]
            else:
                self.tmp.input_var[k] = v

    def forward(self):
        ## set random seed with current_step at each iteration
        try:
            self._set_randomseed(self.randomseed_pool[self.tmp.current_step])
        except:  # workaround for reid task resumed sampler/loader bug damaging newest_checkpoints at the end of training
            time.sleep(60)
            raise ValueError(f"max_iter: {self.config.max_iter} current_step(-1): {self.tmp.current_step} "
                             f"rank: {self.C.rank}, task_id: "
                             f"{self.ginfo.task_id} (<--- I guess its reid task) task_rank: {self.ginfo.task_rank}"
                             f"This error is a reminder that we caught a data_loader length bug (should be from reid "
                             f"task), but the program should end normally with final checkpoint intact")

        tmp = self.tmp
        ginfo = self.ginfo

        oom = False
        try:
            output = self.model(tmp.input_var, tmp.current_step)
        except RuntimeError as mem_error:
            printlog(f"*****\n"
                     f"***** encountered potential mem_error, current node: "
                     f"{os.environ['SLURM_NODEID']} - {os.environ['SLURMD_NODENAME']}"
                     f"task_id: {self.ginfo.task_id}"
                     f"\n*****")
            printlog(f"error_message:\n{mem_error}")
            printlog(traceback.format_exc())
            oom = True
        if oom:
            # python exception object holds a reference to the stack frame where the error was raised, which
            # prevents the original tensor objects from being freed torch.cuda.empty_cache()
            torch.cuda.empty_cache()
            try:
                output = self.model(tmp.input_var, tmp.current_step)
            except RuntimeError as mem_error:
                printlog(f"*****\n"
                         f"***** encountered potential mem_error, **restart attempt failed** current node: "
                         f"{os.environ['SLURM_NODEID']} - {os.environ['SLURMD_NODENAME']}"
                         f"\n*****")
                raise mem_error

        tmp.output = output
        tmp.raw_losses = output['loss']  # TODO: log all losses separately
        if isinstance(tmp.raw_losses, dict):  # only key with loss are used for loss computation
            tmp.raw_loss = sum(tmp.raw_losses[k] for k in tmp.raw_losses.keys() if 'loss' in k) / ginfo.task_size
        else:
            tmp.raw_loss = tmp.raw_losses / ginfo.task_size
            tmp.raw_losses = {"total_loss": tmp.raw_losses}

        if 'top1' in output:
            tmp.raw_top1 = output['top1'] / ginfo.task_size
        else:
            tmp.raw_top1 = torch.zeros(1).cuda()
        tmp.loss = tmp.raw_loss * ginfo.task_weight
        tmp.top1 = tmp.raw_top1

    def backward(self):
        self.optimizer.zero_grad()
        self.tmp.loss.backward()

    def backward_expand_bs(self):
        self.tmp.loss.backward()

    def run_dummy(self):
        raise

    def run(self):

        if DEBUG_MODE:
            self.run_dummy()
            return

        config = self.config
        ginfo = self.ginfo
        tmp = self.tmp

        self.pre_run()

        end = time.time()

        load_flag = False

        for i, tmp.input in enumerate(self.loader):
            tmp.vdata_time.update(time.time() - end)

            self.prepare_data()

            # TODO currently a work around for gpu memory leak when recovering
            if load_flag:
                tmp.current_step = 0
                self.forward()
                self.backward()
                dist.barrier()
                #self.update()
                self.load(self.load_args)
                load_flag = False

            tmp.current_step = self.last_iter + i + 1
            self.lr_scheduler.step(tmp.current_step)
            tmp.current_lr = self.lr_scheduler.get_lr()[0]

            # import pdb; pdb.set_trace();
            self.forward()

            if self.config.get('expand_bs', False):
                self.backward_expand_bs()
                if (tmp.current_step + 1) % 2 == 0:
                    if self.autodenan:
                        is_nan = self.auto_denan()
                        if is_nan:
                            continue

                    self.model.reduce_gradients()
                    self.optimizer.step()
                    self.optimizer.zero_grad()
            else:
                self.backward()
                if self.autodenan:
                    is_nan = self.auto_denan()
                    if is_nan:
                        continue

                self.model.reduce_gradients()
                self.optimizer.step()

            self.gather_result()
            tmp.vbatch_time.update(time.time() - end)
            end = time.time()

            if tmp.current_step % config.print_freq == 0 and ginfo.task_rank == 0:
                if ginfo.task_id == 0:
                    self.tb_logging()
                self.logging()

            self.save()
            self.post_run()


class TesterMultiTaskDev(SolverMultiTaskDev):
    def __init__(self, C_train, C_test):
        torch.cuda.empty_cache()

        train_config = edict(C_train.config['common'])
        ginfo = C_train.ginfo
        config = train_config

        if C_test.config.get('common') is not None:
            recursive_update(config, C_test.config.get('common'))
        config = edict(config)
        if 'out_dir' in config:
            self.out_dir = config['out_dir'] + 'test_results/'
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
            # self.tb_logger = SummaryWriter(self.tb_path)
            project_name = config.get('project_name', os.path.dirname(C_train.config_file).split('/')[-1])
            self.tb_logger = SummaryWriter(log_dir=self.tb_path,name=C_train.config['expname'],project=project_name, overwrite_last_training=True)

        else:
            while not os.path.exists(self.logs_path):
                time.sleep(1)

        if ginfo.task_rank == 0:
            assert C_train.rank == 0, "there shall be only one group"
            self.logger = create_logger('global_logger', '{}/log_task_{}.txt'.format(self.logs_path, ginfo.task_id))

        self.sync = config.get('sync', True)
        self.C = C_train

        self.config = config
        self.ginfo = ginfo
        # change tensor .cuda
        change_tensor_cuda()

        self.tmp = edict()

        ## random seed setting
        rng = np.random.RandomState(self.config.get('random_seed', 0))
        self.randomseed_pool = rng.randint(999999, size=config.max_iter)

        ### VVV deterministic measures VVV

        if self.config.get('deterministic', False):
            if self.config.get('cudnn_deterministic', True):
                cudnn.deterministic = True
                cudnn.benchmark = False
            else:
                cudnn.benchmark = True
            seed = self.config.get('random_seed', 0)
            worker_rank = self.config.get('worker_rank', False)
            if worker_rank:
                worker_init = WorkerInit(self.C.rank, self.config.workers)
            else:
                worker_init = WorkerInit(0, 0)
            self.worker_init_fn = worker_init.func
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            dist.barrier()
            if self.C.rank == 0:
                self.logger.info(f'deterministic mode, seed: {seed}, worker_rank: {worker_rank},\
                                   cudnn_deterministic: {self.config.get("cudnn_deterministic", True)}')
            dist.barrier()
        else:
            self.worker_init_fn = None

    def initialize(self, args):
        self.create_dataset()
        self.create_model()

        self.load_args = args
        self.load(args)

        self.create_dataloader()

    def create_dataloader(self):
        self.test_sampler = DistributedSequentialSampler(self.dataset)

        collate = naive_collate if self.config.get('collate', 'naive') == 'naive' else functools.partial(dev_collate, nested=self.config.dataset.kwargs.get('nested', False))
        self.test_loader = DataLoader(self.dataset, batch_size=self.config.sampler.batch_size,
                                      shuffle=False, drop_last=False, num_workers=self.config.workers,
                                      pin_memory=False, sampler=self.test_sampler, collate_fn=collate)

    def load(self, args):
        if args.load_path == '':
            return
        load_path = args.load_path if args.load_single else args.load_path.replace('ckpt_task_', f'ckpt_task{self.ginfo.task_id}_')

        try:
            checkpoint = torch.load(load_path, 'cpu')
        except:
            raise FileNotFoundError(f'=> no checkpoint found at {load_path}')

        if self.ginfo.task_rank == 0:
            printlog(f"Recovering from {load_path}, keys={list(checkpoint.keys())}")

        if 'state_dict' in checkpoint:
            pretrained_state_dict = checkpoint['state_dict']
        else:
            pretrained_state_dict = checkpoint

        ignores = args.ignore + self.config.get('load_ignore', [])
        if len(ignores) > 0:
            for k in list(pretrained_state_dict.keys()):
                flag = False
                for prefix in ignores:
                    if k.startswith(prefix):
                        flag = True
                        the_prefix = prefix
                        break
                if flag:
                    print('ignoring {} (prefix: {})'.format(k, the_prefix))
                    del pretrained_state_dict[k]
        load_state_model(self.model, pretrained_state_dict, self.ginfo)

    def prepare_data(self):
        self.tmp.input_var = dict()

        if self.config.sampler.batch_size == 1 and self.config.dataset.type != 'PedestrainDetectionDataset':
            self.tmp.input[0]['image'] = self.tmp.input[0]['image'].unsqueeze(0)  #### TODO: ugly hot fix for single gpu###
            for k, v in self.tmp.input[0].items():
                if isinstance(v, np.ndarray) or isinstance(v, str) or isinstance(v, int) or isinstance(v, DataContainer) or k == "img_metas" or k == "filename":
                    self.tmp.input_var[k] = v
                elif not isinstance(v, list):
                    self.tmp.input_var[k] = v.cuda()
                elif k == "instances":
                    self.tmp.input_var[k] = [_v.cuda() for _v in v]
        else:
            for k,v in self.tmp.input.items():
                if isinstance(v, np.ndarray) or isinstance(v, str) or isinstance(v, int) or isinstance(v, DataContainer) or k == "img_metas" or k == "filename":
                    self.tmp.input_var[k] = v
                elif not isinstance(v, list):
                    self.tmp.input_var[k] = v.cuda()
                elif k == "instances":
                    self.tmp.input_var[k] = [_v.cuda() for _v in v]
                else:
                    self.tmp.input_var[k] = v
        # print(f" self.tmp.input: {self.tmp.input}")

    def inference_on_dataset(self, model, evaluator):
        """
        Run model on the data_loader and evaluate the metrics with evaluator.
        Also benchmark the inference speed of `model.__call__` accurately.
        The model will be used in eval mode.

        Args:
            model (callable): a callable which takes an object from
                `data_loader` and returns some outputs.

                If it's an nn.Module, it will be temporarily set to `eval` mode.
                If you wish to evaluate a model in `training` mode instead, you can
                wrap the given model and override its behavior of `.eval()` and `.train()`.
            data_loader: an iterable object with a length.
                The elements it generates will be the inputs to the model.
            evaluator: the evaluator(s) to run. Use `None` if you only want to benchmark,
                but don't want to do any evaluation.

        Returns:
            The return value of `evaluator.evaluate()`
        """
        num_devices = self.C.world_size
        total = len(self.test_loader)  # inference data loader must have a fixed length

        if self.C.rank == 0:
            logger = self.logger
            logger.info("Start inference on {} batches".format(total))

        evaluator.reset()

        num_warmup = min(5, total - 1)
        start_time = time.perf_counter()
        total_data_time = 0
        total_compute_time = 0
        total_eval_time = 0
        with ExitStack() as stack:
            if isinstance(model, nn.Module):
                stack.enter_context(inference_context(model))
            stack.enter_context(torch.no_grad())

            start_data_time = time.perf_counter()
            for idx, self.tmp.input in enumerate(self.test_loader):
                total_data_time += time.perf_counter() - start_data_time
                self.prepare_data()
                if idx == num_warmup:
                    start_time = time.perf_counter()
                    total_data_time = 0
                    total_compute_time = 0
                    total_eval_time = 0

                start_compute_time = time.perf_counter()
                outputs = model(self.tmp.input_var, idx)
                # import pdb; pdb.set_trace();
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                total_compute_time += time.perf_counter() - start_compute_time
                start_eval_time = time.perf_counter()
                evaluator.process(self.tmp.input_var, outputs)
                total_eval_time += time.perf_counter() - start_eval_time

                iters_after_start = idx + 1 - num_warmup * int(idx >= num_warmup)
                data_seconds_per_iter = total_data_time / iters_after_start
                compute_seconds_per_iter = total_compute_time / iters_after_start
                eval_seconds_per_iter = total_eval_time / iters_after_start
                total_seconds_per_iter = (time.perf_counter() - start_time) / iters_after_start
                if idx >= num_warmup * 2 or compute_seconds_per_iter > 5:
                    eta = datetime.timedelta(seconds=int(total_seconds_per_iter * (total - idx - 1)))
                    self.logger.info(f"Inference done {idx + 1}/{total}. "
                                     f"Dataloading: {data_seconds_per_iter:.4f} s/iter. "
                                     f"Inference: {compute_seconds_per_iter:.4f} s/iter. "
                                     f"Eval: {eval_seconds_per_iter:.4f} s/iter. "
                                     f"Total: {total_seconds_per_iter:.4f} s/iter. "
                                     f"ETA={eta}")
                start_data_time = time.perf_counter()

        # Measure the time only for this worker (before the synchronization barrier)
        total_time = time.perf_counter() - start_time
        total_time_str = str(datetime.timedelta(seconds=total_time))
        # NOTE this format is parsed by grep
        self.logger.info(
            "Total inference time: {} ({:.6f} s / iter per device, on {} devices)".format(
                total_time_str, total_time / (total - num_warmup), num_devices
            )
        )
        total_compute_time_str = str(datetime.timedelta(seconds=int(total_compute_time)))
        self.logger.info(
            "Total inference pure compute time: {} ({:.6f} s / iter per device, on {} devices)".format(
                total_compute_time_str, total_compute_time / (total - num_warmup), num_devices
            )
        )

        results = evaluator.evaluate()
        # An evaluator may return None when not in main process.
        # Replace it by an empty dict instead to make it easier for downstream code to handle
        if results is None:
            results = {}
        return results

    def test_with_TTA(self):  # more like a decorator
        # In the end of training, run an evaluation with TTA.
        # self.create_dataloader()
        self.logger.info("Running inference with test-time augmentation ...")
        model = SemanticSegmentorWithTTA(self.config.extra, self.model)
        evaluator = SemSegEvaluator(dataset_name=self.ginfo.task_name, distributed=True,
                                    output_dir=os.path.join(self.ckpt_path, "inference_TTA"), config=self.config)

        res = self.test(model, evaluator=evaluator)
        res = OrderedDict({k + "_TTA": v for k, v in res.items()})
        return res

    def test(self, model, evaluator=None):
        if evaluator is None:
            evaluator = SemSegEvaluator(dataset_name=self.ginfo.task_name, distributed=True,
                                        output_dir=self.ckpt_path, config=self.config)
        results = OrderedDict()

        results_i = self.inference_on_dataset(model, evaluator)
        results[self.ginfo.task_name] = results_i
        if self.C.rank == 0:
            assert isinstance(
                results_i, dict
            ), "Evaluator must return a dict on the main process. Got {} instead.".format(
                results_i
            )
            self.logger.info("Evaluation results for {} in csv format:".format(self.ginfo.task_name))
            print_csv_format(results_i, self.logger)
        if len(results) == 1:
            results = list(results.values())[0]

        return results

    def run(self):
        if self.config.dataset.type == 'COCOStuffSegDatasetDev':
            results = self.test(self.model)
            results.update(self.test_with_TTA())
        elif 'ParsingDataset' in self.config.dataset.type:
            if self.config.dataset.get('bce_cls_test',False):
                evaluator = HumParEvaluator_bce_cls(dataset_name=self.ginfo.task_name, distributed=True,
                                                    output_dir=self.ckpt_path, config=self.config)
            else:
                evaluator = HumParEvaluator(dataset_name=self.ginfo.task_name, distributed=True,
                                            output_dir=self.ckpt_path, config=self.config)
            results = self.test(self.model, evaluator=evaluator)
        elif self.config.dataset.type == 'COCOPosDatasetDev' or self.config.dataset.type == 'MultiPoseDatasetDev':
            self.config.evaluation.cfg.name2id = self.dataset.name2id
            self.config.evaluation.cfg.dataset = self.dataset
            # import pdb;pdb.set_trace()
            evaluator = PoseEvaluator(dataset_name=self.ginfo.task_name, distributed=True,
                                        output_dir=os.path.join(self.ckpt_path,self.C.config['expname']), config=self.config)
            results = self.test(self.model, evaluator=evaluator)
        elif self.config.dataset.type == 'PedestrainDetectionDataset':
            evaluator = PedDetEvaluator(dataset_name=self.ginfo.task_name, distributed=True,
                                        output_dir=self.ckpt_path, config=self.config)
            results = self.test(self.model, evaluator=evaluator)
        elif self.config.dataset.type == 'AttrDataset':
            evaluator = PedAttrEvaluator(dataset_name=self.ginfo.task_name, distributed=True,
                                        output_dir=self.ckpt_path, config=self.config)
            results = self.test(self.model, evaluator=evaluator)
        elif self.config.dataset.type == 'MPIIPosDatasetDev':
            # self.config.evaluation.cfg.name2id = self.dataset.name2id
            self.config.evaluation.cfg.dataset = self.dataset
            evaluator = MPIIPoseEvaluator(dataset_name=self.ginfo.task_name, distributed=True,
                                        output_dir=self.ckpt_path, config=self.config)
            results = self.test(self.model, evaluator=evaluator)
        elif self.config.dataset.type == 'PedestrainDetectionDataset':
            evaluator = PedDetEvaluator(dataset_name=self.ginfo.task_name, distributed=True,
                                        output_dir=self.ckpt_path, config=self.config)
            results = self.test(self.model, evaluator=evaluator)
        else:
            raise NotImplementedError

        print(f"** results: {results}")


class PLMultiTaskDev(SolverMultiTaskDev):
    def __init__(self, C_train, C_test):
        train_config = edict(C_train.config['common'])
        ginfo = C_train.ginfo
        config = train_config

        if C_test.config.get('common') is not None:
            recursive_update(config, C_test.config.get('common'))
        config = edict(config)
        if 'out_dir' in config:
            self.out_dir = config['out_dir'] + 'test_results/'
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
            # self.tb_logger = SummaryWriter(self.tb_path)
            project_name = config.get('project_name', os.path.dirname(C.config_file).split('/')[-1])
            self.tb_logger = SummaryWriter(log_dir=self.tb_path, name=C.config['expname'], project=project_name, overwrite_last_training=True)

        else:
            while not os.path.exists(self.logs_path):
                time.sleep(1)

        if ginfo.task_rank == 0:
            assert C_train.rank == 0, "there shall be only one group"
            self.logger = create_logger('global_logger', '{}/log_task_{}.txt'.format(self.logs_path, ginfo.task_id))

        self.sync = config.get('sync', True)
        self.C = C_train

        self.config = config
        self.ginfo = ginfo
        # change tensor .cuda
        change_tensor_cuda()

        self.tmp = edict()

        ## random seed setting
        rng = np.random.RandomState(self.config.get('random_seed', 0))
        self.randomseed_pool = rng.randint(999999, size=config.max_iter)

        ### VVV deterministic measures VVV

        if self.config.get('deterministic', False):
            if self.config.get('cudnn_deterministic', True):
                cudnn.deterministic = True
                cudnn.benchmark = False
            else:
                cudnn.benchmark = True
            seed = self.config.get('random_seed', 0)
            worker_rank = self.config.get('worker_rank', False)
            if worker_rank:
                worker_init = WorkerInit(self.C.rank, self.config.workers)
            else:
                worker_init = WorkerInit(0, 0)
            self.worker_init_fn = worker_init.func
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            dist.barrier()
            if self.C.rank == 0:
                self.logger.info(f'deterministic mode, seed: {seed}, worker_rank: {worker_rank},\
                                   cudnn_deterministic: {self.config.get("cudnn_deterministic", True)}')
            dist.barrier()
        else:
            self.worker_init_fn = None

    def load(self, args):
        if args.load_path == '':
            return
        load_path = args.load_path if args.load_single else args.load_path.replace('ckpt_task_', f'ckpt_task{self.ginfo.task_id}_')

        try:
            checkpoint = torch.load(load_path, 'cpu')
        except:
            raise FileNotFoundError(f'=> no checkpoint found at {load_path}')

        if self.ginfo.task_rank == 0:
            printlog(f"Recovering from {load_path}, keys={list(checkpoint.keys())}")

        if 'state_dict' in checkpoint:
            pretrained_state_dict = checkpoint['state_dict']
        else:
            pretrained_state_dict = checkpoint
        if len(args.ignore) > 0:
            for k in list(pretrained_state_dict.keys()):
                flag = False
                for prefix in args.ignore:
                    if k.startswith(prefix):
                        flag = True
                        the_prefix = prefix
                        break
                if flag:
                    print('ignoring {} (prefix: {})'.format(k, the_prefix))
                    del pretrained_state_dict[k]
        load_state_model(self.model, pretrained_state_dict, self.ginfo)

    def initialize(self, args):
        self.create_dataset()
        self.create_model()

        self.load_args = args
        self.load(args)

        self.create_dataloader()

    def create_dataloader(self):
        self.test_sampler = DistributedSequentialSampler(self.dataset)

        collate = naive_collate if self.config.get('collate', 'naive') == 'naive' else functools.partial(dev_collate, nested=self.config.dataset.kwargs.get('nested', False))
        self.test_loader = DataLoader(self.dataset, batch_size=self.config.sampler.batch_size,
                                      shuffle=False, drop_last=False, num_workers=self.config.workers,
                                      pin_memory=False, sampler=self.test_sampler, collate_fn=collate)

    def prepare_data(self):
        self.tmp.input_var = dict()

        if self.config.sampler.batch_size == 1:
            self.tmp.input[0]['image'] = self.tmp.input[0]['image'].unsqueeze(0)  #### TODO: ugly hot fix for single gpu###
            for k, v in self.tmp.input[0].items():
                if isinstance(v, np.ndarray) or isinstance(v, str) or isinstance(v, int) or isinstance(v, DataContainer) or k == "img_metas":
                    self.tmp.input_var[k] = v
                elif not isinstance(v, list):
                    self.tmp.input_var[k] = v.cuda()
                elif k == "instances":
                    self.tmp.input_var[k] = [_v.cuda() for _v in v]
        else:
            for k,v in self.tmp.input.items():
                # import pdb;pdb.set_trace()
                if isinstance(v, np.ndarray) or isinstance(v, str) or isinstance(v, int) or isinstance(v, DataContainer) or k == "img_metas":
                    self.tmp.input_var[k] = v
                elif not isinstance(v, list):
                    self.tmp.input_var[k] = v.cuda()
                elif k == "instances":
                    self.tmp.input_var[k] = [_v.cuda() for _v in v]
                else:
                    self.tmp.input_var[k] = v
        # print(f" self.tmp.input: {self.tmp.input}")


    def inference_on_dataset(self, model, evaluator):
        """
        Run model on the data_loader and generate pseudo labels.
        Also benchmark the inference speed of `model.__call__` accurately.
        The model will be used in eval mode.

        Args:
            model (callable): a callable which takes an object from
                `data_loader` and returns some outputs.

                If it's an nn.Module, it will be temporarily set to `eval` mode.
                If you wish to evaluate a model in `training` mode instead, you can
                wrap the given model and override its behavior of `.eval()` and `.train()`.
            data_loader: an iterable object with a length.
                The elements it generates will be the inputs to the model.
            evaluator: the evaluator(s) to run. Use `None` if you only want to benchmark,
                but don't want to do any evaluation.

        Returns:
            None. Pseudo labels should be generated separately in evaluator.generate_pseudo_labels()
            and combined in evaluator.combine_pseudo_labels().
        """
        num_devices = self.C.world_size
        total = len(self.test_loader)  # inference data loader must have a fixed length
        logger = self.logger
        logger.info("Start inference on {} batches".format(total))

        evaluator.reset()

        num_warmup = min(5, total - 1)
        start_time = time.perf_counter()
        total_data_time = 0
        total_compute_time = 0
        total_eval_time = 0
        with ExitStack() as stack:
            if isinstance(model, nn.Module):
                stack.enter_context(inference_context(model))
            stack.enter_context(torch.no_grad())

            start_data_time = time.perf_counter()
            # import pdb;
            # pdb.set_trace()
            try:
                save_dir = self.config.pseudo_save_path
            except:
                save_dir = './'
            os.makedirs(save_dir, exist_ok=True)

            for idx, self.tmp.input in enumerate(self.test_loader):
                total_data_time += time.perf_counter() - start_data_time
                self.prepare_data()
                if idx == num_warmup:
                    start_time = time.perf_counter()
                    total_data_time = 0
                    total_compute_time = 0
                    total_eval_time = 0

                start_compute_time = time.perf_counter()
                outputs = model(self.tmp.input_var, idx)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                total_compute_time += time.perf_counter() - start_compute_time

                start_eval_time = time.perf_counter()
                evaluator.generate_pseudo_labels(self.tmp.input_var, outputs, dataset=self.dataset, save_dir=save_dir)
                total_eval_time += time.perf_counter() - start_eval_time

                iters_after_start = idx + 1 - num_warmup * int(idx >= num_warmup)
                data_seconds_per_iter = total_data_time / iters_after_start
                compute_seconds_per_iter = total_compute_time / iters_after_start
                eval_seconds_per_iter = total_eval_time / iters_after_start
                total_seconds_per_iter = (time.perf_counter() - start_time) / iters_after_start
                if idx >= num_warmup * 2 or compute_seconds_per_iter > 5:
                    eta = datetime.timedelta(seconds=int(total_seconds_per_iter * (total - idx - 1)))
                    self.logger.info(f"Inference done {idx + 1}/{total}. "
                                     f"Dataloading: {data_seconds_per_iter:.4f} s/iter. "
                                     f"Inference: {compute_seconds_per_iter:.4f} s/iter. "
                                     f"Eval: {eval_seconds_per_iter:.4f} s/iter. "
                                     f"Total: {total_seconds_per_iter:.4f} s/iter. "
                                     f"ETA={eta}")
                start_data_time = time.perf_counter()

        # Measure the time only for this worker (before the synchronization barrier)
        total_time = time.perf_counter() - start_time
        total_time_str = str(datetime.timedelta(seconds=total_time))
        # NOTE this format is parsed by grep
        logger.info(
            "Total inference time: {} ({:.6f} s / iter per device, on {} devices)".format(
                total_time_str, total_time / (total - num_warmup), num_devices
            )
        )
        total_compute_time_str = str(datetime.timedelta(seconds=int(total_compute_time)))
        logger.info(
            "Total inference pure compute time: {} ({:.6f} s / iter per device, on {} devices)".format(
                total_compute_time_str, total_compute_time / (total - num_warmup), num_devices
            )
        )

        results = evaluator.combine_pseudo_labels(save_dir=save_dir)
        # An evaluator may return None when not in main process.
        # Replace it by an empty dict instead to make it easier for downstream code to handle
        if results is None:
            results = {}
        return results

    def test_with_TTA(self):  # more like a decorator
        # In the end of training, run an evaluation with TTA.
        # self.create_dataloader()
        self.logger.info("Running inference with test-time augmentation ...")
        model = SemanticSegmentorWithTTA(self.config.extra, self.model)
        evaluator = SemSegEvaluator(dataset_name=self.ginfo.task_name, distributed=True,
                                    output_dir=os.path.join(self.ckpt_path, "inference_TTA"), config=self.config)

        res = self.test(model, evaluator=evaluator)
        res = OrderedDict({k + "_TTA": v for k, v in res.items()})
        return res

    def test(self, model, evaluator=None):
        if evaluator is None:
            evaluator = SemSegEvaluator(dataset_name=self.ginfo.task_name, distributed=True,
                                        output_dir=self.ckpt_path, config=self.config)
        results = OrderedDict()

        results_i = self.inference_on_dataset(model, evaluator)

        # results[self.ginfo.task_name] = results_i
        # if self.C.rank == 0:
        #     assert isinstance(
        #         results_i, dict
        #     ), "Evaluator must return a dict on the main process. Got {} instead.".format(
        #         results_i
        #     )
        #     self.logger.info("Evaluation results for {} in csv format:".format(self.ginfo.task_name))
        #     print_csv_format(results_i, self.logger)
        # if len(results) == 1:
        #     results = list(results.values())[0]

        return 'Final generating pseudo labels.'

    def run(self):
        if self.config.dataset.type == 'COCOStuffSegDatasetDev':
            results = self.test(self.model)
            results.update(self.test_with_TTA())
        elif 'ParsingDataset' in self.config.dataset.type:
            evaluator = HumParEvaluator(dataset_name=self.ginfo.task_name, distributed=True,
                                        output_dir=self.ckpt_path, config=self.config)
            results = self.test(self.model, evaluator=evaluator)
        elif self.config.dataset.type == 'COCOPosDatasetDev':
            self.config.evaluation.cfg.name2id = self.dataset.name2id
            self.config.evaluation.cfg.dataset = self.dataset
            evaluator = PoseEvaluator(dataset_name=self.ginfo.task_name, distributed=True,
                                        output_dir=self.ckpt_path, config=self.config)
            results = self.test(self.model, evaluator=evaluator)
        else:
            raise NotImplementedError

        print(f"** results: {results}")

default_collate_err_msg_format = ("default_collate: batch must contain tensors, numpy arrays, numbers, "
                                  "dicts or lists; found {}")

np_str_obj_array_pattern = re.compile(r'[SaUO]')


def dev_collate(batch, nested=False):  # altered collate_fn to support 'Instance' object within batch
    r"""
        Function that takes in a batch of data and puts the elements within the batch
        into a tensor with an additional outer dimension - batch size. The exact output type can be
        a :class:`torch.Tensor`, a `Sequence` of :class:`torch.Tensor`, a
        Collection of :class:`torch.Tensor`, or left unchanged, depending on the input type.
        This is used as the default function for collation when
        `batch_size` or `batch_sampler` is defined in :class:`~torch.utils.data.DataLoader`.

        Here is the general input type (based on the type of the element within the batch) to output type mapping:
        * :class:`torch.Tensor` -> :class:`torch.Tensor` (with an added outer dimension batch size)
        * NumPy Arrays -> :class:`torch.Tensor`
        * `float` -> :class:`torch.Tensor`
        * `int` -> :class:`torch.Tensor`
        * `str` -> `str` (unchanged)
        * `bytes` -> `bytes` (unchanged)
        * `Mapping[K, V_i]` -> `Mapping[K, dev_collate([V_1, V_2, ...])]`
        * `NamedTuple[V1_i, V2_i, ...]` -> `NamedTuple[dev_collate([V1_1, V1_2, ...]), dev_collate([V2_1, V2_2, ...]), ...]`
        * `Sequence[V1_i, V2_i, ...]` -> `Sequence[dev_collate([V1_1, V1_2, ...]), dev_collate([V2_1, V2_2, ...]), ...]`

        Args:
            batch: a single batch to be collated

        Examples:
            >>> # Example with a batch of `int`s:
            >>> dev_collate([0, 1, 2, 3])
            tensor([0, 1, 2, 3])
            >>> # Example with a batch of `str`s:
            >>> dev_collate(['a', 'b', 'c'])
            ['a', 'b', 'c']
            >>> # Example with `Map` inside the batch:
            >>> dev_collate([{'A': 0, 'B': 1}, {'A': 100, 'B': 100}])
            {'A': tensor([  0, 100]), 'B': tensor([  1, 100])}
            >>> # Example with `NamedTuple` inside the batch:
            >>> Point = namedtuple('Point', ['x', 'y'])
            >>> dev_collate([Point(0, 0), Point(1, 1)])
            Point(x=tensor([0, 1]), y=tensor([0, 1]))
            >>> # Example with `Tuple` inside the batch:
            >>> dev_collate([(0, 1), (2, 3)])
            [tensor([0, 2]), tensor([1, 3])]
            >>> # Example with `List` inside the batch:
            >>> dev_collate([[0, 1], [2, 3]])
            [tensor([0, 2]), tensor([1, 3])]
    """
    elem = batch[0]
    elem_type = type(elem)
    if isinstance(elem, torch.Tensor):
        if elem.ndim == 3 and nested:
            return nested_tensor_from_tensor_list(batch)
        else:
            out = None
            if torch.utils.data.get_worker_info() is not None:
                # If we're in a background process, concatenate directly into a
                # shared memory tensor to avoid an extra copy
                numel = sum(x.numel() for x in batch)
                storage = elem.storage()._new_shared(numel)
                out = elem.new(storage).resize_(len(batch), *list(elem.size()))
            return torch.stack(batch, 0, out=out)
    elif isinstance(elem, Instances) or isinstance(elem, DataContainer):  # ** special treatment for 'Instance' object elements
        return batch
    elif elem_type.__module__ == 'numpy' and elem_type.__name__ != 'str_' \
            and elem_type.__name__ != 'string_':
        if elem_type.__name__ == 'ndarray' or elem_type.__name__ == 'memmap':
            # array of string classes and object
            if np_str_obj_array_pattern.search(elem.dtype.str) is not None:
                raise TypeError(default_collate_err_msg_format.format(elem.dtype))

            return dev_collate([torch.as_tensor(b) for b in batch], nested=nested)
        elif elem.shape == ():  # scalars
            return torch.as_tensor(batch)
    elif isinstance(elem, float):
        return torch.tensor(batch, dtype=torch.float64)
    elif isinstance(elem, int):
        return torch.tensor(batch)
    elif isinstance(elem, string_classes):
        return batch
    elif isinstance(elem, collections.abc.Mapping):
        try:
            return elem_type({key: dev_collate([d[key] for d in batch], nested=nested) for key in elem})
        except TypeError:
            # The mapping type may not support `__init__(iterable)`.
            return {key: dev_collate([d[key] for d in batch], nested=nested) for key in elem}
    elif isinstance(elem, tuple) and hasattr(elem, '_fields'):  # namedtuple
        return elem_type(*(dev_collate(samples, nested=nested) for samples in zip(*batch)))
    elif isinstance(elem, collections.abc.Sequence):
        # check to make sure that the elements in batch have consistent size
        it = iter(batch)
        elem_size = len(next(it))
        if not all(len(elem) == elem_size for elem in it):
            raise RuntimeError('each element in list of batch should be of equal size')
        transposed = list(zip(*batch))  # It may be accessed twice, so we use a list.

        if isinstance(elem, tuple):
            return [dev_collate(samples, nested=nested) for samples in transposed]  # Backwards compatibility.
        else:
            try:
                return elem_type([dev_collate(samples, nested=nested) for samples in transposed])
            except TypeError:
                # The sequence type may not support `__init__(iterable)` (e.g., `range`).
                return [dev_collate(samples, nested=nested) for samples in transposed]

    raise TypeError(default_collate_err_msg_format.format(elem_type))


def det_collate(batch):  # altered collate_fn to support 'Instance' object within batch
    r"""
        Function that takes in a batch of data and puts the elements within the batch
        into a tensor with an additional outer dimension - batch size. The exact output type can be
        a :class:`torch.Tensor`, a `Sequence` of :class:`torch.Tensor`, a
        Collection of :class:`torch.Tensor`, or left unchanged, depending on the input type.
        This is used as the default function for collation when
        `batch_size` or `batch_sampler` is defined in :class:`~torch.utils.data.DataLoader`.

        Here is the general input type (based on the type of the element within the batch) to output type mapping:
        * :class:`torch.Tensor` -> :class:`torch.Tensor` (with an added outer dimension batch size)
        * NumPy Arrays -> :class:`torch.Tensor`
        * `float` -> :class:`torch.Tensor`
        * `int` -> :class:`torch.Tensor`
        * `str` -> `str` (unchanged)
        * `bytes` -> `bytes` (unchanged)
        * `Mapping[K, V_i]` -> `Mapping[K, det_collate([V_1, V_2, ...])]`
        * `NamedTuple[V1_i, V2_i, ...]` -> `NamedTuple[det_collate([V1_1, V1_2, ...]), det_collate([V2_1, V2_2, ...]), ...]`
        * `Sequence[V1_i, V2_i, ...]` -> `Sequence[det_collate([V1_1, V1_2, ...]), det_collate([V2_1, V2_2, ...]), ...]`

        Args:
            batch: a single batch to be collated

        Examples:
            >>> # Example with a batch of `int`s:
            >>> det_collate([0, 1, 2, 3])
            tensor([0, 1, 2, 3])
            >>> # Example with a batch of `str`s:
            >>> det_collate(['a', 'b', 'c'])
            ['a', 'b', 'c']
            >>> # Example with `Map` inside the batch:
            >>> det_collate([{'A': 0, 'B': 1}, {'A': 100, 'B': 100}])
            {'A': tensor([  0, 100]), 'B': tensor([  1, 100])}
            >>> # Example with `NamedTuple` inside the batch:
            >>> Point = namedtuple('Point', ['x', 'y'])
            >>> det_collate([Point(0, 0), Point(1, 1)])
            Point(x=tensor([0, 1]), y=tensor([0, 1]))
            >>> # Example with `Tuple` inside the batch:
            >>> det_collate([(0, 1), (2, 3)])
            [tensor([0, 2]), tensor([1, 3])]
            >>> # Example with `List` inside the batch:
            >>> det_collate([[0, 1], [2, 3]])
            [tensor([0, 2]), tensor([1, 3])]
    """
    elem = batch[0]
    elem_type = type(elem)
    if isinstance(elem, torch.Tensor):
        if elem.ndim == 3:
            return nested_tensor_from_tensor_list(batch)
        else:
            out = None
            if torch.utils.data.get_worker_info() is not None:
                # If we're in a background process, concatenate directly into a
                # shared memory tensor to avoid an extra copy
                numel = sum(x.numel() for x in batch)
                storage = elem.storage()._new_shared(numel)
                out = elem.new(storage).resize_(len(batch), *list(elem.size()))
            return torch.stack(batch, 0, out=out)
    elif isinstance(elem, Instances) or isinstance(elem, DataContainer):  # ** special treatment for 'Instance' object elements
        return batch
    elif elem_type.__module__ == 'numpy' and elem_type.__name__ != 'str_' \
            and elem_type.__name__ != 'string_':
        if elem_type.__name__ == 'ndarray' or elem_type.__name__ == 'memmap':
            # array of string classes and object
            if np_str_obj_array_pattern.search(elem.dtype.str) is not None:
                raise TypeError(default_collate_err_msg_format.format(elem.dtype))

            return det_collate([torch.as_tensor(b) for b in batch])
        elif elem.shape == ():  # scalars
            return torch.as_tensor(batch)
    elif isinstance(elem, float):
        return torch.tensor(batch, dtype=torch.float64)
    elif isinstance(elem, int):
        return torch.tensor(batch)
    elif isinstance(elem, string_classes):
        return batch
    elif isinstance(elem, collections.abc.Mapping):
        try:
            return elem_type({key: det_collate([d[key] for d in batch]) for key in elem})
        except TypeError:
            # The mapping type may not support `__init__(iterable)`.
            return {key: det_collate([d[key] for d in batch]) for key in elem}
    elif isinstance(elem, tuple) and hasattr(elem, '_fields'):  # namedtuple
        return elem_type(*(det_collate(samples) for samples in zip(*batch)))
    elif isinstance(elem, collections.abc.Sequence):
        # check to make sure that the elements in batch have consistent size
        it = iter(batch)
        elem_size = len(next(it))
        if not all(len(elem) == elem_size for elem in it):
            raise RuntimeError('each element in list of batch should be of equal size')
        transposed = list(zip(*batch))  # It may be accessed twice, so we use a list.

        if isinstance(elem, tuple):
            return [det_collate(samples) for samples in transposed]  # Backwards compatibility.
        else:
            try:
                return elem_type([det_collate(samples) for samples in transposed])
            except TypeError:
                # The sequence type may not support `__init__(iterable)` (e.g., `range`).
                return [det_collate(samples) for samples in transposed]

    raise TypeError(default_collate_err_msg_format.format(elem_type))


def naive_collate(batch):
    return batch


@contextmanager
def inference_context(model):
    """
    A context where the model is temporarily changed to eval mode,
    and restored to previous mode afterwards.

    Args:
        model: a torch Module
    """
    training_mode = model.training
    model.eval()
    yield
    model.train(training_mode)


def print_csv_format(results, logger):
    """
    Print main metrics in a format similar to Detectron,
    so that they are easy to copypaste into a spreadsheet.

    Args:
        results (OrderedDict[dict]): task_name -> {metric -> score}
            unordered dict can also be printed, but in arbitrary order
    """
    assert isinstance(results, Mapping) or not len(results), results
    for task, res in results.items():
        if isinstance(res, Mapping):
            # Don't print "AP-category" metrics since they are usually not tracked.
            important_res = [(k, v) for k, v in res.items() if "-" not in k]
            logger.info("copypaste: Task: {}".format(task))
            logger.info("copypaste: " + ",".join([k[0] for k in important_res]))
            logger.info("copypaste: " + ",".join(["{0:.4f}".format(k[1]) for k in important_res]))
        else:
            logger.info(f"copypaste: {task}={res}")
