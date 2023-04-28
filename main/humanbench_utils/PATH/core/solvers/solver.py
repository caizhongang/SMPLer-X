import os
import time
import torch
import torch.backends.cudnn as cudnn
from core import distributed_utils as dist
from torch.utils.data import DataLoader
from pavi import SummaryWriter
from easydict import EasyDict as edict
import numpy as np
import random
import copy

import core
import core.models.decoders as decoders
import core.models.backbones as backbones
import core.models.necks as necks
import core.data.datasets as datasets
import core.optimizers as optimizers
from core.models.model_entry import model_entry
from core.utils import (AverageMeter, accuracy, load_state, load_last_iter,
                       save_state, create_logger, IterLRScheduler,
                       count_parameters_num, freeze_bn,
                       change_tensor_cuda, sync_print)
from core.distributed_utils import DistModule, DistributedGivenIterationSampler, simple_group_split, vreduce, vgather
from core.make_param_group import param_group_multitask
from core.lr_scheduler import lr_scheduler_entry

class Solver(object):

    def __init__(self, C):
        config = edict(C.config['common'])
        ginfo = C.ginfo
        if 'out_dir' in C.config:
            self.out_dir = C.config['out_dir']+'/'
        else:
            self.out_dir = ""

        if 'expname' in C.config:
            self.tb_path = '{}events/{}'.format(self.out_dir, C.config['expname'])
            self.ckpt_path = '{}checkpoints/{}'.format(self.out_dir, C.config['expname'])
            self.logs_path = '{}logs/{}'.format(self.out_dir, C.config['expname'])
        else:
            save_path = config.get('save_path', os.path.dirname(C.config_file))
            self.save_path = save_path
            self.tb_path = '{}/events'.format(save_path)
            self.ckpt_path = '{}/checkpoints'.format(save_path)
            self.logs_path = '{}/logs'.format(save_path)
        if C.rank == 0:
            os.makedirs(self.tb_path, exist_ok=True)
            os.makedirs(self.ckpt_path, exist_ok=True)
            os.makedirs(self.logs_path, exist_ok=True)
            # self.tb_logger = SummaryWriter(self.tb_path)
            project_name = config.get('project_name', os.path.dirname(C.config_file).split('/')[-1])
            # import pdb;pdb.set_trace()
            self.tb_logger = SummaryWriter(log_dir=self.tb_path,name=C.config['expname'],project=project_name)
        else:
            while not os.path.exists(self.logs_path):
                time.sleep(1)

        if ginfo.task_rank == 0:
            self.logger = create_logger('global_logger', '{}/log_task_{}.txt'.format(self.logs_path, ginfo.task_id))

        self.clip_grad_backbone = config.get('clip_grad_backbone', 0.0)
        self.clip_grad_neck = config.get('clip_grad_neck', 0.0)
        self.clip_grad_decoder = config.get('clip_grad_decoder', 0.0)
        self.sync = config.get('sync', False)

        self.fix_bn = config.get('fix_bn', False)

        self.last_iter = -1
        # self.feature_dim = config.model['kwargs']['feature_dim']
        # self.feature_dim = config['feature_dim']
        self.C = C
        self.config = config
        self.ginfo = ginfo

        # for auto_denan
        self.autodenan = self.config.get('auto_denan', True)
        if not self.autodenan and self.C.rank == 0:
            self.logger.info('auto_denan disabled!')
        self.last_state_dict = {}
        self.last_optim_state_dict = {}
        self.last_save_iter = -1

        # for auto_alert
        self.auto_alert = self.config.get('auto_alert', False)
        if self.auto_alert and self.C.rank == 0:
            self.job_name = C.config_path.split('/')[-2]
        if self.auto_alert:
            from core.msg_server import MsgClient
            self.alert('job started with auto alert!')

        # change tensor .cuda
        change_tensor_cuda()

        # lr
        assert config.lr_scheduler.get('use_new_lr', 'deprecated') == 'deprecated'  # redundant config alert
        config.base_lr = config.lr_scheduler.kwargs.base_lr

        self.tmp = edict()

        ## random seed setting
        rng = np.random.RandomState(self.config.get('random_seed', 0))
        self.randomseed_pool = rng.randint(999999, size=config.max_iter)

    def init_msg_client(self):
        with open('server.txt') as f:
            line = f.read().strip()
        ip, port = line.split()
        port = int(port)
        self.msg_client = MsgClient(ip, port)

    def alert(self, msg):
        if self.C.rank == 0:
            try:
                self.msg_client.send('[{}]: {}\n'.format(self.job_name, msg))
            except Exception as e:
                print(e)
                count = 0
                succ = False
                while count < 10:
                    print('reconnecting...')
                    try:
                        if hasattr(self, 'msg_client'):
                            self.msg_client.close()
                        self.init_msg_client()
                    except Exception as e2:
                        print(e2)
                        count += 1
                        time.sleep(1)
                    else:
                        succ = True
                        break
                if succ:
                    self.msg_client.send('[{}]: {}'.format(self.job_name, msg))

    def create_dataset(self):
        ginfo = self.ginfo
        config = self.config
        dataset_args = config.dataset['kwargs']
        dataset_args['ginfo'] = ginfo
        self.dataset = datasets.dataset_entry(config.dataset)
        dist.barrier()

    def create_dataloader(self):
        config = self.config
        ginfo = self.ginfo

        self.sampler = DistributedGivenIterationSampler(
                        self.dataset, config.max_iter, config.sampler.batch_size,
                        world_size=ginfo.task_size, rank=ginfo.task_rank,
                        last_iter=self.last_iter, shuffle_strategy=config.sampler.shuffle_strategy,
                        random_seed=ginfo.task_random_seed, ret_save_path=config.sampler.get('ret_save_path', None))
        self.loader = DataLoader(self.dataset, batch_size=config.sampler.batch_size,
                            shuffle=False, num_workers=config.workers,
                            pin_memory=False, sampler=self.sampler)

    def create_model(self):
        config = self.config
        ginfo = self.ginfo

        backbone_bn_group_size = config.backbone['kwargs'].get('bn_group_size', 1)
        assert backbone_bn_group_size == 1, 'other bn group size not support!'
        backbone_bn_group_comm = self.ginfo.backbone_share_group
        # if backbone_bn_group_size == 1:
        #     backbone_bn_group_comm = None
        # else:
        #     assert self.C.world_size % backbone_bn_group_size == 0
        #     backbone_bn_group_comm = simple_group_split(self.C.world_size, self.C.rank, self.C.world_size // backbone_bn_group_size)

        ## build backbone
        config.backbone['kwargs']['bn_group'] = backbone_bn_group_comm
        backbone_module = backbones.backbone_entry(config.backbone)
        count_parameters_num(backbone_module)

        ## build necks
        neck_bn_group_size = config.backbone['kwargs'].get('bn_group_size', 1)
        assert neck_bn_group_size == 1, 'other bn group size not support!'
        neck_bn_group_comm = self.ginfo.neck_share_group

        # neck_bn_group_size = config.neck['kwargs'].get('bn_group_size', 1)
        # if neck_bn_group_size == 1:
        #     neck_bn_group_comm = None
        # else:
        #     assert self.C.world_size % neck_bn_group_size == 0
        #     neck_bn_group_comm = simple_group_split(self.C.world_size, self.C.rank, self.C.world_size // neck_bn_group_size)

        neck_args = config.neck['kwargs']
        neck_args['backbone'] = backbone_module
        neck_args['bn_group'] = neck_bn_group_comm
        neck_module = necks.neck_entry(config.neck)

        ## add decoder
        decoder_bn_group_size = config.backbone['kwargs'].get('bn_group_size', 1)
        assert decoder_bn_group_size == 1, 'other bn group size not support!'
        decoder_bn_group_comm = self.ginfo.decoder_share_group

        # decoder_bn_group_size = config.neck['kwargs'].get('bn_group_size', 1)
        # if decoder_bn_group_size == 1:
        #     decoder_bn_group_comm = None
        # else:
        #     assert self.C.world_size % decoder_bn_group_size == 0
        #     decoder_bn_group_comm = simple_group_split(self.C.world_size, self.C.rank, self.C.world_size // decoder_bn_group_size)

        decoder_args = config.decoder['kwargs']
        decoder_args['backbone'] = backbone_module
        decoder_args['neck'] = neck_module
        decoder_args['bn_group'] = decoder_bn_group_comm
        decoder_module = decoders.decoder_entry(config.decoder)

        # build
        model = model_entry(backbone_module, neck_module, decoder_module)

        if self.C.rank == 0:
            print(model)

        model = DistModule(model, sync=self.sync, task_grp=self.ginfo.group, \
            share_backbone_group=self.ginfo.backbone_share_group, \
            share_neck_group=self.ginfo.neck_share_group, \
            share_decoder_group=self.ginfo.decoder_share_group)

        self.model = model

    def create_optimizer(self):
        ## param_group
        decoder_optimizer_args = self.config.decoder.kwargs.get('optimizer', self.config.optimizer)
        neck_optimizer_args = self.config.neck.kwargs.get('optimizer', self.config.optimizer)

        param_group = param_group_multitask(self.model)
        param_group[1].update(neck_optimizer_args)
        param_group[2].update(decoder_optimizer_args)

        if self.C.rank == 0:
            self.logger.info('making param_group_backbone, num_parameters:{}, args: {}'.format(len(param_group[0]['params']), self.config.optimizer))
            self.logger.info('making param_group_neck, num_parameters:{}, args: {}'.format(len(param_group[1]['params']), neck_optimizer_args))
            self.logger.info('making param_group_decoder, num_parameters:{}, args: {}'.format(len(param_group[2]['params']), decoder_optimizer_args))
            if len(param_group) > 3:
                self.logger.info('making param_group_other, num_parameters:{}, args: {}'.format(len(param_group[3]['params']), self.config.optimizer))
            else:
                self.logger.info('making param_group_other, num_parameters:{}, args: {}'.format(0, 'No Args!'))

        self.config.optimizer.kwargs.params = param_group
        self.config.optimizer.kwargs.lr = self.config.base_lr
        self.optimizer = optimizers.optim_entry(self.config.optimizer)

    def create_lr_scheduler(self):
        if self.C.rank == 0:
            self.logger.info('using new lr scheduler!')
        self.config.lr_scheduler.kwargs.optimizer = self.optimizer
        self.config.lr_scheduler.kwargs.last_iter = self.last_iter
        self.config.lr_scheduler.kwargs.max_iter = self.config.max_iter
        self.lr_scheduler = lr_scheduler_entry(self.config.lr_scheduler)

    def load(self, args):
        if args.load_path == '':
            return
        if args.recover:
            # self.last_iter = load_state(args.load_path.replace('ckpt_task_', 'ckpt_task{}_'.format(self.ginfo.task_id)), self.model, optimizer=self.optimizer, recover=args.recover)
            self.last_iter = load_state(args.load_path.replace('ckpt_task_', 'ckpt_task{}_'.format(self.ginfo.task_id)), self.model, optimizer=None, recover=args.recover)
            self.last_iter -= 1
        else:
            if args.load_single:
                load_state(args.load_path, self.model, ignore=args.ignore)
            else:
                load_state(args.load_path.replace('ckpt_task_', 'ckpt_task{}_'.format(self.ginfo.task_id)), self.model, ignore=args.ignore)

    def initialize(self, args):
        ## create dataset to get num_classes
        self.create_dataset()
        self.create_model()
        # self.create_optimizer()
        ## load first to get last_iter

        # currently a workaround to get last_iter before sampler and scheduler
        # if args.recover:
        #     self.last_iter = load_last_iter(args.load_path.replace('ckpt_task', 'ckpt_task{}'.format(self.ginfo.task_id)))
        #     self.last_iter -= 1
        self.create_optimizer()
        self.load_args = args

        self.load(args)
        ## then create sampler in dataloader
        self.create_dataloader()

        self.create_lr_scheduler()

    def pre_run(self):
        tmp = self.tmp
        tmp.vbatch_time = AverageMeter(10)
        tmp.vdata_time = AverageMeter(10)
        tmp.vloss = AverageMeter(10)
        tmp.vtop1 = AverageMeter(10)

        tmp.loss_list = [torch.Tensor(1).cuda() for _ in range(self.C.world_size)]
        tmp.top1_list = [torch.Tensor(1).cuda() for _ in range(self.C.world_size)]

        tmp.vbackbone_grad_norm = AverageMeter(10)
        tmp.backbone_grad_norm_list = [torch.Tensor(1).cuda() for _ in range(self.C.world_size)]
        tmp.vneck_grad_norm = AverageMeter(10)
        tmp.neck_grad_norm_list = [torch.Tensor(1).cuda() for _ in range(self.C.world_size)]
        tmp.vdecoder_grad_norm = AverageMeter(10)
        tmp.decoder_grad_norm_list = [torch.Tensor(1).cuda() for _ in range(self.C.world_size)]

        self.model.train()
        # if self.fix_bn:
        #     names = freeze_bn(self.model)
        #     if self.C.rank == 0:
        #         for name in names:
        #             self.logger.info('fixing BN [{}]'.format(name))

    def prepare_data(self):
        ginfo = self.ginfo

        tmp = self.tmp
        tmp.input_var = dict()

        if ginfo.task_type == 'pairwise':
            tmp.input_var['image'] = torch.autograd.Variable(torch.cat((tmp.input['image1'], tmp.input['image2']), 0).cuda())
            tmp.input_var['label'] = torch.autograd.Variable(torch.cat((tmp.input['label'], tmp.input['label']), 0).cuda())
        else:
            for k,v in tmp.input.items():
                if not isinstance(v, list):
                    tmp.input_var[k] = torch.autograd.Variable(v.cuda())

    def _set_randomseed(self, seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    def forward(self):
        ## set random seed with current_step at each iteration
        self._set_randomseed(self.randomseed_pool[self.tmp.current_step])

        tmp = self.tmp
        ginfo = self.ginfo
        tmp.drop_this_iter = False

        output = self.model(tmp.input_var, tmp.current_step)
        tmp.raw_loss = output['loss'] / ginfo.task_size
        if 'top1' in output:
            tmp.raw_top1 = output['top1'] / ginfo.task_size
        else:
            tmp.raw_top1 = torch.zeros(1).cuda()
        tmp.loss = tmp.raw_loss * ginfo.task_weight
        tmp.top1 = tmp.raw_top1

    def backward(self):
        tmp = self.tmp
        ginfo = self.ginfo

        self.optimizer.zero_grad()
        tmp.loss.backward()

    def auto_denan(self):
        dist.barrier()
        if self.auto_denan_check():
            self.auto_denan_recover()
            return True
            # self.forward()
            # self.backward()
        else:
            self.auto_denan_save()
            return False

    def auto_denan_check(self):
        tmp = self.tmp
        ginfo = self.ginfo

        drop_flag = 0
        if np.isnan(tmp.loss.data.item()) or np.isinf(tmp.loss.data.item()):
            drop_flag = 1

        drop_flag = torch.Tensor([drop_flag]).cuda()
        dist.allreduce(drop_flag)

        drop_flag = drop_flag.item()
        if drop_flag > 0:
            return True

        return False

    def auto_denan_recover(self):
        try:
            if self.C.rank == 0:
                self.logger.info('NaN or Inf encountered, recovering from {}\t'.format(self.last_save_iter))
            # recover model
            self.model.load_state_dict(self.last_state_dict, strict=True)
            # recover optimizer
            for g in self.optimizer.param_groups:
                for p in g['params']:
                    self.optimizer.state[p]['momentum_buffer'].copy_(self.last_optim_state_dict['state'][id(p)]['momentum_buffer'])
        except:
            raise RuntimeError('If NaN or Inf at iter 0, try lower lr. Otherwise please contact zhouyucong for a bug fix')

    def auto_denan_save(self):
        if self.last_save_iter < 100 or self.tmp.current_step - self.last_save_iter > 100:
            self.last_state_dict = {}
            self.last_optim_state_dict = {}
            # model state
            for k,v in self.model.state_dict().items():
                self.last_state_dict[k] = v.cpu()
            # optimizer state
            self.last_optim_state_dict['state'] = {k:{'momentum_buffer':v['momentum_buffer'].cpu()} for k,v in self.optimizer.state_dict()['state'].items()}
            #self.last_optim_state_dict['param_groups'] = copy.deepcopy(self.optimizer.state_dict()['param_groups']) # currently this is not needed

            self.last_save_iter = self.tmp.current_step

    def gather_result(self):
        tmp = self.tmp
        ginfo = self.ginfo

        vreduce(tmp.vloss, tmp.raw_loss.data, group=ginfo.group)
        vreduce(tmp.vtop1, tmp.top1, group=ginfo.group)

        vgather(tmp.loss_list, tmp.vloss.avg)
        vgather(tmp.top1_list, tmp.vtop1.avg)

        if self.auto_clip:
            vreduce(tmp.vbackbone_grad_norm, torch.Tensor([tmp.backbone_grad_norm/ginfo.task_size]).cuda(), group=ginfo.group)
            vgather(tmp.backbone_grad_norm_list, tmp.vbackbone_grad_norm.avg)
            vreduce(tmp.vneck_grad_norm, torch.Tensor([tmp.neck_grad_norm/ginfo.task_size]).cuda(), group=ginfo.group)
            vgather(tmp.neck_grad_norm_list, tmp.vneck_grad_norm.avg)
            vreduce(tmp.vdecoder_grad_norm, torch.Tensor([tmp.decoder_grad_norm/ginfo.task_size]).cuda(), group=ginfo.group)
            vgather(tmp.decoder_grad_norm_list, tmp.vdecoder_grad_norm.avg)

            vreduce(tmp.vbackbone_grad_thresh, torch.Tensor([tmp.backbone_grad_thresh/ginfo.task_size]).cuda(), group=ginfo.group)
            vgather(tmp.backbone_grad_thresh_list, tmp.vbackbone_grad_thresh.avg)
            vreduce(tmp.vneck_grad_thresh, torch.Tensor([tmp.neck_grad_thresh/ginfo.task_size]).cuda(), group=ginfo.group)
            vgather(tmp.neck_grad_thresh_list, tmp.vneck_grad_thresh.avg)
            vreduce(tmp.vdecoder_grad_thresh, torch.Tensor([tmp.decoder_grad_thresh/ginfo.task_size]).cuda(), group=ginfo.group)
            vgather(tmp.decoder_grad_thresh_list, tmp.vdecoder_grad_thresh.avg)

        elif self.manual_clip:
            if self.clip_grad_backbone > 0:
                vreduce(tmp.vbackbone_grad_norm, torch.Tensor([tmp.backbone_grad_norm/ginfo.task_size]).cuda(), group=ginfo.group)
                vgather(tmp.backbone_grad_norm_list, tmp.vbackbone_grad_norm.avg)
            if self.clip_grad_neck > 0:
                vreduce(tmp.vneck_grad_norm, torch.Tensor([tmp.neck_grad_norm/ginfo.task_size]).cuda(), group=ginfo.group)
                vgather(tmp.neck_grad_norm_list, tmp.vneck_grad_norm.avg)
            if self.clip_grad_decoder > 0:
                vreduce(tmp.vdecoder_grad_norm, torch.Tensor([tmp.decoder_grad_norm/ginfo.task_size]).cuda(), group=ginfo.group)
                vgather(tmp.decoder_grad_norm_list, tmp.vdecoder_grad_norm.avg)

    def play_with_grads(self):
        if self.clip_grad > 0:
            torch.nn.utils.clip_grad.clip_grad_norm_(self.model.parameters(), max_norm=self.clip_grad)

    def update(self):
        ginfo = self.ginfo
        tmp = self.tmp

        # reduce
        self.model.reduce_gradients()

        if self.clip_grad_backbone > 0:
            tmp.backbone_grad_norm = torch.nn.utils.clip_grad.clip_grad_norm_(\
                self.model.module.backbone_module.parameters(), \
                max_norm=self.clip_grad_backbone*(ginfo.task_size**0.5))
            is_inf = np.isinf(tmp.backbone_grad_norm)
            is_nan = np.isnan(tmp.backbone_grad_norm)
            if ginfo.task_rank == 0 and (is_inf or is_nan):
                self.logger.info('task{} {} backbone_grad_norm inf/nan {}/{}'.format(\
                    ginfo.task_id, ginfo.task_name, is_inf, is_nan))

        if self.clip_grad_neck > 0:
            tmp.neck_grad_norm = torch.nn.utils.clip_grad.clip_grad_norm_(\
                self.model.module.neck_module.parameters(), \
                max_norm=self.clip_grad_neck*(self.C.world_size**0.5))
            is_inf = np.isinf(tmp.neck_grad_norm)
            is_nan = np.isnan(tmp.neck_grad_norm)
            if ginfo.task_rank == 0 and (is_inf or is_nan):
                self.logger.info('task{} {} backbone_grad_norm inf/nan {}/{}'.format(\
                    ginfo.task_id, ginfo.task_name, is_inf, is_nan))

        if self.clip_grad_decoder > 0:
            tmp.decoder_grad_norm = torch.nn.utils.clip_grad.clip_grad_norm_(\
                self.model.module.decoder_module.parameters(), \
                max_norm=self.clip_grad_decoder*(self.C.world_size**0.5))
            is_inf = np.isinf(tmp.decoder_grad_norm)
            is_nan = np.isnan(tmp.decoder_grad_norm)
            if ginfo.task_rank == 0 and (is_inf or is_nan):
                self.logger.info('task{} {} backbone_grad_norm inf/nan {}/{}'.format(\
                    ginfo.task_id, ginfo.task_name, is_inf, is_nan))

        self.optimizer.step()

    def tb_logging(self):
        tmp = self.tmp
        ginfo = self.ginfo

        for tid,ii in enumerate(ginfo.task_root_ranks):
            self.tb_logger.add_scalar('loss_{}'.format(ginfo.task_names[tid]), tmp.loss_list[ii], tmp.current_step)
            self.tb_logger.add_scalar('top1_{}'.format(ginfo.task_names[tid]), tmp.top1_list[ii], tmp.current_step)

            if self.clip_grad_backbone > 0:
                self.tb_logger.add_scalar('backbone_grad_norm_{}'.format(ginfo.task_names[tid]), tmp.backbone_grad_norm_list[ii], tmp.current_step)
            if self.clip_grad_neck > 0:
                self.tb_logger.add_scalar('neck_grad_norm_{}'.format(ginfo.task_names[tid]), tmp.neck_grad_norm_list[ii], tmp.current_step)
            if self.clip_grad_decoder > 0:
                self.tb_logger.add_scalar('decoder_grad_norm_{}'.format(ginfo.task_names[tid]), tmp.decoder_grad_norm_list[ii], tmp.current_step)

        self.tb_logger.add_scalar('lr', tmp.current_lr, tmp.current_step)

    def logging(self):
        tmp = self.tmp
        config = self.config
        ginfo = self.ginfo

        vlosses = tmp.vlosses

        log_msg = '\t'.join([
            'Iter: [{0}/{1}] ',
            'task{task_id:<2}: {task_name}\t'
            'Time: {batch_time.avg:.3f} (ETA:{eta:.2f}h) ({data_time.avg:.3f}) ',
            'Loss: {loss.avg:.4f} '
            'Prec@1: {top1.avg:.3f} '
            'LR: {current_lr} '
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
        config = self.config
        tmp = self.tmp
        ginfo = self.ginfo
        if config.save_interval > 0 and (tmp.current_step+1) % 1000 == 0 and ginfo.task_rank == 0:
            save_state({
                'step': tmp.current_step+1,
                'backbone_args': config.get('backbone', None),
                'neck_args': config.get('neck', None),
                'decoder_args': config.get('decoder', None),
                'state_dict': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
            }, '{}/ckpt_task{}'.format(self.ckpt_path, ginfo.task_id), 'newest')
        if config.save_interval > 0 and (tmp.current_step+1) % config.save_interval == 0 and ginfo.task_rank == 0:
            save_state({
                'step': tmp.current_step+1,
                'backbone_args': config.get('backbone', None),
                'neck_args': config.get('neck', None),
                'decoder_args': config.get('decoder', None),
                'state_dict': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
            }, '{}/ckpt_task{}'.format(self.ckpt_path, ginfo.task_id), tmp.current_step+1)
        if config.save_interval > 0 and tmp.current_step+1 == len(self.loader) and ginfo.task_rank == 0:
            save_state({
                'step': tmp.current_step+1,
                'backbone_args': config.get('backbone', None),
                'neck_args': config.get('neck', None),
                'decoder_args': config.get('decoder', None),
                'state_dict': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
            }, '{}/ckpt_task{}'.format(self.ckpt_path, ginfo.task_id), 'final')

    def post_run(self):
        pass

    def run(self):
        config = self.config
        ginfo = self.ginfo
        tmp = self.tmp

        self.pre_run()

        end = time.time()

        load_flag = True

        for i, tmp.input in enumerate(self.loader):
            tmp.vdata_time.update(time.time() - end)
            self.prepare_data()
            # TODO currently a work around for gpu memory leak when recovering
            if load_flag:
                tmp.current_step = 0
                self.forward()
                self.model.module.decoder_module.ignore_this_iter = True
                self.backward()
                self.model.module.decoder_module.ignore_this_iter = False
                dist.barrier()
                #self.update()
                self.load(self.load_args)
                load_flag = False

            tmp.current_step = self.last_iter + i + 1
            self.lr_scheduler.step(tmp.current_step)
            tmp.current_lr = self.lr_scheduler.get_lr()[0]

            self.forward()
            self.backward()

            if self.autodenan:
                self.auto_denan()

            #self.play_with_grads()
            self.update()
            self.gather_result()

            tmp.vbatch_time.update(time.time() - end)
            end = time.time()

            if tmp.current_step % config.print_freq == 0 and ginfo.task_rank == 0:
                if ginfo.task_id == 0:
                    self.tb_logging()
                self.logging()

            self.save()

        self.post_run()
