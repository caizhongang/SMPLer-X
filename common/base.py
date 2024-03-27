import os.path as osp
import math
import abc
from torch.utils.data import DataLoader
import torch.optim
import torchvision.transforms as transforms
from timer import Timer
from logger import colorlogger
from torch.nn.parallel.data_parallel import DataParallel
from config import cfg
from SMPLer_X import get_model

# ddp
import torch.distributed as dist
from torch.utils.data import DistributedSampler
import torch.utils.data.distributed
from utils.distribute_utils import (
    get_rank, is_main_process, time_synchronized, get_group_idx, get_process_groups
)


class Base(object):
    __metaclass__ = abc.ABCMeta

    def __init__(self, log_name='logs.txt'):
        self.cur_epoch = 0

        # timer
        self.tot_timer = Timer()
        self.gpu_timer = Timer()
        self.read_timer = Timer()

        # logger
        self.logger = colorlogger(cfg.log_dir, log_name=log_name)

    @abc.abstractmethod
    def _make_batch_generator(self):
        return

    @abc.abstractmethod
    def _make_model(self):
        return

class Demoer(Base):
    def __init__(self, test_epoch=None):
        if test_epoch is not None:
            self.test_epoch = int(test_epoch)
        super(Demoer, self).__init__(log_name='test_logs.txt')

    def _make_batch_generator(self, demo_scene):
        # data load and construct batch generator
        self.logger.info("Creating dataset...")
        from data.UBody.UBody import UBody
        testset_loader = UBody(transforms.ToTensor(), "demo", demo_scene) # eval(demoset)(transforms.ToTensor(), "demo")
        batch_generator = DataLoader(dataset=testset_loader, batch_size=cfg.num_gpus * cfg.test_batch_size,
                                     shuffle=False, num_workers=cfg.num_thread, pin_memory=True)

        self.testset = testset_loader
        self.batch_generator = batch_generator

    def _make_model(self):
        self.logger.info('Load checkpoint from {}'.format(cfg.pretrained_model_path))

        # prepare network
        self.logger.info("Creating graph...")
        model = get_model('test')
        model = DataParallel(model).to(cfg.device)
        ckpt = torch.load(cfg.pretrained_model_path, map_location=cfg.device)

        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in ckpt['network'].items():
            if 'module' not in k:
                k = 'module.' + k
            k = k.replace('module.backbone', 'module.encoder').replace('body_rotation_net', 'body_regressor').replace(
                'hand_rotation_net', 'hand_regressor')
            new_state_dict[k] = v
        model.load_state_dict(new_state_dict, strict=False)
        model.eval()

        self.model = model

    def _evaluate(self, outs, cur_sample_idx):
        eval_result = self.testset.evaluate(outs, cur_sample_idx)
        return eval_result

