import copy
import os
import random
import time

import core
import core.models.decoders as decoders
import core.models.backbones as backbones
import core.models.necks as necks
import core.data.datasets as datasets
from core.models.model_entry import model_entry
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from core import distributed_utils as dist

from core.distributed_utils import DistributedGivenIterationSampler

from torch.utils.data import DataLoader

from .solver import Solver

class WorkerInit(object):
    def __init__(self, rank, num_workers):
        self.rank = rank
        self.num_workers = num_workers
    def func(self, pid):
        print(f'[rank{self.rank}] setting worker seed {self.rank*self.num_workers+pid}', flush=True)
        np.random.seed(self.rank*self.num_workers+pid)

class SolverDeter(Solver):

    def __init__(self, C):
        super().__init__(C)

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

    def create_dataloader(self):
        config = self.config
        ginfo = self.ginfo

        self.sampler = DistributedGivenIterationSampler(
                        self.dataset, config.max_iter, config.sampler.batch_size,
                        world_size=ginfo.task_size, rank=ginfo.task_rank,
                        last_iter=self.last_iter,
                        shuffle_strategy=config.sampler.shuffle_strategy,
                        random_seed=ginfo.task_random_seed,
                        ret_save_path=config.sampler.get('ret_save_path', None))


        self.loader = DataLoader(self.dataset, batch_size=config.sampler.batch_size,
                            shuffle=False, num_workers=config.workers,
                            pin_memory=False, sampler=self.sampler, worker_init_fn=self.worker_init_fn)
