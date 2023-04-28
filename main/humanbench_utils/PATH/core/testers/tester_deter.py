import random

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from core import distributed_utils as dist


from .tester import Tester

class WorkerInit(object):
    def __init__(self, rank, num_workers):
        self.rank = rank
        self.num_workers = num_workers
    def func(self, pid):
        print(f'[rank{self.rank}] setting worker seed {self.rank*self.num_workers+pid}', flush=True)
        np.random.seed(self.rank*self.num_workers+pid)

class TesterDeter(Tester):

    def __init__(self, C_train, C_test):
        super().__init__(C_train, C_test)

        if self.config.get('deterministic', False):
            if self.config.get('cudnn_deterministic', True):
                cudnn.deterministic = True
                cudnn.benchmark = False
            else:
                cudnn.benchmark = True
            seed = self.config.get('random_seed', 0)
            worker_rank = self.config.get('worker_rank', False)
            if worker_rank:
                worker_init = WorkerInit(self.C_train.rank, self.config.workers)
            else:
                worker_init = WorkerInit(0, 0)
            self.worker_init_fn = worker_init.func
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            dist.barrier()
            if self.C_train.rank == 0:
                self.logger.info(f'deterministic mode, seed: {seed}, worker_rank: {worker_rank},\
                                   cudnn_deterministic: {self.config.get("cudnn_deterministic", True)}')
            dist.barrier()
        else:
            self.worker_init_fn = None
