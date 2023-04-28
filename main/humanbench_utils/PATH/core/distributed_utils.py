import os, subprocess
import torch
import torch.distributed as dist
from torch.distributed.distributed_c10d import _get_global_rank

import collections
from easydict import EasyDict as edict
from torch.nn import Module
from torch.utils.data.sampler import Sampler
import math
import numpy as np
import multiprocessing as mp
import copy
import random
from collections import defaultdict

from PATH.core.utils import named_buffers, sync_print, printlog

### for code compatability after removing dependence on internal package
new_group = dist.new_group
allreduce = dist.all_reduce
broadcast = dist.broadcast
barrier = dist.barrier
allgather = dist.all_gather

class DistModule(torch.nn.Module):
    def __init__(self, module, sync=False, task_grp=None, share_backbone_group=None, \
            share_neck_group=None, share_decoder_group=None, ignore_bcast=None, \
            task_weight=None, task_size=None):
        super(DistModule, self).__init__()
        self.module = module
        self.sync = sync
        self.task_grp = task_grp
        self.share_backbone_group = share_backbone_group
        self.share_neck_group = share_neck_group
        self.share_decoder_group = share_decoder_group
        self.task_weight = task_weight
        self.task_size = task_size

        if not hasattr(torch.nn.Module, 'named_buffers'):
            printlog('registering named_buffers for nn.Module at DistModule')
            torch.nn.Module.named_buffers = named_buffers

        broadcast_params_multitask(self, self.task_grp, self.share_backbone_group, \
                self.share_neck_group, self.share_decoder_group, ignore_bcast)

        assert sync, "Currently, only sync model is supported!"
        if not sync:
            self._grad_accs = {}
            self._reduce_hooks = {}
            self._register_hooks()

    def forward(self, *inputs, **kwargs):
        return self.module(*inputs, **kwargs)

    def train(self, mode=True):
        super(DistModule, self).train(mode)
        self.module.train(mode)

    def reduce_gradients(self, task_specific=False):
        if self.sync:
            if not task_specific:
                if self.task_grp is not None or self.share_backbone_group is not None \
                    or self.share_neck_group is not None or self.share_decoder_group is not None:
                    for name, param in self.named_parameters():
                        if param.grad is None: param.grad = param.data * 0
                        if param.task_specific and param.requires_grad:
                            allreduce(param.grad.data, group=self.task_grp)
                        elif param.backbone_specific and param.requires_grad:
                            allreduce(param.grad.data, group=self.share_backbone_group)
                        elif param.neck_specific and param.requires_grad:
                            allreduce(param.grad.data, group=self.share_neck_group)
                        elif param.decoder_specific and param.requires_grad:
                            allreduce(param.grad.data, group=self.share_decoder_group)
                        elif param.requires_grad:
                            allreduce(param.grad.data)
                else:
                    for param in self.parameters():
                        if param.requires_grad and param.grad is not None:
                            allreduce(param.grad.data)
            else:
                for name, param in self.named_parameters():
                    if param.requires_grad and param.grad is not None:
                        allreduce(param.grad.data, group=self.task_grp)


def broadcast_params_multitask(model, task_grp, share_backbone_group, share_neck_group, share_decoder_group, ignore):
    """ broadcast multi-task model parameters """
    if task_grp is not None or share_backbone_group is not None \
        or share_neck_group is not None or share_decoder_group is None:

        for name,p in model.named_parameters():
            if ignore and name in ignore:
                printlog('param {} ignored in broadcast'.format(name))
                continue
            assert p.task_specific + p.backbone_specific + p.neck_specific + p.decoder_specific <= 1.5, \
                "param could not be task_specific, backbone_specific, neck_specific, decoder_specific at same time"
            try:
                # import pdb; pdb.set_trace()
                if p.task_specific:
                    printlog(f'broadcasting task-specific param {name}\tgroup_idx={task_grp}')
                    broadcast(p, _get_global_rank(task_grp, 0), group=task_grp)
                elif p.backbone_specific:
                    printlog(f'broadcasting backbone-specific param {name}\tgroup_idx={share_backbone_group}')
                    broadcast(p, _get_global_rank(task_grp, 0), group=share_backbone_group)
                elif p.neck_specific:
                    printlog(f'broadcasting neck-specific param {name}\tgroup_idx={share_neck_group}')
                    broadcast(p, _get_global_rank(task_grp, 0), group=share_neck_group)
                elif p.decoder_specific:
                    printlog(f'broadcasting decoder-specific param {name}\tgroup_idx={share_decoder_group}')
                    broadcast(p, _get_global_rank(task_grp, 0), group=share_decoder_group)
                else:
                    printlog(f'broadcasting non-specific param {name}')
                    broadcast(p, 0)
            except:
                raise RuntimeError('param {} does not have task_specific or backbone_specific or neck_specific or decoder_specific'.format(name))

        for name,b in model.named_buffers():
            if ignore and name in ignore:
                printlog('buffer {} ignored in broadcast'.format(name))
                continue
            assert b.task_specific + b.backbone_specific + b.neck_specific + b.decoder_specific <= 1, \
                "buffer could not be task_specific, backbone_specific, neck_specific, decoder_specific at same time"
            try:
                if b.task_specific:
                    printlog('broadcasting task-specific buffer {}'.format(name))
                    broadcast(b, _get_global_rank(task_grp, 0), group=task_grp)
                elif b.backbone_specific:
                    printlog('broadcasting backbone-specific buffer {}'.format(name))
                    broadcast(b, _get_global_rank(task_grp, 0), group=share_backbone_group)
                elif b.neck_specific:
                    printlog('broadcasting neck-specific buffer {}'.format(name))
                    broadcast(b, _get_global_rank(task_grp, 0), group=share_neck_group)
                elif b.decoder_specific:
                    printlog('broadcasting decoder-specific buffer {}'.format(name))
                    broadcast(b, _get_global_rank(task_grp, 0), group=share_decoder_group)
                else:
                    broadcast(b, 0)
            except:
                raise RuntimeError('buffer {} does not have task_specific'.format(name, id(b)))
    else:
        for name,p in model.named_parameters():
            if ignore and name in ignore:
                printlog('param {} ignored in broadcast'.format(name))
                continue
            try:
                broadcast(p, 0)
            except:
                raise RuntimeError('param {} does not have task_specific'.format(name))

        for name,b in model.named_buffers():
            if ignore and name in ignore:
                printlog('buffer {} ignored in broadcast'.format(name))
                continue
            try:
                broadcast(b, 0)
            except:
                raise RuntimeError('buffer {} does not have task_specific'.format(name, id(b)))

def get_rank():
    return int(os.environ.get('SLURM_PROCID', 0))

def get_local_rank():
    return get_rank() % torch.cuda.device_count()

def get_world_size():
    return int(os.environ.get('SLURM_NTASKS', 1))

def dist_init(method='slurm', port='5671'):
    assert method == 'slurm'
    proc_id = int(os.environ['SLURM_PROCID'])
    num_gpus = torch.cuda.device_count()
    torch.cuda.set_device(proc_id % num_gpus)

    world_size = get_world_size()
    rank = get_rank()

    addr = subprocess.getoutput(
        "scontrol show hostname {} | head -n1".format(os.environ["SLURM_NODELIST"])
    )
    os.environ["MASTER_PORT"] = port
    os.environ["MASTER_ADDR"] = addr
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["RANK"] = str(rank)


    dist.init_process_group(
        backend="nccl",
        world_size=world_size,
        rank=rank,
    )

    return rank, world_size

class DistributedGivenIterationSampler(Sampler):
    def __init__(self, dataset, total_iter, batch_size, world_size=None, rank=None, last_iter=-1,
                shuffle_strategy=0, random_seed=0, imageNumPerClass=4, ret_save_path=None):
        if world_size is None:
            world_size = get_world_size()
        if rank is None:
            rank = get_rank()
        assert rank < world_size
        sync_print('sampler: rank={}, world_size={}, random_seed={}'.format(rank, world_size, random_seed))
        self.dataset = dataset
        self.total_iter = total_iter
        self.batch_size = batch_size
        self.world_size = world_size
        self.rank = rank
        self.last_iter = last_iter
        self.shuffle_strategy = shuffle_strategy
        self.random_seed = random_seed
        self.imageNumPerClass = imageNumPerClass
        self.ret_save_path = ret_save_path
        self.task_name = self.dataset.task_name

        self.total_size = self.total_iter*self.batch_size

        self.call = 0

        # generate indices
        if self.ret_save_path is not None:
            self.this_ret_path = os.path.join(self.ret_save_path, '_'.join([self.task_name, str(self.world_size), str(self.rank)]) + ".pth.tar")
            if os.path.exists(self.this_ret_path):
                ret_file = torch.load(self.this_ret_path)
                # ensure this task and task size is unchanged
                if ret_file['task_name'] == self.task_name and ret_file['task_size'] == self.world_size and ret_file['task_rank'] == self.rank:
                    printlog(" load task sampler from ------> {}".format(self.this_ret_path))
                    self.indices = ret_file['ret_file']
                    self.dataset.received_indices = True
                    return
            else:
                printlog("sampler file ({}) is not existed, and will be generated now--->".format(self.this_ret_path))

        if self.shuffle_strategy in [0,1,3,4,6]:
            self.indices = self.gen_new_list()
            self.dataset.indices = self.indices
            self.dataset.received_indices = True
        elif self.shuffle_strategy == 2:
            self.indices = self.gen_s2()
        elif self.shuffle_strategy == 5:
            self.indices = self.gen_s5()
        else:
            raise Error("Invalid shuffle_strategy!") # todo: undefined 'Error'???

        if self.ret_save_path is not None and not os.path.exists(self.ret_save_path):
            self.save()

    def gen_s2(self):

        np.random.seed(self.rank) # set different random seed

        indices = []
        labels = self.dataset.labels
        printlog('using shuffle strategy 2, initializing class map...')
        class2id = collections.OrderedDict()
        for i,l in enumerate(labels):
            if l in class2id:
                class2id[l].append(i)
            else:
                class2id[l] = [i]
        keys = list(class2id.keys())
        ## random shuffle class keys
        np.random.shuffle(keys)
        num_class = len(keys)
        printlog('class map done.')

        for i in range((self.last_iter+1)*self.batch_size, self.total_size):
            class_id = np.random.randint(0, num_class)
            this_num = len(class2id[keys[class_id]])
            inner_id = np.random.randint(0, this_num)
            # yield class2id[keys[class_id]][inner_id]
            indices.append(class2id[keys[class_id]][inner_id])

        return indices

    def gen_s5(self):
        np.random.seed(self.rank) # set different random seed

        indices = []
        labels = self.dataset.labels
        printlog('using shuffle strategy 5, initializing class map...')
        class2id = collections.OrderedDict()
        for i,l in enumerate(labels):
            if l in class2id:
                class2id[l].append(i)
            else:
                class2id[l] = [i]
        keys = list(class2id.keys())
        ## random shuffle class keys
        np.random.shuffle(keys)
        num_class = len(keys)
        printlog('class map done.')
        printlog('{} class with {} samples in a batch!'.format(self.batch_size//self.imageNumPerClass, self.imageNumPerClass))

        for i in range((self.last_iter+1)*self.batch_size, self.total_size):
            if i % self.imageNumPerClass == 0:
                class_id = np.random.randint(0, num_class)
                this_num = len(class2id[keys[class_id]])
            inner_id = np.random.randint(0, this_num)
            # yield class2id[keys[class_id]][inner_id]
            indices.append(class2id[keys[class_id]][inner_id])

        return indices

    def __iter__(self):
        if self.call == 0:
            self.call = 1
            return iter(self.indices)
            # if self.shuffle_strategy in [2,5]:
            #     return self.indices
            # else:
            #     return iter(self.indices)
        else:
            raise RuntimeError("this sampler is not designed to be called more than once!!")

    def gen_new_list(self):

        # each process shuffle independently
        if self.shuffle_strategy == 0:

            np.random.seed(self.rank)

            indices = np.arange(len(self.dataset))
            indices = indices[:self.total_size]
            num_repeat = (self.total_size-1) // indices.shape[0] + 1
            indices = np.tile(indices, num_repeat)
            indices = indices[:self.total_size]

            for beg in range(0, self.total_size, len(self.dataset)):
                end = min(beg+len(self.dataset), self.total_size)
                np.random.shuffle(indices[beg:end])

        # each process shuffle all list with same seed, and pick one piece according to rank
        elif self.shuffle_strategy == 1:

            np.random.seed(self.random_seed)

            all_size = self.total_size * self.world_size
            indices = np.arange(len(self.dataset))
            indices = indices[:all_size]
            num_repeat = (all_size-1) // indices.shape[0] + 1
            indices = np.tile(indices, num_repeat)
            indices = indices[:all_size]

            np.random.shuffle(indices)
            beg = self.total_size * self.rank
            indices = indices[beg:beg+self.total_size]

        elif self.shuffle_strategy == 3:

            np.random.seed(0)

            all_size = self.total_size * self.world_size

            ## make class map
            labels = self.dataset.labels
            class2id = collections.OrderedDict()
            for i,l in enumerate(labels):
                if l in class2id:
                    class2id[l].append(i)
                else:
                    class2id[l] = [i]
            ## calc mean_num
            class_count = [len(x) for _,x in class2id.items()]
            mean_num = int(np.mean(class_count))
            ## fill class to mean_num
            indices = []
            for _,v in class2id.items():
                if len(v) < mean_num:
                    lack_num = mean_num - len(v)
                    indices.extend(np.random.choice(v, lack_num))
                indices.extend(v)
            indices = np.array(indices)
            indices = indices[:all_size]
            printlog('using strategy 3, mean_num: {}, origin_len: {}, balanced_len: {}'.format(mean_num, len(self.dataset), len(indices)))

            num_repeat = (all_size-1) // indices.shape[0] + 1
            indices = np.tile(indices, num_repeat)
            indices = indices[:all_size]

            np.random.shuffle(indices)
            beg = self.total_size * self.rank
            indices = indices[beg:beg+self.total_size]

        elif self.shuffle_strategy == 6:

            np.random.seed(self.random_seed)

            labels = self.dataset.labels
            printlog('using shuffle strategy 6, initializing class map...')
            class2id = collections.defaultdict(list)
            for i,l in enumerate(labels):
                class2id[l].append(i)

            mini_indices = []
            for pid, idxs in class2id.items():
                if len(idxs) < self.imageNumPerClass:
                    idxs = idxs + list(np.random.choice(idxs, size=self.imageNumPerClass-len(idxs), replace=True))
                elif len(idxs) % self.imageNumPerClass != 0:
                    add_num = int(len(idxs) // self.imageNumPerClass + 1) * self.imageNumPerClass - len(idxs)
                    idxs = idxs + list(np.random.choice(idxs, size=add_num, replace=True))

                assert len(idxs) % self.imageNumPerClass == 0
                mini_indices.extend([idxs[i:i+self.imageNumPerClass] for i in range(0, len(idxs), self.imageNumPerClass)])

            np.random.shuffle(mini_indices)
            indices = np.array(mini_indices).reshape(-1)

            all_size = self.total_size * self.world_size
            indices = indices[:all_size]
            num_repeat = (all_size-1) // indices.shape[0] + 1
            indices = np.tile(indices, num_repeat)
            indices = indices[:all_size]

            beg = self.total_size * self.rank
            indices = indices[beg:beg+self.total_size]
        elif self.shuffle_strategy == 7:

            np.random.seed(self.random_seed)

            labels = self.dataset.labels
            printlog('using shuffle strategy 7, initializing class map...')
            class2id = collections.defaultdict(list)
            for i,l in enumerate(labels):
                class2id[l].append(i)

            mini_indices = []
            for pid, idxs in class2id.items():
                if len(idxs) < self.imageNumPerClass:
                    idxs = idxs + list(np.random.choice(idxs, size=self.imageNumPerClass-len(idxs), replace=True))
                elif len(idxs) % self.imageNumPerClass != 0:
                    add_num = int(len(idxs) // self.imageNumPerClass + 1) * self.imageNumPerClass - len(idxs)
                    idxs = idxs + list(np.random.choice(idxs, size=add_num, replace=True))

                assert len(idxs) % self.imageNumPerClass == 0
                mini_indices.extend([idxs[i:i+self.imageNumPerClass] for i in range(0, len(idxs), self.imageNumPerClass)])

            np.random.shuffle(mini_indices)
            indices = np.array(mini_indices).reshape(-1)

            all_size = self.total_size * self.world_size
            indices = indices[:all_size]
            num_repeat = (all_size-1) // indices.shape[0] + 1

            for repeat in range(num_repeat)-1:
                for pid, idxs in class2id.items():
                    if len(idxs) < self.imageNumPerClass:
                        idxs = idxs + list(np.random.choice(idxs, size=self.imageNumPerClass-len(idxs), replace=True))
                    elif len(idxs) % self.imageNumPerClass != 0:
                        add_num = int(len(idxs) // self.imageNumPerClass + 1) * self.imageNumPerClass - len(idxs)
                        idxs = idxs + list(np.random.choice(idxs, size=add_num, replace=True))

                    assert len(idxs) % self.imageNumPerClass == 0
                    mini_indices.extend([idxs[i:i+self.imageNumPerClass] for i in range(0, len(idxs), self.imageNumPerClass)])

            np.random.shuffle(mini_indices)
            indices = np.array(mini_indices).reshape(-1)

            all_size = self.total_size * self.world_size
            indices = indices[:all_size]

            beg = self.total_size * self.rank
            indices = indices[beg:beg+self.total_size]
        elif self.shuffle_strategy == 8:

            np.random.seed(self.random_seed)

            labels = self.dataset.labels
            print('using shuffle strategy 8, initializing class map...')
            class2id = collections.defaultdict(list)
            for i,l in enumerate(labels):
                class2id[l].append(i)

            pids = set()
            cls_pids_idxs = collections.defaultdict(list)
            for pid, idxs in class2id.items():
                if len(idxs) < self.imageNumPerClass:
                    idxs = idxs + list(np.random.choice(idxs, size=self.imageNumPerClass-len(idxs), replace=True))
                elif len(idxs) % self.imageNumPerClass != 0:
                    add_num = int(len(idxs) // self.imageNumPerClass + 1) * self.imageNumPerClass - len(idxs)
                    idxs = idxs + list(np.random.choice(idxs, size=add_num, replace=True))

                assert len(idxs) % self.imageNumPerClass == 0
                np.random.shuffle(idxs)
                idxs = [idxs[i:(i+self.imageNumPerClass)] for i in range(0, len(idxs), self.imageNumPerClass)]
                ptr = 0
                values = [ptr, idxs]
                cls_pids_idxs[pid] = values
                pids.add(pid)

            indices = []
            classnum_per_batch = self.batch_size // self.imageNumPerClass
            while len(pids) >= classnum_per_batch:
                batch = []
                sub_pids = []
                for i, pid in enumerate(pids):
                    if i >= classnum_per_batch:
                        break
                    sub_pids.append(pid)

                for pid in sub_pids:
                    ptr = cls_pids_idxs[pid][0]
                    idxs = cls_pids_idxs[pid][1]
                    batch.extend(idxs[ptr])
                    if ptr + 1 >= len(idxs):
                        pids.remove(pid)
                    else:
                        cls_pids_idxs[pid][0] += 1
                indices.append(batch)

            np.random.shuffle(indices)
            indices = np.array(indices).reshape(-1)

            all_size = self.total_size * self.world_size
            indices = indices[:all_size]
            num_repeat = (all_size-1) // indices.shape[0] + 1
            indices = np.tile(indices, num_repeat)
            indices = indices[:all_size]

            beg = self.total_size * self.rank
            indices = indices[beg:beg+self.total_size]
        else:
            raise RuntimeError('unknow shuffle strategy')

        assert len(indices) == self.total_size

        return indices[(self.last_iter+1)*self.batch_size:]

    def __len__(self):
        return self.total_size - (self.last_iter+1)*self.batch_size

    def save(self):
        torch.save({'task_name': self.task_name,
                    'task_size': self.world_size,
                    'task_rank': self.rank,
                    'ret_file': self.indices}, self.this_ret_path)
        printlog("save sampler file  ------> {}".format(self.this_ret_path))


class RandomIdentitySampler(Sampler):
    """
    Randomly sample N identities, then for each identity,
    randomly sample K instances, therefore batch size is N*K.
    Args:
    - data_source (list): list of (img_path, pid, camid).
    - imageNumPerClass (int): number of instances per identity in a batch.
    - batch_size (int): number of examples in a batch.
    """
    def __init__(self, dataset, total_iter, batch_size, world_size=None, rank=None, last_iter=-1,
                shuffle_strategy=0, random_seed=0, imageNumPerClass=4, ret_save_path=None):
        self.batch_size = batch_size
        self.world_size = world_size if world_size is not None else get_world_size()
        self.num_instances = imageNumPerClass
        self.num_pids_per_batch = self.batch_size // self.num_instances
        self.index_dic = defaultdict(list)
        self.random_seed = random_seed
        self.total_iter = total_iter
        self.total_size = self.total_iter*self.batch_size
        self.last_iter = last_iter
        self.dataset = dataset

        labels = self.dataset.labels
        printlog('using RandomIdentityBatchSampler, initializing class map...')
        self.index_dic = collections.defaultdict(list)
        for i,l in enumerate(labels):
            self.index_dic[l].append(i)

        self.pids = np.array(list(self.index_dic.keys()))
        self.rank = rank if rank is not None else get_rank()

    def __iter__(self):
        np.random.seed(self.random_seed)
        self._seed = int(self.random_seed)
        final_idxs = self.sample_list()
        length = int(math.ceil(len(final_idxs) * 1.0 / self.world_size))
        #final_idxs = final_idxs[self.rank * length:(self.rank + 1) * length]
        final_idxs = self.__fetch_current_node_idxs(final_idxs, length)
        return iter(final_idxs)


    def __fetch_current_node_idxs(self, final_idxs, length):
        total_num = len(final_idxs)
        block_num = (length // self.batch_size)
        index_target = []
        for i in range(0, block_num * self.world_size, self.world_size):
            index = range(self.batch_size * self.rank + self.batch_size * i, min(self.batch_size * self.rank + self.batch_size * (i+1), total_num))
            index_target.extend(index)
        index_target_npy = np.array(index_target)
        final_idxs = list(np.array(final_idxs)[index_target_npy])
        return final_idxs


    def batch_sample_list(self):
        #np.random.seed(self._seed)
        avai_pids = copy.deepcopy(self.pids.tolist())
        batch_idxs_dict = {}
        batch_indices = []
        while len(avai_pids) >= self.num_pids_per_batch:
            selected_pids = np.random.choice(avai_pids, self.num_pids_per_batch, replace=False).tolist()
            for pid in selected_pids:
                if pid not in batch_idxs_dict or len(batch_idxs_dict[pid]) < self.num_instances:
                    idxs = copy.deepcopy(self.index_dic[pid])
                    if len(idxs) < self.num_instances:
                        idxs = np.random.choice(idxs, size=self.num_instances, replace=True).tolist()
                    np.random.shuffle(idxs)
                    batch_idxs_dict[pid] = idxs

                avai_idxs = batch_idxs_dict[pid]
                for _ in range(self.num_instances):
                    batch_indices.append(avai_idxs.pop(0))

                if len(avai_idxs) < self.num_instances: avai_pids.remove(pid)

        return batch_indices

    def sample_list(self):
        #np.random.seed(self._seed)
        all_size = self.total_size * self.world_size

        all_indices = list()

        while len(all_indices) <= all_size:
            all_indices.extend(self.batch_sample_list())

        return all_indices[:all_size]

    def __len__(self):
        return self.total_size - (self.last_iter+1)*self.batch_size


class RandomIdentityBatchSampler(Sampler):
    """
    Randomly sample N identities, then for each identity,
    randomly sample K instances, therefore batch size is N*K.
    Args:
    - data_source (list): list of (img_path, pid, camid).
    - imageNumPerClass (int): number of instances per identity in a batch.
    - batch_size (int): number of examples in a batch.
    """
    def __init__(self, dataset, total_iter, batch_size, world_size=None, rank=None, last_iter=-1,
                shuffle_strategy=0, random_seed=0, imageNumPerClass=4, ret_save_path=None):
        self.batch_size = batch_size
        self.world_size = world_size if world_size is not None else get_world_size()
        self.num_instances = imageNumPerClass
        self.num_pids_per_batch = self.batch_size // self.num_instances
        self.index_dic = defaultdict(list)
        self.random_seed = random_seed
        self.total_iter = total_iter
        self.total_size = self.total_iter*self.batch_size
        self.last_iter = last_iter
        self.dataset = dataset

        labels = self.dataset.labels
        printlog('using RandomIdentityBatchSampler, initializing class map...')
        self.index_dic = collections.defaultdict(list)
        for i,l in enumerate(labels):
            self.index_dic[l].append(i)

        self.pids = np.array(list(self.index_dic.keys()))
        self.rank = rank if rank is not None else get_rank()

    def __iter__(self):
        np.random.seed(self.random_seed)
        self._seed = int(self.random_seed)
        final_idxs_batches = self.sample_list()
        final_idxs = self.__fetch_current_node_idxs(final_idxs_batches)
        return iter(final_idxs)


    def __fetch_current_node_idxs(self, final_idxs_batches):
        res = []
        for final_idxs in final_idxs_batches:
            total_num = len(final_idxs)
            length = int(math.ceil(len(final_idxs) * 1.0 / self.world_size))
            block_num = (length // self.batch_size)
            index_target = []
            for i in range(0, block_num * self.world_size, self.world_size):
                index = range(self.batch_size * self.rank + self.batch_size * i, min(self.batch_size * self.rank + self.batch_size * (i+1), total_num))
                index_target.extend(index)
            index_target_npy = np.array(index_target)
            final_idxs = list(np.array(final_idxs)[index_target_npy])
            res.extend(final_idxs)
        res = res[:self.total_size]
        return res


    def batch_sample_list(self):
        #np.random.seed(self._seed)
        avai_pids = copy.deepcopy(self.pids.tolist())
        batch_idxs_dict = {}
        batch_indices = []
        while len(avai_pids) >= self.num_pids_per_batch:
            selected_pids = np.random.choice(avai_pids, self.num_pids_per_batch, replace=False).tolist()
            for pid in selected_pids:
                if pid not in batch_idxs_dict or len(batch_idxs_dict[pid]) < self.num_instances:
                    idxs = copy.deepcopy(self.index_dic[pid])
                    if len(idxs) < self.num_instances:
                        idxs = np.random.choice(idxs, size=self.num_instances, replace=True).tolist()
                    np.random.shuffle(idxs)
                    batch_idxs_dict[pid] = idxs

                avai_idxs = batch_idxs_dict[pid]
                for _ in range(self.num_instances):
                    batch_indices.append(avai_idxs.pop(0))

                if len(avai_idxs) < self.num_instances: avai_pids.remove(pid)

        return batch_indices

    def sample_list(self):
        #np.random.seed(self._seed)
        all_size = self.total_size * self.world_size

        all_indices = list()
        all_indices_batch = list()

        while len(all_indices) <= all_size:
            all_indices.extend(self.batch_sample_list())
            all_indices_batch.append(self.batch_sample_list())

        return all_indices_batch

    def __len__(self):
        return self.total_size - (self.last_iter+1)*self.batch_size


class RandomIdentityEpochBatchSampler(Sampler):
    """
    Randomly sample N identities, then for each identity,
    randomly sample K instances, therefore batch size is N*K.
    Args:
    - data_source (list): list of (img_path, pid, camid).
    - imageNumPerClass (int): number of instances per identity in a batch.
    - batch_size (int): number of examples in a batch.
    """
    def __init__(self, dataset, total_iter, batch_size, world_size=None, rank=None, last_iter=-1,
                shuffle_strategy=0, random_seed=0, imageNumPerClass=4, ret_save_path=None):
        self.batch_size = batch_size
        self.world_size = world_size if world_size is not None else get_world_size()
        self.num_instances = imageNumPerClass
        self.num_pids_per_batch = self.batch_size // self.num_instances
        self.index_dic = defaultdict(list)
        self.random_seed = random_seed
        self.total_iter = total_iter
        self.total_size = self.total_iter*self.batch_size
        self.last_iter = last_iter
        self.dataset = dataset

        labels = self.dataset.labels
        printlog('using RandomIdentityBatchSampler, initializing class map...')
        self.index_dic = collections.defaultdict(list)
        for i,l in enumerate(labels):
            self.index_dic[l].append(i)

        self.pids = np.array(list(self.index_dic.keys()))
        self.rank = rank if rank is not None else get_rank()

        # estimate number of examples in an epoch
        self.length = 0
        for pid in self.pids:
            idxs = self.index_dic[pid]
            num = len(idxs)
            if num < self.num_instances:
                num = self.num_instances
            self.length += num - num % self.num_instances
        self.length //= self.world_size

    def __iter__(self):
        np.random.seed(self.random_seed)
        self._seed = int(self.random_seed)
        final_idxs_batches = self.sample_list()
        length = int(math.ceil(len(final_idxs_batches) * 1.0 / self.world_size))
        final_idxs = self.__fetch_current_node_idxs(final_idxs_batches, length)
        self.length = len(final_idxs)
        return iter(final_idxs)

    def set_epoch(self, epoch):
        self.random_seed = self.random_seed + epoch

    def __fetch_current_node_idxs(self, final_idxs, length):
        total_num = len(final_idxs)
        block_num = (length // self.batch_size)
        index_target = []
        for i in range(0, block_num * self.world_size, self.world_size):
            index = range(self.batch_size * self.rank + self.batch_size * i, min(self.batch_size * self.rank + self.batch_size * (i+1), total_num))
            index_target.extend(index)
        index_target_npy = np.array(index_target)
        final_idxs = list(np.array(final_idxs)[index_target_npy])
        return final_idxs

    def sample_list(self):
        #np.random.seed(self._seed)
        avai_pids = copy.deepcopy(self.pids.tolist())
        batch_idxs_dict = {}
        batch_indices = []
        while len(avai_pids) >= self.num_pids_per_batch:
            selected_pids = np.random.choice(avai_pids, self.num_pids_per_batch, replace=False).tolist()
            for pid in selected_pids:
                if pid not in batch_idxs_dict or len(batch_idxs_dict[pid]) < self.num_instances:
                    idxs = copy.deepcopy(self.index_dic[pid])
                    if len(idxs) < self.num_instances:
                        idxs = np.random.choice(idxs, size=self.num_instances, replace=True).tolist()
                    np.random.shuffle(idxs)
                    batch_idxs_dict[pid] = idxs

                avai_idxs = batch_idxs_dict[pid]
                for _ in range(self.num_instances):
                    batch_indices.append(avai_idxs.pop(0))

                if len(avai_idxs) < self.num_instances: avai_pids.remove(pid)

        return batch_indices

    def __len__(self):
        return self.length


class DistributedGivenSizeSampler(Sampler):
    def __init__(self, dataset, given_size=None, dup_shuffle=False, world_size=None, rank=None):
        if world_size is None:
            world_size = get_world_size()
        if rank is None:
            rank = get_rank()
        assert rank < world_size
        self.dataset = dataset
        self.dup_shuffle = dup_shuffle
        self.world_size = world_size
        self.rank = rank
        self.epoch = 0
        if given_size is None:
            self.num_samples = int(math.ceil(len(self.dataset) * 1.0 / self.world_size))
        else:
            self.num_samples = int(math.ceil(given_size * 1.0 / self.world_size))
        self.total_size = self.num_samples * self.world_size

        if self.dup_shuffle:
            self.offset = 0
            self.g = torch.Generator()
            #self.g.manual_seed(self.rank)
            self.indices = self.gen_new_list(self.g)

    def __iter__(self):
        if self.dup_shuffle:
            if self.offset == self.world_size:
                self.indices = self.gen_new_list(self.g)
                self.offset = 0
            beg = self.offset*self.num_samples
            indices = self.indices[beg:beg+self.num_samples]
            self.offset += 1
        else:
            # deterministically shuffle based on epoch
            g = torch.Generator()
            g.manual_seed(self.epoch)
            indices = self.gen_new_list(g)
            #origin_indices = list(torch.randperm(len(self.dataset), generator=g))
            #indices = origin_indices[:]

            ## add extra samples to meet self.total_size
            #indices = indices[:self.total_size] # if total_size < len(indices)
            #extra = self.total_size - len(origin_indices)
            #while self.total_size - len(indices) > 0:
            #    intake = self.total_size - len(indices)
            #    indices += origin_indices[:intake]
            #assert len(indices) == self.total_size

            # subsample
            offset = self.num_samples * self.rank
            indices = indices[offset:offset + self.num_samples]

        assert len(indices) == self.num_samples
        return iter(indices)

    def gen_new_list(self, g):
        origin_indices = list(torch.randperm(len(self.dataset), generator=g))
        indices = origin_indices[:]

        # add extra samples to meet self.total_size
        indices = indices[:self.total_size] # if total_size < len(indices)
        extra = self.total_size - len(origin_indices)
        while self.total_size - len(indices) > 0:
            intake = self.total_size - len(indices)
            indices += origin_indices[:intake]
        assert len(indices) == self.total_size
        return indices

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch

class DistributedGroupSampler(Sampler):
    """Sampler that restricts data loading to a subset of the dataset.

    It is especially useful in conjunction with
    :class:`torch.nn.parallel.DistributedDataParallel`. In such case, each
    process can pass a DistributedSampler instance as a DataLoader sampler,
    and load a subset of the original dataset that is exclusive to it.

    .. note::
        Dataset is assumed to be of constant size.

    Arguments:
        dataset: Dataset used for sampling.
        num_replicas (optional): Number of processes participating in
            distributed training.
        rank (optional): Rank of the current process within num_replicas.
        seed (int, optional): random seed used to shuffle the sampler if
            ``shuffle=True``. This number should be identical across all
            processes in the distributed group. Default: 0.
    """

    def __init__(self,
                 dataset,
                 batch_size=1,
                 num_replicas=None,
                 world_size=None,
                 rank=None,
                 total_iter=-1,
                 random_seed=0,
                 last_iter=-1):

        _rank, _num_replicas = rank, world_size
        if num_replicas is None:
            num_replicas = _num_replicas
        if rank is None:
            rank = _rank
        self.dataset = dataset
        self.samples_per_gpu = batch_size
        self.num_replicas = num_replicas
        self.rank = rank
        self.seed = random_seed if random_seed is not None else 0
        self.total_iter = total_iter
        self.batch_size = batch_size

        assert hasattr(self.dataset, 'flag')
        dataset_flag = self.dataset.flag
        self.flag = np.tile(dataset_flag, (total_iter - 1) * batch_size * self.num_replicas // len(self.dataset) + 1)
        self.flag = self.flag[:total_iter * batch_size * self.num_replicas]

        self.group_sizes = np.bincount(self.flag)

        self.num_samples = 0
        for i, j in enumerate(self.group_sizes):
            self.num_samples += int(
                math.ceil(self.group_sizes[i] * 1.0 / self.samples_per_gpu /
                          self.num_replicas)) * self.samples_per_gpu

        self.total_size = total_iter * batch_size
        self.last_iter = last_iter

    def __iter__(self):
        # deterministically shuffle based on epoch
        g = torch.Generator()
        g.manual_seed(self.seed)
        indices = []
        for i, size in enumerate(self.group_sizes):
            if size > 0:
                indice = np.where(self.flag == i)[0]
                assert len(indice) == size
                # add .numpy() to avoid bug when selecting indice in parrots.
                # TODO: check whether torch.randperm() can be replaced by
                # numpy.random.permutation().
                indice = indice[list(
                    torch.randperm(int(size), generator=g).numpy())].tolist()
                extra = int(
                    math.ceil(
                        size * 1.0 / self.samples_per_gpu / self.num_replicas)
                ) * self.samples_per_gpu * self.num_replicas - len(indice)
                # pad indice
                tmp = indice.copy()
                for _ in range(extra // size):
                    indice.extend(tmp)
                indice.extend(tmp[:extra % size])
                indices.extend(indice)

        assert len(indices) == self.num_samples * self.num_replicas

        indices = [
            indices[j] for i in list(
                torch.randperm(
                    len(indices) // self.samples_per_gpu, generator=g))
            for j in range(i * self.samples_per_gpu, (i + 1) *
                           self.samples_per_gpu)
        ]

        indices = [i % len(self.dataset.flag) for i in indices]
        indices = indices[:self.total_size*self.num_replicas]

        # subsample
        piece = (len(indices) - 1) // self.num_replicas + 1
        offset = piece * self.rank
        indices = indices[offset:offset + piece]

        return iter(indices[(self.last_iter+1)*self.batch_size:])

    def __len__(self):
        return self.total_size - (self.last_iter+1)*self.batch_size


class BatchSampler(Sampler):
    r"""Wraps another sampler to yield a mini-batch of indices.

    Args:
        sampler (Sampler or Iterable): Base sampler. Can be any iterable object
        batch_size (int): Size of mini-batch.
        drop_last (bool): If ``True``, the sampler will drop the last batch if
            its size would be less than ``batch_size``

    Example:
        >>> list(BatchSampler(SequentialSampler(range(10)), batch_size=3, drop_last=False))
        [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9]]
        >>> list(BatchSampler(SequentialSampler(range(10)), batch_size=3, drop_last=True))
        [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
    """

    def __init__(self, sampler, batch_size, drop_last):
        # Since collections.abc.Iterable does not check for `__getitem__`, which
        # is one way for an object to be an iterable, we don't do an `isinstance`
        # check here.
        if not isinstance(batch_size, int) or isinstance(batch_size, bool) or \
                batch_size <= 0:
            raise ValueError("batch_size should be a positive integer value, "
                             "but got batch_size={}".format(batch_size))
        if not isinstance(drop_last, bool):
            raise ValueError("drop_last should be a boolean value, but got "
                             "drop_last={}".format(drop_last))
        self.sampler = sampler
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.flag = self.sampler.flag

    def __iter__(self):
        ret = []
        batch = []
        #from IPython import embed;embed()
        for idx in self.sampler:
            batch.append(idx)
            if len(batch) == self.batch_size:
                batch = tuple(batch)
                for i in batch:
                    if self.flag[i]!=self.flag[batch[0]]:
                        import pdb; pdb.set_trace()
                        # from IPython import embed;embed()
                ret.extend(batch)
                batch = []
        if len(batch) > 0 and not self.drop_last:
            ret.extend(batch)

        return iter(ret)

    def __len__(self):
        # Can only be called if self.sampler has __len__ implemented
        # We cannot enforce this condition, so we turn off typechecking for the
        # implementation below.
        # Somewhat related: see NOTE [ Lack of Default `__len__` in Python Abstract Base Classes ]
        if self.drop_last:
            return len(self.sampler) // self.batch_size
        else:
            return (len(self.sampler) + self.batch_size - 1) // self.batch_size

class DistributedSequentialSampler(Sampler):
    def __init__(self, dataset, world_size=None, rank=None):
        if world_size == None:
            world_size = get_world_size()
        if rank == None:
            rank = get_rank()
        self.dataset = dataset
        self.world_size = world_size
        self.rank = rank
        assert len(self.dataset) >= self.world_size, f'{len(self.dataset)} vs {self.world_size}'
        sub_num = int(math.ceil(len(self.dataset) * 1.0 / self.world_size))
        self.beg = sub_num * self.rank
        self.end = min(self.beg+sub_num, len(self.dataset))

    def __iter__(self):
        indices = list(range(self.beg, self.end))
        return iter(indices)

    def __len__(self):
        return self.end - self.beg

def simple_group_split(world_size, rank, num_groups):
    groups = []
    rank_list = np.split(np.arange(world_size), num_groups)
    rank_list = [list(map(int, x)) for x in rank_list]
    for i in range(num_groups):
        groups.append(new_group(ranks=rank_list[i]))
    group_size = world_size // num_groups
    return groups[rank//group_size]

def specific_group_split(world_size, rank, group_spec):
    ## sanity check
    assert type(group_spec) is list
    assert all(map(lambda x: type(x) is int, group_spec))
    num_groups = len(group_spec)
    splits = np.sum(group_spec)
    assert world_size % splits == 0
    unit = int(world_size / splits)
    ## split
    group_spec = [x*unit for x in group_spec]
    groups = []
    roots = []
    last = 0
    group_info = edict()
    for i,gs in enumerate(group_spec):
        ranks = list(map(int, np.arange(last, last+gs)))
        groups.append(new_group(ranks=ranks))
        roots.append(last)
        if rank in ranks:
            group_info.group = groups[-1]
            group_info.task_size = gs
            group_info.task_id = i
            group_info.task_sub_id = rank - last
            group_info.task_root = last
        last += gs
    group_info.task_roots = roots
    group_info.num_groups = num_groups
    return group_info

def vreduce(x, tensor, group=None):
    y = tensor.clone()
    if group is not None:
        allreduce(y, group=group)
    else:
        allreduce(y)
    x.update(y.item())

def vgather(x_list, x):
    allgather(x_list, torch.Tensor([x]).cuda())

def reduce_dict(input_dict, task_size, task_rank, group=None, average=True):
    """
    Args:
        input_dict (dict): all the values will be reduced
        average (bool): whether to do average or sum
    Reduce the values in the dictionary from all processes so that all processes
    have the averaged results. Returns a dict with the same fields as
    input_dict, after reduction.
    """
    world_size = task_size
    if world_size < 2:
        return input_dict
    with torch.no_grad():
        names = []
        values = []
        # sort the keys so that they are consistent across processes
        for k in sorted(input_dict.keys()):
            names.append(k)
            values.append(input_dict[k])
        values = torch.stack(values, dim=0)
        allreduce(values, group=group)
        if average:
            values /= world_size
        reduced_dict = {k: v for k, v in zip(names, values)}
    return reduced_dict
