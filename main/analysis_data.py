""" Visualize camera pitch distribution of various datasets """

import argparse
import torch
import torch.backends.cudnn as cudnn
from config import cfg
import os.path as osp

# ddp
import torch.distributed as dist
from common.utils.distribute_utils import (
    init_distributed_mode, is_main_process, set_seed
)
import torch.distributed as dist
from mmcv.runner import get_dist_info


def parse_args():
    parser = argparse.ArgumentParser()
    # parser.add_argument('--gpu', type=str, dest='gpu_ids')
    parser.add_argument('--num_gpus', type=int, dest='num_gpus')
    parser.add_argument('--master_port', type=int, dest='master_port')
    parser.add_argument('--exp_name', type=str, default='output/test')
    parser.add_argument('--config', type=str, default='./config/config_base.py')
    args = parser.parse_args()

    return args


def main():
    args = parse_args()
    config_path = osp.join('./config', args.config)
    cfg.get_config_fromfile(config_path)
    cfg.update_config(args.num_gpus, args.exp_name)

    cudnn.benchmark = True
    set_seed(2023)

    # ddp by default in this branch
    distributed, gpu_idx = init_distributed_mode(args.master_port)
    from base import Trainer
    trainer = Trainer(distributed, gpu_idx)

    # ddp
    if distributed:
        trainer.logger_info('### Set DDP ###')
        trainer.logger.info(f'Distributed: {distributed}, init done {gpu_idx}')
    else:
        raise Exception("DDP not setup properly")

    trainer.logger_info(f"Using {cfg.num_gpus} GPUs, batch size {cfg.train_batch_size} per GPU.")

    trainer._make_batch_generator()
    # trainer._make_model()
    setattr(trainer, 'start_epoch', 0)

    trainer.logger_info('### Set some hyper parameters ###')
    for k in cfg.__dict__:
        trainer.logger_info(f'set {k} to {cfg.__dict__[k]}')
        trainer.logger_info(f'train with train_3d={cfg.trainset_3d}')
        trainer.logger_info(f'train with train_2d={cfg.trainset_2d}')
        trainer.logger_info(f'train with trainset_humandata={cfg.trainset_humandata}')

    trainer.logger_info('### Start training ###')

    for epoch in range(trainer.start_epoch, cfg.end_epoch):

        # ddp, align random seed between devices
        trainer.batch_generator.sampler.set_epoch(epoch)

        for itr, (inputs, targets, meta_info) in enumerate(trainer.batch_generator):

            import json
            save_name = 'xxx.json'
            import pdb; pdb.set_trace()
            save_dir = '/mnt/cache/caizhongang/osx/output/data_analysis'
            global_orient = targets['smplx_pose'][:, 0:3].detach().numpy().tolist()
            with open(osp.join(save_dir, save_name), 'w') as f:
                json.dump(global_orient, f)
            exit(0)

        # save model ddp, save model.module on rank 0 only
        if is_main_process():
            trainer.save_model({
                'epoch': epoch,
                'network': trainer.model.state_dict(),
                'optimizer': trainer.optimizer.state_dict(),
            }, epoch)

        dist.barrier()


if __name__ == "__main__":
    main()