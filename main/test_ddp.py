import argparse
import torch
import os
# ddp
from common.utils.distribute_utils import (
    init_distributed_mode, is_main_process, cleanup
)
import torch.distributed as dist

 # ddp
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--device',
        default='cuda',
        help='device to use for training / testing')
    parser.add_argument(
        '--world_size',
        default=1,
        type=int,
        help='number of distributed processes')
    parser.add_argument(
        '--dist_url',
        default='env://',
        help='url used to set up distributed training')

    args = parser.parse_args()
    return args

def main():
    print('### Argument parse and create log ###')
    args = parse_args()
    import pdb; pdb.set_trace()
    dist.init_process_group('nccl', init_method='env://')
    rank = dist.get_rank()
    local_rank = os.environ['LOCAL_RANK']
    master_addr = os.environ['MASTER_ADDR']
    master_port = os.environ['MASTER_PORT']
    print(f"rank = {rank} is initialized in {master_addr}:{master_port}; local_rank = {local_rank}")
    torch.cuda.set_device(rank)
    tensor = torch.tensor([1, 2, 3, 4]).cuda()
    print(tensor)
    import pdb; pdb.set_trace()
    logger = None
    distributed, gpu_idx = \
        init_distributed_mode(args.world_size,
                              args.dist_url, logger)
    return True