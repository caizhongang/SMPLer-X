import argparse
from config import cfg

import torch.backends.cudnn as cudnn

# ddp
from common.utils.distribute_utils import (
    init_distributed_mode, is_main_process, cleanup
)
import torch.distributed as dist
from xrprimer.utils.log_utils import get_logger
from base import Trainer

def parse_args():
    parser = argparse.ArgumentParser()
    # parser.add_argument('--gpu', type=str, dest='gpu_ids')
    parser.add_argument('--gpu_num', type=int, dest='gpu_num')
    parser.add_argument('--lr', type=str, dest='lr', default="1e-4")
    parser.add_argument('--continue', dest='continue_train', action='store_true')
    parser.add_argument('--exp_name', type=str, default='output/test')
    parser.add_argument('--num_thread', type=int, default=16)
    parser.add_argument('--end_epoch', type=int, default=14)
    parser.add_argument('--train_batch_size', type=int, default=32)
    parser.add_argument('--model_type', type=str, default='osx_l', choices=['osx_b', 'osx_l'])
    parser.add_argument('--agora_benchmark', action='store_true')
    parser.add_argument('--pretrained_model_path', type=str, default='../pretrained_models/osx_l.pth.tar')
    
    # ddp
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

    # if not args.gpu_ids:
    #     assert 0, "Please set propoer gpu ids"

    # if not args.lr:
    #     assert 0, "Please set learning rate"

    # if '-' in args.gpu_ids:
    #     gpus = args.gpu_ids.split('-')
    #     gpus[0] = int(gpus[0])
    #     gpus[1] = int(gpus[1]) + 1
    #     args.gpu_ids = ','.join(map(lambda x: str(x), list(range(*gpus))))

    return args

def main():
    args = parse_args()
    cfg.set_args(args.gpu_num, args.lr, args.continue_train)
    cfg.set_additional_args(exp_name=args.exp_name,
                            num_thread=args.num_thread, train_batch_size=args.train_batch_size,
                            model_type=args.model_type,
                            end_epoch=args.end_epoch,
                            pretrained_model_path=args.pretrained_model_path,
                            agora_benchmark=args.agora_benchmark
                            )

    cudnn.benchmark = True
    distributed, gpu_idx = \
        init_distributed_mode(args.world_size, args.dist_url)
    trainer = Trainer(distributed, gpu_idx, args.device)
    
    # ddp
    if distributed:
        trainer.logger_info('### Set DDP ###')
        trainer.logger.info(f'Distributed: {distributed}, init done {gpu_idx}')
    trainer.logger_info(f"Using {cfg.num_gpus} GPUs, batch size {cfg.train_batch_size} per GPU.")
    
    trainer._make_batch_generator()
    trainer._make_model()

    trainer.logger_info('### Set some hyper parameters ###')
    for k in cfg.__dict__:
        trainer.logger_info(f'set {k} to {cfg.__dict__[k]}')

    trainer.logger_info('### Start training ###')
    for epoch in range(trainer.start_epoch, cfg.end_epoch):
        trainer.tot_timer.tic()
        trainer.read_timer.tic()
        for itr, (inputs, targets, meta_info) in enumerate(trainer.batch_generator):
            trainer.read_timer.toc()
            trainer.gpu_timer.tic()

            # forward
            trainer.optimizer.zero_grad()
            loss = trainer.model(inputs, targets, meta_info, 'train')
            loss = {k: loss[k].mean() for k in loss}

            # backward
            sum(loss[k] for k in loss).backward()
            trainer.optimizer.step()
            trainer.scheduler.step()
            
            trainer.gpu_timer.toc()
            if (itr + 1) % cfg.print_iters == 0:
                screen = [
                    'Epoch %d/%d itr %d/%d:' % (epoch, cfg.end_epoch, itr, trainer.itr_per_epoch),
                    'lr: %g' % (trainer.get_lr()),
                    'speed: %.2f(%.2fs r%.2f)s/itr' % (
                        trainer.tot_timer.average_time, trainer.gpu_timer.average_time,
                        trainer.read_timer.average_time),
                    '%.2fh/epoch' % (trainer.tot_timer.average_time / 3600. * trainer.itr_per_epoch),
                ]
                screen += ['%s: %.4f' % ('loss_' + k, v.detach()) for k, v in loss.items()]
                trainer.logger_info(' '.join(screen))

            trainer.tot_timer.toc()
            trainer.tot_timer.tic()
            trainer.read_timer.tic()

        # save model
        if (epoch % 10 == 0 or epoch == (cfg.end_epoch-1)) and is_main_process():
            trainer.save_model({
                'epoch': epoch,
                'network': trainer.model.state_dict(),
                'optimizer': trainer.optimizer.state_dict(),
            }, epoch)

if __name__ == "__main__":
    main()