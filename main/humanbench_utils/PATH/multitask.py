import shutil
import os
import multiprocessing as mp
if mp.get_start_method(allow_none=True) != 'spawn':
    mp.set_start_method('spawn')
import argparse
from core.distributed_utils import dist_init
from core.config import Config
from core.solvers import solver_entry
import torch

parser = argparse.ArgumentParser(description='Multi-Task Training Framework')
parser.add_argument('--load-path', default='', type=str)
parser.add_argument('--ignore', nargs='+', default=[], type=str)
parser.add_argument('--recover', action='store_true')
parser.add_argument('--load-single', action='store_true')
parser.add_argument('--port', default='23456', type=str)
parser.add_argument('--config', default='', type=str)
parser.add_argument('--expname', type=str, default=None, help='experiment name, output folder')
parser.add_argument('--auto-resume', type=str, default=None, help='jobs auto resume from the pattern_path or the folder')
parser.add_argument('--forwardbn', action='store_true', help='just forward for re-calcuating bn values')
parser.add_argument('--finetune',action='store_true')


def main():
    args = parser.parse_args()
    
    dist_init()

    C = Config(args.config)
    if args.expname is not None:
        C.config['expname'] = args.expname

    S = solver_entry(C)
    config_save_to = os.path.join(S.ckpt_path, 'config.yaml')

    # auto resume strategy for spring.submit arun
    if args.auto_resume is not None:
        args.auto_resume = os.path.join(S.out_dir, args.auto_resume)
        if os.path.isdir(args.auto_resume):
            max_iter = 0
            filename = os.listdir(args.auto_resume)
            for file in filename:
                if file.startswith('ckpt_task0') and file.endswith('.pth.tar'):
                    cur_iter = int(file.split('_')[-1].split('.')[0])
                    max_iter = max(max_iter, cur_iter)
            if max_iter > 0:
                args.load_path = os.path.join(args.auto_resume,
                                              'ckpt_task_iter_{}.pth.tar'.format(str(max_iter)))
                args.recover = True
                args.ignore = []
                print('auto-resume from: {}'.format(args.load_path))
        elif args.auto_resume.endswith('.pth.tar'):
            tmpl = args.auto_resume.replace('ckpt_task_', 'ckpt_task*_')
            import glob
            ckpt = glob.glob(tmpl)
            if len(ckpt) > 0:
                args.load_path = args.auto_resume
                args.recover = True
                args.ignore = []
                print('auto-resume from: {}'.format(args.load_path))
        else:
            print('auto-resume not work:{}'.format(args.auto_resume))

    #tmp = torch.Tensor(1).cuda()
    if not os.path.exists(config_save_to):
        shutil.copy(args.config, config_save_to)

    S.initialize(args)

    S.run()


if __name__ == '__main__':
    main()
