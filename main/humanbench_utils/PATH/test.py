import shutil
import os
import multiprocessing as mp
if mp.get_start_method(allow_none=True) != 'spawn':
    mp.set_start_method('spawn')
import argparse
from core.distributed_utils import dist_init
from core.config import Config
from core.testers import tester_entry
import torch
import yaml
import re

parser = argparse.ArgumentParser(description='Multi-Task Training Framework')
parser.add_argument('--spec_ginfo_index', type=int, required=True)
parser.add_argument('--load-path', default='', type=str)
parser.add_argument('--ignore', nargs='+', default=[], type=str)
parser.add_argument('--recover', action='store_true')
parser.add_argument('--load-single', action='store_true')
parser.add_argument('--port', default='23456', type=str)
parser.add_argument('--config', default='', type=str)
parser.add_argument('--test_config', default='', type=str)
parser.add_argument('--expname', type=str, default=None, help='experiment name, output folder')
parser.add_argument('--auto-resume', type=str, default=None, help='jobs auto resume from the pattern_path or the folder')

loader = yaml.SafeLoader
loader.add_implicit_resolver(
    u'tag:yaml.org,2002:float',
    re.compile(u'''^(?:
     [-+]?(?:[0-9][0-9_]*)\\.[0-9_]*(?:[eE][-+]?[0-9]+)?
    |[-+]?(?:[0-9][0-9_]*)(?:[eE][-+]?[0-9]+)
    |\\.[0-9_]+(?:[eE][-+][0-9]+)?
    |[-+]?[0-9][0-9_]*(?::[0-5]?[0-9])+\\.[0-9_]*
    |[-+]?\\.(?:inf|Inf|INF)
    |\\.(?:nan|NaN|NAN))$''', re.X),
    list(u'-+0123456789.'))

def main():
    args = parser.parse_args()
    dist_init()

    # auto resume strategy for spring.submit arun
    if args.auto_resume is not None:
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
    C_train = Config(args.config, spec_ginfo_index=args.spec_ginfo_index)

    with open(args.test_config) as f:
        test_config = yaml.load(f, Loader=loader)
    num_test_tasks = len(test_config['tasks'])

    for test_spec_ginfo_index in range(num_test_tasks):
        C_test = Config(args.test_config, spec_ginfo_index=test_spec_ginfo_index)
        if args.expname is not None:
            C_train.config['expname'] = args.expname

        S = tester_entry(C_train, C_test)
        config_save_to = os.path.join(S.ckpt_path, 'config.yaml')
        test_config_save_to = os.path.join(S.ckpt_path, 'test_config_task{}.yaml'.format(test_spec_ginfo_index))
        if not os.path.exists(config_save_to):
            shutil.copy(args.config, config_save_to)
            shutil.copy(args.test_config, test_config_save_to)

        S.initialize(args)

        S.run()


if __name__ == '__main__':
    main()
