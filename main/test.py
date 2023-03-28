import argparse
from config import cfg

import torch.backends.cudnn as cudnn
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=str, dest='gpu_ids')
    parser.add_argument('--exp_name', type=str, default='output/test')
    parser.add_argument('--test_batch_size', type=int, default=32)
    parser.add_argument('--model_type', type=str, default='osx_l', choices=['osx_b', 'osx_l'])
    parser.add_argument('--agora_benchmark', action='store_true')
    parser.add_argument('--pretrained_model_path', type=str, default='../pretrained_models/osx_l.pth.tar')
    args = parser.parse_args()

    if not args.gpu_ids:
        assert 0, "Please set propoer gpu ids"

    if '-' in args.gpu_ids:
        gpus = args.gpu_ids.split('-')
        gpus[0] = int(gpus[0])
        gpus[1] = int(gpus[1]) + 1
        args.gpu_ids = ','.join(map(lambda x: str(x), list(range(*gpus))))

    return args

def main():
    print('### Argument parse and create log ###')
    args = parse_args()
    cfg.set_args(args.gpu_ids)
    cfg.set_additional_args(exp_name=args.exp_name,
                            test_batch_size=args.test_batch_size,
                            model_type=args.model_type,
                            pretrained_model_path=args.pretrained_model_path,
                            agora_benchmark=args.agora_benchmark
                            )
    cudnn.benchmark = True
    from base import Tester
    from evaluate import evaluate
    tester = Tester()
    tester._make_batch_generator()
    tester._make_model()

    print('### Start testing ###')
    tester.model.eval()
    eval_result = evaluate(tester.model, testset_name='EHF')
    print('EHF dataset:')
    for key in eval_result.keys():
        print(f'{key.upper()}: {np.mean(eval_result[key]):.2f} mm')
    eval_result = evaluate(tester.model, testset_name='AGORA')
    print('AGORA dataset:')
    for key in eval_result.keys():
        print(f'{key.upper()}: {np.mean(eval_result[key]):.2f} mm')
    tester.model.train()

if __name__ == "__main__":
    main()