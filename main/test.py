import argparse
from config import cfg
from tqdm import tqdm
import torch
import torch.backends.cudnn as cudnn
from base import Tester

def parse_args():
    parser = argparse.ArgumentParser()
    # parser.add_argument('--gpu', type=str, dest='gpu_ids')
    parser.add_argument('--gpu_num', type=int, dest='gpu_num')
    parser.add_argument('--exp_name', type=str, default='output/test')
    parser.add_argument('--test_batch_size', type=int, default=32)
    parser.add_argument('--model_type', type=str, default='osx_l', choices=['osx_b', 'osx_l'])
    parser.add_argument('--testset', type=str, default='EHF')
    parser.add_argument('--agora_benchmark', action='store_true')
    parser.add_argument('--pretrained_model_path', type=str, default='../pretrained_models/osx_l.pth.tar')
    args = parser.parse_args()

    # if not args.gpu_ids:
    #     assert 0, "Please set propoer gpu ids"

    # if '-' in args.gpu_ids:
    #     gpus = args.gpu_ids.split('-')
    #     gpus[0] = int(gpus[0])
    #     gpus[1] = int(gpus[1]) + 1
    #     args.gpu_ids = ','.join(map(lambda x: str(x), list(range(*gpus))))

    return args

def main():
    print('### Argument parse and create log ###')
    args = parse_args()
    cfg.set_args(args.gpu_num)
    cfg.set_additional_args(exp_name=args.exp_name,
                            test_batch_size=args.test_batch_size,
                            model_type=args.model_type,
                            pretrained_model_path=args.pretrained_model_path,
                            agora_benchmark=args.agora_benchmark,
                            testset=args.testset,
                            )
    cudnn.benchmark = True
    
    tester = Tester()
    tester._make_batch_generator()
    tester._make_model()

    eval_result = {}
    cur_sample_idx = 0
    for itr, (inputs, targets, meta_info) in enumerate(tqdm(tester.batch_generator)):

        # forward
        with torch.no_grad():
            out = tester.model(inputs, targets, meta_info, 'test')

        # save output
        out = {k: v.cpu().numpy() for k, v in out.items()}
        for k, v in out.items(): batch_size = out[k].shape[0]
        out = [{k: v[bid] for k, v in out.items()} for bid in range(batch_size)]

        # evaluate
        cur_eval_result = tester._evaluate(out, cur_sample_idx)
        for k, v in cur_eval_result.items():
            if k in eval_result:
                eval_result[k] += v
            else:
                eval_result[k] = v
        cur_sample_idx += len(out)

    tester._print_eval_result(eval_result)

if __name__ == "__main__":
    main()