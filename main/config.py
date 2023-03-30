import os
import os.path as osp
import sys

class Config:

    # dataset setting
    dataset_list = ['Human36M', 'MSCOCO', 'MPII', 'AGORA', 'EHF']
    # trainset_3d = ['Human36M']; trainset_2d = ['MSCOCO', 'MPII']; testset = 'EHF'
    trainset_3d = []; trainset_2d = ['MPII']; testset = 'EHF'

    ## model setting
    pretrained_model_path = None
    upscale = 4
    encoder_pretrained_model_path = '../pretrained_models/osx_vit_l.pth'
    hand_pos_joint_num = 20
    face_pos_joint_num = 72
    model_type = 'OSX'
    num_task_token = 24
    feat_dim = 768
    encoder_config_file = 'transformer_utils/configs/osx/encoder/body_encoder_large.py'
    num_noise_sample = 0

    ## UBody setting
    train_sample_interval = 10
    test_sample_interval = 100
    make_same_len = False

    ## input, output size
    input_img_shape = (512, 384)
    input_body_shape = (256, 192)
    output_hm_shape = (16, 16, 12)
    input_hand_shape = (256, 256)
    output_hand_hm_shape = (16, 16, 16)
    output_face_hm_shape = (8, 8, 8)
    input_face_shape = (192, 192)
    focal = (5000, 5000)  # virtual focal lengths
    princpt = (input_body_shape[1] / 2, input_body_shape[0] / 2)  # virtual principal point position
    body_3d_size = 2
    hand_3d_size = 0.3
    face_3d_size = 0.3
    camera_3d_size = 2.5

    ## training config
    end_epoch = 14
    train_batch_size = 48
    print_iters = 100
    lr_mult = 1
    smplx_loss_weight = 1
    agora_benchmark = False

    ## testing config
    test_batch_size = 16

    ## others
    num_thread = 16
    gpu_ids = '0'
    num_gpus = 1
    continue_train = False
    vis = False

    ## directory
    cur_dir = osp.dirname(os.path.abspath(__file__))
    root_dir = osp.join(cur_dir, '..')
    data_dir = osp.join(root_dir, 'dataset')

    output_dir, model_dir, vis_dir, log_dir, result_dir, code_dir = None, None, None, None, None, None
    human_model_path = osp.join(root_dir, 'common', 'utils', 'human_model_files')
    exp_name = 'output/exp1/pre_analysis'

    def set_args(self, gpu_ids, lr=1e-4, continue_train=False):
        self.gpu_ids = gpu_ids
        self.num_gpus = len(self.gpu_ids.split(','))
        self.lr = float(lr)
        self.continue_train = continue_train
        os.environ["CUDA_VISIBLE_DEVICES"] = self.gpu_ids
        print('>>> Using GPU: {}'.format(self.gpu_ids))

    def set_additional_args(self, **kwargs):
        names = self.__dict__
        for k, v in kwargs.items():
            names[k] = v
        self.prepare_dirs(self.exp_name)
        if self.model_type == 'osx_b':
            self.encoder_config_file = 'transformer_utils/configs/osx/encoder/body_encoder_base.py'
            self.encoder_pretrained_model_path = '../pretrained_models/osx_vit_b.pth'
            self.feat_dim = 768
        elif self.model_type == 'osx_l':
            self.encoder_config_file = 'transformer_utils/configs/osx/encoder/body_encoder_large.py'
            self.encoder_pretrained_model_path = '../pretrained_models/osx_vit_l.pth'
            self.feat_dim = 1024
        if 'AGORA' in self.testset:
            self.testset = 'AGORA'
        if self.agora_benchmark:
            self.smplx_loss_weight = 2
            self.trainset_3d = ['AGORA']
            self.trainset_2d = []
            self.testset = 'AGORA'

    def prepare_dirs(self, exp_name):
        self.output_dir = osp.join(self.root_dir, exp_name)
        self.model_dir = osp.join(self.output_dir, 'model_dump')
        self.vis_dir = osp.join(self.output_dir, 'vis')
        self.log_dir = osp.join(self.output_dir, 'log')
        self.code_dir = osp.join(self.output_dir, 'code')
        self.result_dir = osp.join(self.output_dir, 'result')
        make_folder(self.model_dir)
        make_folder(self.vis_dir)
        make_folder(self.log_dir)
        make_folder(self.code_dir)
        make_folder(self.result_dir)
        ## copy some code to log dir as a backup
        copy_files = ['main/config.py', 'main/train.py', 'main/test.py', 'common/base.py',
                      'main/OSX.py', 'common/nets', 'main/OSX_WoDecoder.py',
                      'data/dataset.py', 'data/MSCOCO/MSCOCO.py', 'data/AGORA/AGORA.py']
        for file in copy_files:
            os.system(f'cp -r {self.root_dir}/{file} {self.code_dir}')

cfg = Config()

## add some paths to the system root dir
sys.path.insert(0, osp.join(cfg.root_dir, 'common'))
from utils.dir import add_pypath, make_folder
add_pypath(osp.join(cfg.data_dir))
for i in range(len(cfg.dataset_list)):
    add_pypath(osp.join(cfg.root_dir, 'data', cfg.dataset_list[i]))
add_pypath(osp.join(cfg.root_dir, 'data'))
add_pypath(cfg.data_dir)
