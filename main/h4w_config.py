import os
import os.path as osp
import sys
import numpy as np

class Config:
    
    ## dataset (use names in the `data` folder)
    trainset_3d = ['Human36M'] 
    trainset_2d = ['MSCOCO','MPII'] #, 
    testset = 'EHF'

    ## model setting
    resnet_type = 50
    hand_resnet_type = 50
    face_resnet_type = 18
    
    ## input, output
    input_img_shape = (512, 384) 
    input_body_shape = (256, 192)
    output_hm_shape = (8, 8, 6)
    input_hand_shape = (256, 256)
    output_hand_hm_shape = (8, 8, 8)
    input_face_shape = (192, 192)
    focal = (5000, 5000) # virtual focal lengths
    princpt = (input_body_shape[1]/2, input_body_shape[0]/2) # virtual principal point position
    body_3d_size = 2
    hand_3d_size = 0.3
    face_3d_size = 0.3
    camera_3d_size = 2.5

    ## training config
    lr_dec_factor = 10
    lr_dec_epoch = [4,6] #[40, 60] #[4,6]
    end_epoch = 7 #70 #7
    train_batch_size = 6 # 24

    ## testing config
    test_batch_size = 64

    ## others
    num_thread = 16
    gpu_ids = '0'
    num_gpus = 1
    continue_train = False
    
    ## directory
    cur_dir = osp.dirname(os.path.abspath(__file__))
    root_dir = osp.join(cur_dir, '..')
    data_dir = osp.join(root_dir, 'data')
    output_dir = osp.join(root_dir, 'output')
    model_dir = osp.join(output_dir, 'model_dump')
    vis_dir = osp.join(output_dir, 'vis')
    log_dir = osp.join(output_dir, 'log')
    result_dir = osp.join(output_dir, 'result')
    human_model_path = osp.join(root_dir, 'common', 'utils', 'human_model_files')
    
    def set_args(self, gpu_ids, lr=1e-4, continue_train=False):
        self.gpu_ids = gpu_ids
        self.num_gpus = len(self.gpu_ids.split(','))
        self.lr = float(lr)
        self.continue_train = continue_train
        os.environ["CUDA_VISIBLE_DEVICES"] = self.gpu_ids
        print('>>> Using GPU: {}'.format(self.gpu_ids))

cfg = Config()

sys.path.insert(0, osp.join(cfg.root_dir, 'common'))
from utils.dir import add_pypath, make_folder
add_pypath(osp.join(cfg.data_dir))
for i in range(len(cfg.trainset_3d)):
    add_pypath(osp.join(cfg.data_dir, cfg.trainset_3d[i]))
for i in range(len(cfg.trainset_2d)):
    add_pypath(osp.join(cfg.data_dir, cfg.trainset_2d[i]))
add_pypath(osp.join(cfg.data_dir, cfg.testset))
make_folder(cfg.model_dir)
make_folder(cfg.vis_dir)
make_folder(cfg.log_dir)
make_folder(cfg.result_dir)
