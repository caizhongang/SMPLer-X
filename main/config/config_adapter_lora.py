import os
import os.path as osp

# will be update in exp
num_gpus = -1
exp_name = 'output/exp1/pre_analysis'

# quick access
lr = 1e-4
end_epoch = 14
train_batch_size = 32

# continue
continue_train = False
pretrained_model_path = None

# dataset setting
dataset_list = ['Human36M', 'MSCOCO', 'MPII', 'AGORA', 'EHF', 'SynBody', 'GTA_Human2']
trainset_3d = ['AGORA']
trainset_2d = []
trainset_humandata = []
testset = 'EHF'

# downsample rate
AGORA_train_sample_interval = 1000

# model
smplx_loss_weight = 1 #2 for agora_model
agora_benchmark = 'na' # 'agora_model', 'test_only'
adapter_name = 'lora' # apply adapter. 'lora', 'vit_adapter'

model_type = 'osx_b'
encoder_config_file = 'transformer_utils/configs/osx/encoder/body_encoder_base.py'
encoder_pretrained_model_path = '../pretrained_models/osx_vit_b.pth'
feat_dim = 768

# model_type = 'osx_l'
# encoder_config_file = 'transformer_utils/configs/osx/encoder/body_encoder_large.py'
# encoder_pretrained_model_path = '../pretrained_models/osx_vit_l.pth'
# feat_dim = 1024

## =====FIXED ARGS============================================================
## model setting
upscale = 4
hand_pos_joint_num = 20
face_pos_joint_num = 72
num_task_token = 24
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
print_iters = 100
lr_mult = 1

## testing config
test_batch_size = 32

## others
num_thread = 4
vis = False

## directory
output_dir, model_dir, vis_dir, log_dir, result_dir, code_dir = None, None, None, None, None, None
