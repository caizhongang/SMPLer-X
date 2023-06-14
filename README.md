# SMPLer-X

## Install
```bash
conda create -n smplerx python=3.8 -y
conda activate smplerx
conda install pytorch==1.12.0 torchvision==0.13.0 torchaudio==0.12.0 cudatoolkit=11.3 -c pytorch -y
wget http://download.openmmlab.sensetime.com/mmcv/dist/cu113/torch1.12.0/mmcv_full-1.7.1-cp38-cp38-manylinux1_x86_64.whl
pip install mmcv_full-1.7.1-cp38-cp38-manylinux1_x86_64.whl
rm mmcv_full-1.7.1-cp38-cp38-manylinux1_x86_64.whl
pip install -r requirements.txt

# install mmpose
cd main/transformer_utils
pip install -v -e .
cd ../..
```

## Preparation
- process all datasets into [HumanData](https://github.com/open-mmlab/mmhuman3d/blob/main/docs/human_data.md) format, except the following:
  - AGORA, MSCOCO, MPII, Human3.6M, UBody
- follow [OSX](https://github.com/IDEA-Research/OSX) in preparing pretrained ViT-Pose models. 
- download [SMPL-X](https://smpl-x.is.tue.mpg.de/) body models

The file structure should be like:
```
SMPLer-X/
├── common/
│   └── utils/
│       └── human_model_files/  # body model
├── data/
├── main/
├── pretrained_models/  # pretrained ViT-Pose models
└── dataset/  
    ├── AGORA/       
    ├── ARCTIC/      
    ├── BEDLAM/      
    ├── Behave/      
    ├── CHI3D/       
    ├── CrowdPose/   
    ├── EgoBody/     
    ├── EHF/         
    ├── FIT3D/                
    ├── GTA_Human2/           
    ├── Human36M/             
    ├── HumanSC3D/            
    ├── InstaVariety/         
    ├── LSPET/                
    ├── MPII/                 
    ├── MPI_INF_3DHP/         
    ├── MSCOCO/               
    ├── MTP/                    
    ├── MuCo/                   
    ├── OCHuman/                
    ├── PoseTrack/                
    ├── PROX/                   
    ├── PW3D/                   
    ├── RenBody/
    ├── RICH/
    ├── SPEC/
    ├── SSP3D/
    ├── SynBody/
    ├── Talkshow/
    ├── UBody/
    ├── UP3D/
    └── preprocessed_datasets/  # HumanData files
```

## Training
```bash
cd main
sh slurm_train.sh {JOB_NAME} {NUM_GPU} {CONFIG_FILE}
# logs and ckpts will be saved to ../output/train_{JOB_NAME}_{DATE_TIME}
# config file is the file name under ./config, e.g. ./config/config_base.py
# a copy of current config file wil be saved to ../output/train_{JOB_NAME}_{DATE_TIME}/code/config_base.py
```

## Testing
```bash
cd main
sh slurm_test.sh {JOB_NAME} {NUM_GPU} {TRAIN_OUTPUT_DIR} {CKPT_ID}
# this will eval the model ../output/train_{JOB_NAME}_{DATE_TIME}/model_dump/snapshot_{CKPT_ID}.pth.tar with confing ../output/train_{JOB_NAME}_{DATE_TIME}/code/config_base.py
# logs and results  will be saved to ../output/test_{JOB_NAME}_{DATE_TIME}
```

## References
- [Hand4Whole](https://github.com/mks0601/Hand4Whole_RELEASE)
- [OSX](https://github.com/IDEA-Research/OSX)
- [MMHuman3D](https://github.com/open-mmlab/mmhuman3d)