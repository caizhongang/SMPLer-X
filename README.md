# SMPLer-X: Scaling Up Expressive Human Pose and Shape Estimation

![Teaser](./assets/teaser_complete.png)

## Useful links

<div align="center">
    <a href="https://caizhongang.github.io/projects/SMPLer-X/" class="button"><b>[Homepage]</b></a> &nbsp;&nbsp;&nbsp;&nbsp;
    <a href="https://huggingface.co/spaces/caizhongang/SMPLer-X" class="button"><b>[HuggingFace]</b></a> &nbsp;&nbsp;&nbsp;&nbsp;   
    <a href="https://arxiv.org/abs/2309.17448" class="button"><b>[arXiv]</b></a> &nbsp;&nbsp;&nbsp;&nbsp;
    <a href="https://youtu.be/DepTqbPpVzY" class="button"><b>[Video]</b></a> &nbsp;&nbsp;&nbsp;&nbsp;
    <a href="https://github.com/open-mmlab/mmhuman3d" class="button"><b>[MMHuman3D]</b></a>
</div>

## News
- [2024-02-29] [HuggingFace](https://huggingface.co/spaces/caizhongang/SMPLer-X) demo is online!
- [2023-10-23] Support visualization through SMPL-X mesh overlay and add inference docker. 
- [2023-10-02] [arXiv](https://arxiv.org/abs/2309.17448) preprint is online!
- [2023-09-28] [Homepage](https://caizhongang.github.io/projects/SMPLer-X/) and [Video](https://youtu.be/DepTqbPpVzY) are online!
- [2023-07-19] Pretrained models are released.
- [2023-06-15] Training and testing code is released.

## Gallery
| ![001.gif](./assets/005.gif) | ![001.gif](./assets/002.gif)  | ![001.gif](./assets/006.gif)  |  
|:--------------------------------------:|:-----------------------------:|:-----------------------------:|
|      ![001.gif](./assets/003.gif)      | ![001.gif](./assets/001.gif)  | ![001.gif](./assets/004.gif)  |

![Visualization](./assets/smpler_x_vis1.jpg)




## Install
```bash
conda create -n smplerx python=3.10 -y
conda activate smplerx
conda install cudatoolkit=11.7 -c nvidia -y
pip install -r pre-requirements.txt
pip install mmcv==2.1.0 -f https://download.openmmlab.com/mmcv/dist/cu117/torch2.0.0/index.html
pip install -r requirements.txt

# install mmpose
cd main/transformer_utils
pip install -v -e .
cd ../..
```

## Pretrained Models
|    Model     | Backbone | #Datasets | #Inst. | #Params | MPE  | Download |  FPS  |
|:------------:|:--------:|:---------:|:------:|:-------:|:----:|:--------:|:-----:|
| SMPLer-X-S32 |  ViT-S   |    32 |  4.5M  |   32M | 82.6 | [model](https://pjlab-my.sharepoint.cn/:u:/g/personal/openmmlab_pjlab_org_cn/EbkyKOS5PclHtDSxdZDmsu0BNviaTKUbF5QUPJ08hfKuKg?e=LQVvzs) | 36.17 |
| SMPLer-X-B32 |  ViT-B   |    32 |  4.5M  |  103M | 74.3 | [model](https://pjlab-my.sharepoint.cn/:u:/g/personal/openmmlab_pjlab_org_cn/EVcRBwNOQl9OtWhnCU54l58BzJaYEPxcFIw7u_GnnlPZiA?e=nPqMjz) | 33.09 |
| SMPLer-X-L32 |  ViT-L   |    32 |  4.5M  |  327M | 66.2 | [model](https://pjlab-my.sharepoint.cn/:u:/g/personal/openmmlab_pjlab_org_cn/EWypJXfmJ2dEhoC0pHFFd5MBoSs7LCZmWQjHjbcQJF72fQ?e=Gteus3) | 24.44 |
| SMPLer-X-H32 |  ViT-H   |    32 |  4.5M  |  662M | 63.0 | [model](https://pjlab-my.sharepoint.cn/:u:/g/personal/openmmlab_pjlab_org_cn/Eco7AAc_ZmtBrhAat2e5Ti8BonrR3NVNx-tNSck45ixT4Q?e=nudXrR) | 17.47 |
* MPE (Mean Primary Error): the average of the primary errors on five benchmarks (AGORA, EgoBody, UBody, 3DPW, and EHF)
* FPS (Frames Per Second): the average inference speed on a single Tesla V100 GPU, batch size = 1

## Preparation
- download [SMPL-X](https://smpl-x.is.tue.mpg.de/) and [SMPL](https://smpl.is.tue.mpg.de/) body models.
- download mmdet pretrained [model](https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth) for inference.


The file structure should be like:
```
SMPLer-X/
├── common/
│   └── utils/
│       └── human_model_files/  # body model
│           ├── smpl/
│           │   ├──SMPL_NEUTRAL.pkl
│           │   ├──SMPL_MALE.pkl
│           │   └──SMPL_FEMALE.pkl
│           └── smplx/
│               ├──MANO_SMPLX_vertex_ids.pkl
│               ├──SMPL-X__FLAME_vertex_ids.npy
│               ├──SMPLX_NEUTRAL.pkl
│               ├──SMPLX_to_J14.pkl
│               ├──SMPLX_NEUTRAL.npz
│               ├──SMPLX_MALE.npz
│               └──SMPLX_FEMALE.npz
├── main/
└── pretrained_models/  # pretrained ViT-Pose, SMPLer_X and mmdet models
    ├── mmdet/
    │   ├──faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth
    │   └──mmdet_faster_rcnn_r50_fpn_coco.py
    ├── smpler_x_s32.pth.tar
    ├── smpler_x_b32.pth.tar
    ├── smpler_x_l32.pth.tar
    ├── smpler_x_h32.pth.tar
    ├── vitpose_small.pth
    ├── vitpose_base.pth
    ├── vitpose_large.pth
    └── vitpose_huge.pth
```
## Inference 

```
python demo.py --input_video {VIDEO_FILE} --pretrained_model {PRETRAINED_CKPT} --show_verts

# For inferencing test_video.mp4 (24FPS) with smpler_x_h32
python demo.py --input_video test_video.mp4 --pretrained_model smpler_x_h32 --show_verts

```

## Huggingface
- Replace README.md with README_huggingface.md
- add mmcv into requirements.txt
  - eg: if using zero-gpu, add 'https://download.openmmlab.com/mmcv/dist/cu117/torch2.0.0/mmcv-2.1.0-cp310-cp310-manylinux1_x86_64.whl'

## FAQ
- `RuntimeError: Subtraction, the '-' operator, with a bool tensor is not supported. If you are trying to invert a mask, use the '~' or 'logical_not()' operator instead.`
  
  Follow [this post](https://github.com/mks0601/I2L-MeshNet_RELEASE/issues/6#issuecomment-675152527) and modify `torchgeometry`

- `KeyError: 'SinePositionalEncoding is already registered in position encoding'` or any other similar KeyErrors due to duplicate module registration.

  Manually add `force=True` to respective module registration under `main/transformer_utils/mmpose/models/utils`, e.g. `@POSITIONAL_ENCODING.register_module(force=True)` in [this file](main/transformer_utils/mmpose/models/utils/positional_encoding.py)

- How do I animate my virtual characters with SMPLer-X output (like that in the demo video)? 
  - We are working on that, please stay tuned!
    Currently, this repo supports SMPL-X estimation and a simple visualization (overlay of SMPL-X vertices).

## References
- [Hand4Whole](https://github.com/mks0601/Hand4Whole_RELEASE)
- [OSX](https://github.com/IDEA-Research/OSX)
- [MMHuman3D](https://github.com/open-mmlab/mmhuman3d)
