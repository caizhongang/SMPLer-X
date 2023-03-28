# AGORA dataset parsing process

## Cleaning SMPL parameters
* This process will convert CudaFloatTensor smpl parameters to numpy format.
* Download and unzip `smpl_gt.zip` and `smplx_gt.zip`
* Run `python tensor_to_numpy_parameter.py --dataset_path $PATH1`. $PATH1 denotes AGORA dataset path. 

## Make annotation files
* This code will dump GT 2D/3D joints and 3D vertices of SMPL and SMPL-X in $PATH1. Also, it will generate `AGORA_train.json` and `AGORA_validation.json` in $PATH1.
* Download and unzip `train_SMPL.zip`, `train_SMPLX.zip`, `validation_SMPL.zip`, and `validation_SMPLX.zip` from [here](https://agora.is.tue.mpg.de/download.php).
* Run `python agora2coco.py --dataset_path $PATH1 --human_model_path $PATH2`. $PATH1 denotes AGORA dataset path. $PATH2 denotes human model layer path. 

## Preparing 1280x720 image files
* This code will prepare 1280x720 image files.
* Download and unzip 1280x720 image files.
* Then, make `1280x720` folder in AGORA dataset path.
* For the $i$th zip file of training set, make `train_$i$` folder and move all image files to that folder. For example, make `train_0` folder at AGORA dataset path and move all image files from `train_images_1280x720_0.zip` to that folder.
* For the images of validation and test sets, make `validation` and `test` folders and move all images files to corresponding folders.

## Preparing 3840x2160 image files
* This code will prepare 3840x2160 image files.
* Do the same process of 1280x720 image files
* As the image resolution is too high, you need to crop and resize humans to prevent the dataloader from being stuck.
* To this end, run `python affine_transom.py --dataset_path $PATH1 --out_height 512 --out_width 384`. $PATH1 denotes AGORA dataset path. 

## Download `AGORA_test_bbox.json`
* Download human detection results on test set from [here](https://drive.google.com/file/d/1dGIMsX00xUIwlFTa1gtU9bTxbfTpMt9T/view?usp=share_link)
* The human detection results are from YOLO v5.

## Final directory
```
${PATH1}
|-- AGORA_train.json
|-- AGORA_validation.json
|-- AGORA_test_bbox.json
|-- gt_joints_2d
|-- |-- smpl
|-- |-- smplx
|-- gt_joints_3d
|-- |-- smpl
|-- |-- smplx
|-- gt_verts
|-- |-- smpl
|-- |-- smplx
|-- 1280x720
|   |-- train_0
|   |-- train_1
|   |-- train_2
|   |-- train_3
|   |-- train_4
|   |-- train_5
|   |-- train_6
|   |-- train_7
|   |-- train_8
|   |-- train_9
|   |-- validation
|   |-- test
|-- 3840x2160
|   |-- train_0
|   |-- train_0_crop
|   |-- train_1
|   |-- train_1_crop
|   |-- train_2
|   |-- train_2_crop
|   |-- train_3
|   |-- train_3_crop
|   |-- train_4
|   |-- train_4_crop
|   |-- train_5
|   |-- train_5_crop
|   |-- train_6
|   |-- train_6_crop
|   |-- train_7
|   |-- train_7_crop
|   |-- train_8
|   |-- train_8_crop
|   |-- train_9
|   |-- train_9_crop
|   |-- validation
|   |-- validation_crop
|   |-- test
|   |-- test_crop
```

