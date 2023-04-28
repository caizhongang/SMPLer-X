def print_dataset_specific_training_info(total_iters, dataset_samples, dataset_sample_weight, dataset_imgs_per_gpu, GPU_per_dataset, data_use_ratio=0.1):
    print('total iters: ', total_iters)
    datasets = list(dataset_samples.keys())
    
    print('############### epoch per dataset ###############')
    for i, dataset in enumerate(datasets):
        print('{}: {}: {}'.format(i, dataset, (total_iters * dataset_imgs_per_gpu[dataset] * GPU_per_dataset[dataset]) / int(dataset_samples[dataset] * data_use_ratio)))

    print('############### config ###############')
    print("\t     sw|  gpus|  bs")
    for i, dataset in enumerate(datasets):
        print('{}: {}: {}, {}, {}'.format(i, dataset, dataset_sample_weight[dataset] * dataset_imgs_per_gpu[dataset] * GPU_per_dataset[dataset], GPU_per_dataset[dataset], dataset_imgs_per_gpu[dataset]))

    print('############### total GPUs ###############')
    count = 0
    for per in list(GPU_per_dataset.values()):
        count += per
    print("GPUs: ", count)
    print("nodes: ", count // 8)

# 'LaST_PRCC': 71248+17896

# new_data_path:

# - AIST: sh1424:s3://pose_public/aistplusplus/train.json  1015257
#         sh1424:s3://pose_public/aistplusplus/images/

# - halpe: sh1424:s3://pose_public/Halpe//train.json  41712
#          opnmmlab: 

# - 3dhp: sh1424:s3://pose_public/3dhp/train.json  1031701

# - h36m_pose: openmmlab:s3://openmmlab/datasets/pose/h36m/processed/annotation_body2d/h36m_coco_train.json  312187
#     openmmlab:s3://openmmlab/datasets/pose/h36m/processed/images

# paperroll: s3://parsing_public/PaperDoll/  1035825





## full dataset without merge attr
# dataset_samples = {'coco_pose': 149813, 'aic_pose': 378352, 'posetrack': 97174, 'jrdb2022': 310035, 'mhp_pose': 41128, 'penn_action': 163839, '3dpw': 74620, 'halpe': 41712,'3dhp': 1031701, 'h36m_pose': 312187, 'AIST': 1015257,
#                     'h36m_par': 62668, 'lip_par': 30462, 'cihp_par': 28280, 'vip_par': 18469, 'paper_roll': 1035825,
#                     'deepfashion': 191961, 'modanet': 52245, 
#                     'rap2_attr': 67943, 'pa100k_attr': 90000, 'HARDHC_attr': 28336, 'uavhuman_attr': 16183, 'parse27k_attr': 27482, 'duke_attr': 34183, 'market_attr': 12936,
#                     'reid_5set': 118063, 'DGMarket': 128309, 'LUperson': 5000000,
#                     'crowdhuman_det': 15000, '5set_det': 9000+91500+23892+2975+118287}

# dataset_sample_weight = {'coco_pose': 8000, 'aic_pose': 6000, 'posetrack': 6000, 'jrdb2022': 4000, 'mhp_pose': 4000, 'penn_action': 4000, '3dpw': 4000, 'halpe': 2000,'3dhp': 2000, 'h36m_pose': 2000, 'AIST': 2000,
#                         'h36m_par': 20, 'lip_par': 20, 'cihp_par': 20, 'vip_par': 20, 'paper_roll': 15,
#                         'deepfashion': 15, 'modanet': 15,
#                         'rap2_attr': 0.1, 'pa100k_attr': 0.1, 'HARDHC_attr': 0.05, 'uavhuman_attr': 0.05, 'parse27k_attr': 0.05, 'duke_attr': 0.05, 'market_attr': 0.05,
#                         'reid_5set': 5, 'DGMarket': 1, 'LUperson': 1,
#                         'crowdhuman_det': 10, '5set_det': 10}

# dataset_imgs_per_gpu = {'coco_pose': 224, 'aic_pose': 224, 'posetrack': 224, 'jrdb2022': 224, 'mhp_pose': 96, 'penn_action': 128, '3dpw': 128, 'halpe': 96,'3dhp': 128, 'h36m_pose': 128, 'AIST': 128,
#                         'h36m_par': 20, 'lip_par': 12, 'cihp_par': 16, 'vip_par': 16, 'paper_roll': 16,
#                         'deepfashion': 32, 'modanet': 32,
#                         'rap2_attr': 72, 'pa100k_attr': 72, 'HARDHC_attr': 18, 'uavhuman_attr': 12, 'parse27k_attr': 18, 'duke_attr': 24, 'market_attr': 8,
#                         'reid_5set': 192, 'DGMarket': 128, 'LUperson': 192, 
#                         'crowdhuman_det': 2, '5set_det': 2}

# GPU_per_dataset = {'coco_pose': 2, 'aic_pose': 2, 'posetrack': 1, 'jrdb2022': 1, 'mhp_pose': 1, 'penn_action': 1, '3dpw': 1, 'halpe': 1,'3dhp': 1, 'h36m_pose': 1, 'AIST': 1,
#                     'h36m_par': 4, 'lip_par': 3, 'cihp_par': 3, 'vip_par': 1, 'paper_roll': 2,
#                     'deepfashion': 2, 'modanet': 1,
#                     'rap2_attr': 1, 'pa100k_attr': 1, 'HARDHC_attr': 1, 'uavhuman_attr': 1, 'parse27k_attr': 1, 'duke_attr': 1, 'market_attr': 1,
#                     'reid_5set': 1, 'DGMarket': 1, 'LUperson': 2, 
#                     'crowdhuman_det': 16, '5set_det': 16}


## full dataset merge attr dataset
dataset_samples = {'coco_pose': 149813, 'aic_pose': 378352, 'posetrack': 97174, 'jrdb2022': 310035, 'mhp_pose': 41128, 'penn_action': 163839, '3dpw': 74620, 'halpe': 41712,'3dhp': 1031701, 'h36m_pose': 312187, 'AIST': 1015257,
                    'h36m_par': 62668, 'lip_par': 30462, 'cihp_par': 28280, 'vip_par': 18469, 'paper_roll': 1035825,
                    'deepfashion': 191961, 'modanet': 52245,
                    'rap2_pa100k_attr': 67943 + 90000, '5set_attr': 28336 + 16183 + 27482 + 34183 + 12936,
                    'reid_4set': 67070, 'LaST_PRCC_DGMarket': 128309 + 71248 + 17896, 'LUperson': 5000000, 
                    'crowdhuman_det': 15000, '5set_det': 9000+91500+23892+2975+118287}

dataset_sample_weight = {'coco_pose': 8000, 'aic_pose': 6000, 'posetrack': 6000, 'jrdb2022': 4000, 'mhp_pose': 4000, 'penn_action': 4000, '3dpw': 4000, 'halpe': 2000,'3dhp': 2000, 'h36m_pose': 2000, 'AIST': 2000,
                        'h36m_par': 20, 'lip_par': 20, 'cihp_par': 20, 'vip_par': 20, 'paper_roll': 15,
                        'deepfashion': 15, 'modanet': 15,
                        'rap2_pa100k_attr': 0.1, '5set_attr': 0.1,
                        'reid_4set': 5, 'LaST_PRCC_DGMarket': 0.1, 'LUperson': 1,
                        'crowdhuman_det': 10, '5set_det': 10}

dataset_imgs_per_gpu = {'coco_pose': 224, 'aic_pose': 224, 'posetrack': 224, 'jrdb2022': 224, 'mhp_pose': 96, 'penn_action': 128, '3dpw': 128, 'halpe': 64,'3dhp': 128, 'h36m_pose': 128, 'AIST': 128,
                        'h36m_par': 26, 'lip_par': 32, 'cihp_par': 24, 'vip_par': 16, 'paper_roll': 24,
                        'deepfashion': 32, 'modanet': 32,
                        'rap2_pa100k_attr': 128, '5set_attr': 116,
                        'reid_4set': 112, 'LaST_PRCC_DGMarket': 96, 'LUperson': 192, 
                        'crowdhuman_det': 2, '5set_det': 2}

GPU_per_dataset = {'coco_pose': 2, 'aic_pose': 2, 'posetrack': 1, 'jrdb2022': 1, 'mhp_pose': 1, 'penn_action': 1, '3dpw': 1, 'halpe': 1,'3dhp': 1, 'h36m_pose': 1, 'AIST': 1,
                    'h36m_par': 3, 'lip_par': 2, 'cihp_par': 2, 'vip_par': 1, 'paper_roll': 2,
                    'deepfashion': 2, 'modanet': 1,
                    'rap2_pa100k_attr': 1, '5set_attr': 1,
                    'reid_4set': 1, 'LaST_PRCC_DGMarket': 1, 'LUperson': 2, 
                    'crowdhuman_det': 16, '5set_det': 16}


total_iters = 80000
data_use_ratio = 1
print_dataset_specific_training_info(total_iters, dataset_samples, dataset_sample_weight, dataset_imgs_per_gpu, GPU_per_dataset, data_use_ratio=data_use_ratio)