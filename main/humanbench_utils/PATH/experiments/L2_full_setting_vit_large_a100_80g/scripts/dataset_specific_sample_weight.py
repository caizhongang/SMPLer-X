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

dataset_imgs_per_gpu = {'coco_pose': 448, 'aic_pose': 448, 'posetrack': 224, 'jrdb2022': 224, 'mhp_pose': 96, 'penn_action': 128, '3dpw': 128, 'halpe': 64,'3dhp': 128, 'h36m_pose': 128, 'AIST': 128,
                        'h36m_par': 64, 'lip_par': 36, 'cihp_par': 48, 'vip_par': 16, 'paper_roll': 48,
                        'deepfashion': 64, 'modanet': 32,
                        'rap2_pa100k_attr': 128, '5set_attr': 116,
                        'reid_4set': 112, 'LaST_PRCC_DGMarket': 96, 'LUperson': 384, 
                        'crowdhuman_det': 4, '5set_det': 4}

GPU_per_dataset = {'coco_pose': 1, 'aic_pose': 1, 'posetrack': 1, 'jrdb2022': 1, 'mhp_pose': 1, 'penn_action': 1, '3dpw': 1, 'halpe': 1,'3dhp': 1, 'h36m_pose': 1, 'AIST': 1,
                    'h36m_par': 1, 'lip_par': 1, 'cihp_par': 1, 'vip_par': 1, 'paper_roll': 1,
                    'deepfashion': 1, 'modanet': 1,
                    'rap2_pa100k_attr': 1, '5set_attr': 1,
                    'reid_4set': 1, 'LaST_PRCC_DGMarket': 1, 'LUperson': 1, 
                    'crowdhuman_det': 8, '5set_det': 9}


total_iters = 80000
data_use_ratio = 1
print_dataset_specific_training_info(total_iters, dataset_samples, dataset_sample_weight, dataset_imgs_per_gpu, GPU_per_dataset, data_use_ratio=data_use_ratio)