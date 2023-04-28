## expr2

dataset_samples = {'coco_pose': 149813, 'aic_pose': 378352, 'posetrack': 97174, 'jrdb2022': 310035, 'mhp_pose': 41128, 'penn_action': 163839, '3dpw': 74620,
                    'h36m_par': 62668, 'lip_par': 30462, 'cihp_par': 28280, 'vip_par': 18469,
                    'deepfashion': 191961, 'modanet': 52245,
                    'rap2_attr': 67943, 'pa100k_attr': 90000, 'HARDHC_attr': 28336, 'uavhuman_attr': 16183, 'parse27k_attr': 27482, 'duke_attr': 34183, 'market_attr': 12936,
                    'reid_5set': 118063, 'DGMarket': 128309,
                    'crowdhuman_det': 15000, 'widepersons_cocoperson_det': 9000+64115}

dataset_sample_weight = {'coco_pose': 8000, 'aic_pose': 6000, 'posetrack': 6000, 'jrdb2022': 4000, 'mhp_pose': 4000, 'penn_action': 4000, '3dpw': 4000,
                        'h36m_par': 20, 'lip_par': 20, 'cihp_par': 20, 'vip_par': 20,
                        'deepfashion': 15, 'modanet': 15,
                        'rap2_attr': 0.1, 'pa100k_attr': 0.1, 'HARDHC_attr': 0.05, 'uavhuman_attr': 0.05, 'parse27k_attr': 0.05, 'duke_attr': 0.05, 'market_attr': 0.05,
                        'reid_5set': 5, 'DGMarket': 1,
                        'crowdhuman_det': 30, 'widepersons_cocoperson_det': 10}

dataset_imgs_per_gpu = {'coco_pose': 224, 'aic_pose': 224, 'posetrack': 224, 'jrdb2022': 224, 'mhp_pose': 96, 'penn_action': 128, '3dpw': 128,
                        'h36m_par': 20, 'lip_par': 12, 'cihp_par': 16, 'vip_par': 16,
                        'deepfashion': 32, 'modanet': 32,
                        'rap2_attr': 64, 'pa100k_attr': 64, 'HARDHC_attr': 32, 'uavhuman_attr': 16, 'parse27k_attr': 32, 'duke_attr': 32, 'market_attr': 8,
                        'reid_5set': 192, 'DGMarket': 128,
                        'crowdhuman_det': 2, 'widepersons_cocoperson_det': 2}

GPU_per_dataset = {'coco_pose': 2, 'aic_pose': 2, 'posetrack': 1, 'jrdb2022': 1, 'mhp_pose': 1, 'penn_action': 1, '3dpw': 1,
                    'h36m_par': 4, 'lip_par': 3, 'cihp_par': 3, 'vip_par': 1,
                    'deepfashion': 2, 'modanet': 1,
                    'rap2_attr': 1, 'pa100k_attr': 1, 'HARDHC_attr': 1, 'uavhuman_attr': 1, 'parse27k_attr': 1, 'duke_attr': 1, 'market_attr': 1,
                    'reid_5set': 1, 'DGMarket': 1,
                    'crowdhuman_det': 16, 'widepersons_cocoperson_det': 16}

total_iters = 80000
data_use_ratio = 1

total iters:  80000
############### epoch per dataset ###############
0: coco_pose: 239.23157536395374
1: aic_pose: 94.72660379752189
2: posetrack: 184.41146808817174
3: jrdb2022: 57.79992581482736
4: mhp_pose: 186.73409842443104
5: penn_action: 62.50038147205488
6: 3dpw: 137.22862503350308
7: h36m_par: 102.12548669177252
8: lip_par: 94.54402206027181
9: cihp_par: 135.78500707213578
10: vip_par: 69.30532243218366
11: deepfashion: 26.672084433817286
12: modanet: 48.99990429706192
13: rap2_attr: 75.35728478283266
14: pa100k_attr: 56.888888888888886
15: HARDHC_attr: 90.34443817052512
16: uavhuman_attr: 79.09534696904159
17: parse27k_attr: 93.15188123135142
18: duke_attr: 74.8910277038294
19: market_attr: 49.474335188620906
20: reid_5set: 130.10003133920026
21: DGMarket: 79.80734009305661
22: crowdhuman_det: 170.66666666666666
23: widepersons_cocoperson_det: 35.01333515694454
############### config ###############
             sw|  gpus|  bs
0: coco_pose: 3584000, 2, 224
1: aic_pose: 2688000, 2, 224
2: posetrack: 1344000, 1, 224
3: jrdb2022: 896000, 1, 224
4: mhp_pose: 384000, 1, 96
5: penn_action: 512000, 1, 128
6: 3dpw: 512000, 1, 128
7: h36m_par: 1600, 4, 20
8: lip_par: 720, 3, 12
9: cihp_par: 960, 3, 16
10: vip_par: 320, 1, 16
11: deepfashion: 960, 2, 32
12: modanet: 480, 1, 32
13: rap2_attr: 6.4, 1, 64
14: pa100k_attr: 6.4, 1, 64
15: HARDHC_attr: 1.6, 1, 32
16: uavhuman_attr: 0.8, 1, 16
17: parse27k_attr: 1.6, 1, 32
18: duke_attr: 1.6, 1, 32
19: market_attr: 0.4, 1, 8
20: reid_5set: 960, 1, 192
21: DGMarket: 128, 1, 128
22: crowdhuman_det: 960, 16, 2
23: widepersons_cocoperson_det: 320, 16, 2
############### total GPUs ###############
GPUs:  64
nodes:  8



## sample weight 2

dataset_samples = {'coco_pose': 149813, 'aic_pose': 378352, 'posetrack': 97174, 'jrdb2022': 310035, 'mhp_pose': 41128, 'penn_action': 163839, '3dpw': 74620, 
                    'h36m_par': 62668, 'lip_par': 30462, 'cihp_par': 28280, 'vip_par': 18469,
                    'deepfashion': 191961, 'modanet': 52245, 
                    'rap2_attr': 67943, 'pa100k_attr': 90000, 'HARDHC_attr': 28336, 'uavhuman_attr': 16183, 'parse27k_attr': 27482, 'duke_attr': 34183, 'market_attr': 12936,
                    'reid_5set': 118063, 'DGMarket': 128309,
                    'crowdhuman_det': 15000, '5set_det': 9000+64115}

dataset_sample_weight = {'coco_pose': 8000, 'aic_pose': 6000, 'posetrack': 6000, 'jrdb2022': 4000, 'mhp_pose': 4000, 'penn_action': 4000, '3dpw': 4000, 
                        'h36m_par': 20, 'lip_par': 20, 'cihp_par': 20, 'vip_par': 20,
                        'deepfashion': 15, 'modanet': 15,
                        'rap2_attr': 0.1, 'pa100k_attr': 0.1, 'HARDHC_attr': 0.05, 'uavhuman_attr': 0.05, 'parse27k_attr': 0.05, 'duke_attr': 0.05, 'market_attr': 0.05,
                        'reid_5set': 5, 'DGMarket': 1,
                        'crowdhuman_det': 10, '5set_det': 10}

dataset_imgs_per_gpu = {'coco_pose': 224, 'aic_pose': 224, 'posetrack': 224, 'jrdb2022': 224, 'mhp_pose': 96, 'penn_action': 128, '3dpw': 128, 
                        'h36m_par': 20, 'lip_par': 12, 'cihp_par': 16, 'vip_par': 16,
                        'deepfashion': 32, 'modanet': 32,
                        'rap2_attr': 72, 'pa100k_attr': 72, 'HARDHC_attr': 18, 'uavhuman_attr': 12, 'parse27k_attr': 18, 'duke_attr': 24, 'market_attr': 8,
                        'reid_5set': 216, 'DGMarket': 96,
                        'crowdhuman_det': 2, '5set_det': 2}

GPU_per_dataset = {'coco_pose': 2, 'aic_pose': 2, 'posetrack': 1, 'jrdb2022': 1, 'mhp_pose': 1, 'penn_action': 1, '3dpw': 1,
                    'h36m_par': 4, 'lip_par': 3, 'cihp_par': 3, 'vip_par': 1,
                    'deepfashion': 2, 'modanet': 1,
                    'rap2_attr': 1, 'pa100k_attr': 1, 'HARDHC_attr': 1, 'uavhuman_attr': 1, 'parse27k_attr': 1, 'duke_attr': 1, 'market_attr': 1,
                    'reid_5set': 1, 'DGMarket': 1,
                    'crowdhuman_det': 16, '5set_det': 16}

total iters:  80000
############### epoch per dataset ###############
0: coco_pose: 239.23157536395374
1: aic_pose: 94.72660379752189
2: posetrack: 184.41146808817174
3: jrdb2022: 57.79992581482736
4: mhp_pose: 186.73409842443104
5: penn_action: 62.50038147205488
6: 3dpw: 137.22862503350308
7: h36m_par: 102.12548669177252
8: lip_par: 94.54402206027181
9: cihp_par: 135.78500707213578
10: vip_par: 69.30532243218366
11: deepfashion: 26.672084433817286
12: modanet: 48.99990429706192
13: rap2_attr: 84.77694538068675
14: pa100k_attr: 64.0
15: HARDHC_attr: 50.81874647092038
16: uavhuman_attr: 59.32151022678119
17: parse27k_attr: 52.39793319263518
18: duke_attr: 56.16827077787204
19: market_attr: 49.474335188620906
20: reid_5set: 146.3625352566003
21: DGMarket: 59.85550506979246
22: crowdhuman_det: 170.66666666666666
23: 5set_det: 35.01333515694454
############### config ###############
             sw|  gpus|  bs
0: coco_pose: 3584000, 2, 224
1: aic_pose: 2688000, 2, 224
2: posetrack: 1344000, 1, 224
3: jrdb2022: 896000, 1, 224
4: mhp_pose: 384000, 1, 96
5: penn_action: 512000, 1, 128
6: 3dpw: 512000, 1, 128
7: h36m_par: 1600, 4, 20
8: lip_par: 720, 3, 12
9: cihp_par: 960, 3, 16
10: vip_par: 320, 1, 16
11: deepfashion: 960, 2, 32
12: modanet: 480, 1, 32
13: rap2_attr: 7.2, 1, 72
14: pa100k_attr: 7.2, 1, 72
15: HARDHC_attr: 0.9, 1, 18
16: uavhuman_attr: 0.6000000000000001, 1, 12
17: parse27k_attr: 0.9, 1, 18
18: duke_attr: 1.2000000000000002, 1, 24
19: market_attr: 0.4, 1, 8
20: reid_5set: 1080, 1, 216
21: DGMarket: 96, 1, 96
22: crowdhuman_det: 320, 16, 2
23: 5set_det: 320, 16, 2
############### total GPUs ###############
GPUs:  64
nodes:  8



# exp3-share pos_embed

dataset_samples = {'coco_pose': 149813, 'aic_pose': 378352, 'posetrack': 97174, 'jrdb2022': 310035, 'mhp_pose': 41128, 'penn_action': 163839, '3dpw': 74620, 'halpe': 41712,'3dhp': 1031701, 'h36m_pose': 312187, 'AIST': 1015257,
                    'h36m_par': 62668, 'lip_par': 30462, 'cihp_par': 28280, 'vip_par': 18469, 'paper_roll': 1035825,
                    'deepfashion': 191961, 'modanet': 52245,
                    'rap2_pa100k_attr': 67943 + 90000, '5set_attr': 28336 + 16183 + 27482 + 34183 + 12936,
                    'reid_4set': 13414, 'LaST_PRCC_DGMarket': 128309 + 71248 + 17896, 'LUperson': 5000000, 
                    'crowdhuman_det': 15000, '5set_det': 9000+91500+23892+2975+118287}

dataset_sample_weight = {'coco_pose': 8000, 'aic_pose': 6000, 'posetrack': 6000, 'jrdb2022': 4000, 'mhp_pose': 4000, 'penn_action': 4000, '3dpw': 4000, 'halpe': 2000,'3dhp': 2000, 'h36m_pose': 2000, 'AIST': 2000,
                        'h36m_par': 20, 'lip_par': 20, 'cihp_par': 20, 'vip_par': 20, 'paper_roll': 15,
                        'deepfashion': 15, 'modanet': 15,
                        'rap2_pa100k_attr': 0.1, '5set_attr': 0.1,
                        'reid_4set': 5, 'LaST_PRCC_DGMarket': 0.1, 'LUperson': 1,
                        'crowdhuman_det': 10, '5set_det': 10}

dataset_imgs_per_gpu = {'coco_pose': 224, 'aic_pose': 224, 'posetrack': 224, 'jrdb2022': 224, 'mhp_pose': 96, 'penn_action': 128, '3dpw': 128, 'halpe': 64,'3dhp': 128, 'h36m_pose': 128, 'AIST': 128,
                        'h36m_par': 26, 'lip_par': 18, 'cihp_par': 24, 'vip_par': 16, 'paper_roll': 24,
                        'deepfashion': 32, 'modanet': 32,
                        'rap2_pa100k_attr': 128, '5set_attr': 116,
                        'reid_4set': 96, 'LaST_PRCC_DGMarket': 96, 'LUperson': 192, 
                        'crowdhuman_det': 2, '5set_det': 2}

GPU_per_dataset = {'coco_pose': 2, 'aic_pose': 2, 'posetrack': 1, 'jrdb2022': 1, 'mhp_pose': 1, 'penn_action': 1, '3dpw': 1, 'halpe': 1,'3dhp': 1, 'h36m_pose': 1, 'AIST': 1,
                    'h36m_par': 3, 'lip_par': 2, 'cihp_par': 2, 'vip_par': 1, 'paper_roll': 2,
                    'deepfashion': 2, 'modanet': 1,
                    'rap2_pa100k_attr': 1, '5set_attr': 1,
                    'reid_4set': 1, 'LaST_PRCC_DGMarket': 1, 'LUperson': 2, 
                    'crowdhuman_det': 16, '5set_det': 16}

total iters:  80000
############### epoch per dataset ###############
0: coco_pose: 239.23157536395374
1: aic_pose: 94.72660379752189
2: posetrack: 184.41146808817174
3: jrdb2022: 57.79992581482736
4: mhp_pose: 186.73409842443104
5: penn_action: 62.50038147205488
6: 3dpw: 137.22862503350308
7: halpe: 122.74645186037591
8: 3dhp: 9.925356280550275
9: h36m_pose: 32.80085333470004
10: AIST: 10.086116126261626
11: h36m_par: 99.5723495244782
12: lip_par: 94.54402206027181
13: cihp_par: 135.78500707213578
14: vip_par: 69.30532243218366
15: paper_roll: 3.707189921077402
16: deepfashion: 26.672084433817286
17: modanet: 48.99990429706192
18: rap2_pa100k_attr: 64.83351588864338
19: 5set_attr: 77.90463398253861
20: reid_4set: 572.5361562546593
21: LaST_PRCC_DGMarket: 35.317976758196025
22: LUperson: 6.144
23: crowdhuman_det: 170.66666666666666
24: 5set_det: 10.421161471012073
############### config ###############
             sw|  gpus|  bs
0: coco_pose: 3584000, 2, 224
1: aic_pose: 2688000, 2, 224
2: posetrack: 1344000, 1, 224
3: jrdb2022: 896000, 1, 224
4: mhp_pose: 384000, 1, 96
5: penn_action: 512000, 1, 128
6: 3dpw: 512000, 1, 128
7: halpe: 128000, 1, 64
8: 3dhp: 256000, 1, 128
9: h36m_pose: 256000, 1, 128
10: AIST: 256000, 1, 128
11: h36m_par: 1560, 3, 26
12: lip_par: 720, 2, 18
13: cihp_par: 960, 2, 24
14: vip_par: 320, 1, 16
15: paper_roll: 720, 2, 24
16: deepfashion: 960, 2, 32
17: modanet: 480, 1, 32
18: rap2_pa100k_attr: 12.8, 1, 128
19: 5set_attr: 11.600000000000001, 1, 116
20: reid_4set: 480, 1, 96
21: LaST_PRCC_DGMarket: 9.600000000000001, 1, 96
22: LUperson: 384, 2, 192
23: crowdhuman_det: 320, 16, 2
24: 5set_det: 320, 16, 2
############### total GPUs ###############
GPUs:  64
nodes:  8