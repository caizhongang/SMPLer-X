def cat_dataset(batch_size, iters, dataset_size, imgs_per_gpu, sample_weight):
    print(f"====> dataset_size:\t{dataset_size}")
    print(f"====> sample_weight:\t{sample_weight}")
    print(f"====> iters:\t{iters}")
    print(f"initial batch_size:\t{batch_size}")
    total = sum(dataset_size.values())
    t_wise_batchsize = {k: v / total * batch_size for k, v in dataset_size.items()}
    t_wise_gpuse = {k: v / imgs_per_gpu[k] for k, v in t_wise_batchsize.items()}

    print(f"-----------------------rounded-----------------------")
    gpus = 1
    while gpus % 8 != 0:
        t_wise_batchsize = {k: round(v / total * batch_size / imgs_per_gpu[k]) * imgs_per_gpu[k] for k, v in
                            dataset_size.items()}
        t_wise_gpuse = {k: v // imgs_per_gpu[k] for k, v in t_wise_batchsize.items()}
        t_wise_epochs = {k: iters * t_wise_batchsize[k] / v for k, v in dataset_size.items()}
        batch_size += 1
        gpus = sum(t_wise_gpuse.values())
    loss_weights = {k: v * t_wise_batchsize[k] for k, v in sample_weight.items()}
    print(f"rounded batch_size:\t{sum(t_wise_batchsize.values())}")
    print(f"task-wise batchsize:\t{t_wise_batchsize}")
    print(f"task-wise epoch:\t{t_wise_epochs}\t avg apoch: {sum(t_wise_epochs.values()) / len(t_wise_epochs)}")
    print(f"loss weights:\t{loss_weights}")

    print(f"task-wise gpus:\t{t_wise_gpuse}\ngpus:\t{gpus} ({gpus // 8} nodes)")


# example usage
cat_dataset(batch_size=600,
            iters=10000,
            dataset_size={'reid':118063, 'pose': 149813,
                          'h36_parse': 62668, 'LIP': 30462, 'CIHP': 28280},
            imgs_per_gpu={'reid':96, 'pose': 16,
                          'h36_parse': 7, 'LIP': 7, 'CIHP': 7},
            sample_weight={'reid': 2, 'pose':8000,
                           'h36_parse': 1,
                           'LIP': 1,
                           'CIHP': 1,})
# output:
# ====> dataset_size:	{'reid': 118063, 'pose': 149813, 'h36_parse': 62668, 'LIP': 30462, 'CIHP': 28280}
# ====> sample_weight:	{'reid': 2, 'pose': 8000, 'h36_parse': 1, 'LIP': 1, 'CIHP': 1}
# ====> iters:	10000
# initial batch_size:	600
# -----------------------rounded-----------------------
# rounded batch_size:	658
# task-wise batchsize:	{'reid': 192, 'pose': 256, 'h36_parse': 105, 'LIP': 56, 'CIHP': 49}
# task-wise epoch:	{'reid': 16.262503917400032, 'pose': 17.087969668853837, 'h36_parse': 16.75496266036893, 'LIP': 18.383559845052854, 'CIHP': 17.326732673267326}	 avg apoch: 17.163145752988598
# loss weights:	{'reid': 384, 'pose': 2048000, 'h36_parse': 105, 'LIP': 56, 'CIHP': 49}
# task-wise gpus:	{'reid': 2, 'pose': 16, 'h36_parse': 15, 'LIP': 8, 'CIHP': 7}
# gpus:	48 (6 nodes)