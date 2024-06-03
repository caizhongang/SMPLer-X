import os
import argparse
import cv2
import readline
import tqdm
import time

import torch
from torch.utils.data import DataLoader
from PIL import Image

# from smplerx.main.inference import Inferer
from main.inference import SmplerxData, Inferer


import pdb


def inference(args):

    # load model
    num_gpus = 1 if torch.cuda.is_available() else -1
    inferer = Inferer(args.pretrained_model, num_gpus)

    # test annotations
    annotations = [
        {'image_path': '/home/weichen/wc_workspace/laoyouji/frames/Season_1/S01E01/frame025000.jpg', 'bbox': [0, 0, 1000, 1000]},
        {'image_path': '/home/weichen/wc_workspace/laoyouji/frames/Season_1/S01E01/frame026000.jpg', 'bbox': [0, 0, 1000, 1000]}
    ]

    annotations = annotations*500
    anno_len = len(annotations)
    
    start_time = time.time()
    batch_size = 1
    
    dataset = SmplerxData(annotations=annotations)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    preproces_time = time.time()

    for batch in tqdm.tqdm(dataloader):

        smplx_pred, meta, mesh = inferer.batch_infer_given_bbox(batch['image'], batch['bbox'])

    end_time = time.time()

    # print report, time in seconds
    print(f'Instance number: {anno_len}, Batch size: {batch_size}')
    print(f'Preprocess time: {preproces_time-start_time:02f}, FPS: {anno_len/(preproces_time-start_time):02f}')
    print(f'Inference time: {end_time-preproces_time:02f}, FPS: {anno_len/(end_time-preproces_time):02f}')


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('--in_threshold', type=float, default=0.5)
    parser.add_argument('--pretrained_model', type=str, default='smpler_x_h32')

    args = parser.parse_args()

    inference(args)




