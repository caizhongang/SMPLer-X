import os
import csv
import random

import numpy as np
import torch
from einops import rearrange
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import Resize
from torchvision.transforms import Normalize
from torchvision.transforms import RandomHorizontalFlip


class GraphDataset(Dataset):
    def __init__(self, data_type="train", fold_idx=0, data_path="blood-cells", data_size=1.0):
        cases = []
        labels = []
        if data_type == "train":
            for grade in os.listdir(f"{data_path}/dataset2-master/dataset2-master/images/TRAIN"):
                _cases = os.listdir(f"{data_path}/dataset2-master/dataset2-master/images/TRAIN/{grade}")
                _cases = [f"{data_path}/dataset2-master/dataset2-master/images/TRAIN/{grade}/{_}" for _ in _cases if ".jpeg" in _]
                cases = cases + _cases
        elif data_type == "val":
            for grade in os.listdir(f"{data_path}/dataset2-master/dataset2-master/images/TEST"):
                _cases = os.listdir(f"{data_path}/dataset2-master/dataset2-master/images/TEST/{grade}")
                _cases = [f"{data_path}/dataset2-master/dataset2-master/images/TEST/{grade}/{_}" for _ in _cases if ".jpeg" in _]
                cases = cases + _cases
        elif data_type == "test":
            for grade in os.listdir(f"{data_path}/dataset2-master/dataset2-master/images/TEST_SIMPLE"):
                _cases = os.listdir(f"{data_path}/dataset2-master/dataset2-master/images/TEST_SIMPLE/{grade}")
                _cases = [f"{data_path}/dataset2-master/dataset2-master/images/TEST_SIMPLE/{grade}/{_}" for _ in _cases if ".jpeg" in _]
                cases = cases + _cases
        else:
            print("Dataset type error")
            exit()

        random.shuffle(cases)
        
        if data_type == "train":
            assert ((data_size > 0) and (data_size <= 1.0))
            cases = cases[:int(len(cases) * data_size)]
        
        self.cases = cases

    def __len__(self):
        # return 100
        return len(self.cases)

    def __getitem__(self, idx):
        resize = Resize([224, 224])
        normalize = Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        image = np.array(Image.open(self.cases[idx]).convert("RGB")).astype(np.float32) / 255.0
        image = rearrange(torch.tensor(image, dtype=torch.float32), 'h w c -> c h w')
        image = resize(image)
        image = normalize(image)
        
        label_path = str(self.cases[idx].split("/")[5])
        # NEUTROPHIL 0 MONOCYTE 1 EOSINOPHIL 2 LYMPHOCYTE 3
        if label_path == "NEUTROPHIL":
            label = 0
        elif label_path == "MONOCYTE":
            label = 1
        elif label_path == "EOSINOPHIL":
            label = 2
        elif label_path == "LYMPHOCYTE":
            label = 3
        label = np.array(label)
        label = torch.tensor(label, dtype=torch.long)
        return image, label


def BloodDataloader(cfg):
    train_set = DataLoader(
        GraphDataset(data_type="train", fold_idx=cfg.fold, data_path=cfg.data_path, data_size=cfg.data_size),
        batch_size=cfg.bs,
        shuffle=True,
        num_workers=cfg.num_workers,
        drop_last=True,
    )
    
    val_set = DataLoader(
        GraphDataset(data_type="val", fold_idx=cfg.fold, data_path=cfg.data_path),
        batch_size=cfg.bs,
        shuffle=True,
        num_workers=cfg.num_workers,
        drop_last=True,
    )

    test_set = DataLoader(
        GraphDataset(data_type="test", fold_idx=cfg.fold, data_path=cfg.data_path),
        batch_size=cfg.bs,
        shuffle=False,
        num_workers=cfg.num_workers,
        drop_last=False,
    )

    return train_set, val_set, test_set

