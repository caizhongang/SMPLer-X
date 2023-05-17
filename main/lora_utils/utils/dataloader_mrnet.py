import os

import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import Normalize, RandomHorizontalFlip, Resize


class MRNetDataset(Dataset):
    def __init__(self, data_type="train"):
        self.type = data_type
        self.cases = os.listdir(f"data/MRNet/{data_type}/sagittal")
        self.cases = [f"data/MRNet/{data_type}/sagittal/" + _ for _ in self.cases]
        self.labels = {}
        _labels = np.loadtxt(f"data/MRNet/{data_type}-abnormal.csv", dtype=str, delimiter=",")
        for i in range(len(_labels)):
            self.labels[_labels[i, 0]] = _labels[i, 1]

    def __len__(self):
        # return 10
        return len(self.cases)

    def __getitem__(self, idx):
        resize = Resize([224, 224])
        image = np.load(self.cases[idx]).astype(np.float32) / 255.0
        if image.shape[0] > 30:
            image = image[:30]
        else:
            image = np.concatenate([image, np.zeros_like(image[: 30 - image.shape[0]])], axis=0)
        image = torch.tensor(image, dtype=torch.float32)
        image = image[:, None].repeat(1, 3, 1, 1)
        image = resize(image)

        label = self.labels[self.cases[idx].split("/")[-1].split(".")[0]]
        label = int(label)
        label = torch.tensor(label, dtype=torch.long)

        return image, label


def kneeDataloader(cfg):
    train_set = DataLoader(
        MRNetDataset(data_type="train"),
        batch_size=cfg.bs,
        shuffle=True,
        num_workers=cfg.num_workers,
        drop_last=True,
    )

    val_set = DataLoader(
        MRNetDataset(data_type="valid"),
        batch_size=cfg.bs,
        shuffle=True,
        num_workers=cfg.num_workers,
        drop_last=True,
    )

    return train_set, val_set
