from functools import partial
import numpy as np
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T
import os
from tqdm import tqdm
from PIL import Image
import torch
cxpFinding={
        "Enlarged":0,
        "Cardiomegaly":1,
        "Lung Opacity":2,
        "Lung Lesion":3,
        "Edema":4,
        "Consolidation":5,
        "Pneumonia":6,
        "Atelectasis":7,
        "Pneumothorax":8,
        "Pleural Effusion":9,
        "Pleural Other":10,
        "Fracture":11,
        }
class cxpDataset(Dataset):
    def __init__(self, imgPrefix, labelPath,  mode='train'):
        self.imgPrefix=imgPrefix
        self.mode=mode
        self._label_header = None
        self._image_paths = []
        self._labels = []
        self._mode = mode
        self.Trans=T.Compose([
            T.Resize(size=(384,384)),
            T.RandomHorizontalFlip(p=0.5),
            T.ToTensor(),
            T.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
        ])
        self.testTrans=T.Compose([
            T.Resize(size=(384,384)),
            T.ToTensor(),
            T.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
        ])
        self.dict = [{'1.0': 1, '': 0, '0.0': 0, '-1.0': 0},
                     {'1.0': 1, '': 0, '0.0': 0, '-1.0': 1}, ]
        with open(labelPath) as f:
            header = f.readline().strip('\n').split(',')
            self._label_header = header[6:18]
            for line in tqdm(f):
                labels = []
                fields = line.strip('\n').split(',')
                image_path = fields[0]
                for index, value in enumerate(fields[6:18]):
                    if index == 4 or index == 7:
                        labels.append(self.dict[1].get(value))
                    else:
                        labels.append(self.dict[0].get(value))
                # assert os.path.exists(os.path.join(self.imgPrefix,image_path)), image_path
                self._image_paths.append(os.path.join(self.imgPrefix,image_path))
                self._labels.append(labels)

        self._num_image = len(self._image_paths)

    def __len__(self):
        return self._num_image

    def __getitem__(self, idx):
        img = Image.open(self._image_paths[idx]).convert('RGB')
        if self.mode =='train':
            img=self.Trans(img)
        else:
            img=self.testTrans(img)
        gt=self._labels[idx]
        gt=torch.tensor(gt,dtype=torch.float32)

        return img,gt

def cxpDataloader(cfg):

    cxp=partial(cxpDataset,imgPrefix=cfg.data_path)
    train_set = DataLoader(
        cxp(labelPath="/public_bme/data/CheXpert-v1.0/train.csv",mode="train"),
        batch_size=cfg.bs,
        shuffle=True,
        num_workers=cfg.num_workers,
        drop_last=True,
    )
    val_set = DataLoader(
        cxp(labelPath="/public_bme/data/CheXpert-v1.0/valid.csv",mode="val"),
        batch_size=cfg.bs,
        shuffle=False,
        num_workers=cfg.num_workers,
        drop_last=False,
    )
    test_set = DataLoader(
        cxp(labelPath="/public_bme/data/CheXpert-v1.0/valid.csv",mode="test"),
        batch_size=cfg.bs,
        shuffle=False,
        num_workers=cfg.num_workers,
        drop_last=False,
    )

    return train_set, val_set, test_set