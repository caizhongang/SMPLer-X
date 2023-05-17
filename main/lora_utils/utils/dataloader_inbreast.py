from cgi import test
from nis import maps
import os
from functools import partial
from tkinter.messagebox import NO
import json
import pandas as pd
import numpy as np
import torch
from einops import rearrange
from PIL import Image
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as T
import matplotlib.pyplot as plt
from skimage.exposure import equalize_adapthist

schedule=[
            [[0],[1],[2,3,4]],
            [[1],[2],[3,4,0]],
            [[2],[3],[4,0,1]],
            [[3],[4],[0,1,2]],
            [[4],[0],[1,2,3]],
        ]
mapSchedule={'test':0,'val':1,'train':2}
class InbreastDataset(Dataset):
    def __init__(self, dataPath='../data/INBreast/', dataInfo='foldInfo.json',data_type="train",fold=0):
        self.namelist=json.load(open(os.path.join(dataPath,dataInfo)))
        self.fold=fold
        self.dataPath=dataPath
        self.data_type=data_type
        self.dataset=self.prepareData()
        self.trans=T.Compose([
            T.Resize(size=(384,384)),
            T.RandomHorizontalFlip(p=0.5),
            T.ToTensor(),
            T.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
        ])
    def prepareData(self):
        foldSplit=schedule[self.fold][mapSchedule[self.data_type]]
        if len(foldSplit)==1:
            return self.namelist[foldSplit[0]]
        else:
            repo=self.namelist[foldSplit[0]]+self.namelist[foldSplit[1]]+self.namelist[foldSplit[2]]
            return repo


    def pre_img(self, img):
        img=np.array(img)
        img=img/255
        img_heq = equalize_adapthist(img, clip_limit=0.04, nbins=256)
        img_heq=(img_heq*255).astype(np.uint8)
        # img_clahe=np.stack([img_clahe,img_clahe,img_clahe],axis=0)
        
        img_heq=Image.fromarray(img_heq)
        return img_heq


    def __len__(self):

        return len(self.dataset)
    
    def __getitem__(self, idx):
        info=self.dataset[idx]
        filename=info['img']
        img = Image.open(os.path.join(self.dataPath,filename)).convert("RGB")
        # img=self.pre_img(img)
        img=self.trans(img)
        # # img_clahe=torch.tensor(img_clahe,dtype=torch.float32)
        # img=torch.permute(img,dims=[2,0,1])
        level=info['Bi-RADs']
        if level=='1':
            gt=0
        elif level=='2' or level=='3':
            gt=0
        else:
            gt=1
        
        gt=torch.tensor(gt,dtype=torch.long)

        return img,gt


def InbreastDataloader(cfg,fold):
    # foldInfo=json.load(open(os.path.join(cfg.data_path,cfg.data_info)))
    Inbreast=partial(InbreastDataset,dataPath=cfg.data_path,dataInfo=cfg.data_info,fold=fold)
    train=Inbreast(data_type="train")
    val=Inbreast(data_type="val")
    test=Inbreast(data_type="test")
    train_set = DataLoader(
        train,
        batch_size=cfg.bs, shuffle=True, num_workers=cfg.num_workers, drop_last=True,
    )
    val_set = DataLoader(
        val,
        batch_size=cfg.bs, shuffle=False, num_workers=cfg.num_workers, drop_last=False,
    )
    test_set = DataLoader(
        test,
        batch_size=cfg.bs, shuffle=False, num_workers=cfg.num_workers, drop_last=False,
    )

    return train_set, val_set, test_set




