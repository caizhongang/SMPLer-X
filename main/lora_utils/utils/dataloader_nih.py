from cgi import test
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

disease={"Atelectasis":0,
        "Cardiomegaly":1,
        "Effusion":2,
        "Infiltration":3,
        "Mass":4,
        "Nodule":5,
        "Pneumonia":6,
        "Pneumothorax":7,
        "Consolidation":8,
        "Edema":9,
        "Emphysema":10,
        "Fibrosis":11,
        "Pleural_Thickening":12,
        "Hernia":13,

}

class nihDataset(Dataset):
    def __init__(self, dataPath='../data/NIH_X-ray/',dataInfo='nih_split_712.json',annotation='Data_Entry_2017_jpg.csv',data_type="train"):
        self.namelist=json.load(open(os.path.join(dataPath,dataInfo)))[data_type]
        self.imgPath=os.path.join(dataPath,'images/')
        self.df = pd.read_csv(os.path.join(dataPath,annotation))
        self.trans=T.Compose([
            T.Resize(size=384),
            T.RandomHorizontalFlip(p=0.5),
            T.ToTensor(),
            T.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
        ])


    def __len__(self):

        return len(self.namelist)
    
    def __getitem__(self, idx):
        filename=self.namelist[idx]

        img = Image.open(os.path.join(self.imgPath,filename))
        img=self.trans(img)
        
        findings=self.df.loc[self.df['Image Index'].values==filename]['Finding Labels'].values[0].split("|")
        gt=np.zeros([len(disease)],dtype=np.int64)
        if findings[0]!="No Finding":
            gt[list(map(lambda x: disease[x], findings))]=1
        
        gt=torch.tensor(gt,dtype=torch.float32)

        return img,gt


def nihDataloader(cfg):

    nih=partial(nihDataset,dataPath=cfg.data_path,dataInfo=cfg.data_info,annotation=cfg.annotation)
    train_set = DataLoader(
        nih(data_type="train"),
        batch_size=cfg.bs,
        shuffle=True,
        num_workers=cfg.num_workers,
        drop_last=True,
    )
    val_set = DataLoader(
        nih(data_type="val"),
        batch_size=cfg.bs,
        shuffle=True,
        num_workers=cfg.num_workers,
        drop_last=True,
    )
    test_set = DataLoader(
        nih(data_type="test"),
        batch_size=cfg.bs,
        shuffle=False,
        num_workers=cfg.num_workers,
        drop_last=False,
    )

    return train_set, val_set, test_set

# if __name__=="__main__":
#     prev_case=None
#     dataInfo={}
#     testset=[]
#     with open('../data/NIH_X-ray/test_list_jpg.txt','r') as f:
#         content=f.readlines()
#         for c in content:
#             testset.append(c.strip('\n'))
    
#     trainset=[]
#     valset=[]
#     train_ratio=7/8
#     with open('../data/NIH_X-ray/train_val_list_jpg.txt','r') as f:
#         train_content=f.readlines()
#         trainNum=int(len(train_content)*train_ratio)
#         for i in range(0,trainNum):
#             trainset.append(train_content[i].strip('\n'))
#         for i in range(trainNum,len(train_content)):
#             valset.append(train_content[i].strip('\n'))
#         # for c in content:
#         #     testset.append(c.strip('\n'))
#     # dataInfo['test']=testset
#     dataInfo['meta']={'trainSize':len(trainset),'valSize':len(valset),'testSize':len(testset)}
#     dataInfo['train']=trainset
#     dataInfo['val']=valset
#     dataInfo['test']=testset
    
#     with open('nih_split_712.json', 'w') as json_file:
#         json.dump(dataInfo, json_file,indent = 4)

