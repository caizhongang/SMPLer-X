from functools import partial
import numpy as np
import argparse
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T
import os
import json
from tqdm import tqdm
from PIL import Image
import torch
mimicFinding={
        "Enlarged Cardiomediastinum":0,
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
class mimicDataset(Dataset):
    def __init__(self, imgPrefix='/public_bme/data/physionet.org/files/mimic-cxr-jpg/2.0.0/files/', dataInfo='mimic_val_split.json',  mode='train'):
        
        self.imgPrefix=imgPrefix
        self.mode=mode
        self._label_header = None
        self._mode = mode
        self.info=json.load(open(os.path.join(self.imgPrefix,dataInfo)))
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

        self.findings={
            # "Enlarged Cardiomediastinum":0,
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


    def __len__(self):
        return len(self.info)

    def __getitem__(self, idx):
        img_path=os.path.join(self.imgPrefix,self.info[idx]["img"])
        img = Image.open(img_path).convert('RGB')
        if self.mode =='train':
            img=self.Trans(img)
        else:
            img=self.testTrans(img)
        gt=self.info[idx]['gt_list']
        gt=torch.tensor(gt,dtype=torch.float32)

        return img,gt

def mimicDataloader(cfg):

    mimic=partial(mimicDataset,imgPrefix='/public_bme/data/physionet.org/files/mimic-cxr-jpg/2.0.0/files/')
    # train_set = DataLoader(
    #     cxp(labelPath="../data/CheXpert-v1.0/train.csv",mode="train"),
    #     batch_size=cfg.bs,
    #     shuffle=True,
    #     num_workers=cfg.num_workers,
    #     drop_last=True,
    # )
    val_set = DataLoader(
        mimic(dataInfo='mimic_val_split.json',mode="val"),
        batch_size=cfg.bs,
        shuffle=False,
        num_workers=cfg.num_workers,
        drop_last=False,
    )
    test_set = DataLoader(
        mimic(dataInfo='mimic_test_split.json',mode="test"),
        batch_size=cfg.bs,
        shuffle=False,
        num_workers=cfg.num_workers,
        drop_last=False,
    )
    return  val_set, test_set
    # return train_set, val_set, test_set

if __name__=="__main__":
    val_set=mimicDataset(imgPrefix="../data/mimic-cxr/",dataInfo="mimic_val_split.json",mode="val")
    parser = argparse.ArgumentParser()
    parser.add_argument("-bs", type=int, default=128)
    parser.add_argument("-fold", type=int, default=0)
    parser.add_argument("-data_path",type=str, default='/public_bme/data/')
    # parser.add_argument("-data_path",type=str, default='../data/NIH_X-ray/')
    parser.add_argument("-data_info",type=str,default='nih_split_712.json')
    parser.add_argument("-annotation",type=str,default='Data_Entry_2017_jpg.csv')
    parser.add_argument("-lr", type=float, default=1e-3)
    parser.add_argument("-epochs", type=int, default=20)
    parser.add_argument("-num_workers", type=int, default=4)
    parser.add_argument("-num_classes", "-nc", type=int, default=12)
    parser.add_argument("-backbone", type=str, default='base(384)')
    parser.add_argument("-train_type", "-tt", type=str, default="lora", help="lora: only train lora, full: finetune on all, linear: finetune only on linear layer")
    parser.add_argument("-rank", "-r", type=int, default=4)
    cfg = parser.parse_args()

    valset,testset=mimicDataloader(cfg)