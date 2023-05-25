import argparse
from cgi import test
import logging
from sklearn.model_selection import KFold
from torchvision import models
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.cuda.amp.autocast_mode import autocast
from torch.cuda.amp.grad_scaler import GradScaler
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import numpy as np
from base_vit import ViT
from lora import LoRA_ViT
from utils.dataloader import kneeDataloader
from utils.dataloader_inbreast import InbreastDataset,InbreastDataloader
from utils.result import ResultCLS, ResultMLS
from utils.utils import init, save


def train(epoch,trainset):
    running_loss = 0.0
    this_lr = scheduler.get_last_lr()[0]
    net.train()
    for image, label in tqdm(trainset, ncols=60, desc="train", unit="b", leave=None):
        image, label = image.to(device), label.to(device)
        optimizer.zero_grad()
        with autocast(enabled=True):
            pred = net.forward(image)
            loss = loss_func(pred, label)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        running_loss = running_loss + loss.item()
    scheduler.step()

    loss = running_loss / len(trainset)
    logging.info(f"\n\nEPOCH: {epoch}, LOSS : {loss:.3f}, LR: {this_lr:.2e}")
    return


@torch.no_grad()
def eval(epoch,testset,datatype='val'):
    result.init()
    net.eval()
    for image, label in tqdm(testset, ncols=60, desc=datatype, unit="b", leave=None):
        image, label = image.to(device), label.to(device)
        with autocast(enabled=True):
            pred = net.forward(image)
            result.eval(label, pred)
    result.print(epoch,datatype)
    return

def parseNet(cfg):
    if cfg.train_type=='resnet50':
        model=models.__dict__[cfg.train_type]()
        model.load_state_dict(torch.load('../preTrain/resnet50-19c8e357.pth'))
    else:
        model = ViT('B_16_imagenet1k')
        model.load_state_dict(torch.load('../preTrain/B_16_imagenet1k.pth'))
    
    if cfg.train_type == "lora":
        lora_model = LoRA_ViT(model, r=cfg.rank, num_classes=cfg.num_classes)
        num_params = sum(p.numel() for p in lora_model.parameters() if p.requires_grad)
        logging.info(f"trainable parameters: {num_params/2**20:.1f}M")
        net = lora_model.to(device)
    elif cfg.train_type == "full":
        model.fc = nn.Linear(768, cfg.num_classes)
        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logging.info(f"trainable parameters: {num_params/2**20:.1f}M")
        net = model.to(device)
    elif cfg.train_type == "linear":
        model.fc = nn.Linear(768, cfg.num_classes)
        for param in model.parameters():
            param.requires_grad = False
        for param in model.fc.parameters():
            param.requires_grad = True
        num_params = sum(p.numel() for p in model.fc.parameters())
        logging.info(f"trainable parameters: {num_params/2**20:.3f}M")
        net = model.to(device)
    elif cfg.train_type=='resnet50':
        infeature = model.fc.in_features
        model.fc = nn.Linear(infeature, cfg.num_classes)
        num_params = sum(p.numel() for p in model.fc.parameters())
        logging.info(f"trainable parameters: {num_params/2**20:.3f}M")
        net = model.to(device)
    else:
        print("Wrong training type")
        exit()
    return net


if __name__ == "__main__":
    scaler = GradScaler()
    parser = argparse.ArgumentParser()
    parser.add_argument("-bs", type=int, default=8)
    parser.add_argument("-fold", type=int, default=0)
    parser.add_argument("-data_path",type=str, default='../data/INBreast/')
    parser.add_argument("-data_info",type=str,default='foldInfo.json')
    parser.add_argument("-lr", type=float, default=3e-5)
    parser.add_argument("-epochs", type=int, default=20)
    parser.add_argument("-kfold", type=int, default=5)
    parser.add_argument("-num_workers", type=int, default=4)
    parser.add_argument("-num_classes", "-nc", type=int, default=2)
    parser.add_argument("-train_type", "-tt", type=str, default="full", help="lora: only train lora, full: finetune on all, linear: finetune only on linear layer")
    parser.add_argument("-rank", "-r", type=int, default=4)
    cfg = parser.parse_args()
    ckpt_path = init()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(cfg)

    stat=np.zeros([cfg.kfold,7])
    for k in range(cfg.kfold):
        logging.info(f"============== Fold: {k} ==============")
        net=parseNet(cfg)
        net = torch.nn.DataParallel(net) 
        loss_func = nn.CrossEntropyLoss(label_smoothing=0.1).to(device)
        trainset,valset, testset=InbreastDataloader(cfg,fold=k)
        optimizer = optim.Adam(net.parameters(), lr=cfg.lr)
        scheduler = CosineAnnealingLR(optimizer, cfg.epochs, 1e-6)
        result = ResultCLS(cfg.num_classes)

        for epoch in range(1, cfg.epochs+1):
            train(epoch,trainset)
            if epoch%1==0:
                eval(epoch,valset,datatype='val')
                if result.best_epoch == result.epoch:
                    torch.save(net.state_dict(), ckpt_path.replace(".pt", "_best.pt"))
                    eval(epoch,testset,datatype='test')
                    logging.info(f"BEST VAL: {result.best_val_result:.3f}, TEST ACC: {result.test_acc:.4f}, TEST SEN: {result.test_sen:.4f}, TEST SPE: {result.test_spe:.4f},\
                                   TEST PRE: {result.test_sen:.4f},TEST F1: {result.test_f1:.4f}, TEST AUC: {result.test_auc:.4f}, EPOCH: {(result.best_epoch):3}")
        stat[k][0]=result.test_acc
        stat[k][1]=result.test_sen
        stat[k][2]=result.test_spe
        stat[k][3]=result.test_pre
        stat[k][4]=result.test_f1
        stat[k][5]=result.test_auc
    logging.info(f"============== {cfg.kfold} fold results: ==============")
    logging.info(f"ACC: {stat.mean(axis=0)[0]:.3f}±{stat.std(axis=0)[0]:.3f}")
    logging.info(f"SEN: {stat.mean(axis=0)[1]:.3f}±{stat.std(axis=0)[1]:.3f}")
    logging.info(f"SPE: {stat.mean(axis=0)[2]:.3f}±{stat.std(axis=0)[2]:.3f}")
    logging.info(f"PRE: {stat.mean(axis=0)[3]:.3f}±{stat.std(axis=0)[3]:.3f}")
    logging.info(f"F1: {stat.mean(axis=0)[4]:.3f}±{stat.std(axis=0)[4]:.3f}")
    logging.info(f"AUC: {stat.mean(axis=0)[5]:.3f}±{stat.std(axis=0)[5]:.3f}")

