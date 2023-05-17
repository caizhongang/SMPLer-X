import argparse
from cgi import test
import logging
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

import timm
from lora import LoRA_ViT_timm
from adapter import Adapter_ViT
from utils.dataloader_oai import kneeDataloader
from utils.dataloader_cxr_cn import cxrDataloader
from utils.dataloader_blood_cell import BloodDataloader
from utils.dataloader_nih import nihDataloader
from utils.result import ResultCLS
from utils.utils import init, save

weightInfo={
            # "small":"WinKawaks/vit-small-patch16-224",
            "base":"vit_base_patch16_224.orig_in21k_ft_in1k",
            "base_dino":"vit_base_patch16_224.dino", # 21k -> 1k
            "base_sam":"vit_base_patch16_224.sam", # 1k
            "base_mill":"vit_base_patch16_224_miil.in21k_ft_in1k", # 1k
            "base_beit":"beitv2_base_patch16_224.in1k_ft_in22k_in1k",
            "base_clip":"vit_base_patch16_clip_224.laion2b_ft_in1k", # 1k
            "base_deit":"deit_base_distilled_patch16_224", # 1k
            "large":"google/vit-large-patch16-224",
            "large_clip":"vit_large_patch14_clip_224.laion2b_ft_in1k", # laion-> 1k
            "large_beit":"beitv2_large_patch16_224.in1k_ft_in22k_in1k", 
            "huge_clip":"vit_huge_patch14_clip_224.laion2b_ft_in1k", # laion-> 1k
            "giant_eva":"eva_giant_patch14_224.clip_ft_in1k", # laion-> 1k
            "giant_clip":"vit_giant_patch14_clip_224.laion2b",
            "giga_clip":"vit_gigantic_patch14_clip_224.laion2b"
            }
    
    


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


if __name__ == "__main__":
    scaler = GradScaler()
    parser = argparse.ArgumentParser()
    parser.add_argument("-bs", type=int, default=16)
    parser.add_argument("-fold", type=int, default=0)
    parser.add_argument("-data_path",type=str, default='../data/NIH_X-ray/')
    parser.add_argument("-data_info",type=str,default='nih_split_712.json')
    parser.add_argument("-annotation",type=str,default='Data_Entry_2017_jpg.csv')
    parser.add_argument("-lr", type=float, default=1e-3)
    parser.add_argument("-epochs", type=int, default=20)
    parser.add_argument("-num_workers", type=int, default=4)
    parser.add_argument("-num_classes", "-nc", type=int, default=14)
    parser.add_argument("-train_type", "-tt", type=str, default="linear", help="lora, full, linear, adapter")
    parser.add_argument("-rank", "-r", type=int, default=4)
    parser.add_argument("-vit", type=str, default="base")
    parser.add_argument("-data_size", type=float, default="1.0")
    cfg = parser.parse_args()
    ckpt_path = init()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(cfg)

    if cfg.train_type=='resnet50':
        model=models.__dict__[cfg.train_type]()
        model.load_state_dict(torch.load('../preTrain/resnet50-19c8e357.pth'))

        # model.load_state_dict()
    else:
        if cfg.vit == "base":
            model = timm.create_model("vit_base_patch16_224", pretrained=True)
        elif cfg.vit == "base_dino":
            model = timm.create_model(weightInfo["base_dino"], pretrained=True)
        elif cfg.vit == "base_sam":
            model = timm.create_model(weightInfo["base_sam"], pretrained=True)
        elif cfg.vit == "base_mill":
            model = timm.create_model(weightInfo["base_mill"], pretrained=True)
        elif cfg.vit == "base_beit":
            model = timm.create_model(weightInfo["base_beit"], pretrained=True)
        elif cfg.vit == "base_clip":
            model = timm.create_model(weightInfo["base_clip"], pretrained=True)
        elif cfg.vit == "base_deit":
            model = timm.create_model(weightInfo["base_deit"], pretrained=True)
        elif cfg.vit == "large_clip":
            model = timm.create_model(weightInfo["large_clip"], pretrained=True)
        elif cfg.vit == "large_beit":
            model = timm.create_model(weightInfo["large_beit"], pretrained=True)
        elif cfg.vit == "huge_clip":
            model = timm.create_model(weightInfo["huge_clip"], pretrained=True)
        elif cfg.vit == "giant_eva":
            model = timm.create_model(weightInfo["giant_eva"], pretrained=True)
        elif cfg.vit == "giant_clip":
            model = timm.create_model(weightInfo["giant_clip"], pretrained=True)
        elif cfg.vit == "giga_clip":
            model = timm.create_model(weightInfo["giga_clip"], pretrained=True)
        else:
            print("Wrong training type")
            exit()

    if cfg.train_type == "lora":
        lora_model = LoRA_ViT_timm(model, r=cfg.rank, num_classes=cfg.num_classes)
        num_params = sum(p.numel() for p in lora_model.parameters() if p.requires_grad)
        print(f"trainable parameters: {num_params/2**20:.3f}M")
        net = lora_model.to(device)
    elif cfg.train_type == "adapter":
        adapter_model = Adapter_ViT(model, num_classes=cfg.num_classes)
        num_params = sum(p.numel() for p in adapter_model.parameters() if p.requires_grad)
        print(f"trainable parameters: {num_params/2**20:.3f}M")
        net = adapter_model.to(device)
    elif cfg.train_type == "full":
        model.reset_classifier(cfg.num_classes)
        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"trainable parameters: {num_params/2**20:.3f}M")
        net = model.to(device)
    elif cfg.train_type == "linear":
        model.reset_classifier(cfg.num_classes)
        for param in model.parameters():
            param.requires_grad = False
        for param in model.head.parameters():
            param.requires_grad = True
        num_params = sum(p.numel() for p in model.head.parameters())
        print(f"trainable parameters: {num_params/2**20:.3f}M")
        net = model.to(device)
    elif cfg.train_type=='resnet50':
        infeature = model.fc.in_features
        model.fc = nn.Linear(infeature, cfg.num_classes)
        num_params = sum(p.numel() for p in model.fc.parameters())
        print(f"trainable parameters: {num_params/2**20:.3f}M")
        net = model.to(device)
    else:
        print("Wrong training type")
        exit()
    net = torch.nn.DataParallel(net)
    if cfg.data_path == "OAI-train":
        trainset, valset, testset = kneeDataloader(cfg)
    elif cfg.data_path == "ChinaSet_AllFiles":
        trainset, valset, testset = cxrDataloader(cfg)
    elif cfg.data_path == "blood-cells":
        trainset, valset, testset = BloodDataloader(cfg)
    loss_func = nn.CrossEntropyLoss(label_smoothing=0.1).to(device)
    # trainset,valset, testset=nihDataloader(cfg)
    # loss_func = nn.BCEWithLogitsLoss().to(device)
    optimizer = optim.Adam(net.parameters(), lr=cfg.lr)
    scheduler = CosineAnnealingLR(optimizer, cfg.epochs, 1e-6)
    result = ResultCLS(cfg.num_classes)

    for epoch in range(1, cfg.epochs+1):
        train(epoch,trainset)
        if epoch%1==0:
            eval(epoch,valset,datatype='val')
            if result.best_epoch == result.epoch:
                if cfg.train_type == "lora":
                    net.module.save_lora_parameters(ckpt_path.replace(".pt", ".safetensors"))
                else:
                    torch.save(net.state_dict(), ckpt_path.replace(".pt", "_best.pt"))
                eval(epoch,testset,datatype='test')
                logging.info(f"BEST VAL: {result.best_val_result:.3f}, TEST: {result.test_auc:.3f}, EPOCH: {(result.best_epoch):3}")
                # logging.info(result.test_mls_auc)

