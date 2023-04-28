import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .module_helper import ModuleHelper
from .spatial_ocr_block import SpatialGather_Module, SpatialOCR_Module
from core.utils import accuracy
from core.utils import SegmentationRunningMetrics

from ..losses import loss_entry

class HRT_OCR_V2(nn.Module):
    def __init__(self, num_classes, 
                       bn_type,
                       input_size, 
                       in_channels, 
                       loss_cfg, 
                       bn_group=None,
                       **kwargs):

        super(HRT_OCR_V2, self).__init__()
        
        assert bn_type in ['torchbn', 'syncBN', 'LN']
        self.bn_type = bn_type
        self.num_classes = num_classes
        self.loss_config = loss_cfg
        self.input_h, self.input_w = input_size

        self.conv3x3 = nn.Sequential(
            nn.Conv2d(in_channels, 512, kernel_size=3, stride=1, padding=1),
            ModuleHelper.BNReLU(512, bn_type=self.bn_type, bn_group=bn_group),
        )
        self.ocr_gather_head = SpatialGather_Module(self.num_classes)
        self.ocr_distri_head = SpatialOCR_Module(
            in_channels=512,
            key_channels=256,
            out_channels=512,
            scale=1,
            dropout=0.05,
            bn_type=self.bn_type,
        )
        self.cls_head = nn.Conv2d(
            512, self.num_classes, kernel_size=1, stride=1, padding=0, bias=True
        )
        self.aux_head = nn.Sequential(
            nn.Conv2d(in_channels, 512, kernel_size=3, stride=1, padding=1),
            ModuleHelper.BNReLU(512, bn_type=self.bn_type, bn_group=bn_group),
            nn.Conv2d(
                512, self.num_classes, kernel_size=1, stride=1, padding=0, bias=True
            ),
        )

        # loss
        self.loss = loss_entry(self.loss_config)

    def forward(self, input):
        x = input['neck_output']
        label = input['label']
        _, _, h, w = x[0].size()

        feat1 = x[0]
        feat2 = F.interpolate(x[1], size=(h, w), mode="bilinear", align_corners=True)
        feat3 = F.interpolate(x[2], size=(h, w), mode="bilinear", align_corners=True)
        feat4 = F.interpolate(x[3], size=(h, w), mode="bilinear", align_corners=True)

        feats = torch.cat([feat1, feat2, feat3, feat4], 1)
        out_aux = self.aux_head(feats)

        feats = self.conv3x3(feats)

        context = self.ocr_gather_head(feats, out_aux)
        feats = self.ocr_distri_head(feats, context)

        out = self.cls_head(feats)

        out_aux = F.interpolate(
            out_aux, size=(self.input_h, self.input_w), mode="bilinear", align_corners=True
        )
        out = F.interpolate(
            out, size=(self.input_h, self.input_w), mode="bilinear", align_corners=True
        )
        loss = self.loss((out_aux, out), label)
        top1 = torch.FloatTensor([0]).cuda()
        return {'loss': loss, 'top1': top1}
    

class ViT_OCR_V2(nn.Module):
    def __init__(self, 
                 task,
                 num_classes,
                 bn_type, 
                 input_size, 
                 in_channels, 
                 loss_cfg, 
                 bn_group=None,
                 **kwargs):
        super(ViT_OCR_V2, self).__init__()

        assert bn_type in ['torchbn', 'syncBN', 'LN']
        self.task = task
        self.bn_type = bn_type
        self.num_classes = num_classes
        self.loss_config = loss_cfg
        self.input_h, self.input_w = input_size

        self.conv3x3 = nn.Sequential(
            nn.Conv2d(in_channels, 512, kernel_size=3, stride=1, padding=1),
            ModuleHelper.BNReLU(512, bn_type=self.bn_type, bn_group=bn_group),
        )
        self.ocr_gather_head = SpatialGather_Module(self.num_classes)
        self.ocr_distri_head = SpatialOCR_Module(
            in_channels=512,
            key_channels=256,
            out_channels=512,
            scale=1,
            dropout=0.05,
            bn_type=self.bn_type,
        )
        self.cls_head = nn.Conv2d(
            512, self.num_classes, kernel_size=1, stride=1, padding=0, bias=True
        )
        self.aux_head = nn.Sequential(
            nn.Conv2d(in_channels, 512, kernel_size=3, stride=1, padding=1),
            ModuleHelper.BNReLU(512, bn_type=self.bn_type, bn_group=bn_group),
            nn.Conv2d(
                512, self.num_classes, kernel_size=1, stride=1, padding=0, bias=True
            ),
        )

        # loss
        self.loss = loss_entry(self.loss_config)
        self.seg_eval = SegmentationRunningMetrics(num_classes=num_classes)

    def forward(self, input):
        if self.task=='par':
            if self.training:
                x = input['neck_output']
                label = input['label']
                # import pdb;
                # pdb.set_trace()
                _, _, h, w = x.size()

                # feats shape b,c,h,w

                # feat1 = x[0]
                # feat2 = F.interpolate(x[1], size=(h, w), mode="bilinear", align_corners=True)
                # feat3 = F.interpolate(x[2], size=(h, w), mode="bilinear", align_corners=True)
                # feat4 = F.interpolate(x[3], size=(h, w), mode="bilinear", align_corners=True)
                #
                # feats = torch.cat([feat1, feat2, feat3, feat4], 1)
                feats = x
                out_aux = self.aux_head(feats)

                feats = self.conv3x3(feats)

                context = self.ocr_gather_head(feats, out_aux)
                feats = self.ocr_distri_head(feats, context)

                out = self.cls_head(feats)

                out_aux = F.interpolate(
                    out_aux, size=(self.input_h, self.input_w), mode="bilinear", align_corners=True
                )
                out = F.interpolate(
                    out, size=(self.input_h, self.input_w), mode="bilinear", align_corners=True
                )
                loss = self.loss((out_aux, out), label)
                pred_numpy = out.argmax(dim=1).cpu().detach().numpy()
                label_numpy = label.cpu().detach().numpy()
                self.seg_eval.update(pred_numpy, label_numpy)
                miou = self.seg_eval.get_mean_iou()
                # top1 = torch.FloatTensor([0]).cuda()
                return {'loss': loss, 'top1': torch.FloatTensor([miou]).cuda()}
            else:
                images = input['image']
                x = input['neck_output']
                feats = x
                out_aux = self.aux_head(feats)

                feats = self.conv3x3(feats)

                context = self.ocr_gather_head(feats, out_aux)
                feats = self.ocr_distri_head(feats, context)

                out = self.cls_head(feats)

                # out_aux = F.interpolate(
                #     out_aux, size=(self.input_h, self.input_w), mode="bilinear", align_corners=True
                # )
                out = F.interpolate(
                    out, size=(self.input_h, self.input_w), mode="bilinear", align_corners=True
                )

                # import pdb; pdb.set_trace();
                processed_results = []
                # import pdb;pdb.set_trace()
                for _idx, o in enumerate(out):
                    try:
                        height = input.get("gt", None).shape[-2] #.item()
                        width = input.get("gt", None).shape[-1] #.item()
                    except:
                        height = input['height'][_idx].item()
                        width = input['width'][_idx].item()
                    processed_results.append({})
                    pred = F.interpolate(o.unsqueeze(0), (height, width),mode="bilinear", align_corners=True)[0]
                    processed_results[-1]["sem_seg"] = pred



                return processed_results


class ViT_SimpleUpSampling(nn.Module):
    def __init__(self, 
                 task,
                 num_classes,
                 bn_type, 
                 input_size, 
                 in_channels, 
                 loss_cfg, 
                 bn_group=None,
                 **kwargs):
        super(ViT_SimpleUpSampling, self).__init__()
        assert bn_type in ['torchbn', 'syncBN', 'LN']
        self.task = task
        self.bn_type = bn_type
        self.num_classes = num_classes
        self.loss_config = loss_cfg
        self.input_h, self.input_w = input_size
        
        self.conv3x3 = nn.Sequential(
            nn.Conv2d(in_channels, 256, kernel_size=3, stride=1, padding=1),
            ModuleHelper.BNReLU(256, bn_type=self.bn_type, bn_group=bn_group),
        )
        
        self.upsample1_conv = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            ModuleHelper.BNReLU(256, bn_type=self.bn_type, bn_group=bn_group),
        )
        
        self.upsample2_conv = nn.Conv2d(256, self.num_classes, kernel_size=3, stride=1, padding=1)
        
        # loss
        self.loss = loss_entry(self.loss_config)
        # eval
        self.seg_eval = SegmentationRunningMetrics(num_classes=num_classes)


    def forward(self, input):
        if self.task=='par':
            if self.training:
                x = input['neck_output']
                label = input['label']
                # import pdb;
                # pdb.set_trace()
                _, _, h, w = x.size()

                # upsample1
                feats = self.conv3x3(x)
                feats = self.upsample1_conv(feats)
                feats = F.interpolate(feats, size=(h * 2, w * 2), mode="bilinear", align_corners=True)

                # upsample2
                feats = self.upsample2_conv(feats)
                feats = F.interpolate(feats, size=(self.input_h, self.input_w), mode="bilinear", align_corners=True)
                
                # loss
                loss = self.loss(feats, label)
                
                # metric
                pred_numpy = feats.argmax(dim=1).cpu().detach().numpy()
                label_numpy = label.cpu().detach().numpy()
                self.seg_eval.update(pred_numpy, label_numpy)
                miou = self.seg_eval.get_mean_iou()
                # top1 = torch.FloatTensor([0]).cuda()
                return {'loss': loss, 'top1': torch.FloatTensor([miou]).cuda()}
            else:
                x = input['neck_output']
                _, _, h, w = x.size()
                # upsample1
                feats = self.conv3x3(x)
                feats = self.upsample1_conv(feats)
                feats = F.interpolate(feats, size=(h * 2, w * 2), mode="bilinear", align_corners=True)

                # upsample2
                feats = self.upsample2_conv(feats)
                feats = F.interpolate(feats, size=(self.input_h, self.input_w), mode="bilinear", align_corners=True)

                # import pdb; pdb.set_trace();
                processed_results = []
                # import pdb;pdb.set_trace()
                for _idx, o in enumerate(feats):
                    try:
                        height = input.get("gt", None).shape[-2] #.item()
                        width = input.get("gt", None).shape[-1] #.item()
                    except:
                        height = input['height'][_idx].item()
                        width = input['width'][_idx].item()
                    processed_results.append({})
                    pred = F.interpolate(o.unsqueeze(0), (height, width),mode="bilinear", align_corners=True)[0]
                    processed_results[-1]["sem_seg"] = pred
                    
                return processed_results


