import numpy as np
import torch
from torch import nn
from torch.nn import init
from collections import OrderedDict

__all__ = ['ECAAttention', 'CBAMBlock', 'SEAttention']

class ECAAttention(nn.Module):
    def __init__(self, kernel_size=3, use_cls_token=False):
        super().__init__()
        self.gap=nn.AdaptiveAvgPool2d(1)
        self.conv=nn.Conv1d(1,1,kernel_size=kernel_size,padding=(kernel_size-1)//2)
        self.sigmoid=nn.Sigmoid()
        self.use_cls_token = use_cls_token

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x, H, W):
        if self.use_cls_token:
            cls_feat = x[:, :1, :]
            x = x[:, 1:, :]
        
        B = x.shape[0]
        x = x.permute(0, 2, 1).reshape(B, -1, H, W)
        y = self.gap(x) #bs,c,1,1
        y = y.squeeze(-1).permute(0,2,1) #bs,1,c
        y = self.conv(y) #bs,1,c
        y = self.sigmoid(y) #bs,1,c
        y = y.permute(0,2,1).unsqueeze(-1) #bs,c,1,1
        y = x * y.expand_as(x)
        
        if self.use_cls_token:
            return torch.cat([cls_feat, y.reshape(B, -1, H * W).permute(0, 2, 1)], dim=1)
        else:
            return y.reshape(B, -1, H * W).permute(0, 2, 1)


class ChannelAttention(nn.Module):
    def __init__(self,channel,reduction=16):
        super().__init__()
        self.maxpool=nn.AdaptiveMaxPool2d(1)
        self.avgpool=nn.AdaptiveAvgPool2d(1)
        self.se=nn.Sequential(
            nn.Conv2d(channel,channel//reduction,1,bias=False),
            nn.ReLU(),
            nn.Conv2d(channel//reduction,channel,1,bias=False)
        )
        self.sigmoid=nn.Sigmoid()
    
    def forward(self, x) :
        max_result=self.maxpool(x)
        avg_result=self.avgpool(x)
        max_out=self.se(max_result)
        avg_out=self.se(avg_result)
        output=self.sigmoid(max_out+avg_out)
        return output

class SpatialAttention(nn.Module):
    def __init__(self,kernel_size=7):
        super().__init__()
        self.conv=nn.Conv2d(2,1,kernel_size=kernel_size,padding=kernel_size//2)
        self.sigmoid=nn.Sigmoid()
    
    def forward(self, x) :
        max_result,_=torch.max(x,dim=1,keepdim=True)
        avg_result=torch.mean(x,dim=1,keepdim=True)
        result=torch.cat([max_result,avg_result],1)
        output=self.conv(result)
        output=self.sigmoid(output)
        return output


class CBAMBlock(nn.Module):
    def __init__(self, channel=512,reduction=16,kernel_size=7, use_cls_token=False):
        super().__init__()
        self.ca = ChannelAttention(channel=channel,reduction=reduction)
        self.sa = SpatialAttention(kernel_size=kernel_size)
        self.use_cls_token = use_cls_token


    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x, H, W):
        # import pdb; pdb.set_trace()
        if self.use_cls_token:
            cls_feat = x[:, :1, :]
            x = x[:, 1:, :]
        B = x.shape[0]
        x = x.permute(0, 2, 1).reshape(B, -1, H, W)
        b, c, _, _ = x.size()
        residual = x
        out = x * self.ca(x)
        out = out * self.sa(out) + residual
        if self.use_cls_token:
            return torch.cat([cls_feat, out.reshape(B, -1, H * W).permute(0, 2, 1)], dim=1)
        else:
            return out.reshape(B, -1, H * W).permute(0, 2, 1)


class SEAttention(nn.Module):

    def __init__(self, channel=512, reduction=16, use_cls_token=False):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )
        self.use_cls_token = use_cls_token

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x, H, W):
        if self.use_cls_token:
            cls_feat = x[:, :1, :]
            x = x[:, 1:, :]
        B = x.shape[0]
        x = x.permute(0, 2, 1).reshape(B, -1, H, W)
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        y = x * y.expand_as(x)
        
        if self.use_cls_token:
            return torch.cat([cls_feat, y.reshape(B, -1, H * W).permute(0, 2, 1)], dim=1)
        else:
            return y.reshape(B, -1, H * W).permute(0, 2, 1)