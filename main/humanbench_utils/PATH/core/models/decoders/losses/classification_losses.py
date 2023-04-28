import torch
import math
import torch.nn as nn
from torch.nn.parameter import Parameter
import numpy as np
import random
from core.utils import (accuracy, sync_print)

__all__ = ['MarginCosineProductLoss', 'ArcFaceLoss', 'TripletLoss', 'MarginCosineProductLoss_TripletLoss']

import random
import numpy as np
import torch

def _set_randomseed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def euclidean_dist(x, y):
    m, n = x.size(0), y.size(0)
    xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
    yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
    dist = xx + yy
    dist.addmm_(1, -2, x, y.t())
    dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
    return dist


def cosine_dist(x, y):
    bs1, bs2 = x.size(0), y.size(0)
    frac_up = torch.matmul(x, y.transpose(0, 1))
    frac_down = (torch.sqrt(torch.sum(torch.pow(x, 2), 1))).view(bs1, 1).repeat(1, bs2) * \
                (torch.sqrt(torch.sum(torch.pow(y, 2), 1))).view(1, bs2).repeat(bs1, 1)
    cosine = frac_up / frac_down
    return 1 - cosine

def _batch_hard(mat_distance, mat_similarity, indice=False):
    sorted_mat_distance, positive_indices = torch.sort(mat_distance + (-9999999.) * (1 - mat_similarity), dim=1,
                                                       descending=True)
    hard_p = sorted_mat_distance[:, 0]
    hard_p_indice = positive_indices[:, 0]
    sorted_mat_distance, negative_indices = torch.sort(mat_distance + (9999999.) * (mat_similarity), dim=1,
                                                       descending=False)
    hard_n = sorted_mat_distance[:, 0]
    hard_n_indice = negative_indices[:, 0]
    if (indice):
        return hard_p, hard_n, hard_p_indice, hard_n_indice
    return hard_p, hard_n


class MarginCosineProductLoss(nn.Module):
    def __init__(self, in_features, out_features, scale, margin, with_theta, label_smooth=-1):
        super(MarginCosineProductLoss, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.scale = scale
        self.margin = margin
        self.with_theta = with_theta
        self.thetas = []
        self.classifier = Parameter(torch.Tensor(out_features, in_features))
        self.reset_parameters()
        self.ce = torch.nn.CrossEntropyLoss()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.classifier.size(1))
        self.classifier.data.uniform_(-stdv, stdv)

    def forward(self, input, label):
        cosine = self.cosine_sim(input, self.classifier)
        thetas = [math.acos(cosine[i, int(label[i])].item()) / math.pi * 180 for i in range(cosine.size(0))]
        self.thetas.append(thetas)
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, label.view(-1, 1), 1.0)
        output = self.scale * (cosine - one_hot * self.margin)
        loss = self.ce(output, label)
        top1 = accuracy(output.data, label.cuda(), topk=(1, 5))[0]
        if self.with_theta:
            return {'logits': output, 'thetas': thetas, 'loss': loss, 'top1': top1}
        else:
            return {'logits': output, 'loss': loss, 'top1': top1}

    def cosine_sim(self, x1, x2, dim=1, eps=1e-8):
        ip = torch.mm(x1, x2.t())
        w1 = torch.norm(x1, 2, dim)
        w2 = torch.norm(x2, 2, dim)
        return ip / torch.ger(w1,w2).clamp(min=eps)

    def __repr__(self):
        return self.__class__.__name__ + '(' \
            + 'in_features=' + str(self.in_features) \
            + ', out_features=' + str(self.out_features) \
            + ', scale=' + str(self.scale) \
            + ', margin='+ str(self.margin) + ')'

class ArcFaceLoss(nn.Module):
    def __init__(self, in_features, out_features, scale, margin, easy_margin, \
                with_theta=False, with_no_margin_logits=False, fc_std=0.001):
        super(ArcFaceLoss, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.scale = scale
        self.margin = margin
        self.easy_margin = easy_margin
        self.with_theta = with_theta
        self.classifier = Parameter(torch.Tensor(out_features, in_features))
        self.fc_std = fc_std
        self.reset_parameters()
        self.with_no_margin_logits = with_no_margin_logits
        self.thresh = math.cos(math.pi-self.margin)
        self.mm = math.sin(math.pi-self.margin) * self.margin
        self.ce = torch.nn.CrossEntropyLoss()

    def reset_parameters(self):
        self.classifier.data.normal_(std=self.fc_std)
        # stdv = 1. / math.sqrt(self.classifier.size(1))
        # self.classifier.data.uniform_(-stdv, stdv)

    def forward(self, input, label):
        ex = input / torch.norm(input, 2, 1, keepdim=True)
        ew = self.classifier / torch.norm(self.classifier, 2, 1, keepdim=True)
        cos = torch.mm(ex, ew.t())

        a = torch.zeros_like(cos)
        thetas = []
        if self.easy_margin == -1:
            for i in range(a.size(0)):
                lb = int(label[i])
                a[i, lb] = a[i, lb] + self.margin
                theta = math.acos(cos[i, lb].item()) / math.pi * 180
                thetas.append(theta)
            output = self.scale * torch.cos(torch.acos(cos) + a)
            loss = self.ce(output, label)
            top1 = accuracy(output.data, label.cuda(), topk=(1, 5))[0]
            if self.with_theta:
                return {'logits': output, 'thetas': thetas, 'loss': loss, 'top1': top1}
            elif self.with_no_margin_logits:
                return {'logits': output, 'no_margin_logits': self.scale * cos, \
                        'thetas': thetas, 'loss': loss, 'top1': top1}
            else:
                return {'logits': output, 'loss': loss, 'top1': top1}
        elif self.easy_margin is True:
            for i in range(a.size(0)):
                lb = int(label[i])
                if cos[i, lb].data[0] > 0:
                    a[i, lb] = a[i, lb] + self.margin
                theta = math.acos(cos[i, lb].data[0]) / math.pi * 180
                thetas.append(theta)
            output = self.scale * torch.cos(torch.acos(cos) + a)
            loss = self.ce(output, label)
            top1 = accuracy(output.data, label.cuda(), topk=(1, 5))[0]
            if self.with_theta:
                return {'logits': output, 'thetas': thetas, 'loss': loss, 'top1': top1}
            else:
                return {'logits': output, 'loss': loss, 'top1': top1}
        else:
            b = torch.zeros_like(cos)
            for i in range(a.size(0)):
                lb = int(label[i])
                theta = math.acos(cos[i, lb].data[0]) / math.pi * 180
                thetas.append(theta)
                if cos[i, lb].data[0] > self.thresh:
                    a[i, lb] = a[i, lb] + self.margin
                else:
                    #print(theta)
                    b[i, lb] = b[i, lb] - self.mm
            output = self.scale * (torch.cos(torch.acos(cos) + a) + b)
            loss = self.ce(output, label)
            top1 = accuracy(output.data, label.cuda(), topk=(1, 5))[0]
            if self.with_theta:
                return {'logits': output, 'thetas': thetas, 'loss': loss, 'top1': top1}
            else:
                return {'logits': output, 'loss': loss, 'top1': top1}

    def __repr__(self):
        return self.__class__.__name__ + '(' \
            + 'in_features=' + str(self.in_features) \
            + ', out_features=' + str(self.out_features) \
            + ', scale=' + str(self.scale) \
            + ', margin=' + str(self.margin) + ')'


class TripletLoss(nn.Module):

    def __init__(self, margin):
        super(TripletLoss, self).__init__()
        self.margin = margin
        if margin is not None:
            self.margin_loss = nn.MarginRankingLoss(margin=margin).cuda()
        else:
            self.margin_loss = nn.SoftMarginLoss().cuda()

    def forward(self, emb, label):
        mat_dist = euclidean_dist(emb, emb)
        assert mat_dist.size(0) == mat_dist.size(1)
        N = mat_dist.size(0)
        mat_sim = label.expand(N, N).eq(label.expand(N, N).t()).float()

        dist_ap, dist_an = _batch_hard(mat_dist, mat_sim)
        assert dist_an.size(0) == dist_ap.size(0)
        y = torch.ones_like(dist_ap)
        if self.margin is not None:
            loss = self.margin_loss(dist_an, dist_ap, y)
        else:
            loss = self.margin_loss(dist_an - dist_ap, y)
        prec = (dist_an.data > dist_ap.data).sum() * 1. / y.size(0)
        return {'loss': loss, 'top1': prec}

    def __repr__(self):
        return self.__class__.__name__ + '(' \
            + ', margin=' + str(self.margin) + ')'

class MarginCosineProductLoss_TripletLoss(nn.Module):
    def __init__(self, in_features, out_features, scale, margin, \
                easy_margin, with_theta=False, with_no_margin_logits=False, \
                tri_margin=0, balance_weight=1):
        super(MarginCosineProductLoss_TripletLoss, self).__init__()
        self.MarginCosineProductLoss = MarginCosineProductLoss(in_features, \
                            out_features, scale, margin, easy_margin,
                            with_theta=False, with_no_margin_logits=False)
        self.TripletLoss = TripletLoss(tri_margin)
        self.balance_weight = balance_weight

    def forward(self, inputs, targets):
        MarginCosineProductLoss_output = self.MarginCosineProductLoss(inputs, targets)
        TripletLoss_output = self.TripletLoss(inputs, targets)
        output = {'cos_loss': MarginCosineProductLoss_output['loss'], \
                'top1': MarginCosineProductLoss_output['top1'], \
                'tri_loss': TripletLoss_output['loss'], \
                'tri_acc': TripletLoss_output['top1'], \
                'loss': MarginCosineProductLoss_output['loss'] + \
                    self.balance_weight*TripletLoss_output['loss']}
        return output

    def __repr__(self):
        return self.MarginCosineProductLoss.__repr__() \
            + self.TripletLoss.__repr__() + \
            self.__class__.__name__ + '(' \
            + ', balance_weight=' + str(self.balance_weight) + ')'

class Softmax(nn.Module):
    def __init__(self, in_features, out_features):
        super(Softmax, self).__init__()
        self.classifier = nn.Linear(in_features, out_features, bias=False)
        self.in_features = in_features
        self.out_features = out_features
        self.ce = torch.nn.CrossEntropyLoss()
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.classifier.weight, std=0.001)
        if self.classifier.bias:
            nn.init.constant_(self.classifier.bias, 0.0)

    def forward(self, inputs, targets):
        output = self.classifier(inputs)
        loss = self.ce(output, targets)
        top1 = accuracy(output.data, targets.data, topk=(1, 5))[0]
        return {'loss': loss, 'logits': output, 'top1': top1}

    def __repr__(self):
        return self.__class__.__name__ + '(' \
            + 'in_features=' + str(self.in_features) \
            + ', out_features=' + str(self.out_features)  + ')'

class Softmax_TripletLoss(nn.Module):
    def __init__(self, in_features, out_features, tri_margin=0.3, balance_weight=1):
        super(Softmax_TripletLoss, self).__init__()
        self.SoftmaxLoss = Softmax(in_features, out_features)
        self.TripletLoss = TripletLoss(tri_margin)
        self.balance_weight = balance_weight

    def forward(self, inputs):
        SoftmaxLoss_output = self.SoftmaxLoss(inputs['feature'], inputs['label'])
        TripletLoss_output = self.TripletLoss(inputs['feature_nobn'], inputs['label'])
        output = {'Softmax_loss': SoftmaxLoss_output['loss'], \
                'top1': SoftmaxLoss_output['top1'], \
                'tri_loss': TripletLoss_output['loss'], \
                'tri_acc': TripletLoss_output['top1'], \
                'loss': SoftmaxLoss_output['loss'] + \
                    self.balance_weight*TripletLoss_output['loss']}
        return output

    def __repr__(self):
        return self.SoftmaxLoss.__repr__() \
            + self.TripletLoss.__repr__() + \
            self.__class__.__name__ + '(' \
            + ', balance_weight=' + str(self.balance_weight) + ')'


class Softmax_TripletLoss_wBN(nn.Module):
    def __init__(self, in_features, out_features, tri_margin=0.3, balance_weight=1, cfg=None):
        super(Softmax_TripletLoss_wBN, self).__init__()
        self.SoftmaxLoss = Softmax(in_features, out_features)
        self.TripletLoss = TripletLoss(tri_margin)
        self.balance_weight = balance_weight
        if cfg.out_norm == 'layer_norm':
            self.norm = nn.LayerNorm(in_features)
        elif cfg.out_norm == 'batch_norm':
            def BN(*args, **kwargs):
                class SyncBatchNorm1d(torch.nn.SyncBatchNorm):
                    def forward(self, input):
                        assert input.dim() == 2
                        output = super(SyncBatchNorm1d, self).forward(input.unsqueeze(-1).unsqueeze(-1))
                        return output.squeeze(dim=2).squeeze(dim=2)
                return SyncBatchNorm1d(*args, **kwargs, process_group=bn_group, momentum=0.1, eps=0.00001)

            def weights_init_kaiming(m):
                classname = m.__class__.__name__
                if classname.find('Linear') != -1:
                    nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
                    nn.init.constant_(m.bias, 0.0)

                elif classname.find('Conv') != -1:
                    nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0.0)
                elif classname.find('BatchNorm') != -1:
                    if m.affine:
                        nn.init.constant_(m.weight, 1.0)
                        nn.init.constant_(m.bias, 0.0)
            self.norm = BN(in_features)
            self.norm.bias.requires_grad_(False)
            self.norm.apply(weights_init_kaiming)
        self.cfg = cfg

    def forward(self, inputs):
        SoftmaxLoss_output = self.SoftmaxLoss(inputs['feature'], inputs['label'])
        TripletLoss_output = self.TripletLoss(inputs['feature_nobn'], inputs['label'])
        loss = SoftmaxLoss_output['loss'] + self.balance_weight*TripletLoss_output['loss']

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if "aux_outputs" in inputs and self.cfg.get('aux_loss', True):
            for aux_outputs in inputs["aux_outputs"]:
                SoftmaxLoss_output = self.SoftmaxLoss(aux_outputs['feature'], inputs['label'])
                TripletLoss_output = self.TripletLoss(aux_outputs['feature_nobn'], inputs['label'])
                loss = loss + SoftmaxLoss_output['loss'] + self.balance_weight*TripletLoss_output['loss']

        output = {'Softmax_loss': SoftmaxLoss_output['loss'], \
                'top1': SoftmaxLoss_output['top1'], \
                'tri_loss': TripletLoss_output['loss'], \
                'tri_acc': TripletLoss_output['top1'], \
                'loss': loss}
        return output

    def __repr__(self):
        return self.SoftmaxLoss.__repr__() \
            + self.TripletLoss.__repr__() + \
            self.__class__.__name__ + '(' \
            + ', balance_weight=' + str(self.balance_weight) + ')'


class DoNothing(nn.Module):
    def __init__(self):
        super(DoNothing, self).__init__()

    def forward(self, x):
        return x
