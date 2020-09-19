from __future__ import absolute_import

import torch
from torch import nn
from torch.nn import functional as F, ParameterList, ModuleList, Parameter
from torch.autograd import Variable
import torchvision
import numpy as np
from torch.utils import model_zoo
from torchvision.models import inception, Inception3
from torchvision.models.inception import BasicConv2d, InceptionA, InceptionB, InceptionC, InceptionAux, InceptionD, \
    InceptionE, model_urls

from torch.nn.parameter import Parameter
def gem(x, p=3, eps=1e-6):
    return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1./p)
class GeM(nn.Module):
    def __init__(self, p=3, eps=1e-6):
        super(GeM,self).__init__()
        self.p = Parameter(torch.ones(1)*p)
        self.eps = eps
    def forward(self, x):
        return gem(x, p=self.p, eps=self.eps)       
    def __repr__(self):
        return self.__class__.__name__ + '(' + 'p=' + '{:.4f}'.format(self.p.data.tolist()[0]) + ', ' + 'eps=' + str(self.eps) + ')'

class Baseline_single(nn.Module):
    def __init__(self, num_classes, loss_type="single BCE", **kwargs):
        super(Baseline_single, self).__init__()
        self.loss_type = loss_type
        resnet50 = torchvision.models.densenet121(pretrained=True)

        self.base = nn.Sequential(*list(resnet50.children())[:-1])
        self.feature_dim = 512*2
        #self.feature_dim = 1664
        #self.feature_dim=1920
        # self.reduce_conv = nn.Conv2d(self.feature_dim * 4, self.feature_dim, 1)
        if self.loss_type == "single BCE":
            #self.ap = nn.AdaptiveAvgPool2d(1)
            self.ap=GeM()
            self.classifiers = nn.Linear(in_features=self.feature_dim, out_features=num_classes)
            self.sigmoid = nn.Sigmoid()
            self.dropout=nn.Dropout(0.5)
            self.cal_score=nn.Linear(in_features=num_classes, out_features=1)
        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        #         nn.init.kaiming_normal(m.weight, mode='fan_in', nonlinearity='relu')
        #         # import scipy.stats as stats
        #         # stddev = m.stddev if hasattr(m, 'stddev') else 0.1
        #         # X = stats.truncnorm(-2, 2, scale=stddev)
        #         # values = torch.Tensor(X.rvs(m.weight.data.numel()))
        #         # values = values.view(m.weight.data.size())
        #         # m.weight.data.copy_(values)
        #     elif isinstance(m, nn.BatchNorm2d):
        #         m.weight.data.fill_(1)
        #         m.bias.data.zero_()
    def freeze_base(self):
        for p in self.base.parameters():
            p.requires_grad = False

    def unfreeze_all(self):
        for p in self.parameters():
            p.requires_grad = True
    def forward(self, x1):
        x = self.base(x1)
        #print(x.shape)
        map_feature=torch.tensor(x)
        # x = self.reduce_conv(x)
        if self.loss_type == "single BCE":
            x = self.ap(x)
            #print(x.shape)
            x = self.dropout(x)
            x = x.view(x.size(0), -1)
            ys = self.classifiers(x)
            #y = self.sigmoid(y)
            #score=self.cal_score(ys)
        return ys
