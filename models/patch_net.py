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
from efficientnet_pytorch import EfficientNet

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
#model = se_resnet50(num_classes=1000, pretrained='imagenet')
#model.avg_pool = GeM()

class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class Model(nn.Module):
    def __init__(self,arch,num_classes=6):
        super().__init__()
        m = self.get_model(arch)
        self.enc = nn.Sequential(*list(m.children()))[:-2]
        n_channels = [46080,11520,2880]
        layers = []
        for i in range(len(n_channels) - 1):
            fc = nn.Linear(n_channels[i],n_channels[i+1],bias=True)
            relu = Swish()
            bn = nn.BatchNorm1d(n_channels[i+1])
            layers.extend([fc,relu,bn])
            if i == len(n_channels) - 2:
                dropout = nn.Dropout(p=0.5)
                layers.append(dropout)
        self.head = nn.Sequential(*list(layers))
        self.fc=nn.Linear(n_channels[-1],num_classes,bias=True)

    def forward(self,x):
        shape = x.shape
        bs = shape[0]
        n = shape[1]
        x = x.view(-1,shape[2],shape[3],shape[4])
        x = self.enc(x)  # bs*n X 1280
        x = x.view(bs,n,-1) #bs X n X 1280
        x = x.view(bs,-1) # bs X n*1280
        #x = torch.cat((x1.reshape(1, -1), x2.reshape(1, -1)), dim=0) # 2*46080
        feat = self.head(x)
        x = self.fc(feat)
        return x,feat


    def get_model(self,arch):
        net = EfficientNet.from_pretrained(arch)
        #net = EfficientNet.from_name('efficientnet-b0')
        feature = net._fc.in_features
        net._fc = nn.Linear(in_features=feature,out_features=6,bias=True)
        net._avg_pooling=GeM(p=2.3)
        return net