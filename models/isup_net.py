from __future__ import absolute_import
from efficientnet_pytorch import EfficientNet
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
#model = se_resnet50(num_classes=1000, pretrained='imagenet')
#model.avg_pool = GeM()

class SwishImplementation(torch.autograd.Function):
    @staticmethod
    def forward(ctx, i):
        result = i * torch.sigmoid(i)
        ctx.save_for_backward(i)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        i = ctx.saved_variables[0]
        sigmoid_i = torch.sigmoid(i)
        return grad_output * (sigmoid_i * (1 + i * (1 - sigmoid_i)))


class MemoryEfficientSwish(nn.Module):
    def forward(self, x):
        return SwishImplementation.apply(x)

class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

class isupnet(nn.Module):
    def __init__(self, num_classes,base_model, **kwargs):
        super(isupnet, self).__init__()
        
        
        self.base_model_name=base_model
        self.avg_pooling=GeM(p=3)
        self.max_pooling=nn.AdaptiveMaxPool2d(1)
        if base_model[0]=='e':
            self.base = EfficientNet.from_pretrained(base_model)
            feature = self.base._fc.in_features
        #self.base = EfficientNet.from_name('efficientnet-b0')
        else:
            base_net=torch.hub.load('facebookresearch/semi-supervised-ImageNet1K-models', 'resnext50_32x4d_ssl')
            feature = base_net.fc.in_features
            self.base=nn.Sequential(*list(base_net.children())[:-2])  
        self.fc = nn.Linear(in_features=2*feature,out_features=feature,bias=True)
        self.classifiers = nn.Linear(in_features=feature, out_features=num_classes)
        self.sigmoid = MemoryEfficientSwish()
        self.dropout=nn.Dropout(0.4)
        self.bn=nn.BatchNorm1d(feature)

    def freeze_base(self):
        for p in self.base.parameters():
            p.requires_grad = False

    def unfreeze_all(self):
        for p in self.parameters():
            p.requires_grad = True
    def forward(self, x1):
        bs=x1.size(0)
        if self.base_model_name[0]=='e':
            x2 = self.base.extract_features(x1)
            feat1=self.avg_pooling(x2)
            feat2=self.max_pooling(x2)
            feat1=feat1.view(bs,-1)
            feat2=feat2.view(bs,-1)
            feat=torch.cat([feat1,feat2],1)
            
            feat = self.fc(feat)
            #feat = self.sigmoid(feat)
            #feat = self.bn(feat)
            feat = self.dropout(feat)
            ys = self.classifiers(feat)

            #ys=self.sigmoid(ys)
        else:
            '''
            x=x1
            x=x.view(bs,3,-1,224,224)
            x=x.permute(0,2,1,3,4).contiguous()
            shape = [36,3,224,224]
            #print(x.shape)
            n = x.shape[1]
            x = x.view(-1,shape[1],shape[2],shape[3])
            #print(x.shape)
            #x: bs*N x 3 x 128 x 128
            x = self.base(x)
            #x: bs*N x C x 4 x 4
            shape = x.shape
            #print(shape)
            #concatenate the output for tiles into a single map
            x = x.view(-1,n,shape[1],shape[2],shape[3]).permute(0,2,1,3,4).contiguous()\
              .view(-1,shape[1],shape[2]*n,shape[3])
            #x: bs x C x N*4 x 4
            #print(x.shape)
            '''
            x=self.base(x1)
            feat1=self.avg_pooling(x)
            feat2=self.max_pooling(x)
            feat1=feat1.view(bs,-1)
            feat2=feat2.view(bs,-1)
            feat=torch.cat([feat1,feat2],1)
            
            feat = self.fc(feat)
            #feat = self.sigmoid(feat)
            #feat = self.bn(feat)
            feat = self.dropout(feat)
            ys = self.classifiers(feat)
            #x: bs x n
        

        return ys