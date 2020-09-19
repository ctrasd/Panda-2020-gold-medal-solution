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

class SingleClassifierWithFodus(nn.Module):
    def __init__(self, feature_dim, attr_len):
        super(SingleClassifierWithFodus, self).__init__()
        self.feature_dim = feature_dim
        self.conv1 = nn.Conv2d(self.feature_dim, self.feature_dim, 1)
        self.bn1 = nn.BatchNorm2d(num_features=self.feature_dim)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(self.feature_dim, self.feature_dim // 2, 1)
        self.bn2 = nn.BatchNorm2d(num_features=self.feature_dim // 2)
        self.relu2 = nn.ReLU()
        # self.ap = nn.AdaptiveAvgPool2d(1)
        self.ap = nn.AdaptiveMaxPool2d(1)
        # self.dropout = nn.Dropout2d(p=0.5)

        # self.conv1_fodus = nn.Conv2d(self.feature_dim, self.feature_dim // 2, 1)
        # self.bn1_fodus = nn.BatchNorm2d(num_features=self.feature_dim // 2)
        # self.relu1_fodus = nn.ReLU()
        # self.conv2_fodus = nn.Conv2d(self.feature_dim // 2, self.feature_dim // 4, 1)
        # self.bn2_fodus = nn.BatchNorm2d(num_features=self.feature_dim // 4)
        # self.relu2_fodus = nn.ReLU()
        # self.ap_fodus = nn.AdaptiveMaxPool2d(1)

        # self.classifier = nn.Linear(feature_dim // 4 * 3, attr_len)
        self.classifier = nn.Linear(feature_dim // 2, attr_len)

        # for m in self.modules():
        #
        #     if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        #         nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
        #         # import scipy.stats as stats
        #         # stddev = m.stddev if hasattr(m, 'stddev') else 0.1
        #         # X = stats.truncnorm(-2, 2, scale=stddev)
        #         # values = torch.Tensor(X.rvs(m.weight.data.numel()))
        #         # values = values.view(m.weight.data.size())
        #         # m.weight.data.copy_(values)
        #     elif isinstance(m, nn.BatchNorm2d):
        #         m.weight.data.fill_(1)
        #         m.bias.data.zero_()

    def forward(self, x, f_x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.bn1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.bn2(x)
        x = self.ap(x).view(x.size(0), -1)
        # x = self.dropout(x)
        # f_x = self.conv1_fodus(f_x)
        # f_x = self.relu1_fodus(f_x)
        # f_x = self.bn1_fodus(f_x)
        # f_x = self.conv2_fodus(f_x)
        # f_x = self.relu2_fodus(f_x)
        # f_x = self.bn2_fodus(f_x)
        # f_x = self.ap_fodus(f_x).view(x.size(0), -1)
        #
        # x = torch.cat([x, f_x], 1)
        return self.classifier(x), x


class Baseline(nn.Module):
    def __init__(self, num_classes, loss_type="single BCE", **kwargs):
        super(Baseline, self).__init__()
        self.loss_type = loss_type
        resnet50_1 = torchvision.models.resnet34(pretrained=True)
        resnet50_2 = torchvision.models.resnet34(pretrained=True)

        self.left_base = nn.Sequential(*list(resnet50_1.children())[:-2])
        self.right_base = nn.Sequential(*list(resnet50_2.children())[:-2])
        # self.left_fodus_base = nn.Sequential(*list(resnet50_1.children())[:-2])
        # self.right_fodus_base = nn.Sequential(*list(resnet50_2.children())[:-2])
        # self.dropout = nn.Dropout2d(p=0.5)
        self.feature_dim = 512 * 2
        # self.classifier = nn.Linear(self.hidden_dim, num_classes)
        if self.loss_type == "single BCE":
            self.ap = nn.AdaptiveAvgPool2d(1)
            self.classifiers = nn.Linear(in_features=self.feature_dim, out_features=num_classes)
            self.sigmoid = nn.Sigmoid()
        elif self.loss_type == "multi CE":
            self.classifiers = []
            self.left_classifiers = []
            self.right_classifiers = []
            for i in range(num_classes):
                self.classifiers.append(SingleClassifierWithFodus(feature_dim=self.feature_dim, attr_len=2))
            for i in range(num_classes):
                self.left_classifiers.append(SingleClassifierWithFodus(feature_dim=self.feature_dim // 2, attr_len=2))
            for i in range(num_classes):
                self.right_classifiers.append(SingleClassifierWithFodus(feature_dim=self.feature_dim // 2, attr_len=2))
            self.classifiers = nn.ModuleList(self.classifiers)
            self.left_classifiers = nn.ModuleList(self.left_classifiers)
            self.right_classifiers = nn.ModuleList(self.right_classifiers)

    def forward(self, x1, x2, x3, x4):
        left_x = self.left_base(x1)
        right_x = self.right_base(x2)

        # l_fodus_x = self.left_fodus_base(x3)
        # r_fodus_x = self.right_fodus_base(x4)

        if self.loss_type == "single BCE":
            left_x = self.ap(left_x)
            right_x = self.ap(right_x)
            left_x = left_x.view(left_x.size(0), -1)
            right_x = right_x.view(right_x.size(0), -1)
            x = torch.cat([left_x, right_x], 1)
            y = self.classifiers(x)
            y = self.sigmoid(y)
        elif self.loss_type == "multi CE":
            y = []
            left_y = []
            right_y = []

            # left_x = self.dropout(left_x)
            # right_x = self.dropout(right_x)


            x = torch.cat([left_x, right_x], 1)
            # f_x = torch.cat([l_fodus_x, r_fodus_x], 1)
            for c in self.left_classifiers:
                # y.append(c(x, f_x))
                left_y.append(c(left_x, None))

            for c in self.right_classifiers:
                # y.append(c(x, f_x))
                right_y.append(c(right_x, None))

            for c in self.classifiers:
                # y.append(c(x, f_x))
                y.append(c(x, None))


        return y, left_y, right_y


class Baseline_single(nn.Module):
    def __init__(self, num_classes, loss_type="single BCE", **kwargs):
        super(Baseline_single, self).__init__()
        self.loss_type = loss_type
        resnet50 = torchvision.models.resnet34(pretrained=True)

        self.base = nn.Sequential(*list(resnet50.children())[:-2])
        self.feature_dim = 512
        # self.reduce_conv = nn.Conv2d(self.feature_dim * 4, self.feature_dim, 1)
        if self.loss_type == "single BCE":
            self.ap = nn.AdaptiveMaxPool2d(1)
            self.classifiers = nn.Linear(in_features=self.feature_dim, out_features=num_classes)
            self.sigmoid = nn.Sigmoid()
        elif self.loss_type == "multi CE":
            self.classifiers = []
            self.left_classifiers = []
            self.right_classifiers = []
            for i in range(num_classes):
                self.classifiers.append(SingleClassifierWithFodus(feature_dim=self.feature_dim, attr_len=2))
            self.classifiers = nn.ModuleList(self.classifiers)


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
        # x = self.reduce_conv(x)
        if self.loss_type == "single BCE":
            x = self.ap(x)
            x = x.view(x.size(0), -1)
            ys = self.classifiers(x)
            ys = self.sigmoid(ys)

        if self.loss_type == "multi CE":
            ys = []
            fs = []
            for c in self.classifiers:
                # y.append(c(x, f_x))
                y, f = c(x, None)
                ys.append(y)
                fs.append(f)
        return ys