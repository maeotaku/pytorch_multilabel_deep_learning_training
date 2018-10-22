import torch.nn as nn
import torch

from .ResNetCIFAR import ResNet_Cifar, BasicBlock
from .USM import *

def resnet20_cifar_usm(pretrained=False, num_classes=100, cuda=True, **kwargs):
    return ResNetCIFARUSM(BasicBlock, [3, 3, 3], num_classes=num_classes, pretrained=pretrained, cuda=cuda, **kwargs)

class ResNetCIFARUSM(ResNet_Cifar):

    def __init__(self, block, layers, num_classes=100, pretrained=True, cuda=True):
        ResNet_Cifar.__init__(self, block=block, layers=layers, num_classes=num_classes)
        #if pretrained:
        #    self.load_state_dict(model_zoo.load_url(resnet_urls['resnet18']))
        self.filter = USM(in_channels=3, kernel_size=5, fixed_coeff=True, sigma=1.667, cuda=cuda, requires_grad=True)
        #self.filter.assign_weight(1.33)
        #self.filter_conv1 = USM(in_channels=64, kernel_size=5, fixed_coeff=True, sigma=1.667, cuda=True)

    def forward(self, x):
        x = self.filter(x)
        #x = self.filter_conv1(x)
        return super().forward(x)
