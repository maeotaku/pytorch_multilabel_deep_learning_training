import torch.nn as nn
import torch.utils.model_zoo as model_zoo
from torchvision.models.resnet import ResNet, BasicBlock, model_urls as resnet_urls

import torch
import torch.nn.init as init
import torch.nn.functional as F
from torch.autograd import Variable

from .USM import *

def resnet18_usm(pretrained=False, num_classes=1000, cuda=True):
    return ResNetUSM(pretrained, cuda)

class ResNetUSM(ResNet):

    def __init__(self, pretrained=True, cuda=True):
        #super(ResNet, self).__init__( block=BasicBlock, layers=[2, 2, 2, 2] )
        ResNet.__init__(self, block=BasicBlock, layers=[2, 2, 2, 2])
        if pretrained:
            self.load_state_dict(model_zoo.load_url(resnet_urls['resnet18']))
        self.filter = USM(in_channels=3, kernel_size=5, fixed_coeff=True, sigma=1.667, cuda=cuda, requires_grad=True)
        #self.filter.assign_weight(1.33)
        self.filter_conv1 = USM(in_channels=64, kernel_size=5, fixed_coeff=True, sigma=1.667, cuda=cuda)

    def forward(self, x):
        x = self.filter(x)
        #x = self.filter_conv1(x)
        return super().forward(x)
