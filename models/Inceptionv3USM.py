import torch.nn as nn
import torch.utils.model_zoo as model_zoo
from torchvision.models.inception import Inception3, model_urls as inception_urls

import torch
import torch.nn.init as init
import torch.nn.functional as F
from torch.autograd import Variable

from .USM import *

def inception_v3_usm(pretrained=True, **kwargs):
    return InceptionUSM(pretrained, **kwargs)

class InceptionUSM(Inception3):

    def __init__(self, pretrained=True, num_classes=1000, aux_logits=True, transform_input=False, cuda=True):
        Inception3.__init__(self, num_classes=num_classes, aux_logits=aux_logits, transform_input=transform_input)
        if pretrained:
            self.load_state_dict(model_zoo.load_url(inception_urls['inception_v3_google']))
        self.aux_logit=False
        self.filter = USM(in_channels=3, kernel_size=5, fixed_coeff=True, sigma=1.667, cuda=cuda, requires_grad=True)

    def forward(self, x):
        x = self.filter(x)
        return super().forward(x)
