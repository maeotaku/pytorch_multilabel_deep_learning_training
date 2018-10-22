import torch.nn as nn
import torch
import torch.nn.init as init
import torch.nn.functional as F
from torch.autograd import Variable

from .USM import *

def lenet_usm(pretrained=False, num_classes=1000):
    return LeNetUSM()

class LeNetUSM(nn.Module):
    def __init__(self, num_classes, cuda=True):
        super(LeNet, self).__init__()
        self.filter = USM(in_channels=3, kernel_size=5, fixed_coeff=True, sigma=1.667, cuda=cuda, requires_grad=False)
        #self.filter.assign_weight(4.759330)
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1   = nn.Linear(16*5*5, 120)
        self.fc2   = nn.Linear(120, 84)
        self.fc3   = nn.Linear(84, num_classes)

    def forward(self, x):
        out = self.filter(x)
        out = F.relu(self.conv1(out))
        #out = F.relu(self.conv1(x))
        out = F.max_pool2d(out, 2)
        out = F.relu(self.conv2(out))
        out = F.max_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        return(out)
