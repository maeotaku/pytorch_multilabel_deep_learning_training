
import torch.nn as nn
import torchvision.models as models

from .Inceptionv3USM import *
from .ResNetUSM import *
from .LeNetUSM import *
from .ResNetCIFARUSM import *
#from .CapsNetUSM import *

def predefined_model(args, cuda=True):
    # create model
    '''
    if args.pretrained:
        print("=> using pre-trained model '{}'".format(args.arch))
        model = models.__dict__[args.arch](pretrained=True)
    elif args.arch.startswith('resnext'):
        model = models.__dict__[args.arch](
                    baseWidth=args.base_width,
                    cardinality=args.cardinality,
                )
    else:
        print("=> creating model '{}'".format(args.arch))
        model = models.__dict__[args.arch]()
    '''
    model = resnet20_cifar_usm(pretrained=True, cuda=cuda)
    #model = resnet18_usm(pretrained=True, cuda=cuda)
    #model = inception_v3_usm(pretrained=True, cuda=cuda)
    print("Model Loaded.")
    return model


def create_model(model, num_classes, cuda=True):
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)

    if cuda:
        model.cuda()
    print('    Total params: %.2fM' % (sum(p.numel() for p in model.parameters())/1000000.0))
    return model
