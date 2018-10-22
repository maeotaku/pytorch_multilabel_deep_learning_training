#python3 main.py 2>&1 | tee output.txt

from __future__ import print_function, division

import os
import time
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim

#import models.imagenet as customized_models
from utils import Bar, Logger, AverageMeter, accuracy, mkdir_p, savefig

import numpy as np

#from funcs import *
from tsoftmax import *
from data import *
from training import *
from arguments import args, state
from models import chooser


# Use CUDA
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
use_cuda = torch.cuda.is_available()
print("CUDA available", use_cuda)

# Random seed
if args.manualSeed is None:
    args.manualSeed = random.randint(1, 10000)
random.seed(args.manualSeed)
torch.manual_seed(args.manualSeed)
if use_cuda:
    torch.cuda.manual_seed_all(args.manualSeed)

def save_checkpoint(state, is_best, checkpoint='checkpoint', filename='checkpoint.pth.tar'):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, 'model_best.pth.tar'))

def adjust_learning_rate(optimizer, epoch, args):
    global state
    if epoch in args.schedule:
        state['lr'] *= args.gamma
        for param_group in optimizer.param_groups:
            param_group['lr'] = state['lr']
def main():
    title = 'Plant-' + args.arch
    best_acc = 0
    cudnn.benchmark = True
    start_epoch = args.start_epoch  # start from epoch 0 or last checkpoint epoch
    num_classes = 100
    '''
    num_classes = dset.hierarchy.get_class_level_size(0)
    parent_num_classes = dset.hierarchy.get_class_level_size(2)
    hierarchy_matrix = dset.hierarchy.get_hierarchy_mask(2, 0)
    if use_cuda:
        hierarchy_matrix = torch.FloatTensor(hierarchy_matrix)
    else:
        hierarchy_matrix = torch.FloatTensor(hierarchy_matrix)
    '''

    if not os.path.isdir(args.checkpoint):
        mkdir_p(args.checkpoint)

    model = chooser.create_model(chooser.predefined_model(args, cuda=use_cuda), num_classes, cuda=use_cuda)

    #criterion = CrossEntropyLossTSoftmax(hierarchy_matrix=hierarchy_matrix)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    if args.resume:
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        assert os.path.isfile(args.resume), 'Error: no checkpoint directory found!'
        args.checkpoint = os.path.dirname(args.resume)
        checkpoint = torch.load(args.resume)
        best_acc = checkpoint['best_acc']
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        logger = Logger(os.path.join(args.checkpoint, 'log.txt'), title=title, resume=True)
    else:
        logger = Logger(os.path.join(args.checkpoint, 'log.txt'), title=title)
        logger.set_names(['Learning Rate', 'Train Loss', 'Valid Loss', 'Train Acc. Top 1', 'Train Acc. Top 5', 'Valid Acc. Top 1', 'Valid Acc. Top 5', 'USM Alpha'])

    if args.evaluate:
        print('\nEvaluation only')
        test_loss, test_acc = test(dataloaders['val'], model, criterion, start_epoch, use_cuda)
        print(' Test Loss:  %.8f, Test Acc:  %.2f' % (test_loss, test_acc))
        return

    # Train and val
    for epoch in range(start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch, args)

        print('\nEpoch: [%d | %d] LR: %f' % (epoch + 1, args.epochs, state['lr']))

        train_loss, train_acc, train_acc5 = train(dataloaders['train'], model, criterion, optimizer, epoch, use_cuda)
        test_loss, test_acc, test_acc5 = test(dataloaders['val'], model, criterion, epoch, use_cuda)

        # append logger file
        logger.append([state['lr'], train_loss, test_loss, train_acc, train_acc5, test_acc, test_acc5, float(model.filter.alpha.detach().data)])

        # save model
        is_best = test_acc > best_acc
        best_acc = max(test_acc, best_acc)
        save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'acc': test_acc,
                'best_acc': best_acc,
                'optimizer' : optimizer.state_dict(),
            }, is_best, checkpoint=args.checkpoint)

    logger.close()
    logger.plot()
    savefig(os.path.join(args.checkpoint, 'log.eps'))

    print('Best acc:')
    print(best_acc)

if __name__ == '__main__':
    main()
