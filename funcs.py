import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import numpy as np

from layers import HierarchicalSoftmax

def hierarchical_log_loss(pred, soft_targets, soft_parents):
    y_onehot = torch.FloatTensor(pred.shape[0], pred.shape[1])
    y_onehot.zero_()
    softmax = nn.Softmax()
    logsoftmax = nn.LogSoftmax()
    probs = softmax(pred)

    #all children at genus level
    batch_size = soft_targets.shape[0]
    #print(soft_parents)
    for i in range(batch_size):
        the_one = int(soft_parents.data[i])
        idx = torch.LongTensor(dset.hierarchy.get_children_idx_at_class_level_idx(1, the_one, 0))
        #print(idx)
        #y = soft_targets.view(soft_targets.shape[0], 1)
        #y_onehot.scatter_(1, y, 1)
        y_onehot.data[i, idx] = 1.0 / len(idx)
        y_onehot.data[i, the_one] = 1.0
    #print(y_onehot)
    return torch.mean(torch.sum(- y_onehot * torch.log(probs), 1))
'''
def hierarchical_log_loss(pred, soft_targets, soft_parents):
    y_onehot = torch.FloatTensor(pred.shape[0], pred.shape[1])
    y_onehot.zero_()
    softmax = nn.Softmax()
    logsoftmax = nn.LogSoftmax()
    probs = softmax(pred)

    #all children at genus level
    batch_size = soft_targets.shape[0]
    #print(soft_parents)
    for i in range(batch_size):
        the_one = int(soft_parents.data[i])
        idx = torch.LongTensor(dset.hierarchy.get_children_idx_at_class_level_idx(1, the_one, 0))
        #print(idx)
        #y = soft_targets.view(soft_targets.shape[0], 1)
        #y_onehot.scatter_(1, y, 1)
        y_onehot.data[i, idx] = 1
    #print(y_onehot)
    return torch.mean(torch.sum(- y_onehot * torch.log(probs), 1))
'''

def ce(soft_preds, unused):
    return torch.mean(-torch.log(soft_preds))

'''
def cross_entropy(pred, soft_targets, unused):
    y = soft_targets.view(soft_targets.shape[0], 1)
    #y_onehot = torch.cuda.FloatTensor(pred.shape[0], pred.shape[1])
    y_onehot = torch.FloatTensor(pred.shape[0], pred.shape[1])
    y_onehot.zero_()
    y_onehot.scatter_(1, y, 1)
    logsoftmax = nn.LogSoftmax()
    return torch.mean(torch.sum(- y_onehot * logsoftmax(pred), 1))
'''

def cross_entropy_h(pred, soft_targets, unused):
    #print(pred)
    y = soft_targets.view(soft_targets.shape[0], 1)
    #y_onehot = torch.cuda.FloatTensor(pred.shape[0], pred.shape[1])
    y_onehot = torch.FloatTensor(pred.shape[0], pred.shape[1])
    y_onehot.zero_()
    y_onehot.scatter_(1, y, 1)
    return torch.mean(torch.sum(- y_onehot * torch.log(pred), 1))


def cross_entropy_2d(pred, soft_targets):
    torch.set_printoptions(threshold=5000)
    #s = hierarchical_softmax
    
    print(pred)
    #print(softmax2d(pred))
    
    y = soft_targets.view(soft_targets.shape[0], soft_targets.shape[1], 1)
    #y_onehot = torch.cuda.FloatTensor(pred.shape[0], pred.shape[1])
    y_onehot = torch.FloatTensor(pred.shape[0], pred.shape[1], pred.shape[2])
    #print(soft_targets.size(), y.size(), y_onehot.size())
    y_onehot.zero_()
    #y_onehot.scatter_(1, y, 1)
    for i in range(y_onehot.size(0)):
        #print(soft_targets[i, 0])
        y_onehot[i, soft_targets[i, 1], soft_targets[i, 0]] = 1
    return torch.mean(torch.sum(- y_onehot * torch.log(pred), 1))

'''
def cross_entropy_2d(pred, soft_targets):
    torch.set_printoptions(threshold=5000)
    #s = HierarchicalSoftmax(8, 10, 4)
    s = hierarchical_softmax
    
    #print(pred.type())
    #print(softmax2d(pred))
    
    y = soft_targets.view(soft_targets.shape[0], soft_targets.shape[1], 1)
    #y_onehot = torch.cuda.FloatTensor(pred.shape[0], pred.shape[1])
    y_onehot = torch.FloatTensor(pred.shape[0], pred.shape[1], pred.shape[2])
    #print(soft_targets.size(), y.size(), y_onehot.size())
    y_onehot.zero_()
    #y_onehot.scatter_(1, y, 1)
    for i in range(y_onehot.size(0)):
        #print(soft_targets[i, 0])
        y_onehot[i, soft_targets[i, 1], soft_targets[i, 0]] = 1
    return torch.mean(torch.sum(- y_onehot * torch.log(s(pred)), 1))
'''

def softmax2d(input):
    e = torch.exp(input) #- input.max(dim=2, keepdim=True))
    return e / torch.sum(e, dim=1, keepdim=True)


def to_list(x):
    if x is None:
        return ()
    if not type(x) is tuple:
        return x
    a, b = x
    return (to_list(a),) + to_list(b)

def accuracy2d(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)
    output_l = output.numpy() #easier to work with numpy this stuff, no need for tensors
    
    res = [0, 0]
    for i in range(batch_size):
        x = output_l[i]
        idx = np.unravel_index(np.argsort(x.ravel())[-maxk:], x.shape)
        idx = np.column_stack(idx)
        ordered_target = np.flip(target[i], 0)

        if np.all(ordered_target == idx[0]):
            res[0] += 1

        j=0
        while j<maxk:
            if np.all(ordered_target == idx[j]):
                res[1] += 1
                j=maxk
            j+=1
        
        #idx = x.view(-1).topk(2, -1, True, True)
        #rows = idx / x.size(0)
        #cols = idx % x.size(1)

    res[0] = (res[0] * 100.0) / batch_size
    res[1] = (res[1] * 100.0) / batch_size

    #_, pred = output.topk(maxk, 2, True, True)
    #print(pred)
    #print(target)
    #pred = pred.t()
    #correct = pred.eq(target.view(1, -1).expand_as(pred))

    
    #for k in topk:
    #    correct_k = correct[:k].view(-1).float().sum(0)
    #    res.append(correct_k.mul_(100.0 / batch_size))
    return res