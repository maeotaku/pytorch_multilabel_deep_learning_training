import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn

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

def cross_entropy(pred, soft_targets, unused):
    y = soft_targets.view(soft_targets.shape[0], 1)
    y_onehot = torch.FloatTensor(pred.shape[0], pred.shape[1])
    y_onehot.zero_()
    y_onehot.scatter_(1, y, 1)
    logsoftmax = nn.LogSoftmax()
    return torch.mean(torch.sum(- y_onehot * logsoftmax(pred), 1))
