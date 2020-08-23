# -*- coding: utf-8 -*-
'''
@time: 2019/9/12 15:16

@ author: javis
'''
import torch
import numpy as np
import time,os
from sklearn.metrics import f1_score,roc_auc_score
from torch import nn


def mkdirs(path):
    if not os.path.exists(path):
        os.makedirs(path)

#计算F1score,每一类单独计算最后计算平均的auc
def calc_auc(y_true, y_pre, threshold=0.5):
    labels = y_true.cpu().detach().numpy().astype(np.int)
    outputs = y_pre.cpu().detach().numpy() 
    return roc_auc_score(labels, outputs,'micro')
    
    
#打印时间
def print_time_cost(since):
    time_elapsed = time.time() - since
    return '{:.0f}m{:.0f}s\n'.format(time_elapsed // 60, time_elapsed % 60)


# 调整学习率
def adjust_learning_rate(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

#多标签使用类别权重
class WeightedMultilabel(nn.Module):
    def __init__(self, weights: torch.Tensor):
        super(WeightedMultilabel, self).__init__()
        self.cerition = nn.BCEWithLogitsLoss(reduction='none')
        self.weights = weights

    def forward(self, outputs, targets):
        loss = self.cerition(outputs, targets)
        return (loss * self.weights).mean()
