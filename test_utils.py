from functools import reduce
import torch
from sklearn import metrics
import numpy as np


def set_requires_grad(nets, requires_grad=False):
    if not isinstance(nets, list):
        nets = [nets]
    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad


def summary(nets):
    if not isinstance(nets, list):
        nets = [nets]
    for net in nets:
        if net is not None:
            for param in net.parameters():
                print(param.shape)



def Selected_ReconLoss(batch_size, criterion, int_label, pre_residual, gt_residual, positive=True):
    num = 0.
    loss_tmp_recon = 0.
    if positive:
        for i in range(batch_size):
            num += 1.
            if int_label[i] == 0:
                loss_tmp_recon +=  criterion(pre_residual[i], gt_residual[i]) * 1.
            else:
                loss_tmp_recon += criterion(pre_residual[i], gt_residual[i]) * 0.001
        return loss_tmp_recon
    else:
        for i in range(batch_size):
            if int_label[i] == 0:
                num += 1.
                loss_tmp_recon += criterion(pre_residual[i], gt_residual[i])
            else:
                continue
        return loss_tmp_recon


def compute_AUC(y, pred, n_class=1):
    ## compute one score
    if n_class == 1:
        auc = metrics.roc_auc_score(y, pred)

    ## compute two-class
    elif n_class == 2:
        # pos = pred[:, 1]
        auc = metrics.roc_auc_score(y, pred)
    return auc


def compute_NegAUC(y, pred):
    ## compute one score
    # auc = metrics.roc_auc_score(y, pred)

    ## compute two-class
    neg = pred[:, 0]
    auc = metrics.roc_auc_score(y, neg)
    return auc

def compute_ACC(y, pred, n_class=2, t=0.5):
    ## compute one score
    if n_class == 1:
        pred[pred >= t] = 1
        pred[pred < t] = 0
        acc = metrics.accuracy_score(y, pred)

    ## compute two-class
    elif n_class == 2:
        index = torch.argmax(pred, dim=1)
        # index[index!=0]=1
        acc = metrics.accuracy_score(y, index)
        # acc = metrics.f1_score(y, index)
    
    return acc


def compute_F1(y, pred, n_class=1, t=0.5):
    ## compute one score
    if n_class == 1:
        pred[pred >= t] = 1
        pred[pred < t] = 0
        f1 = metrics.f1_score(y, pred)
        return f1
    else:
        ## compute two-class
        index = torch.argmax(pred, dim=1)
        # index[index!=0]=1
        f1 = metrics.f1_score(y, index)
    
    return f1

def compute_recall(y, pred, n_class=1, t=0.5):
    if n_class == 1:
        pred[pred >= t] = 1
        pred[pred < t] = 0
        recall = metrics.recall_score(y, pred)
        return recall
    else:
        index = torch.argmax(pred, dim=1)
        recall = metrics.recall_score(y_true=y, y_pred=index)
        return recall

def compute_precision(y, pred, n_class=1, t=0.5):
    if n_class == 1:
        pred[pred >= t] = 1
        pred[pred < t] = 0
        pre = metrics.precision_score(y, pred)
        return pre
    else:
        index = torch.argmax(pred, dim=1)
        pre = metrics.precision_score(y_true=y, y_pred=index)
    return pre

