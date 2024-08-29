import gc
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

import sys, os
from ..Dataset.FFPP_aug_dataset import FFPP_Dataset


import random
import time

from torch.utils.tensorboard import SummaryWriter
from sklearn import metrics
from torch.cuda.amp import autocast, GradScaler
from test_utils import compute_F1, compute_ACC, compute_AUC, compute_recall, compute_precision


import torchvision
from torch.utils import data
import os
import cv2
os.environ['CUDA_LAUNCH_BLOCKING'] = '1' # 下面老是报错 shape 不一致

def compute_AUC(y, pred, n_class=1):
    # compute one score
    if n_class == 1:
        auc = metrics.roc_auc_score(y, pred)

    # compute two-class
    elif n_class == 2:
        # pos = pred[:, 1]
        auc = metrics.roc_auc_score(y, pred)
    return auc

def set_requires_grad(nets, requires_grad=False):
    if not isinstance(nets, list):
        nets = [nets]
    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad

image_size = 224
test_dataset = FFPP_Dataset(phase='test',image_size=image_size, compress='c23', tamper='all')
batch_size = 64
Test_set = torch.utils.data.DataLoader(test_dataset,
                                            batch_size = batch_size//2,
                                            shuffle=False,
                                            pin_memory=True,
                                            num_workers=8,
                                            collate_fn=test_dataset.collate_fn,
                                            worker_init_fn=test_dataset.worker_init_fn,
                                            drop_last = True
                                            )

test_len = len(Test_set)
print('length of TestSet is ', test_len)

# -----------------Build optimizerr-----------------
lr = 2e-5
wd = 5e-3
device = "cuda" if torch.cuda.is_available() else "cpu"
from CLIP import clip
from CLIP_Model import Face_Clip
clip_model, preprocess = clip.load("ViT-B/16", \
    device=torch.device("cpu"), download_root="../model/clip/clip_model")#ViT-B/16
model = Face_Clip(clip_model).cuda()
check = True
if check:
    wp = '../weight/Face_clip_soft_learnable_Epoch162.pth'
    model.load_state_dict(torch.load(wp)) 
    print('parameters inherited from ', wp)
    
model.eval()
outputs = None
testargets = None
with torch.no_grad():
    for step_id, datas in enumerate(Test_set):
        img = datas['img'].to(device, non_blocking=True).float()
        targets = datas['label'].to(device, non_blocking=True).float()
        output = model.test_forward(img)
        cls_final = torch.sigmoid(output)
        outputs = cls_final if outputs is None else torch.cat(
            (outputs, cls_final), dim=0)
        testargets = targets if testargets is None else torch.cat(
            (testargets, targets), dim=0)
        
t = 0.5
cdfauc = compute_AUC(testargets.cpu().detach(), outputs.cpu().detach(), n_class=2)
ap = metrics.average_precision_score(testargets.cpu().detach(), outputs.cpu().detach())
acc = compute_ACC(testargets.cpu().detach(), outputs.cpu().detach(), n_class=1, t=t)
f1 = compute_F1(testargets.cpu().detach(), outputs.cpu().detach(), n_class=1, t=t)
recall = compute_recall(testargets.cpu().detach(), outputs.cpu().detach(), n_class=1, t=t)
pre = compute_precision(testargets.cpu().detach(), outputs.cpu().detach(), n_class=1, t=t)
print(f'test AUC:{cdfauc:.4f}; AP: {ap:.4f}')
print(f'acc : {acc:.4f} ; f1 : {f1:.4f} ; ')
