import gc
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

import sys, os
from Dataset.FFPP_aug_dataset import FFPP_Dataset

import random
import time

from torch.utils.tensorboard import SummaryWriter
from sklearn import metrics


import torchvision
from torch.utils import data
import os
import cv2
os.environ['CUDA_LAUNCH_BLOCKING'] = '1' 

seed = 10
random.seed(seed)
torch.manual_seed(seed)
np.random.seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.enabled = False
print('Random seed :', seed)
torch.use_deterministic_algorithms(True)
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

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
batch_size = 64
train_dataset = FFPP_Dataset(phase='train',image_size=image_size, compress='c23', tamper='all')
train_set = torch.utils.data.DataLoader(train_dataset,
                                            batch_size = batch_size//2,
                                            shuffle=True,
                                            pin_memory=True,
                                            num_workers=8,
                                            collate_fn=train_dataset.collate_fn,
                                            worker_init_fn=train_dataset.worker_init_fn,
                                            drop_last = True
                                            )
val_dataset = FFPP_Dataset(phase='val',image_size=image_size, compress='c23', tamper='all')
val_set = torch.utils.data.DataLoader(val_dataset,
                                            batch_size = batch_size//2,
                                            shuffle=True,
                                            pin_memory=True,
                                            num_workers=8,
                                            collate_fn=train_dataset.collate_fn,
                                            worker_init_fn=train_dataset.worker_init_fn,
                                            drop_last = True
                                            )
train_len = len(train_set)
print('length of TrainSet is ', train_len)

criterion_dict = dict()
criterion_dict['bce'] = nn.BCEWithLogitsLoss().cuda()
criterion_dict['ce'] = nn.CrossEntropyLoss().cuda()
criterion_dict['mse'] = nn.MSELoss().cuda()
criterion_dict['l1'] = nn.L1Loss().cuda()
# -----------------Build optimizerr-----------------
lr = 2e-5 
wd = 5e-3 
device = "cuda" if torch.cuda.is_available() else "cpu"
from model.CLIP import clip
from model.CLIP_Model import Face_Clip
clip_model, preprocess = clip.load("ViT-B/16", \
    device=torch.device("cpu"), download_root="../model/pretrained_clip_weight")#ViT-B/16
model = Face_Clip(clip_model).cuda()

best_auc = 0.0


print(
    f'Set of Optimizer is Batch_size:{batch_size}, lr:{lr}, weight_decay:{wd}')
model_params = [{'params': model.parameters(), 'lr': lr},
                ]

optims = 'adan'
if optims == 'adan':
    from adan import Adan
    optimizer = Adan(model_params, betas=(0.98, 0.92, 0.99),
                     weight_decay=wd, max_grad_norm=0.)
elif optims == 'sgd':
    optimizer = optim.SGD(model_params, momentum=0.9, weight_decay=wd)
elif optims == 'adamw':
    optimizer = optim.AdamW(model_params, betas=(0.9, 0.999), weight_decay=wd)
elif optims == 'adam':
    optimizer = optim.Adam(model_params, betas=(0.9, 0.999), weight_decay=wd)

print('Current Optimizer is', optims)

Epoch = 1000
step = 0
step2 = 0
step_v = 0

device = torch.device('cuda')
writer = SummaryWriter('/path/')
gc.collect()
torch.cuda.empty_cache()
print("Let's start training!")

scheduler1 = optim.lr_scheduler.ConstantLR(
    optimizer, factor=1., total_iters=75, verbose=True)
scheduler2 = optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=25, eta_min=lr*0.01, verbose=True)
scheduler = optim.lr_scheduler.SequentialLR(
    optimizer, [scheduler1, scheduler2], milestones=[75])

max_val_auc = 0.

for e in range(0, Epoch):
    start = time.time()
    model.train()
    for step_id, data in enumerate(train_set):
        img = data['img'].to(device, non_blocking=True).float()
        label = data['label'].to(device, non_blocking=True).float()
        nor1 = data['nor1'].to(device, non_blocking=True).float()
        nor2 = data['nor2'].to(device, non_blocking=True).float()
        nor3 = data['nor3'].to(device, non_blocking=True).float()
        nor4 = data['nor4'].to(device, non_blocking=True).float()
        residual = data['residual_map'].to(device, non_blocking=True).float()
        probs, cls_real, cls1, cls2, cls3, cls4,residual_feat = model(img)
        
        
        loss_cls = criterion_dict['bce'](probs, label.float().squeeze(1))
        loss_cls1 = criterion_dict['bce'](cls1, nor1.float())
        loss_cls2 = criterion_dict['bce'](cls2, nor2.float())
        loss_cls3 = criterion_dict['bce'](cls3, nor3.float())
        loss_cls4 = criterion_dict['bce'](cls4, nor4.float())
        loss_residual = criterion_dict['l1'](residual_feat, residual)
        loss =  loss_cls*1. + loss_cls1*1. + loss_cls2*1. + loss_cls3*1. + loss_cls4*1. + loss_residual*1.
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if not (step_id+1) % 30:
            print(f"epoch: {e} / {Epoch},step {step_id} / {len(train_set)}, loss: {loss.detach().cpu().numpy():.4f}")
    
    model.eval()
    outputs = None
    testargets = None
    with torch.no_grad():
        for step_id, datas in enumerate(val_set):
            img = data['img'].to(device, non_blocking=True).float()
            targets = data['label'].to(device, non_blocking=True).float()
            output = model.test_forward(img)
            cls_final = torch.sigmoid(output)
            outputs = cls_final if outputs is None else torch.cat(
                (outputs, cls_final), dim=0)
            testargets = targets if testargets is None else torch.cat(
                (testargets, targets), dim=0)
    cdfauc = compute_AUC(testargets.cpu().detach(), outputs.cpu().detach())
    print(f'CelebDF test AUC:{cdfauc:.4f}')
    if best_auc < cdfauc:
        best_auc = cdfauc
        torch.save(model.state_dict(
        ), f'/path/model/model_weight/Face_clip_soft_learnable_Epoch{e}_cdf{cdfauc:.4f}.pth')
    writer.add_scalar('test/AUC', cdfauc, e)
    end = time.time()
    print(f"epoch: {e} end ; cost time: {(end - start)/60.:.4f} min")
    gc.collect()
    torch.cuda.empty_cache()
    model.train()
    scheduler.step()
print('train ended !')
writer.close()
