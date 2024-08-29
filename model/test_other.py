'''
对于视频里截取的图片有多张人脸进行测试
'''
import gc
import time

import torch

from ..Dataset.test_dataset import Get_DataLoader
import cv2
import numpy as np
import random

seed = 1
random.seed(seed)
torch.manual_seed(seed)
np.random.seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
from test_utils import compute_F1, compute_ACC, compute_AUC, compute_recall, compute_precision
from sklearn import metrics


def Tensor2cv(img_tensor):
    img_tensor = img_tensor.permute(1, 2, 0).cpu()
    img_numpy = img_tensor.numpy() * 255
    img_numpy = np.uint8(img_numpy)
    img_numpy = cv2.cvtColor(img_numpy, cv2.COLOR_RGB2BGR)
    return img_numpy

img_size = 224
check = True
from CLIP import clip
from CLIP_Model import Face_Clip

clip_model, preprocess = clip.load("ViT-B/16", \
    device=torch.device("cpu"), download_root="../model/clip/clip_model")#ViT-B/16
learn_prompts = Face_Clip(clip_model).to('cuda')
if check:
    wp = '../weight/Face_clip_soft_learnable_Epoch162.pth' 
    learn_prompts.load_state_dict(torch.load(wp)) 
    print('parameters inherited from ', wp)

'''
通过idx判断使用VideoCDF、VideoDFDC还是VideoDFV1、VideoDFD、VideoDFDCP
若idx为False,则测试VideoDFV1、VideoDFD、VideoDFDCP
若idx为True,则测试VideoCDF、VideoDFDC
'''

frame_level = False
# frame_level = True

# test_name = "VideoCDF"
# test_name = "VideoDFDC"
# test_name = 'VideoDFV1'
test_name = 'VideoDFD'
# test_name = 'VideoDFDCP'
if test_name in ["VideoCDF","VideoDFDC"]:
    idx = True
if test_name in ["VideoDFV1","VideoDFD","VideoDFDCP"]:
    idx = False

TestSet = Get_DataLoader(dataset_name=test_name,
                          root='',
                          mode='val',
                          bz=1, 
                          shuffle=False,
                          image_size=img_size)

print(test_name)
print(idx)


if frame_level:
    print('testing in frame level now!')
else:
    print('testing in video level now!')

learn_prompts.eval()
outputs = None
testargets = None
print('start testing...')
print(len(TestSet))
cnt = 0
feats = None

with torch.no_grad():
    start = time.time()
    for step_id, datas in enumerate(TestSet):
        gc.collect()
        torch.cuda.empty_cache()
        if datas[0] is None:
            continue
        img = datas[0].cuda()
        targets = datas[1].cuda()
        if idx:
            idx_path = datas[2]
            idx_list = np.load(idx_path).tolist()
        output = learn_prompts.test_forward(img)
        cls_final = torch.sigmoid(output)
        
        if idx:
            pred_list=[]
            idx_img=-1
            for i in range(len(cls_final)):
                if idx_list[i]!=idx_img:
                    pred_list.append([])
                    idx_img=idx_list[i]
                pred_list[-1].append(cls_final[i].item())

            pred_res=np.zeros(len(pred_list))
            for i in range(len(pred_res)):
                pred_res[i]=max(pred_list[i])
            if frame_level:
                pred=pred_res
                n_frames = len(pred)
                targets = torch.full((n_frames, 1), int(targets.cpu().numpy()), dtype=torch.float32, device=img.device)
            else:
                pred=pred_res.mean()
            cls_final = torch.tensor(pred, dtype=torch.float32).unsqueeze(-1)
        else:
            if frame_level:
                n_frames = img.shape[0]
                targets = torch.full((n_frames, 1), int(targets.cpu().numpy()), dtype=torch.float32, device=img.device)
            else:
                cls_final = cls_final.mean().unsqueeze(0)
        outputs = cls_final if outputs is None else torch.cat((outputs, cls_final), dim=0)
        testargets = targets if testargets is None else torch.cat((testargets, targets), dim=0)
        
    
        if not (step_id+1) % 100:
            now_percent = int(step_id / len(TestSet) * 100)
            print(f"Test: complete {now_percent} %")


cdfauc = compute_AUC(testargets.cpu().detach(), outputs.cpu().detach(), n_class=2)
ap = metrics.average_precision_score(testargets.cpu().detach(), outputs.cpu().detach())
t = 0.5
print('threshold :', t)
acc = compute_ACC(testargets.cpu().detach(), outputs.cpu().detach(), n_class=1, t=t)
f1 = compute_F1(testargets.cpu().detach(), outputs.cpu().detach(), n_class=1, t=t)
recall = compute_recall(testargets.cpu().detach(), outputs.cpu().detach(), n_class=1, t=t)
pre = compute_precision(testargets.cpu().detach(), outputs.cpu().detach(), n_class=1, t=t)
print(f'{test_name} test AUC:{cdfauc:.4f}; AP: {ap:.4f}')
print(f'acc : {acc:.4f} ; f1 : {f1:.4f} ; recall : {recall:.4f} ; precision : {pre:.4f}')


end = time.time()
spent = (end - start) / 60
print(f'spent time: {spent:.4f} min')
print('ending')
gc.collect()
torch.cuda.empty_cache()
exit(0)

