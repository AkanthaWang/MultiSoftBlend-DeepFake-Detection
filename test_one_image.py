import gc
import time

import torch
import torch.utils.data as data
import torchvision.transforms as tr
from sklearn import metrics

from torchvision.transforms import InterpolationMode
from PIL import Image
import random
import numpy as np
import torch.nn.functional as F
import cv2
import torchvision.transforms.functional as TF
import os
from functools import reduce
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

# 测试图像路径
img_path = '../test_image/Real face/6.png'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
from model.CLIP import clip
from model.CLIP_Model import Face_Clip
clip_model, preprocess = clip.load("ViT-B/16", \
    device=device, download_root="../model/pretrianed_clip_weight")#ViT-B/16
model = Face_Clip(clip_model).to(device)
check = True
if check:
    wp = './weight/Face_clip_soft_learnable_Epoch162.pth'
    model_state_dict = torch.load(wp, map_location=device) 
    model.load_state_dict(model_state_dict)
    print('parameters inherited from ', wp)

print('model weight loaded!')

def Tensor2cv(img_tensor):
    img_tensor = img_tensor.squeeze(0).permute(1, 2, 0).cpu()
    img_numpy = img_tensor.numpy() * 255
    img_numpy = np.uint8(img_numpy)
    img_numpy = cv2.cvtColor(img_numpy, cv2.COLOR_RGB2BGR)
    return img_numpy

def TensorRead(path, size=(224, 224)):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, dsize=size)
    
    img_tensor = torch.from_numpy(img).float() / 255.
    img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0)
    return img_tensor

outputs = None
testargets = None
print('start testing...')

start = time.time()
cnt = 0
import torchvision.transforms as tr
with torch.no_grad():
    print(img_path)
    img = TensorRead(img_path, size=(224, 224)).to(device)
    cls_final = model.test_forward(img)
    cls_final = torch.sigmoid_(cls_final)
    end = time.time()
    spent = (end - start) 
    
print(f'spent time: {spent:.4f} s')
p = float(cls_final[0].cpu().numpy()*100.)
print(f'the probability of face image is fake: {p:.2f} %')

