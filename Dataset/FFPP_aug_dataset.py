import torch
from torchvision import datasets,transforms,utils
from torch.utils.data import Dataset,IterableDataset
import torchvision.transforms as tr
from glob import glob
import os
import numpy as np
import random
import cv2
import json
import io
from PIL import Image
import time
from torchvision.transforms import InterpolationMode
import sys
import warnings
import albumentations as alb
warnings.filterwarnings('ignore')


def IoUfrom2bboxes(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)
    # return the intersection over union value
    return iou

def crop_face(img,landmark=None,bbox=None,margin=False,crop_by_bbox=True,abs_coord=False,only_img=False,phase='train'):
    assert phase in ['train','val','test']

    #crop face------------------------------------------
    H,W=len(img),len(img[0])

    assert landmark is not None or bbox is not None

    H,W=len(img),len(img[0])
    
    if crop_by_bbox:
        x0,y0=bbox[0]
        x1,y1=bbox[1]
        w=x1-x0
        h=y1-y0
        w0_margin=w/4#0#np.random.rand()*(w/8)
        w1_margin=w/4
        h0_margin=h/4#0#np.random.rand()*(h/5)
        h1_margin=h/4
    else:
        x0,y0=landmark[:68,0].min(),landmark[:68,1].min()
        x1,y1=landmark[:68,0].max(),landmark[:68,1].max()
        w=x1-x0
        h=y1-y0
        w0_margin=w/32#w/8#0#np.random.rand()*(w/8)
        w1_margin=w/32#w/8
        h0_margin=h/4.5#h/2#0#np.random.rand()*(h/5)
        h1_margin=h/16# h/5

    if margin:
        w0_margin*=4
        w1_margin*=4
        h0_margin*=2
        h1_margin*=2
    elif phase=='train':
        
        w0_margin*=(np.random.rand()*0.6+4.)#np.random.rand()
        w1_margin*=(np.random.rand()*0.6+4.)#np.random.rand()
        h0_margin*=(np.random.rand()*0.6+2.)#np.random.rand()
        h1_margin*=(np.random.rand()*0.6+2.)#np.random.rand()
    else:
        w0_margin*=0.5
        w1_margin*=0.5
        h0_margin*=0.5
        h1_margin*=0.5
            
    y0_new=max(0,int(y0-h0_margin))
    y1_new=min(H,int(y1+h1_margin)+1)
    x0_new=max(0,int(x0-w0_margin))
    x1_new=min(W,int(x1+w1_margin)+1)
    
    img_cropped=img[y0_new:y1_new,x0_new:x1_new]
    if landmark is not None:
        landmark_cropped=np.zeros_like(landmark)
        for i,(p,q) in enumerate(landmark):
            landmark_cropped[i]=[p-x0_new,q-y0_new]
    else:
        landmark_cropped=None
    if bbox is not None:
        bbox_cropped=np.zeros_like(bbox)
        for i,(p,q) in enumerate(bbox):
            bbox_cropped[i]=[p-x0_new,q-y0_new]
    else:
        bbox_cropped=None

    if only_img:
        return img_cropped
    if abs_coord:
        return img_cropped,landmark_cropped,bbox_cropped,(y0-y0_new,x0-x0_new,y1_new-y1,x1_new-x1),y0_new,y1_new,x0_new,x1_new
    else:
        return img_cropped,landmark_cropped,bbox_cropped,(y0-y0_new,x0-x0_new,y1_new-y1,x1_new-x1)

def init_fff(phase, dataset_paths, DF, NT, FS, FF, n_frames=8):
    landmark_path = '/home/liu/wsy/Data/Dataset/FFPP_16/landmarks/'
    image_list = []
    DFs_list = []
    NTs_list = []
    FSs_list = []
    FFs_list = []
    landmark_list=[]
    folder_list = sorted(glob(os.path.join(dataset_paths,'*')))
    landmark_folder_list = sorted(glob(landmark_path+'*'))
    DF_list = sorted(glob(os.path.join(DF,'*')))
    DF_list = sorted(DF_list, key=lambda x: int(x.split('/')[-1].split('_')[0]))
    NT_list = sorted(glob(os.path.join(NT,'*')))
    NT_list = sorted(NT_list, key=lambda x: int(x.split('/')[-1].split('_')[0]))
    FS_list = sorted(glob(os.path.join(FS,'*')))
    FS_list = sorted(FS_list, key=lambda x: int(x.split('/')[-1].split('_')[0]))
    FF_list = sorted(glob(os.path.join(FF,'*')))
    FF_list = sorted(FF_list, key=lambda x: int(x.split('/')[-1].split('_')[0]))
    filelist = []
    list_dict = json.load(open(f'/home/liu/wsy/Data/Dataset/FFplus/{phase}.json','r'))
    for i in list_dict:
        filelist+=i
    folder_list = [i for i in folder_list if os.path.basename(i)[:3] in filelist]
    landmark_folder_list = [i for i in landmark_folder_list if os.path.basename(i)[:3] in filelist]
    DF_list = [i for i in DF_list if os.path.basename(i)[:7].split('/')[-1].split('_')[0] in filelist]
    NT_list = [i for i in NT_list if os.path.basename(i)[:7].split('/')[-1].split('_')[0] in filelist]
    FS_list = [i for i in FS_list if os.path.basename(i)[:7].split('/')[-1].split('_')[0] in filelist]
    FF_list = [i for i in FF_list if os.path.basename(i)[:7].split('/')[-1].split('_')[0] in filelist]
    for i in range(len(folder_list)):
        images_temp=sorted(glob(folder_list[i]+'/*.png'))
        DF_temp = sorted(glob(DF_list[i]+'/*.png'))
        NT_temp = sorted(glob(NT_list[i]+'/*.png'))
        FS_temp = sorted(glob(FS_list[i]+'/*.png'))
        FF_temp = sorted(glob(FF_list[i]+'/*.png'))
        landmarks_temp=sorted(glob(landmark_folder_list[i]+'/*.npy'))
        if n_frames<len(images_temp):
            images_temp=[images_temp[round(i)] for i in np.linspace(0,len(images_temp)-1,n_frames)]
            DF_temp=[DF_temp[round(i)] for i in np.linspace(0,len(DF_temp)-1,n_frames)]
            NT_temp=[NT_temp[round(i)] for i in np.linspace(0,len(NT_temp)-1,n_frames)]
            FS_temp=[FS_temp[round(i)] for i in np.linspace(0,len(FS_temp)-1,n_frames)]
            FF_temp=[FF_temp[round(i)] for i in np.linspace(0,len(FF_temp)-1,n_frames)]
            landmarks_temp=[landmarks_temp[round(i)] for i in np.linspace(0,len(landmarks_temp)-1,n_frames)]     
        image_list+=images_temp
        DFs_list+=DF_temp
        NTs_list+=NT_temp
        FSs_list+=FS_temp
        FFs_list+=FF_temp
        landmark_list+=landmarks_temp
    return image_list,DFs_list,NTs_list,FSs_list,FFs_list,landmark_list

class FFPP_Dataset(Dataset):
    def __init__(self,compress='raw',image_size=256, phase = "train", tamper="all"):# raw、c23、c40
        super().__init__()
        self.original_root = f'/path/FFPPDataset/original_sequences/{compress}/images/'
        self.Deepfakes_root = f'/path/FFPPDataset/manipulated_sequences/Deepfakes/{compress}/images/'
        self.NeuralTextures_root = f'/path/FFPPDataset/manipulated_sequences/NeuralTextures/{compress}/images/'
        self.FaseSwap_root = f'/path/FFPPDataset/manipulated_sequences/FaceSwap/{compress}/images/'
        self.Face2Face_root = f'/path/FFPPDataset/manipulated_sequences/Face2Face/{compress}/images/'
        print('start')
        st = time.time()
        original_list,Deepfakes_list,NeuralTextures_list,FaseSwap_list,Face2Face_list,landmark_list = init_fff(phase=phase,
                                                                                                               dataset_paths=self.original_root,
                                                                                                               DF=self.Deepfakes_root,
                                                                                                               NT=self.NeuralTextures_root,
                                                                                                               FS=self.FaseSwap_root,
                                                                                                               FF=self.Face2Face_root,
                                                                                                               n_frames=8)
        self.real_frame_list = original_list
        self.landmark_list = landmark_list
        self.tamper = tamper
        if tamper == 'all':
            self.fake_frame_list = Deepfakes_list + NeuralTextures_list + FaseSwap_list + Face2Face_list
        elif tamper == 'DF':
            self.fake_frame_list = Deepfakes_list
        elif tamper == 'NT':
            self.fake_frame_list = NeuralTextures_list
        elif tamper == 'FS':
            self.fake_frame_list = FaseSwap_list
        elif tamper == 'FF':
            self.fake_frame_list = Face2Face_list
        ed = time.time()
        print(f'load... {ed -st:.2f} s')
        print('Real samples : {}'.format(len(self.real_frame_list)))
        print('Fake samples : {}'.format(len(self.fake_frame_list)))
        
        
        self.hflip = tr.RandomHorizontalFlip(1)
        self.trans = [
            tr.ToTensor(),
            tr.Resize((image_size, image_size), antialias=True)

        ]
        self.trans = tr.Compose(self.trans)
        # self.tamper = tamper
        self.idx_real = 0
        self.max_real = len(self.real_frame_list)
        # self.compress = compress
        self.img_size = image_size
        
        self.landmark_root = "/path/FFPP_16/landmarks/"
        self.real_root = self.original_root
        self.augmentation = self.get_transforms()
        self.augmentation2 = alb.ImageCompression(quality_lower=40,quality_upper=100,p=0.2)
        self.phase = phase
        

        
    def __len__(self):
        return len(self.fake_frame_list)
    
    def RandomGaussianNoiseInjection(self, x, p=0.05):
        _p = np.random.random(1)
        noise_weight = [0.01, 0.02]
        w = random.choice(noise_weight)
        if _p <= p:
            noise = w * torch.randn(size=x.shape, requires_grad=False)
            x = x + noise
            x[x > 1] = 1
            x[x < 0] = 0
        return x
    
    def RandomCompress(self, src, p=0.5):
        _p = np.random.random(1)
        if _p <= p:
            quality = np.random.randint(40, 100)
            compressed = self.JPEGCompress(src, quality=quality)
            return compressed
        return src
    
    @staticmethod
    def JPEGCompress(src, quality=70):
        x = tr.ToPILImage()(src)
        tmp = io.BytesIO()
        x.save(tmp, format="JPEG", quality=quality)
        tmp.seek(0)
        x = Image.open(tmp)
        x = tr.ToTensor()(x)
        return x
    
    def get_transforms(self):
        return alb.Compose([
            alb.RGBShift((-20,20),(-20,20),(-20,20),p=0.3),
            alb.HueSaturationValue(hue_shift_limit=(-0.3,0.3), sat_shift_limit=(-0.3,0.3), val_shift_limit=(-0.3,0.3), p=0.3),
            alb.RandomBrightnessContrast(brightness_limit=(-0.3,0.3), contrast_limit=(-0.3,0.3), p=0.3),
            alb.ImageCompression(quality_lower=40,quality_upper=100,p=0.2),
            alb.GaussianBlur(p=0.05),
            alb.GaussNoise(p=0.03),
        ], 
        additional_targets={f'image1': 'image'},
        p=1.)
        
    def get_patch_mask(self, real, fake):   # for prediction of num of modified pixels in a patch
        ph = pw = 16
        pixel_num_of_patch = ph * pw
        mask = np.abs(real - fake)
        mask = np.sum(mask, axis=0)
        H, W = mask.shape
        mask[mask!=0] = 1
        mask = mask.reshape(H//ph, ph, W//pw, pw)
        mask = mask.transpose((0, 2, 1, 3)) # H, W, patch_h, patch_w
        mask = mask.reshape(H//ph, W//pw, ph*pw) # H, W, patch_h*patch_w
        patch_score = np.sum(mask, axis=2) / pixel_num_of_patch
        patch_score = patch_score.reshape((H//ph)*(W//pw))
        
        return mask    
    
    def get_patch_margin_mask(self, residual):
        ph = pw = 16
        pixel_num_of_patch = ph * pw
        C, H, W = residual.shape
        mask = np.sum(residual, axis=0)
        mask[mask!=0] = 1
        mask = mask.reshape(H//ph, ph, W//pw, pw)
        mask = mask.transpose((0, 2, 1, 3)) # H, W, patch_h, patch_w
        mask = mask.reshape(H//ph, W//pw, ph*pw) # H, W, patch_h*patch_w
        
        margin_mask = np.sum(mask, axis=2) / pixel_num_of_patch
        margin_mask = margin_mask[None, ...]
        return margin_mask
    
    def residual_augment(self,real_img,fake_image):
        random_numbers = [random.random() for _ in range(4)]
        total = sum(random_numbers)
        normalized_numbers = [(num / total) for num in random_numbers]
        fake_img = 0.
        for i in range(4):
            fake = fake_image[i]
            real = real_img.numpy()
            fake = fake.numpy()
            real = real.astype(np.float32)
            fake = fake.astype(np.float32)
            fake_img += ((real - fake))*normalized_numbers[i]
        fake_img = real_img-fake_img
        fake_img = fake_img*255.
        fake_img[fake_img > 255] = 255
        fake_img[fake_img < 0] = 0
        fake_img = fake_img/255.
        return fake_img,normalized_numbers
    
    def __getitem__(self, idx):
        flag = True
        fake_image = []
        while flag:
            idx = torch.randint(low=0,high=len(self.real_frame_list),size=(1,)).item()
            try:
                img_path = self.real_frame_list[idx]
                real_lamk_path = self.landmark_list[idx]
                img = cv2.imread(img_path)
                real_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                landmark=np.load(real_lamk_path)[0]
                if self.phase == 'train':
                    for i in range(4):
                        fake_img_path = self.fake_frame_list[idx+i*len(self.real_frame_list)]
                        fake_img = cv2.imread(fake_img_path)
                        fake_image.append(cv2.cvtColor(fake_img, cv2.COLOR_BGR2RGB))
                else:
                    if self.tamper != 'all':
                        fake_img_path = self.fake_frame_list[idx]
                        fake_img = cv2.imread(fake_img_path)
                        fake_image=cv2.cvtColor(fake_img, cv2.COLOR_BGR2RGB)
                    else:
                        idx = torch.randint(low=0,high=4,size=(1,)).item()
                        fake_img_path = self.fake_frame_list[idx+i*len(self.real_frame_list)]
                        fake_img = cv2.imread(fake_img_path)
                        fake_image=cv2.cvtColor(fake_img, cv2.COLOR_BGR2RGB)
                flag = False
            except Exception as e:
                print(e)
                idx = torch.randint(low=0,high=len(self.real_frame_list),size=(1,)).item()
        # 真实图像       
        real_img, _, _, _, y0_new,y1_new,x0_new,x1_new=crop_face(real_img,landmark,bbox=None,margin=False,crop_by_bbox=False,abs_coord=True,phase=self.phase)
        real_img = self.trans(real_img)# .numpy()
        if self.phase == 'train':
            fake_all_list = []
            for i in range(4):
                fake = fake_image[i]
                fake = fake[y0_new:y1_new,x0_new:x1_new]
                fake = self.trans(fake)
                fake_all_list.append(fake)
                
            fake_img,normalized_numbers = self.residual_augment(real_img=real_img,fake_image=fake_all_list)
            fake_all_list.append(fake_img)
            random_number = torch.rand(1)
            index = int(random_number // 0.2)
            fake_img = fake_all_list[index]
            if index != 4:
                normalized_numbers = [0, 0, 0, 0]
                normalized_numbers[index] = 1
        else:
            fake_img = fake_img[y0_new:y1_new,x0_new:x1_new]
            fake_img = self.trans(fake)

        if np.random.rand() < 0.5:
            real_img = self.hflip(real_img)
            fake_img = self.hflip(fake_img)
        real_img = real_img.numpy()
        fake_img = fake_img.numpy()

        if self.phase == 'train':
            real_img *= 255
            fake_img *= 255
            fake_img = fake_img.transpose((1, 2, 0))
            real_img = real_img.transpose((1, 2, 0))
            transformed=self.augmentation(image=fake_img.astype('uint8'),\
                image1=real_img.astype('uint8'))
            real_img=transformed['image1']/ 255.
            fake_img=transformed['image']/ 255.
            real_img=real_img.transpose((2, 0, 1))
            fake_img=fake_img.transpose((2, 0, 1))
            residual = real_img - fake_img
            residual,_ = self.get_patch_margin_mask(residual)
        else:
            residual=0   
        return fake_img, real_img, normalized_numbers[0],normalized_numbers[1],normalized_numbers[2],normalized_numbers[3],residual
        
    
    def collate_fn(self,batch):
        img_f,img_r,nor1,nor2,nor3,nor4,residual=zip(*batch)
        data={}
        # 原图像
        data['img']=torch.cat([torch.tensor(img_r).float(),torch.tensor(img_f).float()],0)
        # 真伪GT
        data['label']=torch.tensor([0]*len(img_r)+[1]*len(img_f)).unsqueeze(1)
        zero = torch.tensor([0]*len(img_r)).unsqueeze(1)
        nor1 = torch.tensor(nor1).unsqueeze(1)
        data['nor1'] = torch.cat([zero,nor1],dim=0)
        
        nor2 = torch.tensor(nor2).unsqueeze(1)
        data['nor2'] = torch.cat([zero,nor2],dim=0)
        
        nor3 = torch.tensor(nor3).unsqueeze(1)
        data['nor3'] = torch.cat([zero,nor3],dim=0)
        
        nor4 = torch.tensor(nor4).unsqueeze(1)
        data['nor4'] = torch.cat([zero,nor4],dim=0)
         
        residual = torch.tensor(residual).float()
        real_zero = torch.zeros_like(residual)
        data['residual_map'] = torch.cat([real_zero, residual], dim=0)
        
        return data

    def worker_init_fn(self,worker_id):
        np.random.seed(np.random.get_state()[1][0] + worker_id)


       
if __name__ == "__main__":
    
    train_dataset = FFPP_Dataset(phase='train',image_size=224,compress='c23',tamper='all')
    batch_size_sbi = 64
    train_set = torch.utils.data.DataLoader(train_dataset,
                                            batch_size = batch_size_sbi//2,
                                            shuffle=False,
                                            pin_memory=True,
                                            num_workers=1,
                                            collate_fn=train_dataset.collate_fn,
                                            worker_init_fn=train_dataset.worker_init_fn
                                            )
    print(len(train_set))
    print("begin")
    for step_id, data in enumerate(train_set):
        img = data['img']
        print(img.shape)
        label = data['label']
        print(label.shape)
    
