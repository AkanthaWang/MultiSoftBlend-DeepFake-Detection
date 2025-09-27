import os
import cv2
import torch
import torch.utils.data as data
import torchvision.transforms as tr
import numpy as np

class VideoDFD(data.Dataset):
    def __init__(self,image_size=224,
                      normalize=True,
                      interpolation=cv2.INTER_CUBIC) -> None:
        super().__init__()
        root = 'Data/VideoDFD'
        idx_root = os.path.join(root,'idx')
        txt_root = os.path.join(root,'all.txt')
        allData = np.loadtxt(txt_root, dtype='str', delimiter='\t')
        
        self.image_size = image_size
        self.normalize = normalize
        self.interpolation = interpolation

        self.video_list = []
        self.target_list = []
        self.idx_list = []
        
        
        for _root, _, files in os.walk(idx_root):
            if files:
                for file in files:
                    idx_path = os.path.join(_root, file)
                    self.idx_list.append(idx_path)
        self.idx_list = list(sorted(self.idx_list))
            
        for i in range(len(self.idx_list)):
            video_path = self.idx_list[i].replace("/idx", "").replace(".npy", "")
            self.video_list.append(video_path)
        for j in range(len(self.video_list)):
            for i in range(len(allData)):
                if self.video_list[j] == allData[i][0]:
                    self.target_list.append(int(allData[i][1]))
                    break
        
        if normalize == 'clip':
            print("Normalize with CLIP")
            self.normalize = True
            self.norm = tr.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))
        elif normalize == "imagenet":
            print("Normalize with ImageNet")
            self.normalize=True
            self.norm = tr.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        elif normalize == "efficient":
            print("Normalize with Efficient")
            self.normalize=True
            self.norm = tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))

        
        
    def __len__(self):
        return len(self.video_list)
    
    def __getitem__(self, index):
        video_path = self.video_list[index]
        target = self.target_list[index]
        idx_path = self.idx_list[index]
        
        video = None
        try:
            for _root, _, files in os.walk(video_path):
                if files:
                    for file in files:
                        img_path = os.path.join(_root, file)                  
                        image = cv2.imread(img_path)
                        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                        image = cv2.resize(image, dsize=(self.image_size, self.image_size), interpolation=self.interpolation)
                
                        image = image.transpose((2, 0, 1)) / 255.
                        img = torch.tensor(image).float()
                        if self.normalize:
                            img = self.norm(img)
                        img = img.unsqueeze_(0)
                        video = img if video is None else torch.cat((video, img), dim=0)
        except Exception as e:
            print(e)
                    
        return video, target, idx_path
    
    def collate_fn(self, batch):
        video, target, idx_path = zip(*batch)
        video = video[0]
        idx_path = idx_path[0]
        target = torch.tensor(target).unsqueeze(0)
        return video, target, idx_path