import os
import torch
import torch.utils.data as data
import torchvision.transforms as tr
from torchvision.transforms import InterpolationMode
from PIL import Image
import random
import numpy as np
import torch.nn.functional as F
import cv2
import torchvision.transforms.functional as TF

from numpy.linalg import lstsq, inv
from numpy.linalg import matrix_rank as rank
import cv2
import os
import numpy as np

class WildDeepfake(data.Dataset):
    def __init__(self,image_size=384) -> None:
        super().__init__()
        
        txt_root = '/home/liu/sdb/wildDeepfake/WildDeefake.txt'
        allData = np.loadtxt(txt_root, dtype='str', delimiter='\t')
        
        self.video_list = []
        self.target_list = []
        self.idx_list = []
        
        self.totensor = [
            tr.ToTensor(),
            # tr.Resize((256,256), antialias=True),
            # tr.Resize((384, 384), antialias=True),
            # tr.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5]),
            # tr.Resize((image_size, image_size),antialias=True)
        ]
        self.image_size = image_size
        self.totensor = tr.Compose(self.totensor)

        allData = list(sorted(allData, key=lambda x:x[0]))
        # print(allData)
        for i in range(len(allData)):
            parts = allData[i].rsplit(' ', 1)
            self.video_list.append(parts[0])
            self.target_list.append(int(parts[1].strip()))
            # break
        self.length = len(self.video_list)
        
        
    def __len__(self):
        return self.length
    
    def __getitem__(self, index):
        video_path = self.video_list[index]
        target = self.target_list[index]
        
        video = None
        # try:
        for _root, _, files in os.walk(video_path):
            if files:
                for file in files:
                    # file = file.decode()
                    img_path = os.path.join(_root, file)
                    # img = Image.open(img_path)
                    # img = self.totensor(img)
                    # img = img.unsqueeze(0)
                    
                    image = cv2.imread(img_path)
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    inter_flag = cv2.INTER_CUBIC
                    image = cv2.resize(image, dsize=(self.image_size, self.image_size), interpolation=inter_flag)
            
                    image = image.transpose((2, 0, 1)) / 255.
                    img = torch.tensor(image).float()
                    # img = self.norm(img)
                    img = img.unsqueeze_(0)
                    video = img if video is None else torch.cat((video, img), dim=0)
        if video is None:
            print(video_path)
        # except Exception as e:
        #     print(e)
                    
        return video, target
                    

    def collate_fn(self, batch):
        video, target = zip(*batch)
        video = video[0]
        target = torch.tensor(target).unsqueeze(0)
        return video, target


if __name__ == "__main__":
    image_size = 224
    dataset = WildDeepfake( image_size = image_size)
    TrainSet = data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=8, collate_fn=dataset.collate_fn)
    print(len(TrainSet))
    for step_id, datas in enumerate(TrainSet):
        print(step_id)
        # print(datas)
    
            