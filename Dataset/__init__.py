from VideoCDF import VideoCDF
from VideoDFD import VideoDFD
from VideoDFDC import VideoDFDC
from VideoDFDCP import VideoDFDCP
from VideoDFV1 import VideoDFV1

import torch.utils.data as data
import cv2

def Get_DataLoader(dataset_name="forgery_detect", image_size=224, normalize='clip', interpolation=cv2.INTER_CUBIC):
    if dataset_name =="VideoCDF":
        dataset = VideoCDF(image_size=image_size,
                            normalize=normalize,
                            interpolation=interpolation)
        return data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=8, collate_fn=dataset.collate_fn)
    elif dataset_name =="VideoDFD":
        dataset = VideoDFD(image_size=image_size,
                            normalize=normalize,
                            interpolation=interpolation)
        return data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=10, collate_fn=dataset.collate_fn)
    elif dataset_name =="VideoDFDC":
        dataset = VideoDFDC(image_size=image_size,
                            normalize=normalize,
                            interpolation=interpolation)
        return data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=10, collate_fn=dataset.collate_fn)
    elif dataset_name =="VideoDFDCP":
        dataset = VideoDFDCP(image_size=image_size,
                            normalize=normalize,
                            interpolation=interpolation)
        return data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=10, collate_fn=dataset.collate_fn)
    elif dataset_name =="VideoDFV1":
        dataset = VideoDFV1(image_size=image_size,
                            normalize=normalize,
                            interpolation=interpolation)
        return data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=10, collate_fn=dataset.collate_fn)
    else:
        raise NotImplementedError("No this kind of dataset!")
