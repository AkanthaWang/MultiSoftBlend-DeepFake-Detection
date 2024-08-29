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
import io
from functools import reduce
from skimage.transform import PiecewiseAffineTransform, warp

from numpy.linalg import lstsq, inv
from numpy.linalg import matrix_rank as rank
import cv2
import os
import numpy as np


def findNonreflectiveSimilarity_for_cv2(uv, xy, K=2):  # uv输入点 xy参考点

    x = xy[:, 0].reshape((-1, 1))  # use reshape to keep a column vector
    y = xy[:, 1].reshape((-1, 1))  # use reshape to keep a column vector
    Y = np.vstack((x, y))
    # print('Y :', Y)

    n = uv.shape[0]
    u = uv[:, 0].reshape((-1, 1))  # use reshape to keep a column vector
    v = uv[:, 1].reshape((-1, 1))  # use reshape to keep a column vector
    tmp1 = np.hstack((u, -v, np.ones((n, 1)), np.zeros((n, 1))))
    tmp2 = np.hstack((v, u, np.zeros((n, 1)), np.ones((n, 1))))
    X = np.vstack((tmp1, tmp2))
    # print('X :', X)

    # We know that X * r = Y
    if rank(X) >= 2 * K:
        r, _, _, _ = lstsq(X, Y, rcond=-1)
        r = np.squeeze(r)
    else:
        raise Exception('cp2tform:twoUniquePointsReq')

    sc = r[0]
    ss = r[1]
    tx = r[2]
    ty = r[3]

    final = np.array([
        [sc, -ss, tx],
        [ss, sc, ty]
    ])

    return final

def FaceAlign(src_lm, target_lm, img):
    similar_trans_matrix = findNonreflectiveSimilarity_for_cv2(src_lm, target_lm)

    # img = cv2.imread(img_path)
    # img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    aligned_face = cv2.warpAffine(img.copy(), similar_trans_matrix, (256, 256), flags=cv2.INTER_LINEAR)
    # aligned_face = cv2.cvtColor(aligned_face, cv2.COLOR_RGB2BGR)
    # cv2.imwrite(save_path, aligned_face)
    return aligned_face

def random_deform(imageSize, nrows, ncols, mean=0, std=5):
    '''
    e.g. where nrows = 6, ncols = 7
    *_______*______*_____*______*______*_________*
    |                                            |
    |                                            |
    *       *      *     *      *      *         *
    |                                            |
    *       *      *     *      *      *         *
    |                                            |
    *       *      *     *      *      *         *
    |                                            |
    *       *      *     *      *      *         *
    |                                            |
    *_______*______*_____*______*______*_________*

    '''
    h, w = imageSize
    rows = np.linspace(0, h-1, nrows).astype(np.int32)
    cols = np.linspace(0, w-1, ncols).astype(np.int32)
    rows, cols = np.meshgrid(rows, cols)
    anchors = np.vstack([rows.flat, cols.flat]).T
    assert anchors.shape[1] == 2 and anchors.shape[0] == ncols * nrows
    deformed = anchors + np.random.normal(mean, std, size=anchors.shape)
    np.clip(deformed[:,0], 0, h-1, deformed[:,0])
    np.clip(deformed[:,1], 0, w-1, deformed[:,1])
    return anchors, deformed.astype(np.int32)


def piecewise_affine_transform(image, srcAnchor, tgtAnchor):
    trans = PiecewiseAffineTransform()
    trans.estimate(srcAnchor, tgtAnchor)
    warped = warp(image, trans)
    return warped

def remove_border(mask):
    mask = mask[0, :, :]

    non_zero_index = torch.argwhere(mask > 0)
    x_ind = non_zero_index[:, 1]
    y_ind = non_zero_index[:, 0]
    right = torch.max(x_ind)
    left = torch.min(x_ind)
    top = torch.min(y_ind)
    bottom = torch.max(y_ind)

    # center_x = (right + left) // 2
    # center_y = (bottom + top) // 2

    return [[left, top], [right, bottom]]

def color_transfer(source, target, clip=True, preserve_paper=True, mask=None, e=1e-8):
	"""
	Transfers the color distribution from the source to the target
	image using the mean and standard deviations of the L*a*b*
	color space.
	This implementation is (loosely) based on to the "Color Transfer
	between Images" paper by Reinhard et al., 2001.
	Parameters:
	-------
	source: NumPy array
		OpenCV image in BGR color space (the source image)
	target: NumPy array
		OpenCV image in BGR color space (the target image)
	clip: Should components of L*a*b* image be scaled by np.clip before 
		converting back to BGR color space?
		If False then components will be min-max scaled appropriately.
		Clipping will keep target image brightness truer to the input.
		Scaling will adjust image brightness to avoid washed out portions
		in the resulting color transfer that can be caused by clipping.
	preserve_paper: Should color transfer strictly follow methodology
		layed out in original paper? The method does not always produce
		aesthetically pleasing results.
		If False then L*a*b* components will scaled using the reciprocal of
		the scaling factor proposed in the paper.  This method seems to produce
		more consistently aesthetically pleasing results 
	Returns:
	-------
	transfer: NumPy array
		OpenCV image (w, h, 3) NumPy array (uint8)
	"""
	# convert the images from the RGB to L*ab* color space, being
	# sure to utilizing the floating point data type (note: OpenCV
	# expects floats to be 32-bit, so use that instead of 64-bit)
	if source is not None:
		source = cv2.cvtColor(source, cv2.COLOR_BGR2LAB).astype("float32")
	target = cv2.cvtColor(target, cv2.COLOR_BGR2LAB).astype("float32")

	# compute color statistics for the source and target images
	(lMeanSrc, lStdSrc, aMeanSrc, aStdSrc, bMeanSrc, bStdSrc) = image_stats(source, mask)
	(lMeanTar, lStdTar, aMeanTar, aStdTar, bMeanTar, bStdTar) = image_stats(target, mask)

	# subtract the means from the target image
	(l, a, b) = cv2.split(target)
	l -= lMeanTar
	a -= aMeanTar
	b -= bMeanTar

	if preserve_paper:
		# scale by the standard deviations using paper proposed factor
		l = (lStdTar / (lStdSrc+e)) * l
		a = (aStdTar / (lStdSrc+e)) * a
		b = (bStdTar / (lStdSrc+e)) * b
	else:
		# scale by the standard deviations using reciprocal of paper proposed factor
		l = (lStdSrc / lStdTar) * l
		a = (aStdSrc / aStdTar) * a
		b = (bStdSrc / bStdTar) * b

	# add in the source mean
	l += lMeanSrc
	a += aMeanSrc
	b += bMeanSrc

	# clip/scale the pixel intensities to [0, 255] if they fall
	# outside this range
	l = _scale_array(l, clip=clip)
	a = _scale_array(a, clip=clip)
	b = _scale_array(b, clip=clip)

	# merge the channels together and convert back to the RGB color
	# space, being sure to utilize the 8-bit unsigned integer data
	# type
	transfer = cv2.merge([l, a, b])
	transfer = cv2.cvtColor(transfer.astype("uint8"), cv2.COLOR_LAB2BGR)
	
	# return the color transferred image
	return transfer

def image_stats(image, mask=None):
	"""
	Parameters:
	-------
	image: NumPy array
		OpenCV image in L*a*b* color space
	Returns:
	-------
	Tuple of mean and standard deviations for the L*, a*, and b*
	channels, respectively
	"""
	# compute the mean and standard deviation of each channel
	(l, a, b) = cv2.split(image)
	if mask is not None:
		# import pdb
		# pdb.set_trace()
		l, a, b = l.reshape(-1), a.reshape(-1), b.reshape(-1)
		mask = mask.reshape(-1)
		l, a, b = l[mask], a[mask], b[mask]
		# import pdb
		# pdb.set_trace()
	(lMean, lStd) = (l.mean(), l.std())
	(aMean, aStd) = (a.mean(), a.std())
	(bMean, bStd) = (b.mean(), b.std())

	# return the color statistics
	return (lMean, lStd, aMean, aStd, bMean, bStd)

def _min_max_scale(arr, new_range=(0, 255)):
	"""
	Perform min-max scaling to a NumPy array
	Parameters:
	-------
	arr: NumPy array to be scaled to [new_min, new_max] range
	new_range: tuple of form (min, max) specifying range of
		transformed array
	Returns:
	-------
	NumPy array that has been scaled to be in
	[new_range[0], new_range[1]] range
	"""
	# get array's current min and max
	mn = arr.min()
	mx = arr.max()

	# check if scaling needs to be done to be in new_range
	if mn < new_range[0] or mx > new_range[1]:
		# perform min-max scaling
		scaled = (new_range[1] - new_range[0]) * (arr - mn) / (mx - mn) + new_range[0]
	else:
		# return array if already in range
		scaled = arr

	return scaled

def _scale_array(arr, clip=True):
	"""
	Trim NumPy array values to be in [0, 255] range with option of
	clipping or scaling.
	Parameters:
	-------
	arr: array to be trimmed to [0, 255] range
	clip: should array be scaled by np.clip? if False then input
		array will be min-max scaled to range
		[max([arr.min(), 0]), min([arr.max(), 255])]
	Returns:
	-------
	NumPy array that has been scaled to be in [0, 255] range
	"""
	if clip:
		scaled = np.clip(arr, 0, 255)
	else:
		scale_range = (max([arr.min(), 0]), min([arr.max(), 255]))
		scaled = _min_max_scale(arr, new_range=scale_range)

	return scaled

def forge(srcRgb, targetRgb, mask):
    # blend_list=[0.92,0.99,0.97,1,1,1,1,1,1,1]
    # blend_ratio = blend_list[np.random.randint(len(blend_list))]
    # mask*=blend_ratio
    res = (mask * targetRgb + (1 - mask) * srcRgb)
    res[res > 255] = 255
    res[res < 0] = 0
    res = res.astype(np.uint8)
    return res

def dynamic_forge(srcRgb, targetRgb, mask):
    blend_list=[0.25,0.5,0.75,1,1,1]
    blend_ratio = blend_list[np.random.randint(len(blend_list))]
    mask*=blend_ratio
    res = (mask * targetRgb + (1 - mask) * srcRgb)
    res[res > 255] = 255
    res[res < 0] = 0
    res = res.astype(np.uint8)
    return res

def RandomReplace(src, tar, src_mask, flag=0):  # 0 easy; 1 simple; 2 hard
    colortrans_list = [2]
    src = cv2.cvtColor(src, cv2.COLOR_RGB2BGR)
    tar = cv2.cvtColor(tar, cv2.COLOR_RGB2BGR)
    # ---------------random deform mask -------------------------
    # src_mask = np.transpose(src_mask, (1, 2, 0))
    warp_list = [4, 5, 6]
    warp_size = random.choice(warp_list)
    anchors, deformedAnchors = random_deform(src_mask.shape[:2], warp_size, warp_size)
    warped_mask = piecewise_affine_transform(src_mask, anchors, deformedAnchors)
    kernel_list = [15, 15, 15, 15, 16, 17, 18]
    kernel = random.choice(kernel_list)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel, kernel))
    anti_warped = 1 - warped_mask
    warped_mask = cv2.dilate(anti_warped, kernel)
    warped_mask = 1 - warped_mask
    warped_mask_numpy = warped_mask.copy()
    kernel_size_list = [15, 15, 15, 15, 11, 13, 9]
    kernel_size = random.choice(kernel_size_list)
    warped_mask_numpy = cv2.GaussianBlur(warped_mask_numpy, (kernel_size,kernel_size), np.random.randint(5,46))
    warped_mask_numpy = warped_mask_numpy[:, :, None]
    warped_mask = torch.tensor(warped_mask, dtype=torch.float32).unsqueeze(0)
    # ---------------color transfer face region -------------------------
    bbox = remove_border(warped_mask)
    x0,y0=bbox[0]
    x1,y1=bbox[1]
    src_face=src[y0:y1,x0:x1, :]
    tar_face=tar[y0:y1,x0:x1, :]
    if flag in colortrans_list:
        tar_face = color_transfer(src_face, tar_face)
    tar[y0:y1,x0:x1, :] = tar_face
    #-----------------replace--------------------------------------
    fake = forge(src, tar, warped_mask_numpy)
    # else:
    #     x0, x1, y0, y1 = x0.numpy(), x1.numpy(), y0.numpy(), y1.numpy()
    #     center_x = (x0 + x1) // 2
    #     center_y = (y0 + y1) // 2
    #     center = [int(center_x), int(center_y)]
    #     tp = warped_mask_numpy.copy()
    #     tp = tp[y0:y1,x0:x1, :]
    #     fake = cv2.seamlessClone(tar_face, src, np.uint8(tp*255), center, cv2.MONOCHROME_TRANSFER)
    fake = cv2.cvtColor(fake, cv2.COLOR_BGR2RGB)
   
    return fake, warped_mask_numpy

class VideoCDF(data.Dataset):
    def __init__(self,image_size=384) -> None:
        super().__init__()
        idx_root = '/path/Dataset/VideoCDF/idx'
        txt_root = '/path/Dataset/VideoCDF/all.txt'
        allData = np.loadtxt(txt_root, dtype='str', delimiter='\t')

        self.video_list = []
        self.target_list = []
        self.idx_list = []
        
        self.totensor = [
        tr.ToTensor(),
        tr.Resize((image_size, image_size),antialias=True)
        ]
        self.totensor = tr.Compose(self.totensor)
        
        for _root, _, files in os.walk(idx_root):
            if files:
                for file in files:
                    idx_path = os.path.join(_root, file)
                    self.idx_list.append(idx_path)
        self.idx_list = list(sorted(self.idx_list))

        allData = list(sorted(allData, key=lambda x:x[0]))
    
        
        for i in range(len(self.idx_list)):
            video_path = self.idx_list[i].replace("/idx", "").replace(".npy", "")
            self.video_list.append(video_path)
        for j in range(len(self.video_list)):
            for i in range(len(allData)):
                if self.video_list[j] == allData[i][0]:
                    self.target_list.append(int(allData[i][1]))
                    break
        self.length = len(self.video_list)
        
    def __len__(self):
        return self.length
    
    def __getitem__(self, index):
        video_path = self.video_list[index]
        target = self.target_list[index]
        idx_path = self.idx_list[index]
        
        video = None
        for _root, _, files in os.walk(video_path):
            if files:
                for file in files:
                    # file = file.decode()
                    img_path = os.path.join(_root, file)
                    img = Image.open(img_path)
                    img = self.totensor(img)
                    img = img.unsqueeze(0)
                    video = img if video is None else torch.cat((video, img), dim=0)
        if video is None:
            print(video_path)
                    
        return video, target, idx_path
                    

    
    def collate_fn(self, batch):
        video, target, idx_path = zip(*batch)
        video = video[0]
        idx_path = idx_path[0]
        target = torch.tensor(target).unsqueeze(0)
        return video, target, idx_path
            
    

import albumentations as alb
class RandomDownScale(alb.core.transforms_interface.ImageOnlyTransform):
	def apply(self,img,**params):
		return self.randomdownscale(img)

	def randomdownscale(self,img):
		keep_ratio=True
		keep_input_shape=True
		H,W,C=img.shape
		ratio_list=[2,4]
		r=ratio_list[np.random.randint(len(ratio_list))]
		img_ds=cv2.resize(img,(int(W/r),int(H/r)),interpolation=cv2.INTER_NEAREST)
		if keep_input_shape:
			img_ds=cv2.resize(img_ds,(W,H),interpolation=cv2.INTER_LINEAR)

		return img_ds


class VideoDFDC(data.Dataset):
    def __init__(self,image_size = 384) -> None:
        super().__init__()
        idx_root = '/path/Dataset/VideoDFDC/idx'
        txt_root = '/path/Dataset/VideoDFDC/all.txt'
        allData = np.loadtxt(txt_root, dtype='str', delimiter='\t')
        
        self.video_list = []
        self.target_list = []
        self.idx_list = []
        
        self.totensor = [
        tr.ToTensor(),
        tr.Resize((image_size, image_size),antialias=True)
        ]
        self.totensor = tr.Compose(self.totensor)
        
        for _root, _, files in os.walk(idx_root):
            if files:
                for file in files:
                    idx_path = os.path.join(_root, file)
                    self.idx_list.append(idx_path)
        self.idx_list = list(sorted(self.idx_list))

        allData = list(sorted(allData, key=lambda x:x[0]))
        
        for i in range(len(self.idx_list)):
            video_path = self.idx_list[i].replace("/idx", "").replace(".npy", "")
            self.video_list.append(video_path)
        for j in range(len(self.video_list)):
            for i in range(len(allData)):
                if self.video_list[j] == allData[i][0]:
                    self.target_list.append(int(allData[i][1]))
                    break
        self.length = len(self.video_list)
        
        
        
    def __len__(self):
        return self.length
    
    def __getitem__(self, index):
        video_path = self.video_list[index]
        target = self.target_list[index]
        idx_path = self.idx_list[index]
        
        video = None
        for _root, _, files in os.walk(video_path):
            if files:
                for file in files:
                    # file = file.decode()
                    img_path = os.path.join(_root, file)
                    img = Image.open(img_path)
                    img = self.totensor(img)
                    img = img.unsqueeze(0)
                    video = img if video is None else torch.cat((video, img), dim=0)
        if video is None:
            print(video_path)
                    
        return video, target, idx_path
    
    def collate_fn(self, batch):
        video, target, idx_path = zip(*batch)
        video = video[0]
        idx_path = idx_path[0]
        target = torch.tensor(target).unsqueeze(0)
        return video, target, idx_path

class VideoDFDCP(data.Dataset):
    def __init__(self,image_size=384) -> None:
        super().__init__()
        root = '/path/Dataset/VideoDFDCP'
        txt_root = '/path/Dataset/VideoDFDCP/all.txt'
        allData = np.loadtxt(txt_root, dtype='str', delimiter='\t')
        self.video_list = []
        self.target_list = []
        
        self.totensor = [
        tr.ToTensor(),
        tr.Resize((image_size, image_size),antialias=True)
        ]
        self.totensor = tr.Compose(self.totensor)
        
        for i in range(len(allData)):
            self.video_list.append(allData[i, 0])
            
            if allData[i, 1] == 'True':
                label = 1
            elif allData[i, 1] == 'False':
                label = 0
            self.target_list.append(label)
        self.length = len(self.video_list)

    def __len__(self):
        return self.length
    
    def __getitem__(self, index):
        video_path = self.video_list[index]
        target = self.target_list[index]
        
        video = None
        for _root, _, files in os.walk(video_path):
            if files:
                for file in files:
                    file = file.decode()
                    img_path = os.path.join(_root, file)
                    img = Image.open(img_path)
                    img = self.totensor(img)
                    img = img.unsqueeze(0)
                    video = img if video is None else torch.cat((video, img), dim=0)
        return video, target
    
    def collate_fn(self, batch):
        video, target = zip(*batch)
        video = video[0]
        target = torch.tensor(target).unsqueeze(0)
        return video, target

class VideoDFV1(data.Dataset):
    def __init__(self,image_size = 384) -> None:
        super().__init__()
        root = '/path/Dataset/VideoDFV1'
        txt_root = '/path/Dataset/VideoDFV1/all.txt'
        allData = np.loadtxt(txt_root, dtype='str', delimiter='\t')
        self.video_list = []
        self.target_list = []
        
        self.totensor = [
        tr.ToTensor(),
        tr.Resize((image_size, image_size),antialias=True)
        ]
        self.totensor = tr.Compose(self.totensor)
        
        for i in range(len(allData)):
            self.video_list.append(allData[i, 0])
            self.target_list.append(int(allData[i, 1]))
        self.length = len(self.video_list)

    def __len__(self):
        return self.length
    
    def __getitem__(self, index):
        video_path = self.video_list[index]
        target = self.target_list[index]
        
        video = None
        for _root, _, files in os.walk(video_path):
            if files:
                for file in files:
                    file = file.decode()
                    img_path = os.path.join(_root, file)
                    img = Image.open(img_path)
                    img = self.totensor(img)
                    img = img.unsqueeze(0)
                    video = img if video is None else torch.cat((video, img), dim=0)
                    
        return video, target
    
    def collate_fn(self, batch):
        video, target = zip(*batch)
        video = video[0]
        target = torch.tensor(target).unsqueeze(0)
        return video, target

class VideoDFD(data.Dataset):
    def __init__(self, image_size = 384) -> None:
        super().__init__()
        root = '/path/Dataset/VideoDFD'
        txt_root = '/path/Dataset/VideoDFD/all.txt'
        allData = np.loadtxt(txt_root, dtype='str', delimiter='\t')
        
        self.video_list = []
        self.target_list = []
        
        self.totensor = [
        tr.ToTensor(),
        tr.Resize((image_size, image_size),antialias=True)
        ]
        self.totensor = tr.Compose(self.totensor)
        
        for i in range(len(allData)):
            self.video_list.append(allData[i, 0])

            self.target_list.append(int(allData[i, 1]))
        self.length = len(self.video_list)
        
    def __len__(self):
        return self.length
    
    def __getitem__(self, index):
        video_path = self.video_list[index]
        target = self.target_list[index]
        
        video = None
        for _root, _, files in os.walk(video_path):
            if files:
                for file in files:
                    file = file.decode()
                    img_path = os.path.join(_root, file)
                    img = Image.open(img_path)
                    img = self.totensor(img)
                    img = img.unsqueeze(0)
                    video = img if video is None else torch.cat((video, img), dim=0)
                    
        return video, target
    
    def collate_fn(self, batch):
        video, target = zip(*batch)
        video = video[0]
        target = torch.tensor(target).unsqueeze(0)
        return video, target

def Get_DataLoader(dataset_name="forgery_detect", root='/path/forgery_detect', mode='train', image_size = 384, bz=1,
                   shuffle=False):
    if dataset_name =="VideoCDF":
        dataset = VideoCDF( image_size = image_size)
        return data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=8, collate_fn=dataset.collate_fn)
    elif dataset_name =="VideoDFD":
        dataset = VideoDFD( image_size = image_size)
        return data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=10, collate_fn=dataset.collate_fn)
    elif dataset_name =="VideoDFDC":
        dataset = VideoDFDC( image_size = image_size)
        return data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=10, collate_fn=dataset.collate_fn)
    elif dataset_name =="VideoDFDCP":
        dataset = VideoDFDCP( image_size = image_size)
        return data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=10, collate_fn=dataset.collate_fn)
    elif dataset_name =="VideoDFV1":
        dataset = VideoDFV1( image_size = image_size)
        return data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=10, collate_fn=dataset.collate_fn)
    else:
        raise NotImplementedError("No this kind of dataset!")


if __name__ == '__main__':
    TrainSet = Get_DataLoader(dataset_name="VideoDFDC",
                              mode='train',
                              bz=32,
                              shuffle=True)
    print(len(TrainSet))
    for step_id, datas in enumerate(TrainSet):
        print(step_id)
