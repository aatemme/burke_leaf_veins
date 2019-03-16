from os.path import join, basename
import torch
import torch.utils.data
import random as rand
import numpy as np
import json
import glob
from skimage import io, util

from albumentations import (
    Compose,
    Cutout,
    ChannelShuffle,
    HorizontalFlip,
    VerticalFlip,
    RandomRotate90,
    Transpose,
    ElasticTransform,
    GaussianBlur,
    NoOp
)

# aug_all = iaa.Sequential([
#     iaa.Fliplr(0.5), # horizontally flip 50% of the images
#     iaa.Flipud(0.5), # vertically flip 50% of the images
# ])

def normalize(img):
    '''
        Normalize an image to mean of 0 and std of 1

        Returns:
            The normalized image
    '''
    mean = torch.mean(img[:])
    std = torch.std(img[:])
    img = (img - mean)/std
    return img

def crop(img,top,size):
    '''
        Crop img to size size, with the top left most pixel of top

        Arguments:
            img (C x X x Y array): image of size C channels by X by Y
            top (tuple (x,y)): Top left most pixel to include in the cropped image
            size (tuple (x,y)): Size of cropped image
    '''
    xstop = top[0]+size[0]
    ystop = top[1]+size[1]

    if(xstop >= img.shape[0]):
        raise ValueError('x dim of image too small for crop')
    if(ystop >= img.shape[1]):
        raise ValueError('y dim of image too small for crop')

    return img[top[0]:xstop,top[1]:ystop]

class PairedImages(torch.utils.data.Dataset):
    """
    """
    def __init__(self,root,augment = True):
        self.root = root
        self.images = [basename(x) for x in glob.glob(join(root,'target',"*.png"))]
        self.length = len(self.images)

        if augment:
            self.aug = Compose([
                HorizontalFlip(),
                VerticalFlip(),
                Transpose(),
                RandomRotate90(),
                ElasticTransform(),
                Cutout(num_holes=10,max_h_size=50,max_w_size=50),
                ChannelShuffle(),
                GaussianBlur()
            ])
        else:
            self.aug = NoOp()

    def __getitem__(self, index):
        img = self.images[index]
        real = io.imread(join(self.root,'real',img.split('.')[0]+".jpeg"))
        target = io.imread(join(self.root,'target',img))

        #functions applied to numpy arrays
        upper_left = (rand.randint(0,real.shape[0]-1 - 388),
                      rand.randint(0,real.shape[1]-1 - 388))

        pad_width = (
            (92,92), # x
            (92,92),  # y
            (0,0),   # C channel
        )
        real = util.pad(real,pad_width,'reflect')
        real = crop(real,upper_left,(572,572))
        target = crop(target,upper_left,(388,388))

        augmented = aug(image = real, mask = target)

        #Functions applied to torch tensors
        real = normalize(torch.from_numpy(augmented['image']).permute(2,0,1).float())
        target = torch.from_numpy(augmented['mask']).float() / 255

        return real, target

    def __len__(self):
        return self.length
