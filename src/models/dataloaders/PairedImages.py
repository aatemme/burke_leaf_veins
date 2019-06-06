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
    Blur,
    ChannelShuffle,
    HorizontalFlip,
    VerticalFlip,
    RandomRotate90,
    Transpose,
    ElasticTransform,
    NoOp
)

# aug_all = iaa.Sequential([
#     iaa.Fliplr(0.5), # horizontally flip 50% of the images
#     iaa.Flipud(0.5), # vertically flip 50% of the images
# ])

def cv_split(data,n_way,n):
    '''
        Split data into N-way cross validation sets. Returns the data split
        into test and train data for the n^th cross-validation split

        The the results are deterministic, given the order of data does not
        change from call to call.

        Args:
            data (list): data to split
            n_way (int): cross validation split in the range of [0,n_way).
            n (int): The cross validation split to return

        Returns:
            (train_data, test_data)
    '''
    n = int(n)
    n_way = int(n_way)

    if(n >= n_way or n < 0):
        raise ValueError("Split must be in the range [0,n_way).")

    N = len(data)

    bin_size = int(N/n_way)
    remainder = N % n_way

    #The min(_,_) adds 1 point to each bin until the remainder is used up
    test_start = bin_size * (n) + min(remainder,n)
    test_stop  = bin_size * (n + 1) + min(remainder,n+1)

    train_data = data[:test_start] + data[test_stop:]
    test_data =  data[test_start:test_stop]

    return (train_data, test_data)

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
    def __init__(self,root,augment = True,cv=None,n_split=None,cv_test=False):
        '''
            Args:
                root (str): path to the folder containing the target and real
                    folders, which contain the masks and real images.
                augment (bool): Augment the images during training
                cv (int): Split the data for N-way cross validation, where
                    cv=N. Used in conjunction with n_split.
                    (default=None, use all data during training)
                n_split (int): Train on the n^th n-way corss validation set,
                    where n_split=n. Used in conjunction with cv
                    (default=None, use all data during training)
                cv_test (bool): if True, return the test data set, otherwise
                    return the train dataset for the n^th cross validation set.
        '''
        self.root = root
        self.images = [basename(x) for x in glob.glob(join(root,'target',"*.png"))]

        if cv is not None and n_split is not None:
            train, test = cv_split(self.images,cv,n_split)
            if cv_test:
                self.images = test
            else:
                self.images = train

        self.length = len(self.images)

        if augment:
            self.aug = Compose([
                HorizontalFlip(),
                VerticalFlip(),
                Transpose(),
                RandomRotate90(),
                Blur(blur_limit=25),
                Cutout(num_holes=7,max_h_size=100,max_w_size=100),
                ChannelShuffle()
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

        augmented = self.aug(image = real, mask = target)

        #Functions applied to torch tensors
        real = normalize(torch.from_numpy(augmented['image']).permute(2,0,1).float())
        target = torch.from_numpy(augmented['mask']).float() / 255

        return real, target

    def __len__(self):
        return self.length
