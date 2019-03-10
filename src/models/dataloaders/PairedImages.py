from os.path import join, basename
import torch
import torch.utils.data
import random as rand
import numpy as np
import json
import glob
from skimage.external import tifffile

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


class PairedImages(torch.utils.data.Dataset):
    """
    """
    def __init__(self,root):
        self.root = root
        self.images = [basename(x) for x in glob.glob(join(root,'real',"*.tiff"))]
        self.length = len(self.images)

    def __getitem__(self, index):
        img = self.images[index]
        real = tifffile.imread(join(self.root,'real',img))
        target = tifffile.imread(join(self.root,'target',img))

        real = torch.from_numpy(real).permute(2,0,1).float()
        real = normalize(real)
        
        target = torch.from_numpy(target).float() / 255

        return real, target

    def __len__(self):
        return self.length
