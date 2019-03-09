import os
from os import path
import shutil
import numpy as np
from skimage import util
from skimage import io
from skimage.external import tifffile
import skimage.morphology as mph

def mkdir(dir):
    if path.exists(dir):
        shutil.rmtree(dir)
    os.makedirs(dir)

def clean_target(target, disk_size = 10):
    '''
        Creates a binary target, removes objects smaller than 100 px^2,
        and trys to connected lines.
    '''
    closed_image = mph.remove_small_objects(target > 0.5, min_size = 100)
    closed_image = mph.closing(
                        util.pad(closed_image,disk_size,'constant'),
                        selem=mph.disk(disk_size))
    return closed_image[disk_size:-disk_size,disk_size:-disk_size]

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

def extract_tiles(real,target,step=42,real_size=572,target_size=388):
    '''
        Extract tiled images from real and target

        returns: a generator yielding each paired cropped (real,target) tile
        extracted from the input images.
    '''
    real_x_starts = np.arange(0,real.shape[0]-real_size,step)
    target_x_starts = np.arange(0,target.shape[0]-target_size,step)

    real_y_starts = np.arange(0,real.shape[1]-real_size,step)
    target_y_starts = np.arange(0,target.shape[1]-target_size,step)

    for real_x,target_x in zip(real_x_starts,target_x_starts):
        for real_y,target_y in zip(real_y_starts,target_y_starts):
            # print("Real: %s,%s"%(real_x,real_y))
            # print("Target: %s,%s"%(target_x,target_y))
            real_cropped = crop(real,
                                (real_x,real_y),
                                (real_size,real_size))
            target_cropped = crop(target,
                                  (target_x,target_y),
                                  (target_size,target_size))
            yield (real_cropped,target_cropped)

def process_image(real_path,target_path,results_path):
    '''
        Args:
            real_path (str): path to real image
            target_path (str): path to target
            results_path (str): folder to save process images in
    '''
    real = io.imread(real_path)
    target = io.imread(target_path)

    pad_width = (
        (92,92), # x
        (92,92),  # y
        (0,0),   # C channel
    )
    real = util.pad(real,pad_width,'reflect')

    for (i,img) in enumerate(extract_tiles(real,target,step=100)):
        t, r = img
        output_filename = "%s_%s.tiff"%(path.basename(real_path).split('.')[0],i)
        tifffile.imsave(path.join(
                          results_path,
                          'real',
                          output_filename
                        ),
                        t,
                       )
        tifffile.imsave(path.join(
                          results_path,
                          'target',
                          output_filename
                        ),
                        r,
                       )
