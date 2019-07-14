# -*- coding: utf-8 -*-
'''
    Makes images binary and removes small gaps between lines that should be
    connected. 
'''
from os.path import join, basename, exists
import csv
import multiprocessing as mp
import glob
import logging
from pathlib import Path
from utils import mkdir
from tqdm import tqdm

import numpy as np
from skimage import io
from skimage.external import tifffile
import skimage.morphology as mph
from skimage.util import pad


def main(project_dir):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('Cleaning Targets.')

    skel_paths = join(project_dir,
                    'data','raw',
                    'Veins machine learning with Chris',
                    'Veins skeleton',
                    '*.jpg')

    result_path = join(project_dir,'data','interim','veins')
    logger.info('Cleaned targets will be placed in %s/target'%(result_path))

    mkdir(result_path)
    mkdir(join(result_path,'target'))

    #Remove blacklisted images
    blacklist_file = join(project_dir,
                          'data','raw',
                          'Veins machine learning with Chris','blacklist.csv')
    image_list = glob.glob(skel_paths)
    if(exists(blacklist_file)):
        #Remove files in blacklist
        logger.info("Blacklist file found")
        with open(blacklist_file,'r') as fin:
            blacklist = list(csv.reader(fin))[0]
        logger.info("Images %s found in blacklist file, will be skipped."%(blacklist))
        basenames = [basename(b).split("_")[0] for b in image_list]
        image_list = [image_list[i] for (i,p) in enumerate(basenames) if p not in blacklist]

    for traced_image in tqdm(image_list):
        base = basename(traced_image)

        image = io.imread(traced_image)
        seg = image > 200
        seg = mph.remove_small_objects(seg,500)
        seg = mph.binary_closing(pad(seg,(20,20),mode='constant'), mph.disk(15))
        seg = seg[20:-20,20:-20]

        seg = seg.astype('uint8') * 255

        if np.sum(seg) < 60000:
            logger.info('Image %s has very few veins, please visually check' % (base))

        output_filename = "%s.png"%(base.split('.')[0])
        io.imsave(join(
                          result_path,
                          'target',
                          output_filename
                        ),
                        seg,
                        check_contrast = False
                       )

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    main(project_dir)
