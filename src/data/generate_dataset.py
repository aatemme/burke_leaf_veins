# -*- coding: utf-8 -*-
'''
    Creates the file data set to train on. This pulls
    the segmentation masks from data/interim/veins/target and the
    real images from data/raw/Veins machine learning with Chris/Untraced Veins.

    Only real images with segmentation masks are copied, and the real image
    file extension is normalized to '.jpeg'

    The result are saved in data/processed/target and data/processed/real
'''
from os.path import join, basename, exists
from shutil import copyfile
import glob
import logging
from pathlib import Path
from utils import mkdir
from tqdm import tqdm

def move_files(paths,raw_path,loc):

    target_path = join(loc,'target')
    real_path = join(loc,'real')
    mkdir(target_path)
    mkdir(real_path)

    for traced_image in tqdm(paths):
        base = basename(traced_image).split('_')[0]

        real_raw_path = join(raw_path,base + '.jpeg')
        if not exists(real_raw_path):
            real_raw_path = join(raw_path,base + '.jpg')

        copyfile(real_raw_path,join(real_path,base + '.jpeg'))
        copyfile(traced_image, join(target_path,base + '.png'))

def main(project_dir):
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw and interim data')

    skel_paths = join(project_dir,
                    'data','interim',
                    'veins',
                    'target',
                    '*.png')

    raw_path = join(project_dir,
                    'data','raw',
                    'Veins machine learning with Chris',
                    'Untraced Veins')

    result_path = join(project_dir,'data','processed','all')

    image_paths = glob.glob(skel_paths)

    move_files(image_paths,raw_path,result_path)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    main(project_dir)
