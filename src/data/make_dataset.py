# -*- coding: utf-8 -*-
from os.path import join, basename, exists
import multiprocessing as mp
import glob
import click
import logging
import csv
from pathlib import Path
from utils import process_image, mkdir
from tqdm import tqdm

def main(project_dir):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')

    skel_paths = join(project_dir,
                    'data','raw',
                    'Veins machine learning with Chris',
                    'Veins skeleton',
                    '*.jpg')
    raw_path = join(project_dir,
                    'data','raw',
                    'Veins machine learning with Chris',
                    'Untraced Veins')

    result_path = join(project_dir,'data','processed','veins')

    mkdir(result_path)
    mkdir(join(result_path,'real'))
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

        real_path = join(raw_path,base.split('_')[0] + '.jpeg')
        if not exists(real_path):
            real_path = join(raw_path,base.split('_')[0] + '.jpg')
        target_path = traced_image
        #logger.info('processing image %s'%(base))
        try:
            process_image(real_path,target_path,result_path)
        except Exception as error:
            logger.warning('Image %s failed with message %s'%(base,error))

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    main(project_dir)
