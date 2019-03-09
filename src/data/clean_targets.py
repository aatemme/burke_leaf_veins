# -*- coding: utf-8 -*-
from os.path import join, basename, exists
import multiprocessing as mp
import glob
import logging
from pathlib import Path
from utils import clean_target
from tqdm import tqdm
from skimage import io

def main(project_dir):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('cleaning target images')

    skel_paths = join(project_dir,
                    'data','processed',
                    'veins',
                    'target',
                    '*.png')

    result_path = join(project_dir,'data','processed','veins','cleaned_targets')
    mkdir(result_path)

    for traced_image in tqdm(glob.glob(skel_paths)):
        try:
            image = io.imread(traced_image)
            cleaned = clean_target(image)
            io.imsave(
                join(
                    result_path,
                    basename(traced_image).split()[0] + ".tiff"
                ),
                cleaned
            )
        except Exception as error:
            logger.warning('Image %s failed with message %s'%(base,error))

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    main(project_dir)
