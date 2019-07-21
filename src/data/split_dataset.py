# -*- coding: utf-8 -*-
'''
    Splits the data set created by generate_dataset (files in data/processed/all/)
    into training and validation sets. 

    The validation set is created by randomlly selecting 1 image from 
    each genotype.

    The training set comprises of all images no in the validation set

    The result are saved in data/processed/valiation and data/processed/training
'''
RANDOM_SEED = 1448
import random
random.seed(RANDOM_SEED)

from os.path import join, basename, exists
from shutil import copyfile
import glob
import logging
from pathlib import Path
from utils import mkdir
from tqdm import tqdm

logger = logging.getLogger(__name__)
logger.info('Splitting veins into training and validation sets')

def move_files(image_names,from_loc,to_loc):
    """
    Copy training images listed in image_names (real and target)  
    from the from_loc to the to_loc

    Args:
        image_name (list of str): Name of images to copy, these can 
            be of either .jpeg (real images) or .png (targets) extensions
        from_loc (str): The directory which contains the "real" and
            "target" folders from which the images will be copied from
        to_loc (str): The diretory in which new "real" and "target"
            folders will be created and the imates in image_names copied.
    """
    target_to_path = join(to_loc,'target')
    real_to_path = join(to_loc,'real')
    mkdir(target_to_path)
    mkdir(real_to_path)

    target_from_path = join(from_loc,'target')
    real_from_path = join(from_loc,'real')
    
    with open(join(to_loc,'file_list.csv'),'w') as f_out:
        for item in image_names:
            f_out.write("%s, " % item)

    for image in tqdm(image_names):
        base = image.split('.')[0]
        real_image = base + ".jpeg"
        target_image = base + ".png"
       
        copyfile(join(real_from_path, real_image),
                 join(real_to_path, real_image))
        copyfile(join(target_from_path, target_image),
                 join(target_to_path, target_image))

def training_validation_split(project_dir, size=100):
    """
     Split images into validation and training sets
    
     Args:
        project_dir (str): absolute path to the project 
            base directory
        size (int): Number of genotypes to choose images
            from for the validaiton set. Also the final 
            size of the validation set (default: 100)i

    Returns:
        Tuple (valdation_set, test_set) containing the list of image filenames in the 
        validation and test sets. 
    """
    
    skel_paths = join(project_dir,
                    'data','processed',
                    'all',
                    'target',
                    '*.png')
    
    image_paths = glob.glob(skel_paths)
    images = [basename(name) for name in image_paths]

    genotypes = {}
    plants = images.copy()
    plants.sort()
    while len(plants) > 0:
        genotype = plants[0].split('-')[0]
        replicates = []
        while len(plants) and plants[0].split('-')[0] == genotype:
            replicate = plants[0].split('-')[1].split('.')[0]
            replicates.append(replicate)
            del plants[0]

        genotypes[genotype] = replicates

    keys = genotypes.keys()
    testing_genotypes = random.sample(list(keys),k=100)

    validation_set = []
    for genotype in testing_genotypes:
        replicate = random.choice(genotypes[genotype])
        validation_set.append("%s-%s.png"%(genotype,replicate))

    training_set = [s for s in images if s not in validation_set]
    return (validation_set, training_set)

def main(project_dir):
    val_set, train_set = training_validation_split(project_dir)

    logger.info("Random Seed: %d"%(RANDOM_SEED, ))
    logger.info("%d images selected for validation set" % (len(val_set),))
    logger.info("%d images left for training set" % (len(train_set),))
           
    print("Copying validation images:")
    val_dir_path = join(project_dir,'data','processed','validation')
    mkdir(val_dir_path)
    move_files(val_set,
               join(project_dir,'data','processed','all'),
               val_dir_path)

    print("Copying training images:")
    train_dir_path = join(project_dir,'data','processed','training')
    mkdir(train_dir_path)
    move_files(train_set,
               join(project_dir,'data','processed','all'),
               train_dir_path)

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    main(project_dir)
