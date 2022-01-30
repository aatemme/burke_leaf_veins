import math
import click
import sys
from os.path import abspath, join, basename, exists
from pathlib import Path
from tqdm import tqdm

import numpy as np
from skimage.morphology import medial_axis
from skimage import io

@click.command()
@click.argument('medial_axis_folder')
@click.argument('raw_images', required=True, nargs=-1)
def main(raw_images, medial_axis_folder):
    '''
        Loads the given medial axis images and overlays them on
        the raw images.

        Resulting images are saved in the same folder as the medial axis images.
    '''

    for image_path in tqdm(raw_images):
        raw_img = io.imread(image_path)

        image_id = basename(image_path).split(".")[0]

        medaxis_path = join(medial_axis_folder,image_id + "_probs.png")
        if not exists(medaxis_path):
            print("%s does not exist"%(medaxis_path,))
            continue

        medaxis = io.imread(medaxis_path) / 2**16

        seg = medial_axis(medaxis > 0.1)
        raw_img[seg,1] = 2**16

        io.imsave(
            join(medial_axis_folder,
                 basename(image_path).split(".")[0] + "_medialAxisOverlay.png"),
            raw_img
        )

if __name__ == '__main__':
    main()
