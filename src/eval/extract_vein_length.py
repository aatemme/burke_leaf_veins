'''
    Extracts vein length in pixels from vein segmentation masks
    created by neural network.

    Psudocode:
        * extra medial axis from vein segmentation mask
        * Vein length (in pixels) = number of pixels in medial axis.

    See notebooks/ComareToData.ipynb for exploration of this process.
'''
import dataset
from os.path import abspath, join, basename
import click
from tqdm import tqdm
from skimage import io
from skimage.morphology import medial_axis
import numpy as np

@click.command()
@click.argument('database_conn')
@click.argument('table_name')
@click.argument('masks', required=True, nargs=-1)
def main(masks,database_conn,table_name):
    db = dataset.connect(database_conn)
    table = db[table_name]
    table.delete()

    for image in tqdm(masks):
        filename = basename(image)

        plant = filename.split('-')[0]
        replicate = filename.split('-')[1][0]

        probs = io.imread(image)
        seg = probs/2**16 > 0.4
        med = medial_axis(seg)
        length = np.sum(med)
        table.insert(dict(plant=plant, replicate=replicate, length=int(length)))

if __name__ == '__main__':
    main()
