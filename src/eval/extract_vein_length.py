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
@click.argument('threshold', type=float)
@click.argument('masks', required=True, nargs=-1)
def main(masks,database_conn,table_name, threshold):
    '''
        Extracts vein length in pixels and placeses
        the results in an sql table.

        Table has the structure:
            plant (str): first part of file name
            replicate (str): second part of file name
            length (int): total vein length

        Args:
            masks (list of strings): list of image file paths to extract
                vein length from. These should be the probability values
                created by the NN model.
            table_name (str): name of sql table
            database_conn (str): any database connection supported by
                sqlalchemy (e.g. sqlite:///results.sqlite for writing to a file)
            threshold (float): mask segmentation threshold between 0 and 1.
    '''
    db = dataset.connect(database_conn)
    table = db[table_name]
    table.delete()

    for image in tqdm(masks):
        filename = basename(image)

        plant = filename.split('-')[0]
        replicate = filename.split('-')[1][0]

        probs = io.imread(image)
        seg = probs/2**16 > threshold
        med = medial_axis(seg)
        length = np.sum(med)
        table.insert(dict(plant=plant, replicate=replicate, length=int(length)))

if __name__ == '__main__':
    main()
