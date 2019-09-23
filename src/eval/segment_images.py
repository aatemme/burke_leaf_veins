import math
import click
import sys
from os.path import abspath, join, basename
from pathlib import Path
from tqdm import tqdm

import torch
from skimage import io
from skimage.util import pad
from skimage.transform import rescale

project_dir = Path(__file__).resolve().parents[2]
sys.path.append(join(project_dir,'src','models','dataloaders'))
from PairedImages import normalize, cv_split

def pad_val(size):
    """
        Calculates the amount of padding needed for a dimension to
        be a multiple of 572 * N + (4 * (N-1))

        Includes the 90 pixels of padding beyond that segemetned.
    """
    size = size + 180
    N = math.ceil(size / 572)
    total_padding = 572 * N + (4 * (N-1)) - size
    left_pad = math.floor(total_padding / 2)
    right_pad = left_pad + (total_padding % 2)
    return left_pad + 90,right_pad + 90

def real_indices(y_shape,overlay_shape):
    """
        Returns the array indexing start and stop to get get the real
        image size out of the padded segmentation
    """
    padding = (y_shape - overlay_shape)
    left_pad = math.floor(padding / 2)
    right_pad = math.floor(padding / 2) + padding % 2
    return left_pad, -right_pad

def segment(net,image):
    image_shape = image.shape

    x_pad = pad_val(image.shape[0])
    y_pad = pad_val(image.shape[1])

    pad_width = (
        x_pad, # x
        y_pad,  # y
        (0,0),   # C channel
    )
    image = pad(image,pad_width,'reflect')

    with torch.no_grad():
        image = torch.from_numpy(image).permute(2,0,1).float()
        image = normalize(image)
        y = net(image.unsqueeze(0))

    x_start, x_stop = real_indices(y.shape[2],image_shape[0])
    y_start, y_stop = real_indices(y.shape[3], image_shape[1])
    y = y[0,0,x_start:x_stop,y_start:y_stop].numpy()

    return y

def load_state(nets,state):
    '''
        Loads saved model weights into the given networks
    '''
    for key,val in nets.items():
        state_dict = torch.load(state + '/' + key + '.net', map_location=lambda storage, loc: storage)
        val.load_state_dict(state_dict)

@click.command()
@click.argument('model')
@click.argument('state')
@click.argument('results_folder')
@click.argument('images', required=True, nargs=-1)
@click.option('--threshold', type=float, default=0.4)
@click.option('--fold_cv', type=int, default=None)
@click.option('--fold', type=int, default=None)
def main(model,state,images,results_folder, threshold, fold_cv, fold):
    '''
        Loads raw images and segments them using the given network and trained weights (state).

        Saves the segmentation probabilities as P * 2^16 as a uint16.
        Saves the segmentation mask overplayed on the image for debugging.
    '''
    sys.path.append(abspath(model))
    from models import UNet

    net = UNet()
    load_state({'Net': net},state)
    net = net.eval()

    if fold:
        assert fold_cv is not None, "fold and fold_cv must be set togeather"

    if fold_cv:
        assert fold is not None, "fold and fold_cv must be set togeather"
        _, image_list = cv_split(images,fold_cv,fold)
    else:
        image_list = images

    for image_path in tqdm(image_list):
        image = io.imread(image_path)

        y = segment(net,image)

        io.imsave(
            join(results_folder,
                 basename(image_path).split(".")[0] + "_probs.png"),
            (y * 2**16).astype('uint16')
        )

        seg = y > threshold
        image[seg,1] = 1


        io.imsave(
            join(results_folder,
                 basename(image_path).split(".")[0] + "_overlay.png"),
            image
        )

if __name__ == '__main__':
    main()
