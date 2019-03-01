import pytest
import numpy as np

def test_crop_output_dim():
    from utils import crop

    img = np.ones((3,2584,1936))

    output = crop(img,(22,22),(100,100))

    assert output.shape == (3, 100, 100)

def test_crop_edge():
    from utils import crop

    img = np.ones((3,300,300))

    output = crop(img,(199,199),(100,100))

    assert output.shape == (3, 100, 100)

def test_crop_out_of_bounds():
    from utils import crop

    img = np.ones((3,300,300))

    with pytest.raises(ValueError):
        crop(img,(200,200),(100,100))


def test_extract_tiles_output_dim():
    from utils import extract_tiles

    img_width = 2584
    img_height = 1936
    padding = 184

    target = np.ones((3,img_width,img_height))
    real = np.ones((3,img_width + padding,img_height + padding))

    Ntiles = 0
    for (r,t) in extract_tiles(real,target):
        assert r.shape == (3,572,572)
        assert t.shape == (3,388,388)
        Ntiles = Ntiles + 1

    total_tiles = np.ceil((img_width-388) / 42) * np.ceil((img_height-388) / 42)
    assert Ntiles == total_tiles
