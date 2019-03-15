import torch
from dataloaders.PairedImages import normalize
from dataloaders import PairedImages
import pytest

def test_normalize():
    image = torch.FloatTensor(3,100,100)
    image.random_(0,255)

    img = normalize(image)

    assert torch.mean(img[:]).item() == pytest.approx(0,abs=1e-6)
    assert torch.std(img[:]).item() == pytest.approx(1,rel=1e-4)

def test_PairedImages_size():
    train_data = PairedImages('../../data/processed/veins/test/')

    real,target = train_data.__getitem__(1)

    assert real.size() == torch.Size([3,572,572])
    assert target.size() == torch.Size([388,388])
