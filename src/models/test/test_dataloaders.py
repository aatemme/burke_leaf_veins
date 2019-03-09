import torch
from dataloaders.PairedImages import normalize
import pytest

def test_normalize():
    image = torch.FloatTensor(3,100,100)
    image.random_(0,255)

    img = normalize(image)

    assert torch.mean(img[:]).item() == pytest.approx(0,abs=1e-6)
    assert torch.std(img[:]).item() == pytest.approx(1,rel=1e-4)
