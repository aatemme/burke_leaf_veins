from pathlib import Path
from os.path import join
from os import makedirs
import torch
from dataloaders.PairedImages import normalize, cv_split
from dataloaders import PairedImages
import pytest

def create_fake_files(tmp_path, N=5):
    '''
        Args:
            N (int): number of files

        Returns:
            Base path to files
    '''

    makedirs(join(tmp_path,'target'))

    files = []
    for i in range(N):
        file_path = join(tmp_path,'target',str(i) + ".png")
        Path(file_path).touch()
        files.append(file_path)

    return tmp_path

def test_normalize():
    image = torch.FloatTensor(3,100,100)
    image.random_(0,255)

    img = normalize(image)

    assert torch.mean(img[:]).item() == pytest.approx(0,abs=1e-6)
    assert torch.std(img[:]).item() == pytest.approx(1,rel=1e-4)

def test_cv_split_size():
    data = list(range(101))

    train_data, test_data = cv_split(data,10,0)

    assert len(test_data) == 11
    assert len(train_data) == 101-11

    train_data, test_data = cv_split(data,10,2)

    assert len(test_data) == 10
    assert len(train_data) == 101-10

    train_data, test_data = cv_split(data,10,9)

    assert len(test_data) == 10
    assert len(train_data) == 101-10

def test_cv_split_values():
    data = list(range(101))

    train_data, test_data = cv_split(data,10,1)

    assert test_data == list(range(11,21))
    print(list(range(10)) + list(range(21,101)))
    print(train_data)
    assert train_data == (list(range(11)) + list(range(21,101)))

def test_cv_split_input_checks():
    data = list(range(101))

    with pytest.raises(ValueError) as e:
        train_data, test_data = cv_split(data,10,11)

    with pytest.raises(ValueError) as e:
        train_data, test_data = cv_split(data,10,10)

def test_PairedImages_init(tmp_path):
    dataloader = PairedImages(create_fake_files(tmp_path,10))

    assert len(dataloader) == 10

def test_PairedImages_init_cv(tmp_path):
    files = create_fake_files(tmp_path,101)

    dataloader = PairedImages(files,
                              cv=10,
                              n_split=0)

    assert len(dataloader) == 101-11

    dataloader = PairedImages(files,
                              cv=10,
                              n_split=1)

    assert len(dataloader) == 101-10

    dataloader = PairedImages(files,
                              cv=10,
                              n_split=0,
                              cv_test=True)

    assert len(dataloader) == 11

# def test_PairedImages_size():
#     train_data = PairedImages('../../data/processed/veins/test/')
#
#     real,target = train_data.__getitem__(1)
#
#     assert real.size() == torch.Size([3,572,572])
#     assert target.size() == torch.Size([388,388])
