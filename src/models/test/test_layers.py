import pytest
import torch
from pytorch_utils import tests

import layers
from fixtures import *

def test_down1_output_dim(batch_size):
    module = layers.Down1()

    batch = torch.FloatTensor(batch_size,
                              3,
                              572,
                              572)

    x,y = module(batch)

    assert x.size() == torch.Size([batch_size,layers.NGF,284,284])
    assert y.size() == torch.Size([batch_size,layers.NGF,392,392])

def test_down2_output_dim(batch_size):
    module = layers.Down2()

    batch = torch.FloatTensor(batch_size,
                              layers.NGF,
                              284,
                              284)

    x,y = module(batch)

    assert x.size() == torch.Size([batch_size,layers.NGF*2,140,140])
    assert y.size() == torch.Size([batch_size,layers.NGF*2,200,200])

def test_down3_output_dim(batch_size):
    module = layers.Down3()

    batch = torch.FloatTensor(batch_size,
                              layers.NGF*2,
                              140,
                              140)

    x,y = module(batch)

    assert x.size() == torch.Size([batch_size,layers.NGF*4,68,68])
    assert y.size() == torch.Size([batch_size,layers.NGF*4,104,104])

def test_down4_output_dim(batch_size):
    module = layers.Down4()

    batch = torch.FloatTensor(batch_size,
                              layers.NGF*4,
                              68,
                              68)

    x,y = module(batch)

    assert x.size() == torch.Size([batch_size,layers.NGF*8,32,32])
    assert y.size() == torch.Size([batch_size,layers.NGF*8,56,56])

def test_up5_output_dim(batch_size):
    module = layers.Up5()

    batch = torch.FloatTensor(batch_size,
                              layers.NGF*8,
                              32,
                              32)

    x = module(batch)

    assert x.size() == torch.Size([batch_size,layers.NGF*8,56,56])

def test_up4_output_dim(batch_size):
    module = layers.Up4()

    batch = torch.FloatTensor(batch_size,
                              layers.NGF*8,
                              56,
                              56)

    x = module(batch,batch)

    assert x.size() == torch.Size([batch_size,layers.NGF*4,104,104])

def test_up3_output_dim(batch_size):
    module = layers.Up3()

    batch = torch.FloatTensor(batch_size,
                              layers.NGF*4,
                              56,
                              56)

    x = module(batch,batch)

    assert x.size() == torch.Size([batch_size,layers.NGF*2,104,104])

def test_up2_output_dim(batch_size):
    module = layers.Up2()

    batch = torch.FloatTensor(batch_size,
                              layers.NGF*2,
                              200,
                              200)

    x = module(batch,batch)

    assert x.size() == torch.Size([batch_size,layers.NGF,392,392])


def test_up1_output_dim(batch_size):
    module = layers.Up1()

    batch = torch.FloatTensor(batch_size,
                              layers.NGF,
                              392,
                              392)

    x = module(batch,batch)

    assert x.size() == torch.Size([batch_size,1,388,388])
