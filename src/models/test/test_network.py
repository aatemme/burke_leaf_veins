import pytest
import torch
from pytorch_utils import tests

from fixtures import *
import models

def test_net_output_dim(batch_size):
    module = models.UNet()

    batch = torch.FloatTensor(batch_size,
                              3,
                              572,
                              572)

    x = module(batch)

    assert x.size() == torch.Size([batch_size,1,388,388])

def test_net_weights_learn(batch_size):
    x = torch.FloatTensor(batch_size,
                              3,
                              572,
                              572)
    x.normal_(0,1)
    
    ##Test
    net = models.UNet()
    tests.is_learning(net,x)
