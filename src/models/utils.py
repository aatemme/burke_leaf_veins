# pylint: disable=missing-docstring, invalid-name, bad-continuation
import os
import argparse
import scipy.stats
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch import nn
from pytorch_utils import init
from dataloaders import PairedImages

def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=16, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--epochs', type=int, default=1e7, metavar='N',
                        help='number of epochs to train (default: 1e7)')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--save-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before checkpointing')
    parser.add_argument('--resume', action='store_true', default=False,
                        help='resume training from checkpoint')
    parser.add_argument('--comment', default='',
                        help="Comment to pass to tensorbaordX")
    parser.add_argument('--logdir', default=None,
                        help="Where to save log data")
    parser.add_argument('--dataset', default="VEINS_100", type=str,
                        help="Data set to train on (MNIST or SVHN) default: MNIST")
    return  parser.parse_args()

def VEINS_100_loaders(args):
    train_data = PairedImages('../../data/processed/veins/')

    train_loader = DataLoader(train_data, batch_size=args.batch_size,
                              shuffle=True, num_workers=4, pin_memory=True)

    return train_loader, train_loader

def grad_norm(parameters, norm_type=2):
    parameters = list(filter(lambda p: p.grad is not None, parameters))

    norm_type = float(norm_type)
    total_norm = 0
    for p in parameters:
        param_norm = p.grad.data.norm(norm_type)
        total_norm += param_norm.item() ** norm_type
    total_norm = total_norm ** (1. / norm_type)

    return total_norm
