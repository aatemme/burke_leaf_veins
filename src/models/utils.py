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
import pickle

def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=1, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--epochs', type=int, default=1e7, metavar='N',
                        help='number of epochs to train (default: 1e7)')
    parser.add_argument('--lr', type=float, default=0.0001, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--save-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before checkpointing')
    parser.add_argument('--resume', type=str, default=None,
                        help='resume training from checkpoint [--resume /path/to/save/folder/]')
    parser.add_argument('--comment', default='',
                        help="Comment to pass to tensorbaordX")
    parser.add_argument('--logdir', default=None,
                        help="Where to save log data")
    parser.add_argument('--dataset', default="VEINS_100", type=str,
                        help="Data set to train on (MNIST or SVHN) default: MNIST")
    parser.add_argument('--augment', action='store_true', default=False,
                        help="Augment dataset")
    parser.add_argument('--weighted-ce', action='store_true', default=False,
                        help="Use weighted cross-entropy loss, default is unweighted")
    parser.add_argument('--cv', default=None, type=int,
                        help=("Setup dataloader for a n-way cross-validation,"
                        " where N is given as --cv N, used in conjunction with"
                        "--n_cv"))
    parser.add_argument('--n_cv', default=0, type=int,
                        help=("Train/Test on the n^th cross-validation set "
                        "where n is given as --n_cv n, used in conjunction with"
                        "--cv"))
    return  parser.parse_args()

def VEINS_100_loaders(args):
    train_data = PairedImages('../../data/processed/veins/',
                               augment = args.augment,
                               cv=args.cv,
                               n_split=args.n_cv)

    train_loader = DataLoader(train_data, batch_size=args.batch_size,
                              shuffle=True, num_workers=4, pin_memory=True)

    test_data = PairedImages('../../data/processed/veins/',
                              augment = False,
                              cv=args.cv,
                              n_split=args.n_cv,
                              cv_test=True)

    test_loader = DataLoader(test_data, batch_size=args.batch_size,
                              shuffle=True, num_workers=4, pin_memory=True)


    return train_loader, test_loader

def load_nets(dir,net):
    state_dict = torch.load(dir + '/Net.net')
    net.load_state_dict(state_dict)

    data = pickle.load(open(dir + '/TrainingLog.pkl','rb'))

    return data['args'], data['optimizers']['inference']

def grad_norm(parameters, norm_type=2):
    parameters = list(filter(lambda p: p.grad is not None, parameters))

    norm_type = float(norm_type)
    total_norm = 0
    for p in parameters:
        param_norm = p.grad.data.norm(norm_type)
        total_norm += param_norm.item() ** norm_type
    total_norm = total_norm ** (1. / norm_type)

    return total_norm
